# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque

import rsl_rl
from rsl_rl.algorithms import PPO, Distillation
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    ActorCriticMLP,
    ActorCriticRNN,
    ActorCriticAtten,
    EmpiricalNormalization,
    StudentTeacher,
    StudentTeacherRecurrent,
    StudentTeacherRecurrentCustom,
)
from rsl_rl.utils import store_code_state


def calculate_rollout_storage_size(storage):
    total_bytes = 0
    # 遍历所有属性
    for name, value in vars(storage).items():
        if torch.is_tensor(value):
            total_bytes += value.numel() * value.element_size()
        elif isinstance(value, list):
            for v in value:
                if torch.is_tensor(v):
                    total_bytes += v.numel() * v.element_size()
    return total_bytes / (1024**2), total_bytes / (1024**3)  # 返回 MB 和 GB


class OnPolicyRunner:
    """On-policy runner for training and evaluation.
    
    Modified to support: Bootstrapping Reinforcement Learning with Imitation for Vision-Based Agile Flight
    Implements Algorithm 1 flow control.
    """

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        # [修改] 1. 兼容性配置读取
        if "runner" in train_cfg:
            self.cfg = train_cfg["runner"]
        else:
            self.cfg = train_cfg
            
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        
        # [新增] 初始化 git repos 列表
        self.git_status_repos = []

        # [新增] 配置多 GPU
        self._configure_multi_gpu()

        # [修复] 只有当 num_privileged_obs > 0 时才认为有特权观测
        if self.env.num_privileged_obs is not None and self.env.num_privileged_obs > 0:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
            
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))  # ActorCritic
        actor_critic: ActorCritic = actor_critic_class(
            self.env.num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)
        
        # [修改] 2. 提前读取 num_steps_per_env
        self.num_steps_per_env = self.cfg["num_steps_per_env"]

        # PPO 算法初始化
        alg_class = eval(self.cfg.pop("algorithm_class_name", "PPO"))
        
        # [修改] 3. 准备参数副本，并注入缺失的存储参数
        ppo_args = self.alg_cfg.copy()
        
        # (A) 移除不必要参数
        if "class_name" in ppo_args: ppo_args.pop("class_name")
        if "critic_warmup_iterations" in ppo_args: ppo_args.pop("critic_warmup_iterations")
        
        # (B) 注入 PPO.__init__ 必需的存储参数
        ppo_args["num_transitions_per_env"] = self.num_steps_per_env
        ppo_args["obs_shape"] = [self.env.num_obs]
        ppo_args["privileged_obs_shape"] = [num_critic_obs]
        ppo_args["actions_shape"] = [self.env.num_actions]

        self.alg: PPO = alg_class(actor_critic, device=self.device, **ppo_args)
        
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg.get("empirical_normalization", False)
        
        if self.empirical_normalization:
            # [修复] 删除不支持的 layout 和 device 参数，并手动 to(device)
            self.obs_normalizer = EmpiricalNormalization(shape=[self.env.num_obs])
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs])
            
            self.obs_normalizer.to(self.device)
            self.critic_obs_normalizer.to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity()
            self.critic_obs_normalizer = torch.nn.Identity()
            
        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions],
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

    # [辅助函数] 安全获取特权观测
    def _get_privileged_obs_safe(self):
        if hasattr(self.env, "get_privileged_observations"):
            return self.env.get_privileged_observations()
        return None

    # [辅助函数] 递归移动数据到 Device
    def _to_device(self, data):
        if data is None:
            return None
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, dict):
            return {k: self._to_device(v) for k, v in data.items()}
        elif isinstance(data, (tuple, list)):
            return type(data)(self._to_device(v) for v in data)
        return data

    # [辅助函数] 智能归一化处理
    def _apply_norm(self, obs, normalizer):
        if isinstance(obs, torch.Tensor):
            return normalizer(obs)
        elif isinstance(obs, (tuple, list)):
            normed_first = normalizer(obs[0])
            return (normed_first, *obs[1:])
        elif isinstance(obs, dict):
            if "obs" in obs:
                obs["obs"] = normalizer(obs["obs"])
            return obs
        else:
            return obs

    # [新增关键函数] 将 Tuple/List 展平成单个 Tensor
    def _flatten_obs(self, obs):
        """
        如果观测是 Tuple/List (例如 [proprio, latent])，将其在最后一维拼接。
        """
        if isinstance(obs, (tuple, list)):
            # 假设所有元素都是 Tensor，且第一维是 Batch，在 dim=-1 拼接
            return torch.cat(obs, dim=-1)
        return obs

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        
        obs = self.env.get_observations()
        privileged_obs = self._get_privileged_obs_safe()
        
        critic_obs = privileged_obs if privileged_obs is not None else obs
        
        obs = self._to_device(obs)
        critic_obs = self._to_device(critic_obs)
        
        self.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        start_iter = self.current_learning_iteration

        # =====================================================================================
        # [Step 1] Initial Reward Collection (r_init)
        # =====================================================================================
        print(f"[Adaptive] Collecting initial reward (r_init) to establish baseline...")
        
        init_steps_collected = 0
        steps_to_collect = 2000 
        
        self.eval_mode() 
        
        with torch.inference_mode():
            while init_steps_collected < steps_to_collect:
                # 1. 归一化 (可能返回 Tuple)
                normed_obs = self._apply_norm(obs, self.obs_normalizer)
                normed_critic_obs = self._apply_norm(critic_obs, self.critic_obs_normalizer)
                
                # 2. [关键修复] 拼接成 Tensor
                flat_obs = self._flatten_obs(normed_obs)
                flat_critic_obs = self._flatten_obs(normed_critic_obs)
                
                actions = self.alg.act(flat_obs, flat_critic_obs)
                
                obs, rewards, dones, infos = self.env.step(actions)
                obs = self._to_device(obs)
                
                privileged_obs = self._get_privileged_obs_safe()
                critic_obs = privileged_obs if privileged_obs is not None else obs
                critic_obs = self._to_device(critic_obs)
                
                rewards, dones = rewards.to(self.device), dones.to(self.device)

                cur_reward_sum += rewards
                cur_episode_length += 1
                new_ids = (dones > 0).nonzero(as_tuple=False)
                cur_reward_sum[new_ids] = 0
                cur_episode_length[new_ids] = 0
                
                if "log" in infos:
                    if "episode" in infos["log"]:
                        rewbuffer.extend(infos["log"]["episode"]["r"])
                        lenbuffer.extend(infos["log"]["episode"]["l"])

                init_steps_collected += self.env.num_envs

        if len(rewbuffer) > 0:
            r_init = statistics.mean(rewbuffer)
        else:
            r_init = 1.0 
            print("[Adaptive] Warning: Failed to collect completed episodes for r_init. Defaulting to 1.0.")
            
        print(f"[Adaptive] r_init collected: {r_init:.4f}")
        
        self.alg.storage.clear()
        
        # =====================================================================================
        # [Step 2] Critic Warm-up
        # =====================================================================================
        critic_warmup_iterations = self.alg_cfg.get("critic_warmup_iterations", 0)
        
        self.train_mode()
        
        if critic_warmup_iterations > 0:
            print(f"[Adaptive] Starting Critic Warm-up for {critic_warmup_iterations} iterations...")
            
            for it in range(critic_warmup_iterations):
                start = time.time()
                with torch.inference_mode():
                    for i in range(self.num_steps_per_env):
                        # 1. 归一化
                        normed_obs = self._apply_norm(obs, self.obs_normalizer)
                        normed_critic_obs = self._apply_norm(critic_obs, self.critic_obs_normalizer)
                        
                        # 2. [关键修复] 拼接
                        flat_obs = self._flatten_obs(normed_obs)
                        flat_critic_obs = self._flatten_obs(normed_critic_obs)
                        
                        actions = self.alg.act(flat_obs, flat_critic_obs)
                        
                        obs, rewards, dones, infos = self.env.step(actions)
                        obs = self._to_device(obs)
                        
                        privileged_obs = self._get_privileged_obs_safe()
                        critic_obs = privileged_obs if privileged_obs is not None else obs
                        critic_obs = self._to_device(critic_obs)
                        
                        rewards, dones = rewards.to(self.device), dones.to(self.device)
                        self.alg.process_env_step(rewards, dones, infos)
                        
                        if "log" in infos:
                            if "episode" in infos["log"]:
                                rewbuffer.extend(infos["log"]["episode"]["r"])
                                lenbuffer.extend(infos["log"]["episode"]["l"])

                stop = time.time()
                collection_time = stop - start
                
                start = stop
                
                # [关键修复] 计算 Returns 时也需要拼接后的 Critic Obs
                normed_critic_obs = self._apply_norm(critic_obs, self.critic_obs_normalizer)
                flat_critic_obs = self._flatten_obs(normed_critic_obs)
                self.alg.compute_returns(flat_critic_obs)
                
                loss_dict = self.alg.update(update_actor=False)
                
                stop = time.time()
                learn_time = stop - start
                
                print(f"[Warmup] Iter {it+1}/{critic_warmup_iterations} | Value Loss: {loss_dict['value_function']:.4f}")
                
                if self.log_dir is not None and self.writer is not None:
                    self.writer.add_scalar("Warmup/Value_Loss", loss_dict['value_function'], it)
        
        self.alg.storage.clear()

        # =====================================================================================
        # [Step 3] Main Adaptive Loop
        # =====================================================================================
        print(f"[Adaptive] Starting Main Training Loop...")
        self.train_mode() 
        
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    # 1. 归一化
                    normed_obs = self._apply_norm(obs, self.obs_normalizer)
                    normed_critic_obs = self._apply_norm(critic_obs, self.critic_obs_normalizer)
                    
                    # 2. [关键修复] 拼接
                    flat_obs = self._flatten_obs(normed_obs)
                    flat_critic_obs = self._flatten_obs(normed_critic_obs)
                    
                    actions = self.alg.act(flat_obs, flat_critic_obs)
                    
                    obs, rewards, dones, infos = self.env.step(actions)
                    obs = self._to_device(obs)
                    
                    privileged_obs = self._get_privileged_obs_safe()
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    critic_obs = self._to_device(critic_obs)
                    
                    rewards, dones = rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            if "episode" in infos["log"]:
                                rewbuffer.extend(infos["log"]["episode"]["r"])
                                lenbuffer.extend(infos["log"]["episode"]["l"])
                                cur_reward_sum += rewards
                                cur_episode_length += 1
                                new_ids = (dones > 0).nonzero(as_tuple=False)
                                cur_reward_sum[new_ids] = 0
                                cur_episode_length[new_ids] = 0

            stop = time.time()
            collection_time = stop - start

            start = stop
            # [关键修复] Compute returns 拼接
            normed_critic_obs = self._apply_norm(critic_obs, self.critic_obs_normalizer)
            flat_critic_obs = self._flatten_obs(normed_critic_obs)
            self.alg.compute_returns(flat_critic_obs)

            # [Algorithm 1] 计算 Performance Ratio (Alpha)
            if len(rewbuffer) > 0:
                current_avg_reward = statistics.mean(rewbuffer)
            else:
                current_avg_reward = 0.0
            
            if abs(r_init) > 1e-5:
                alpha = current_avg_reward / r_init
            else:
                alpha = 1.0

            loss_dict = self.alg.update(performance_ratio=alpha, update_actor=True)
            
            stop = time.time()
            learn_time = stop - start
            
            if self.log_dir is not None:
                self.log(locals())
                if self.writer is not None:
                    self.writer.add_scalar("Adaptive/Alpha", alpha, it)
                    self.writer.add_scalar("Adaptive/r_init", r_init, it)
                    self.writer.add_scalar("Adaptive/LR_Actor", loss_dict.get("param/lr_actor", 0), it)
                    self.writer.add_scalar("Adaptive/LR_Critic", loss_dict.get("param/lr_critic", 0), it)
                    self.writer.add_scalar("Adaptive/Clip_Range", loss_dict.get("param/clip_range", 0), it)

            if it < 2500:
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))
            else:
                if it % 5000 == 0:
                    self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))
            
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, "model_{}.pt".format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.policy.action_std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        self.writer.add_scalar("Loss/value_function", locs["loss_dict"]["value_function"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["loss_dict"]["surrogate"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.cfg.get("empirical_normalization", False):
                self.writer.add_scalar("Train/mean_obs_norm", self.obs_normalizer.mean_sq.mean().item(), locs["it"])
                self.writer.add_scalar("Train/mean_critic_obs_norm", self.critic_obs_normalizer.mean_sq.mean().item(), locs["it"])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['loss_dict']['value_function']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['loss_dict']['surrogate']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item() :.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            log_string += (
                f"""{'-' * width}\n"""
                f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
            )

        print(log_string)

    def save(self, path, infos=None):
        if self.empirical_normalization:
            torch.save(
                {
                    "model_state_dict": self.alg.policy.state_dict(),
                    "optimizer_state_dict": self.alg.optimizer.state_dict(),
                    "iter": self.current_learning_iteration,
                    "infos": infos,
                    "obs_norm_state_dict": self.obs_normalizer.state_dict(),
                    "critic_obs_norm_state_dict": self.critic_obs_normalizer.state_dict(),
                },
                path,
            )
        else:
            torch.save(
                {
                    "model_state_dict": self.alg.policy.state_dict(),
                    "optimizer_state_dict": self.alg.optimizer.state_dict(),
                    "iter": self.current_learning_iteration,
                    "infos": infos,
                },
                path,
            )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]
    def load_std(self, init_noise_std: float):
        if self.alg.policy.noise_std_type == "scalar":
            self.alg.policy.std.data.fill_(init_noise_std)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.alg.policy.noise_std_type}. Should be 'scalar' or 'log'")

    def get_inference_policy(self, device=None):
        self.alg.policy.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.policy.act_inference

    # Helper functions
    def train_mode(self):
        # -- PPO
        self.alg.policy.train()
        # -- RND
        if self.alg.rnd:
            self.alg.rnd.train()
        # -- Normalization
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()

    def eval_mode(self):
        # -- PPO
        self.alg.policy.eval()
        # -- RND
        if self.alg.rnd:
            self.alg.rnd.eval()
        # -- Normalization
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    def _configure_multi_gpu(self):
        """Configure multi-gpu training."""
        # check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # if not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # make a configuration dictionary
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # rank of the main process
            "local_rank": self.gpu_local_rank,  # rank of the current process
            "world_size": self.gpu_world_size,  # total number of processes
        }

        # check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # validate multi-gpu configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # initialize torch distributed
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        # set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)