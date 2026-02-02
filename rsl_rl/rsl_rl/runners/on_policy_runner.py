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
    return total_bytes / (1024 ** 2), total_bytes / (1024 ** 3)  # 返回 MB 和 GB

class OnPolicyRunner:
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # check if multi-gpu is enabled
        self._configure_multi_gpu()

        # resolve training type depending on the algorithm
        if self.alg_cfg["class_name"] == "PPO":
            self.training_type = "rl"
        elif self.alg_cfg["class_name"] == "Distillation":
            self.training_type = "distillation"
        else:
            raise ValueError(f"Training type not found for algorithm {self.alg_cfg['class_name']}.")

        # resolve dimensions of observations
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]

        # resolve type of privileged observations
        if self.training_type == "rl":
            if "critic" in extras["observations"]:
                self.privileged_obs_type = "critic"  # actor-critic reinforcement learnig, e.g., PPO
            else:
                self.privileged_obs_type = None
        if self.training_type == "distillation":
            if "teacher" in extras["observations"]:
                self.privileged_obs_type = "teacher"  # policy distillation
            else:
                self.privileged_obs_type = None

        # resolve dimensions of privileged observations
        if self.privileged_obs_type is not None:
            num_privileged_obs = extras["observations"][self.privileged_obs_type].shape[1]
        else:
            num_privileged_obs = num_obs

        # evaluate the policy class
        policy_class = eval(self.policy_cfg.pop("class_name"))
        policy: ActorCritic | ActorCriticRecurrent | ActorCriticMLP | ActorCriticRNN | StudentTeacher | StudentTeacherRecurrent | StudentTeacherRecurrentCustom = policy_class(
            num_obs, num_privileged_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # resolve dimension of rnd gated state
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            # check if rnd gated state is present
            rnd_state = extras["observations"].get("rnd_state")
            if rnd_state is None:
                raise ValueError("Observations for the key 'rnd_state' not found in infos['observations'].")
            # get dimension of rnd gated state
            num_rnd_state = rnd_state.shape[1]
            # add rnd gated state to config
            self.alg_cfg["rnd_cfg"]["num_states"] = num_rnd_state
            # scale down the rnd weight with timestep (similar to how rewards are scaled down in legged_gym envs)
            self.alg_cfg["rnd_cfg"]["weight"] *= env.unwrapped.step_dt

        # if using symmetry then pass the environment config object
        if "symmetry_cfg" in self.alg_cfg and self.alg_cfg["symmetry_cfg"] is not None:
            # this is used by the symmetry function for handling different observation terms
            self.alg_cfg["symmetry_cfg"]["_env"] = env

        # initialize algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        self.alg: PPO | Distillation = alg_class(
            policy, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )

        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(
                self.device
            )
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization

        # init storage and model
        self.alg.init_storage(
            self.training_type,
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_privileged_obs],
            [self.env.num_actions],
        )

        # Decide whether to disable logging
        # We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]
    
    ##绝对式
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False, init_reward: float = None):  # noqa: C901
        
        # =================================================================================
        # [Adaptive Loop Config] 
        # =================================================================================
        # 只有传入了有效的 init_reward 才开启自适应
        use_adaptive = (init_reward is not None and init_reward > 1.0) 
        
        # 1. 读取基础参数 (Base Values) - 作为计算的锚点
        # 注意：必须深拷贝或直接读取数值，防止被后续修改覆盖
        if self.alg_cfg is not None:
             base_lr = float(self.alg_cfg.get("learning_rate", 1e-3))
             base_clip = float(self.alg_cfg.get("clip_param", 0.2))
        else:
             # Fallback
             base_lr = 1e-3
             base_clip = 0.2

        # 2. 定义敏感度系数 (Coefficients) - 对应论文中的 c 参数
        # 这些系数决定了"进步一点点，参数变多少"
        c_actor = 1.0     # 策略进步越快，学习率加得越快
        c_critic = 1.0    # 策略进步越快，Critic 越需要稳 (LR减小)
        c_clip = 0.2      # 策略越稳，允许的 Clip 范围越大

        # 3. 定义安全边界 (Safety Bounds)
        lr_max = 0.005        # Actor LR 上限 (防止梯度爆炸)
        lr_min_critic = 1e-6  # Critic LR 下限 (防止停止学习)
        clip_max = 0.4        # Clip 上限

        if use_adaptive:
            print(f"\n[{'='*30}]")
            print(f"[INFO] Adaptive Loop: ENABLED")
            print(f"[INFO] Base Reward (r_init): {init_reward:.4f}")
            print(f"[INFO] Base LR: {base_lr} | Base Clip: {base_clip}")
            print(f"[{'='*30}]\n")
        else:
            print(f"[INFO] Adaptive Loop: DISABLED (Using fixed LR={base_lr})")

        # =================================================================================

        # Initialize best mean reward for tracking the best model
        best_mean_reward = float('-inf')
        best_model_path = os.path.join(self.log_dir, "best_model.pt") if self.log_dir is not None else None
        
        # initialize writer
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        # check if teacher is loaded
        if self.training_type == "distillation" and not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs, extras = self.env.get_observations()
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # create buffers for logging extrinsic and intrinsic rewards
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    actions = self.alg.act(obs, privileged_obs)
                    # Step the environment
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # perform normalization
                    obs = self.obs_normalizer(obs)
                    if self.privileged_obs_type is not None:
                        privileged_obs = self.privileged_obs_normalizer(
                            infos["observations"][self.privileged_obs_type].to(self.device)
                        )
                    else:
                        privileged_obs = obs

                    # process the step
                    self.alg.process_env_step(rewards, dones, infos)

                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    # book keeping
                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        # Update rewards
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        # -- common
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        # -- intrinsic and extrinsic rewards
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # compute returns
                if self.training_type == "rl":
                    self.alg.compute_returns(privileged_obs)

            print(f"Rollout storage size: {calculate_rollout_storage_size(self.alg.storage)} MB, {calculate_rollout_storage_size(self.alg.storage)[1]} GB")
            
            # update policy
            loss_dict = self.alg.update()

            # =================================================================================
            # [Core Logic] Phase III: Adaptive Update (Run Every Iteration)
            # =================================================================================
            # 默认值 (用于 Log)
            current_alpha = 1.0
            current_actor_lr = base_lr
            current_critic_lr = base_lr
            current_clip = base_clip

            if use_adaptive and len(rewbuffer) > 0:
                # 1. 计算当前的性能 (r_rollout)
                # 使用 rewbuffer 的均值可以有效平滑波动，比单次 iteration 的均值更稳定
                r_rollout = statistics.mean(rewbuffer)
                
                # 2. 计算比率 alpha
                current_alpha = r_rollout / init_reward
                
                # 3. 计算相对进步幅度 Delta (只在进步时触发)
                # delta = max(alpha - 1, 0)
                delta = max(current_alpha - 1.0, 0.0)
                
                # 4. 计算新的超参数 (Formulas)
                
                # Formula A: Actor LR (Linear Increase)
                # LR_pi = Base * (1 + c * delta)
                target_actor_lr = base_lr * (1.0 + c_actor * delta)
                
                # Formula B: Critic LR (Inverse Decay)
                # LR_v = Base / (1 + c * delta)
                target_critic_lr = base_lr / (1.0 + c_critic * delta)
                
                # Formula C: Clip Range (Linear Increase)
                # Epsilon = Base * (1 + c * delta)
                target_clip = base_clip * (1.0 + c_clip * delta)
                
                # 5. 安全截断 (Bounds)
                current_actor_lr = min(target_actor_lr, lr_max)       # 上限截断
                current_critic_lr = max(target_critic_lr, lr_min_critic) # 下限截断
                current_clip = min(target_clip, clip_max)             # 上限截断
                
                # 6. 执行修改 (Call PPO Interface)
                if hasattr(self.alg, 'update_hyperparameters'):
                    self.alg.update_hyperparameters(current_actor_lr, current_critic_lr, current_clip)

            # =================================================================================

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                
                # [NEW] Log Adaptive Parameters to WandB/Tensorboard
                if use_adaptive:
                    self.writer.add_scalar("Adaptive/Alpha", current_alpha, it)
                    self.writer.add_scalar("Adaptive/Delta", max(current_alpha - 1.0, 0.0), it)
                    self.writer.add_scalar("Adaptive/Actor_LR", current_actor_lr, it)
                    self.writer.add_scalar("Adaptive/Critic_LR", current_critic_lr, it)
                    self.writer.add_scalar("Adaptive/Clip_Param", current_clip, it)

                # Save model
                if len(rewbuffer) > 0:
                    current_mean_reward = statistics.mean(rewbuffer)
                    if current_mean_reward > best_mean_reward:
                        best_mean_reward = current_mean_reward
                        self.save(best_model_path)
                        print(f"New best model saved with mean_reward: {best_mean_reward:.4f} at iteration {it}")

                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

        # 返回最后阶段的平均奖励
        if len(rewbuffer) > 0:
            return statistics.mean(rewbuffer)
        else:
            return 0.0

    ##增量式
    # def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False, init_reward: float = None):  # noqa: C901
        
    #     # =================================================================================
    #     # [Adaptive Loop Config] 
    #     # =================================================================================
    #     use_adaptive = (init_reward is not None and init_reward > 1.0) 
        
    #     # 1. 基础参数 (Base Values)
    #     if self.alg_cfg is not None:
    #          base_lr = float(self.alg_cfg.get("learning_rate", 1e-3))
    #          base_clip = float(self.alg_cfg.get("clip_param", 0.2))
    #     else:
    #          base_lr = 1e-3
    #          base_clip = 0.2

    #     # 2. 状态变量初始化 (Stateful Initialization)
    #     # [关键] 这里初始化为 Base，后面会在此基础上累积变化
    #     current_actor_lr = base_lr
    #     current_critic_lr = base_lr
    #     current_clip = base_clip

    #     # 3. 敏感度系数 (Coefficients) - 论文中的 "Step Size"
    #     # 注意：在累积模式下，这些系数要小一点，否则几次迭代就爆炸了
    #     c_actor = 1e-5     # 每次进步，Actor LR 增加的幅度
    #     c_critic = 1e-5    # 每次进步，Critic LR 减少的幅度
    #     c_clip = 1e-3      # 每次进步，Clip 增加的幅度

    #     # 4. 安全边界 (Safety Bounds)
    #     lr_max = 0.005         # Actor LR 上限
    #     lr_min_critic = 1e-6   # Critic LR 下限
    #     clip_max = 0.4         # Clip 上限

    #     if use_adaptive:
    #         print(f"\n[{'='*30}]")
    #         print(f"[INFO] Adaptive Loop: ENABLED (Cumulative/Ratchet Mode)")
    #         print(f"[INFO] Base Reward (r_init): {init_reward:.4f}")
    #         print(f"[INFO] Initial Params: LR={base_lr} | Clip={base_clip}")
    #         print(f"[{'='*30}]\n")
    #     else:
    #         print(f"[INFO] Adaptive Loop: DISABLED (Using fixed LR={base_lr})")

    #     # =================================================================================

    #     best_mean_reward = float('-inf')
    #     best_model_path = os.path.join(self.log_dir, "best_model.pt") if self.log_dir is not None else None
        
    #     # initialize writer
    #     if self.log_dir is not None and self.writer is None and not self.disable_logs:
    #         self.logger_type = self.cfg.get("logger", "tensorboard")
    #         self.logger_type = self.logger_type.lower()
    #         if self.logger_type == "neptune":
    #             from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter
    #             self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
    #             self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
    #         elif self.logger_type == "wandb":
    #             from rsl_rl.utils.wandb_utils import WandbSummaryWriter
    #             self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
    #             self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
    #         elif self.logger_type == "tensorboard":
    #             from torch.utils.tensorboard import SummaryWriter
    #             self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

    #     if self.training_type == "distillation" and not self.alg.policy.loaded_teacher:
    #         raise ValueError("Teacher model parameters not loaded.")

    #     if init_at_random_ep_len:
    #         self.env.episode_length_buf = torch.randint_like(
    #             self.env.episode_length_buf, high=int(self.env.max_episode_length)
    #         )

    #     obs, extras = self.env.get_observations()
    #     privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
    #     obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
    #     self.train_mode()

    #     ep_infos = []
    #     rewbuffer = deque(maxlen=100)
    #     lenbuffer = deque(maxlen=100)
    #     cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
    #     cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

    #     if self.alg.rnd:
    #         erewbuffer = deque(maxlen=100)
    #         irewbuffer = deque(maxlen=100)
    #         cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
    #         cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

    #     if self.is_distributed:
    #         self.alg.broadcast_parameters()

    #     start_iter = self.current_learning_iteration
    #     tot_iter = start_iter + num_learning_iterations
        
    #     for it in range(start_iter, tot_iter):
    #         start = time.time()
    #         # Rollout
    #         with torch.inference_mode():
    #             for _ in range(self.num_steps_per_env):
    #                 actions = self.alg.act(obs, privileged_obs)
    #                 obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
    #                 obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
    #                 obs = self.obs_normalizer(obs)
    #                 if self.privileged_obs_type is not None:
    #                     privileged_obs = self.privileged_obs_normalizer(
    #                         infos["observations"][self.privileged_obs_type].to(self.device)
    #                     )
    #                 else:
    #                     privileged_obs = obs
    #                 self.alg.process_env_step(rewards, dones, infos)
                    
    #                 if self.alg.rnd:
    #                     intrinsic_rewards = self.alg.intrinsic_rewards
                    
    #                 if self.log_dir is not None:
    #                     if "episode" in infos: ep_infos.append(infos["episode"])
    #                     elif "log" in infos: ep_infos.append(infos["log"])
                        
    #                     if self.alg.rnd:
    #                         cur_ereward_sum += rewards
    #                         cur_ireward_sum += intrinsic_rewards
    #                         cur_reward_sum += rewards + intrinsic_rewards
    #                     else:
    #                         cur_reward_sum += rewards
    #                     cur_episode_length += 1
                        
    #                     new_ids = (dones > 0).nonzero(as_tuple=False)
    #                     rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
    #                     lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
    #                     cur_reward_sum[new_ids] = 0
    #                     cur_episode_length[new_ids] = 0
                        
    #                     if self.alg.rnd:
    #                         erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
    #                         irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
    #                         cur_ereward_sum[new_ids] = 0
    #                         cur_ireward_sum[new_ids] = 0

    #             stop = time.time()
    #             collection_time = stop - start
    #             start = stop

    #             if self.training_type == "rl":
    #                 self.alg.compute_returns(privileged_obs)

    #         print(f"Rollout storage size: {calculate_rollout_storage_size(self.alg.storage)} MB")
            
    #         # update policy
    #         loss_dict = self.alg.update()

    #         # =================================================================================
    #         # [Core Logic] Phase III: Adaptive Update (Cumulative / Ratchet)
    #         # =================================================================================
    #         current_alpha = 1.0
            
    #         # 至少积累一定数据再调整
    #         if use_adaptive and len(rewbuffer) >= 10:
    #             # 1. 计算性能比率
    #             r_rollout = statistics.mean(rewbuffer)
    #             current_alpha = r_rollout / init_reward
                
    #             # 2. 计算相对进步幅度
    #             delta = max(current_alpha - 1.0, 0.0)
                
    #             if delta > 0:
    #                 # 3. 累积更新 (Cumulative Update) - 这就是"记忆"
    #                 # 只有在进步时才更新，而且是在 current 值基础上加减
                    
    #                 # Actor: 增加
    #                 current_actor_lr += (delta * c_actor)
                    
    #                 # Critic: 减少
    #                 current_critic_lr -= (delta * c_critic)
                    
    #                 # Clip: 增加
    #                 current_clip += (delta * c_clip)
                    
    #                 # 4. 严格边界限制 (Hard Clamping)
    #                 current_actor_lr = min(current_actor_lr, lr_max)
    #                 current_critic_lr = max(current_critic_lr, lr_min_critic)
    #                 current_clip = min(current_clip, clip_max)
                
    #             # 5. 执行修改
    #             if hasattr(self.alg, 'update_hyperparameters'):
    #                 self.alg.update_hyperparameters(current_actor_lr, current_critic_lr, current_clip)

    #         # =================================================================================

    #         stop = time.time()
    #         learn_time = stop - start
    #         self.current_learning_iteration = it
            
    #         if self.log_dir is not None and not self.disable_logs:
    #             self.log(locals())
                
    #             if use_adaptive:
    #                 self.writer.add_scalar("Adaptive/Alpha", current_alpha, it)
    #                 self.writer.add_scalar("Adaptive/Delta", max(current_alpha - 1.0, 0.0), it)
    #                 self.writer.add_scalar("Adaptive/Actor_LR", current_actor_lr, it)
    #                 self.writer.add_scalar("Adaptive/Critic_LR", current_critic_lr, it)
    #                 self.writer.add_scalar("Adaptive/Clip_Param", current_clip, it)

    #             if len(rewbuffer) > 0:
    #                 current_mean_reward = statistics.mean(rewbuffer)
    #                 if current_mean_reward > best_mean_reward:
    #                     best_mean_reward = current_mean_reward
    #                     self.save(best_model_path)
    #                     print(f"New best model saved with mean_reward: {best_mean_reward:.4f} at iteration {it}")

    #             if it % self.save_interval == 0:
    #                 self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

    #         ep_infos.clear()
    #         if it == start_iter and not self.disable_logs:
    #             git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
    #             if self.logger_type in ["wandb", "neptune"] and git_file_paths:
    #                 for path in git_file_paths:
    #                     self.writer.save_file(path)

    #     if self.log_dir is not None and not self.disable_logs:
    #         self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    #     if len(rewbuffer) > 0:
    #         return statistics.mean(rewbuffer)
    #     else:
    #         return 0.0

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # -- Episode info
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # -- Losses
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        # -- Policy
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # -- Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # -- Distillation phase info (for two-stage training)
        if self.training_type == "distillation" and hasattr(self.alg, 'get_training_phase_info'):
            phase_info = self.alg.get_training_phase_info()
            if phase_info["use_two_stage_training"]:
                self.writer.add_scalar("Distillation/training_phase", phase_info["training_phase"], locs["it"])
                self.writer.add_scalar("Distillation/current_iteration", phase_info["current_iteration"], locs["it"])

        # -- Training
        if len(locs["rewbuffer"]) > 0:
            # separate logging for intrinsic and extrinsic rewards
            if self.alg.rnd:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])
            # everything else
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            # -- Losses
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            # -- Rewards
            if self.alg.rnd:
                log_string += (
                    f"""{'Mean extrinsic reward:':>{pad}} {statistics.mean(locs['erewbuffer']):.2f}\n"""
                    f"""{'Mean intrinsic reward:':>{pad}} {statistics.mean(locs['irewbuffer']):.2f}\n"""
                )
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            # -- episode info
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        log_string += ep_string
        
        # Add distillation phase info for two-stage training
        if self.training_type == "distillation" and hasattr(self.alg, 'get_training_phase_info'):
            phase_info = self.alg.get_training_phase_info()
            if phase_info["use_two_stage_training"]:
                log_string += (
                    f"""{'-' * width}\n"""
                    f"""{'Training Phase:':>{pad}} Phase {phase_info['training_phase']} ({phase_info['action_source']} actions)\n"""
                    f"""{'Phase Iteration:':>{pad}} {phase_info['current_iteration']}/{phase_info['phase1_iterations']} (Phase 1)\n"""
                )
        
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime(
                "%H:%M:%S",
                time.gmtime(
                    self.tot_time / (locs['it'] - locs['start_iter'] + 1)
                    * (locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])
                )
            )}\n"""
        )
        print(log_string)

    def save(self, path: str, infos=None):
        # -- Save model
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # -- Save RND model if used
        if self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        # -- Save observation normalizer if used
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()

        # save model
        torch.save(saved_dict, path)

        # upload model to external logging service
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, weights_only=False)
        # -- Load model
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        # -- Load RND model if used
        if self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        # -- Load observation normalizer if used
        if self.empirical_normalization:
            if resumed_training:
                # if a previous training is resumed, the actor/student normalizer is loaded for the actor/student
                # and the critic/teacher normalizer is loaded for the critic/teacher
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
            else:
                # if the training is not resumed but a model is loaded, this run must be distillation training following
                # an rl training. Thus the actor normalizer is loaded for the teacher model. The student's normalizer
                # is not loaded, as the observation space could differ from the previous rl training.
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        # -- load optimizer if used
        if load_optimizer and resumed_training:
            # -- algorithm optimizer
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            # -- RND optimizer if used
            if self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        # -- load current learning iteration
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]
    
    def load_std(self, init_noise_std: float):
        if self.alg.policy.noise_std_type == "scalar":
            self.alg.policy.std.data.fill_(init_noise_std)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.alg.policy.noise_std_type}. Should be 'scalar' or 'log'")

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.policy.to(device)
        policy = self.alg.policy.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.policy.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy

    def train_mode(self):
        # -- PPO
        self.alg.policy.train()
        # -- RND
        if self.alg.rnd:
            self.alg.rnd.train()
        # -- Normalization
        if self.empirical_normalization:
            self.obs_normalizer.train() # Student 的 normalizer 可以更新
            
            # [修改] 如果是蒸馏任务，Teacher 的 Normalizer 必须冻结(eval模式)！
            if self.training_type == "distillation":
                self.privileged_obs_normalizer.eval() 
            else:
                self.privileged_obs_normalizer.train()

    def eval_mode(self):
        # -- PPO
        self.alg.policy.eval()
        # -- RND
        if self.alg.rnd:
            self.alg.rnd.eval()
        # -- Normalization
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    """
    Helper functions.
    """

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
