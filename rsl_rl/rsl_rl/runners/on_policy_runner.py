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
    """计算并返回 Rollout 存储器占用的显存大小（以 MB 和 GB 为单位）。"""
    total_bytes = 0
    # 遍历存储器对象中的所有属性
    for name, value in vars(storage).items():
        # 如果属性是 PyTorch 张量，计算其占用字节数
        if torch.is_tensor(value):
            total_bytes += value.numel() * value.element_size()
        # 如果属性是列表，遍历列表中的张量并累加字节数
        elif isinstance(value, list):
            for v in value:
                if torch.is_tensor(v):
                    total_bytes += v.numel() * v.element_size()
    return total_bytes / (1024 ** 2), total_bytes / (1024 ** 3)  # 返回 MB 和 GB

class OnPolicyRunner:
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        """初始化运行器，接收环境、训练配置、日志路径和计算设备。"""
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # 检查是否启用了多 GPU 分布式训练，并进行相应配置
        self._configure_multi_gpu()

        # 根据算法配置中的 class_name 字段解析当前的训练类型（rl 或 distillation）
        if self.alg_cfg["class_name"] == "PPO":
            self.training_type = "rl"
        elif self.alg_cfg["class_name"] == "Distillation":
            self.training_type = "distillation"
        else:
            raise ValueError(f"Training type not found for algorithm {self.alg_cfg['class_name']}.")

        # 从环境中获取初始观测值和附加信息字典，解析普通观测的维度大小
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]

        # 根据训练类型确定特权观测（Privileged Observations）的键名类型
        if self.training_type == "rl":
            if "critic" in extras["observations"]:
                self.privileged_obs_type = "critic"  # PPO 强化学习中通常使用 critic 特权观测
            else:
                self.privileged_obs_type = None
        if self.training_type == "distillation":
            if "teacher" in extras["observations"]:
                self.privileged_obs_type = "teacher"  # 策略蒸馏任务中使用 teacher 特权观测
            else:
                self.privileged_obs_type = None

        # 如果存在特权观测类型，则获取其特征维度大小；否则维数与普通观测相同
        if self.privileged_obs_type is not None:
            num_privileged_obs = extras["observations"][self.privileged_obs_type].shape[1]
        else:
            num_privileged_obs = num_obs

        # 动态评估并实例化策略网络类（例如 StudentTeacherRecurrentCustom 或 ActorCritic）
        policy_class = eval(self.policy_cfg.pop("class_name"))
        policy: ActorCritic | ActorCriticRecurrent | ActorCriticMLP | ActorCriticRNN | StudentTeacher | StudentTeacherRecurrent | StudentTeacherRecurrentCustom = policy_class(
            num_obs, num_privileged_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # 如果算法配置中启用了 RND（随机网络蒸馏）模块，则解析其状态维度
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            # 检查 info 中是否存在 rnd_state 键
            rnd_state = extras["observations"].get("rnd_state")
            if rnd_state is None:
                raise ValueError("Observations for the key 'rnd_state' not found in infos['observations'].")
            # 获取 rnd gated 状态的维度
            num_rnd_state = rnd_state.shape[1]
            # 将其加入到 rnd 算法配置中
            self.alg_cfg["rnd_cfg"]["num_states"] = num_rnd_state
            # 结合仿真步长对 rnd 权重进行缩放
            self.alg_cfg["rnd_cfg"]["weight"] *= env.unwrapped.step_dt

        # 如果启用了对称性约束（Symmetry），则将环境配置对象传入对称配置中
        if "symmetry_cfg" in self.alg_cfg and self.alg_cfg["symmetry_cfg"] is not None:
            self.alg_cfg["symmetry_cfg"]["_env"] = env

        # 动态评估并实例化算法类（例如 PPO 或 Distillation）
        alg_class = eval(self.alg_cfg.pop("class_name"))
        self.alg: PPO | Distillation = alg_class(
            policy, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )

        # 存储基础训练超参数（单环境采样步数、保存间隔、是否启用经验归一化）
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            # 为普通观测和特权观测分别初始化经验归一化器
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(
                self.device
            )
        else:
            # 若不启用经验归一化，则使用恒等映射（不作任何处理）
            self.obs_normalizer = torch.nn.Identity().to(self.device)  
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)  

        # 初始化算法内部的 Rollout 存储器（Storage）
        self.alg.init_storage(
            self.training_type,
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_privileged_obs],
            [self.env.num_actions],
        )

        # 仅在分布式训练的 Rank 0 主进程上启用日志记录，其他 GPU 进程禁用
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        # 初始化日志记录相关变量
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        """执行主要的训练迭代循环函数。"""
        
        # 初始化用于跟踪历史最优模型的平均奖励和保存路径
        best_mean_reward = float('-inf')
        best_model_path = os.path.join(self.log_dir, "best_model.pt") if self.log_dir is not None else None

        
        # 初始化日志写入器（Writer）
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # 获取日志记录器类型，默认为 tensorboard
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

        # 针对蒸馏任务，检查策略网络是否已经成功加载了教师模型参数，若未加载则报错
        if self.training_type == "distillation" and not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        # 如果开启了随机初始回合长度，则对环境的步数缓冲区进行随机化赋值（用于增强探索性）
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # 获取当前环境的观测值和特权观测值，并将它们转移至计算设备上
        obs, extras = self.env.get_observations()
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
        self.train_mode()  # 切换到训练模式（例如启用 Dropout 等）

        # 初始化用于统计和记录数据的双端队列与张量
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # 如果使用了 RND 模块，则额外初始化内外侧奖励的统计缓冲区
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # 多 GPU 分布式训练时，确保所有进程之间的参数保持同步
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # 开始正式的迭代训练循环
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # 轨迹采集阶段 (Rollout Collection)
            with torch.inference_mode(): # 在推理模式下进行采样，不计算梯度以提升速度
                for _ in range(self.num_steps_per_env): # 每个环境循环采样的步数
                    # 算法根据普通观测和特权观测采样动作
                    actions = self.alg.act(obs, privileged_obs)
                    # 将动作输入环境执行，推进仿真，获取下一帧的观测、奖励、终止标志和附加信息
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    # 将环境返回的数据移动到指定的计算设备上
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # 对普通观测进行经验归一化
                    obs = self.obs_normalizer(obs)
                    # 对特权观测进行经验归一化（若存在）
                    if self.privileged_obs_type is not None:
                        privileged_obs = self.privileged_obs_normalizer(
                            infos["observations"][self.privileged_obs_type].to(self.device)
                        )
                    else:
                        privileged_obs = obs

                    # 将当前步的数据交由算法处理并存入存储器中
                    self.alg.process_env_step(rewards, dones, infos)

                    # 若开启 RND，提取内在奖励用于日志记录
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    # 记录和统计各个环境的回合信息与奖励
                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        # 累加奖励
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        # 累加回合长度步数
                        cur_episode_length += 1
                        # 检查并清理已完成（dones > 0）的回合数据
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        # 统计内外侧奖励缓冲区
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # 如果是强化学习任务（PPO），在采样结束后计算回报（Returns / Advantages）
                if self.training_type == "rl":
                    self.alg.compute_returns(privileged_obs)

            # 打印当前 Rollout 存储器占用的显存大小
            print(f"Rollout storage size: {calculate_rollout_storage_size(self.alg.storage)} MB, {calculate_rollout_storage_size(self.alg.storage)[1]} GB")
            # 执行算法更新（对于 PPO 则是更新策略与价值网络，对于蒸馏则是计算学生与教师的模仿损失并更新学生）
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            
            # 记录日志并保存模型检查点
            if self.log_dir is not None and not self.disable_logs:
                # 调用 log 函数将数据写入 TensorBoard / Wandb 并打印到终端
                self.log(locals())
                
                # 如果当前平均奖励超过了历史最优，保存为 best_model.pt
                if len(rewbuffer) > 0:
                    current_mean_reward = statistics.mean(rewbuffer)
                    if current_mean_reward > best_mean_reward:
                        best_mean_reward = current_mean_reward
                        self.save(best_model_path)
                        print(f"New best model saved with mean_reward: {best_mean_reward:.4f} at iteration {it}")

                # 按照指定的保存间隔（save_interval）保存阶段性模型权重文件
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # 清空回合附加信息
            ep_infos.clear()
            # 在首个迭代保存代码状态（Git diff 文件），便于复现
            if it == start_iter and not self.disable_logs:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # 训练全部完成后，保存最终的模型权重文件
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        """记录并格式化输出各项训练指标（损失、奖励、性能等）到日志和终端。"""
        # 计算当前采集的总步数大小
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # 更新全局总步数和总耗时
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # 处理回合信息（Episode info）
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # 写入日志并拼接字符串
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # 记录各项损失（Losses）
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        # 记录策略噪声标准差
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # 记录性能指标（FPS、采集时间、学习时间）
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # 记录蒸馏任务中的两阶段训练阶段信息
        if self.training_type == "distillation" and hasattr(self.alg, 'get_training_phase_info'):
            phase_info = self.alg.get_training_phase_info()
            if phase_info["use_two_stage_training"]:
                self.writer.add_scalar("Distillation/training_phase", phase_info["training_phase"], locs["it"])
                self.writer.add_scalar("Distillation/current_iteration", phase_info["current_iteration"], locs["it"])

        # 记录训练奖励和回合长度
        if len(locs["rewbuffer"]) > 0:
            if self.alg.rnd:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        # 格式化终端输出的日志字符串
        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            if self.alg.rnd:
                log_string += (
                    f"""{'Mean extrinsic reward:':>{pad}} {statistics.mean(locs['erewbuffer']):.2f}\n"""
                    f"""{'Mean intrinsic reward:':>{pad}} {statistics.mean(locs['irewbuffer']):.2f}\n"""
                )
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
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
        """保存模型权重、优化器状态及归一化器参数到指定路径的文件中。"""
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # 如果使用了 RND，一并保存其状态
        if self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        # 如果启用了经验归一化，保存观测归一化器的状态
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()

        # 将字典保存到本地文件
        torch.save(saved_dict, path)

        # 若使用了外部日志平台（Neptune/Wandb），将模型文件同步上传
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True):
        """从指定路径加载模型检查点，用于恢复训练或迁移学习。"""
        loaded_dict = torch.load(path, weights_only=False)
        # 加载策略网络权重
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        # 加载 RND 模型状态
        if self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        # 加载经验归一化器状态
        if self.empirical_normalization:
            if resumed_training:
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
            else:
                # 若不是恢复训练而是蒸馏任务加载教师模型，则将检查点中的归一化状态加载给特权/教师归一化器
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        # 加载优化器状态
        if load_optimizer and resumed_training:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            if self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        # 恢复当前的迭代轮数
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]
    
    def load_std(self, init_noise_std: float):
        """设置策略动作标准差的初始值。"""
        if self.alg.policy.noise_std_type == "scalar":
            self.alg.policy.std.data.fill_(init_noise_std)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.alg.policy.noise_std_type}. Should be 'scalar' or 'log'")

    def get_inference_policy(self, device=None):
        """获取用于推理和部署的纯净策略函数（包含归一化处理）。"""
        self.eval_mode()  # 切换至评估模式
        if device is not None:
            self.alg.policy.to(device)
        policy = self.alg.policy.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.policy.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy

    def train_mode(self):
        """将网络和归一化器切换至训练模式（在蒸馏任务中会特殊冻结教师的归一化器）。"""
        self.alg.policy.train()
        if self.alg.rnd:
            self.alg.rnd.train()
        if self.empirical_normalization:
            self.obs_normalizer.train() # 学生/普通观测归一化器可以正常更新
            
            # 蒸馏任务中，教师/特权归一化器必须强制固定为 eval 模式，防止其统计量漂移！
            if self.training_type == "distillation":
                self.privileged_obs_normalizer.eval() 
            else:
                self.privileged_obs_normalizer.train()

    def eval_mode(self):
        """将所有网络和归一化器切换至评估模式。"""
        self.alg.policy.eval()
        if self.alg.rnd:
            self.alg.rnd.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        """添加代码仓库路径到 Git 追踪列表中。"""
        self.git_status_repos.append(repo_file_path)

    """
    Helper functions.
    """

    def _configure_multi_gpu(self):
        """配置多 GPU 分布式训练环境。"""
        # 检查是否启用了分布式训练（通过环境变量 WORLD_SIZE）
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # 若未启用分布式训练，则将本地和全局 Rank 设为 0 并直接返回
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # 获取当前进程的本地 Rank 和全局 Rank
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # 构建多 GPU 配置字典
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # 主进程的全局 Rank
            "local_rank": self.gpu_local_rank,  # 当前进程的本地 Rank
            "world_size": self.gpu_world_size,  # 进程总数
        }

        # 检查指定的计算设备是否与当前本地 Rank 的 GPU 匹配
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # 验证多 GPU 配置的合法性
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # 初始化 PyTorch 分布式进程组（使用 NCCL 后端）
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        # 将当前计算设备绑定到对应的本地 GPU 上
        torch.cuda.set_device(self.gpu_local_rank)