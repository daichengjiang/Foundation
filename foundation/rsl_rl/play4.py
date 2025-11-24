# # Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# """Script to play a checkpoint if an RL agent from RSL-RL."""

# """Launch Isaac Sim Simulator first."""

# import argparse
# import sys
# import numpy as np
# from isaaclab.app import AppLauncher

# # local imports
# import cli_args  # isort: skip

# # add argparse arguments
# parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
# parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
# parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
# parser.add_argument(
#     "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
# )
# parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
# parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# parser.add_argument(
#     "--use_pretrained_checkpoint",
#     action="store_true",
#     help="Use the pre-trained checkpoint from Nucleus.",
# )
# parser.add_argument("--realtime", action="store_true", default=False, help="Run in real-time, if possible.")
# # append RSL-RL cli arguments
# cli_args.add_rsl_rl_args(parser)
# # append AppLauncher cli args
# AppLauncher.add_app_launcher_args(parser)
# args_cli, hydra_args = parser.parse_known_args()

# # always enable cameras to record video
# if args_cli.video:
#     args_cli.enable_cameras = True

# # clear out sys.argv for Hydra
# sys.argv = [sys.argv[0]] + hydra_args

# # launch omniverse app
# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app

# """Rest everything follows."""

# import gymnasium as gym
# import os
# import time
# import torch

# from rsl_rl.runners import OnPolicyRunner

# from isaaclab.envs import (
#     DirectMARLEnv,
#     DirectMARLEnvCfg,
#     DirectRLEnvCfg,
#     ManagerBasedRLEnvCfg,
#     multi_agent_to_single_agent,
# )
# from isaaclab.utils.assets import retrieve_file_path
# from isaaclab.utils.dict import print_dict
# from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

# from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

# import isaaclab_tasks  # noqa: F401
# from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
# from isaaclab_tasks.utils.hydra import hydra_task_config
# from e2e_drone import tasks
# import copy
# # PLACEHOLDER: Extension template (do not remove this comment)

# class ActorWrapper(torch.nn.Module):
#     def __init__(self, onpolicyrunner: OnPolicyRunner, device):
#         super().__init__()
#         if hasattr(onpolicyrunner, "obs_normalizer"):
#             self.obs_normalizer = copy.deepcopy(onpolicyrunner.obs_normalizer).to(device)
#         else:
#             self.obs_normalizer = torch.nn.Identity().to(device)
#         self.actor_cnn = copy.deepcopy(onpolicyrunner.alg.policy.actor_cnn).to(device)
#         self.actor_rnn = copy.deepcopy(onpolicyrunner.alg.policy.actor_memory.rnn).to(device)
#         self.actor = copy.deepcopy(onpolicyrunner.alg.policy.actor).to(device)
#         self.hidden_states = torch.zeros(1, 1, 512, device=device)

#     def forward(self, observations):
#         observations = self.obs_normalizer(observations)
#         obs = observations[:, :20]
#         depth_obs = observations[:, 20:].reshape(-1, 1, 60, 80)
#         depth_feature = self.actor_cnn(depth_obs)
#         feature = torch.cat([obs, depth_feature], dim=-1)
#         feature, self.hidden_states = self.actor_rnn(feature.unsqueeze(0), self.hidden_states)
#         actions_mean = self.actor(feature.squeeze(0))
#         actions = torch.clamp(actions_mean, -1.0, 1.0)  # 确保动作在[-1, 1]范围内
#         actions[:, 3] = (actions[:, 3] + 1.0) * 0.5
#         return actions
    
# @hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
# def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
#     """Play with RSL-RL agent."""
#     agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
#     env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
#     env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device
#     env_cfg.sim.use_fabric = not args_cli.disable_fabric if args_cli.disable_fabric is not None else env_cfg.sim.use_fabric
    
#     # specify directory for logging experiments
#     log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
#     log_root_path = os.path.abspath(log_root_path)
#     print(f"[INFO] Loading experiment from directory: {log_root_path}")
#     if args_cli.use_pretrained_checkpoint:
#         resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
#         if not resume_path:
#             print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
#             return
#     elif args_cli.checkpoint:
#         resume_path = retrieve_file_path(args_cli.checkpoint)
#     else:
#         resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

#     log_dir = os.path.dirname(resume_path)

#     # create isaac environment
#     env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

#     # convert to single-agent instance if required by the RL algorithm
#     if isinstance(env.unwrapped, DirectMARLEnv):
#         env = multi_agent_to_single_agent(env)

#     # wrap for video recording
#     if args_cli.video:
#         video_kwargs = {
#             "video_folder": os.path.join(log_dir, "videos", "play"),
#             "step_trigger": lambda step: step == 0,
#             "video_length": args_cli.video_length,
#             "disable_logger": True,
#         }
#         print("[INFO] Recording videos during training.")
#         print_dict(video_kwargs, nesting=4)
#         env = gym.wrappers.RecordVideo(env, **video_kwargs)

#     # wrap around environment for rsl-rl
#     env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

#     print(f"[INFO]: Loading model checkpoint from: {resume_path}")
#     # load previously trained model
#     ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
#     ppo_runner.load(resume_path)

#     # obtain the trained policy for inference
#     policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

#     # extract the neural network module
#     # we do this in a try-except to maintain backwards compatibility.
#     try:
#         # version 2.3 onwards
#         policy_nn = ppo_runner.alg.policy
#     except AttributeError:
#         # version 2.2 and below
#         policy_nn = ppo_runner.alg.actor_critic

#     # export policy to onnx/jit
#     export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
#     # export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
#     # export_policy_as_onnx(
#     #     policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
#     # )

#     obs_input = torch.randn(1, 4820).to(agent_cfg.device)
#     model = ActorWrapper(onpolicyrunner=ppo_runner, device=agent_cfg.device)
#     model.eval()
#     with torch.inference_mode():
#         trace_model = torch.jit.script(model, obs_input)
#         modelfile = log_root_path + "/actor_deploy.pt"
#         trace_model.save(modelfile)
#         print(f"模型已成功保存至 {modelfile}")

#     dt = env.unwrapped.step_dt

#     # reset environment
#     obs, _ = env.get_observations()
#     timestep = 0
#     # simulate environment
#     while simulation_app.is_running():
#         start_time = time.time()
#         # run everything in inference mode
#         with torch.inference_mode():
#             # agent stepping
#             actions = policy(obs)

#             first_obs = obs[0:1]  # 保持batch维度 [1, 4820]
#             deploy_actions = trace_model(first_obs)
            
#             # 比较第一个无人机的动作
#             policy_action = actions[0]      # 第一个无人机的policy动作
#             deploy_action = deploy_actions[0]  # trace_model的输出动作
#             policy_action = torch.clamp(policy_action, -1.0, 1.0)
#             policy_action[3] = (policy_action[3] + 1.0) * 0.5  # 确保动作在[-1, 1]范围内
#             # 计算差异
#             action_diff = torch.abs(policy_action - deploy_action)
#             max_diff = torch.max(action_diff).item()
#             mean_diff = torch.mean(action_diff).item()
            
#             print(f"  Policy Action:  {policy_action}")
#             print(f"  Deploy Action:  {deploy_action}")
#             print(f"  Difference:     {action_diff}")
#             print(f"  Max Diff: {max_diff:.6f}, Mean Diff: {mean_diff:.6f}")
#             print("-" * 50)            
#             # env stepping
#             obs, _, _, _ = env.step(actions)
#         # if timestep % 10 == 0:
#             # print(f"[INFO] Velocity: {np.mean(np.linalg.norm(env.unwrapped._robot.data.root_lin_vel_w.cpu().numpy(), axis=1))}")
#             # rot_matrix_b2w = matrix_from_quat(env.unwrapped._robot.data.root_quat_w)  # Shape: (num_envs, 3, 3)
#             # direction_to_goal_w = env.unwrapped._desired_pos_w - env.unwrapped._robot.data.root_state_w[:, :3]
#             # direction_to_goal_xy_w = direction_to_goal_w[:, :2]
#             # # 计算 R_b2w 的转置（即 R_w2b）
#             # rot_matrix_w2b = rot_matrix_b2w.transpose(1, 2)  # [num_envs, 3, 3]

#             # # 将 direction_to_goal_xy_w 扩展为 3D 向量（z 分量为 0）
#             # direction_to_goal_w_3d = torch.cat([
#             #     direction_to_goal_xy_w,  # [num_envs, 2]
#             #     torch.zeros(env.unwrapped.num_envs, 1, device=direction_to_goal_xy_w.device)  # [num_envs, 1]
#             # ], dim=1)  # [num_envs, 3]

#             # # 转换到机体坐标系：v_b = R_w2b @ v_w
#             # direction_to_goal_b_3d = torch.bmm(rot_matrix_w2b, direction_to_goal_w_3d.unsqueeze(-1)).squeeze(-1)  # [num_envs, 3]

#             # # 提取 xy 分量
#             # direction_to_goal_xy_b = direction_to_goal_b_3d[:, :2]  # [num_envs, 2]

#         if args_cli.video:
#             timestep += 1
#             # Exit the play loop after recording one video
#             if timestep == args_cli.video_length:
#                 break

#         # time delay for real-time evaluation
#         sleep_time = dt - (time.time() - start_time)
#         # if args_cli.realtime and sleep_time > 0:
#         #     time.sleep(sleep_time)
#         # else:
#         #     print("is too slow!!!")

#     # close the simulator
#     env.close()


# if __name__ == "__main__":
#     # run the main function
#     main()
#     # close sim app
#     simulation_app.close()


# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import numpy as np
import csv  # 添加csv模块导入
from isaaclab.app import AppLauncher
import math
import matplotlib.pyplot as plt
from collections import deque
import threading

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--realtime", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from e2e_drone import tasks
import copy
# PLACEHOLDER: Extension template (do not remove this comment)

class RealTimePlotter:
    def __init__(self, max_points=500):
        self.max_points = max_points

        # 数据存储 - 角速度 (rad/s)
        self.time_data = deque(maxlen=max_points)
        # Roll rate (p)
        self.roll_rate_sp_data = deque(maxlen=max_points)      # 目标角速度
        self.roll_rate_actual_data = deque(maxlen=max_points)  # 实际角速度
        # Pitch rate (q)
        self.pitch_rate_sp_data = deque(maxlen=max_points)
        self.pitch_rate_actual_data = deque(maxlen=max_points)
        # Yaw rate (r)
        self.yaw_rate_sp_data = deque(maxlen=max_points)
        self.yaw_rate_actual_data = deque(maxlen=max_points)

        # 设置交互式模式
        plt.ion()

        # 创建图形和子图 (3x1 布局)
        self.fig, self.axes = plt.subplots(3, 1, figsize=(14, 10))
        self.fig.suptitle('Real-time Angular Velocity Tracking (Body Frame)',
                         fontsize=16, fontweight='bold')

        # 初始化线条对象
        self.lines = {}

        # Roll rate subplot (p - rad/s)
        self.lines['roll_rate_sp'], = self.axes[0].plot([], [], 'b-', linewidth=2.5,
                                                         label='Target (rate_sp)', alpha=0.9)
        self.lines['roll_rate_actual'], = self.axes[0].plot([], [], 'r--', linewidth=2,
                                                             label='Actual (ang_vel_b)', alpha=0.8)
        self.axes[0].set_title('Roll Rate (p-axis)', fontsize=13, fontweight='bold', pad=10)
        self.axes[0].set_ylabel('Angular Velocity (rad/s)', fontsize=11)
        self.axes[0].legend(loc='upper right', fontsize=10, framealpha=0.9)
        self.axes[0].grid(True, alpha=0.3, linestyle='--')
        self.axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

        # Pitch rate subplot (q - rad/s)
        self.lines['pitch_rate_sp'], = self.axes[1].plot([], [], 'b-', linewidth=2.5,
                                                          label='Target (rate_sp)', alpha=0.9)
        self.lines['pitch_rate_actual'], = self.axes[1].plot([], [], 'r--', linewidth=2,
                                                              label='Actual (ang_vel_b)', alpha=0.8)
        self.axes[1].set_title('Pitch Rate (q-axis)', fontsize=13, fontweight='bold', pad=10)
        self.axes[1].set_ylabel('Angular Velocity (rad/s)', fontsize=11)
        self.axes[1].legend(loc='upper right', fontsize=10, framealpha=0.9)
        self.axes[1].grid(True, alpha=0.3, linestyle='--')
        self.axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

        # Yaw rate subplot (r - rad/s)
        self.lines['yaw_rate_sp'], = self.axes[2].plot([], [], 'b-', linewidth=2.5,
                                                        label='Target (rate_sp)', alpha=0.9)
        self.lines['yaw_rate_actual'], = self.axes[2].plot([], [], 'r--', linewidth=2,
                                                            label='Actual (ang_vel_b)', alpha=0.8)
        self.axes[2].set_title('Yaw Rate (r-axis)', fontsize=13, fontweight='bold', pad=10)
        self.axes[2].set_xlabel('Time (s)', fontsize=11)
        self.axes[2].set_ylabel('Angular Velocity (rad/s)', fontsize=11)
        self.axes[2].legend(loc='upper right', fontsize=10, framealpha=0.9)
        self.axes[2].grid(True, alpha=0.3, linestyle='--')
        self.axes[2].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

        plt.tight_layout()

    def add_data_point(self, current_time,
                       roll_rate_sp, pitch_rate_sp, yaw_rate_sp,
                       roll_rate_actual, pitch_rate_actual, yaw_rate_actual):
        """添加角速度数据点 (单位: rad/s)"""
        self.time_data.append(current_time)
        # 目标角速度
        self.roll_rate_sp_data.append(roll_rate_sp)
        self.pitch_rate_sp_data.append(pitch_rate_sp)
        self.yaw_rate_sp_data.append(yaw_rate_sp)
        # 实际角速度
        self.roll_rate_actual_data.append(roll_rate_actual)
        self.pitch_rate_actual_data.append(pitch_rate_actual)
        self.yaw_rate_actual_data.append(yaw_rate_actual)

    def update_plot(self):
        if len(self.time_data) < 2:
            return

        time_list = list(self.time_data)

        # 更新Roll rate图
        self.lines['roll_rate_sp'].set_data(time_list, list(self.roll_rate_sp_data))
        self.lines['roll_rate_actual'].set_data(time_list, list(self.roll_rate_actual_data))

        # 更新Pitch rate图
        self.lines['pitch_rate_sp'].set_data(time_list, list(self.pitch_rate_sp_data))
        self.lines['pitch_rate_actual'].set_data(time_list, list(self.pitch_rate_actual_data))

        # 更新Yaw rate图
        self.lines['yaw_rate_sp'].set_data(time_list, list(self.yaw_rate_sp_data))
        self.lines['yaw_rate_actual'].set_data(time_list, list(self.yaw_rate_actual_data))

        # 自动调整坐标轴范围
        for ax in self.axes.flat:
            ax.relim()
            ax.autoscale_view()

        # 刷新图形
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class AttitudePlotter:
    """显示网络输出的mean_action (度) 和实际姿态角度 (度)"""
    def __init__(self, max_points=500):
        self.max_points = max_points

        # 数据存储 - 姿态角度 (度)
        self.time_data = deque(maxlen=max_points)
        # Roll angle
        self.roll_cmd_data = deque(maxlen=max_points)      # 网络输出命令
        self.roll_actual_data = deque(maxlen=max_points)   # 实际姿态角
        # Pitch angle
        self.pitch_cmd_data = deque(maxlen=max_points)
        self.pitch_actual_data = deque(maxlen=max_points)
        # Yaw angle
        self.yaw_cmd_data = deque(maxlen=max_points)
        self.yaw_actual_data = deque(maxlen=max_points)

        # 设置交互式模式
        plt.ion()

        # 创建第二个窗口 (3x1 布局)
        self.fig, self.axes = plt.subplots(3, 1, figsize=(14, 10))
        self.fig.suptitle('Network Output (mean_action) vs Actual Attitude Angles',
                         fontsize=16, fontweight='bold')

        # 初始化线条对象
        self.lines = {}

        # Roll angle subplot (degrees)
        self.lines['roll_cmd'], = self.axes[0].plot([], [], 'g-', linewidth=2.5,
                                                     label='Network Output (mean_action)', alpha=0.9)
        self.lines['roll_actual'], = self.axes[0].plot([], [], 'm--', linewidth=2,
                                                        label='Actual Attitude', alpha=0.8)
        self.axes[0].set_title('Roll Angle', fontsize=13, fontweight='bold', pad=10)
        self.axes[0].set_ylabel('Angle (degrees)', fontsize=11)
        self.axes[0].legend(loc='upper right', fontsize=10, framealpha=0.9)
        self.axes[0].grid(True, alpha=0.3, linestyle='--')
        self.axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

        # Pitch angle subplot (degrees)
        self.lines['pitch_cmd'], = self.axes[1].plot([], [], 'g-', linewidth=2.5,
                                                      label='Network Output (mean_action)', alpha=0.9)
        self.lines['pitch_actual'], = self.axes[1].plot([], [], 'm--', linewidth=2,
                                                         label='Actual Attitude', alpha=0.8)
        self.axes[1].set_title('Pitch Angle', fontsize=13, fontweight='bold', pad=10)
        self.axes[1].set_ylabel('Angle (degrees)', fontsize=11)
        self.axes[1].legend(loc='upper right', fontsize=10, framealpha=0.9)
        self.axes[1].grid(True, alpha=0.3, linestyle='--')
        self.axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

        # Yaw angle subplot (degrees)
        self.lines['yaw_cmd'], = self.axes[2].plot([], [], 'g-', linewidth=2.5,
                                                    label='Network Output (mean_action)', alpha=0.9)
        self.lines['yaw_actual'], = self.axes[2].plot([], [], 'm--', linewidth=2,
                                                       label='Actual Attitude', alpha=0.8)
        self.axes[2].set_title('Yaw Angle', fontsize=13, fontweight='bold', pad=10)
        self.axes[2].set_xlabel('Time (s)', fontsize=11)
        self.axes[2].set_ylabel('Angle (degrees)', fontsize=11)
        self.axes[2].legend(loc='upper right', fontsize=10, framealpha=0.9)
        self.axes[2].grid(True, alpha=0.3, linestyle='--')
        self.axes[2].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

        plt.tight_layout()

    def add_data_point(self, current_time,
                       roll_cmd, pitch_cmd, yaw_cmd,
                       roll_actual, pitch_actual, yaw_actual):
        """添加姿态角度数据点 (单位: 度)"""
        self.time_data.append(current_time)
        # 网络输出命令 (mean_action)
        self.roll_cmd_data.append(roll_cmd)
        self.pitch_cmd_data.append(pitch_cmd)
        self.yaw_cmd_data.append(yaw_cmd)
        # 实际姿态角
        self.roll_actual_data.append(roll_actual)
        self.pitch_actual_data.append(pitch_actual)
        self.yaw_actual_data.append(yaw_actual)

    def update_plot(self):
        if len(self.time_data) < 2:
            return

        time_list = list(self.time_data)

        # 更新Roll angle图
        self.lines['roll_cmd'].set_data(time_list, list(self.roll_cmd_data))
        self.lines['roll_actual'].set_data(time_list, list(self.roll_actual_data))

        # 更新Pitch angle图
        self.lines['pitch_cmd'].set_data(time_list, list(self.pitch_cmd_data))
        self.lines['pitch_actual'].set_data(time_list, list(self.pitch_actual_data))

        # 更新Yaw angle图
        self.lines['yaw_cmd'].set_data(time_list, list(self.yaw_cmd_data))
        self.lines['yaw_actual'].set_data(time_list, list(self.yaw_actual_data))

        # 自动调整坐标轴范围
        for ax in self.axes.flat:
            ax.relim()
            ax.autoscale_view()

        # 刷新图形
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

class StepResponseTester:
    """
    Step response testing module for roll, pitch, and yaw axes.

    This class replaces policy-based control with systematic step inputs
    to each axis (roll, pitch, yaw) for controller performance evaluation.

    Test Sequence:
    1. Maintain hover (2s) -> Apply roll step (5s) -> Record metrics
    2. Return to hover (2s) -> Apply pitch step (5s) -> Record metrics
    3. Return to hover (2s) -> Apply yaw step (5s) -> Record metrics

    Performance Metrics Calculated:
    - Overshoot: Percentage peak exceeds final value
    - Rise Time: Time from 10% to 90% of final value
    - Settling Time: Time to reach and stay within ±2% of final value

    Output:
    - Saves plots to log_dir: step_response_roll.png, step_response_pitch.png, step_response_yaw.png
    - Each plot displays actual response vs setpoint with metrics overlay

    Usage:
        tester = StepResponseTester(dt=0.005, log_dir="/path/to/logs")
        while tester.is_testing_active():
            cmd = tester.get_command(current_time)
            # Apply cmd to simulator
            # Extract actual angle from simulator
            tester.record_data(current_time, actual_angle, setpoint_angle)

    Configuration:
        - hover_duration: 2.0s (time to stabilize before step)
        - step_duration: 5.0s (time to record response after step)
        - step_magnitude: 0.3 (normalized command value, maps to ±45° for roll/pitch, ±90° for yaw)
    """
    def __init__(self, dt: float, log_dir: str):
        self.dt = dt
        self.log_dir = log_dir

        # Test configuration
        self.hover_duration = 2.0  # seconds to maintain hover before step
        self.step_duration = 5.0   # seconds to record response after step
        self.step_magnitude = 0.3  # step magnitude (normalized, -1 to 1)

        # Test sequence: [axis_name, axis_index, step_value]
        # Added 'hover' test at the beginning
        self.test_sequence = [
            ('hover', -1, 0.0),  # -1 indicates no axis movement, just hover
            ('roll', 0, self.step_magnitude),
            ('pitch', 1, self.step_magnitude),
            ('yaw', 2, self.step_magnitude)
        ]

        # Current test state
        self.current_test_idx = 0
        self.test_phase = 'hover'  # 'hover', 'step', 'done'
        self.phase_start_time = 0.0

        # Data recording
        self.time_data = []
        self.response_data = []
        self.setpoint_data = []

        # Hover test: record all three axes
        self.hover_roll_data = []
        self.hover_pitch_data = []
        self.hover_yaw_data = []

        print(f"[StepResponseTester] Initialized with {len(self.test_sequence)} tests")
        print(f"  Hover duration: {self.hover_duration}s")
        print(f"  Step duration: {self.step_duration}s")
        print(f"  Step magnitude: {self.step_magnitude}")

    def get_command(self, current_time: float, hover_thrust: float = 0.5) -> torch.Tensor:
        """
        Get command for current test phase.

        Args:
            current_time: Current simulation time (seconds)
            hover_thrust: Hover thrust value [0, 1]

        Returns:
            Command tensor [roll, pitch, yaw, thrust] (normalized)
        """
        # Initialize hover command
        cmd = torch.zeros(4)
        cmd[3] = hover_thrust

        if self.test_phase == 'hover':
            # Check if hover duration elapsed
            if current_time - self.phase_start_time >= self.hover_duration:
                # Transition to step phase
                self.test_phase = 'step'
                self.phase_start_time = current_time
                self.time_data = []
                self.response_data = []
                self.setpoint_data = []
                # Clear hover test data
                self.hover_roll_data = []
                self.hover_pitch_data = []
                self.hover_yaw_data = []

                axis_name, _, _ = self.test_sequence[self.current_test_idx]
                print(f"\n[StepResponseTester] Starting {axis_name.upper()} step response test")

        elif self.test_phase == 'step':
            # Apply step input to current axis
            axis_name, axis_idx, step_value = self.test_sequence[self.current_test_idx]

            # For hover test, keep all axes at zero (no step applied)
            if axis_name != 'hover' and axis_idx >= 0:
                cmd[axis_idx] = step_value

            # Check if step duration elapsed
            if current_time - self.phase_start_time >= self.step_duration:
                # Save results and move to next test
                self._save_step_response(axis_name)

                self.current_test_idx += 1
                if self.current_test_idx >= len(self.test_sequence):
                    # All tests completed
                    self.test_phase = 'done'
                    print("\n[StepResponseTester] All step response tests completed!")
                else:
                    # Move to next test (hover phase)
                    self.test_phase = 'hover'
                    self.phase_start_time = current_time

        return cmd

    def record_data(self, current_time: float, response_value: float, setpoint_value: float):
        """
        Record data during step response test.

        Args:
            current_time: Current simulation time (seconds)
            response_value: Actual response value (e.g., roll angle in degrees)
            setpoint_value: Setpoint value (e.g., commanded roll in degrees)
        """
        if self.test_phase == 'step':
            relative_time = current_time - self.phase_start_time
            self.time_data.append(relative_time)
            self.response_data.append(response_value)
            self.setpoint_data.append(setpoint_value)

    def record_hover_data(self, current_time: float, roll_deg: float, pitch_deg: float, yaw_deg: float):
        """
        Record data during hover stability test (all three axes).

        Args:
            current_time: Current simulation time (seconds)
            roll_deg: Roll angle in degrees
            pitch_deg: Pitch angle in degrees
            yaw_deg: Yaw angle in degrees
        """
        if self.test_phase == 'step':  # During hover test, phase is 'step'
            axis_name = self.test_sequence[self.current_test_idx][0]
            if axis_name == 'hover':
                relative_time = current_time - self.phase_start_time
                self.time_data.append(relative_time)
                self.hover_roll_data.append(roll_deg)
                self.hover_pitch_data.append(pitch_deg)
                self.hover_yaw_data.append(yaw_deg)

    def _calculate_metrics(self, time_array, response_array, setpoint_value):
        """
        Calculate performance metrics from step response data.

        Args:
            time_array: Time data (numpy array)
            response_array: Response data (numpy array)
            setpoint_value: Final setpoint value

        Returns:
            dict: Performance metrics (overshoot, settling_time, rise_time)
        """
        metrics = {}

        # Final value (average of last 10% of data)
        final_idx_start = int(0.9 * len(response_array))
        final_value = np.mean(response_array[final_idx_start:])

        # Overshoot: percentage over final value
        peak_value = np.max(response_array)
        if abs(final_value) > 1e-3:
            overshoot = ((peak_value - final_value) / abs(final_value)) * 100.0
        else:
            overshoot = 0.0
        metrics['overshoot'] = overshoot
        metrics['peak_value'] = peak_value
        metrics['final_value'] = final_value

        # Rise time: time to go from 10% to 90% of final value
        threshold_10 = 0.1 * final_value
        threshold_90 = 0.9 * final_value

        idx_10 = np.where(response_array >= threshold_10)[0]
        idx_90 = np.where(response_array >= threshold_90)[0]

        if len(idx_10) > 0 and len(idx_90) > 0:
            rise_time = time_array[idx_90[0]] - time_array[idx_10[0]]
        else:
            rise_time = float('nan')
        metrics['rise_time'] = rise_time

        # Settling time: time to reach and stay within ±2% of final value
        settling_threshold = 0.02 * abs(final_value)
        settling_band = np.abs(response_array - final_value) <= settling_threshold

        # Find first index where response enters and stays in settling band
        settling_idx = None
        for i in range(len(settling_band) - 10):  # Check at least 10 points ahead
            if np.all(settling_band[i:]):
                settling_idx = i
                break

        if settling_idx is not None:
            settling_time = time_array[settling_idx]
        else:
            settling_time = float('nan')
        metrics['settling_time'] = settling_time

        return metrics

    def _save_step_response(self, axis_name: str):
        """
        Save step response plot with performance metrics and step signal.

        Args:
            axis_name: Name of the axis being tested ('hover', 'roll', 'pitch', 'yaw')
        """
        if len(self.time_data) < 10:
            print(f"[StepResponseTester] WARNING: Insufficient data for {axis_name} test")
            return

        # Convert to numpy arrays
        time_array = np.array(self.time_data)
        response_array = np.array(self.response_data)
        setpoint_array = np.array(self.setpoint_data)

        # Calculate setpoint value (should be constant during step)
        setpoint_value = np.mean(setpoint_array)

        # For hover test, use different analysis (no step, measure stability)
        if axis_name == 'hover':
            # Convert hover data to numpy arrays
            roll_array = np.array(self.hover_roll_data)
            pitch_array = np.array(self.hover_pitch_data)
            yaw_array = np.array(self.hover_yaw_data)
            self._save_hover_test(time_array, roll_array, pitch_array, yaw_array)
            return

        # Calculate metrics for step response
        metrics = self._calculate_metrics(time_array, response_array, setpoint_value)

        # Create plot with two subplots: Step Signal + Response
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1, 2])
        fig.suptitle(f'{axis_name.upper()} Axis Step Response Test',
                    fontsize=16, fontweight='bold', y=0.995)

        # === Subplot 1: Step Signal (Command) ===
        # Create idealized step signal (0 before step, constant after step)
        step_signal = np.zeros_like(time_array)
        step_signal[:] = setpoint_value  # Step applied from t=0 in this phase

        ax1.plot(time_array, step_signal, 'g-', linewidth=2.5,
                label='Step Command', alpha=0.9)
        ax1.set_ylabel('Command (degrees)', fontsize=11, fontweight='bold')
        ax1.set_title('Step Input Signal', fontsize=12, fontweight='bold', pad=8)
        ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.axhline(y=setpoint_value, color='k', linestyle=':', linewidth=1, alpha=0.5)
        ax1.set_xlim(time_array[0], time_array[-1])

        # === Subplot 2: System Response ===
        # Plot response and setpoint
        ax2.plot(time_array, response_array, 'b-', linewidth=2.5,
                label='Actual Response', alpha=0.9)
        ax2.plot(time_array, setpoint_array, 'r--', linewidth=2,
                label='Desired Setpoint', alpha=0.8)

        # Add settling band (±2%)
        final_value = metrics['final_value']
        settling_threshold = 0.02 * abs(final_value)
        ax2.fill_between(time_array,
                        final_value - settling_threshold,
                        final_value + settling_threshold,
                        color='green', alpha=0.2, label='±2% Settling Band')

        # Add reference lines
        ax2.axhline(y=final_value, color='k', linestyle=':', linewidth=1, alpha=0.5)
        ax2.axhline(y=metrics['peak_value'], color='orange', linestyle=':',
                  linewidth=1, alpha=0.5, label=f"Peak: {metrics['peak_value']:.2f}°")

        # Labels and title
        ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Angle (degrees)', fontsize=12, fontweight='bold')
        ax2.set_title('System Response', fontsize=12, fontweight='bold', pad=8)
        ax2.legend(loc='best', fontsize=10, framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xlim(time_array[0], time_array[-1])

        # Add metrics text box
        metrics_text = (
            f"Performance Metrics:\n"
            f"─────────────────────\n"
            f"Overshoot:      {metrics['overshoot']:.2f}%\n"
            f"Rise Time:      {metrics['rise_time']:.3f}s\n"
            f"Settling Time:  {metrics['settling_time']:.3f}s\n"
            f"Final Value:    {metrics['final_value']:.2f}°\n"
            f"Peak Value:     {metrics['peak_value']:.2f}°"
        )

        # Position text box at upper right
        ax2.text(0.98, 0.97, metrics_text, transform=ax2.transAxes,
               fontsize=10, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               family='monospace')

        plt.tight_layout()

        # Save figure
        filename = f"step_response_{axis_name}.png"
        filepath = os.path.join(self.log_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"[StepResponseTester] Saved {axis_name} step response to: {filepath}")
        print(f"  Overshoot: {metrics['overshoot']:.2f}%")
        print(f"  Rise Time: {metrics['rise_time']:.3f}s")
        print(f"  Settling Time: {metrics['settling_time']:.3f}s")

    def _save_hover_test(self, time_array, roll_array, pitch_array, yaw_array):
        """
        Save hover stability test plot with all three axes (roll, pitch, yaw).

        Args:
            time_array: Time data (numpy array)
            roll_array: Roll angle data (numpy array) in degrees
            pitch_array: Pitch angle data (numpy array) in degrees
            yaw_array: Yaw angle data (numpy array) in degrees
        """
        # Create figure with 3 subplots (one for each axis)
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        fig.suptitle('Hover Stability Test - All Axes',
                    fontsize=16, fontweight='bold', y=0.995)

        # Data for each axis
        angle_data = [
            ('Roll', roll_array, 'b'),
            ('Pitch', pitch_array, 'r'),
            ('Yaw', yaw_array, 'g')
        ]

        metrics_all = []

        for idx, (axis_name, angle_array, color) in enumerate(angle_data):
            ax = axes[idx]

            # Calculate stability metrics
            mean_value = np.mean(angle_array)
            std_dev = np.std(angle_array)
            max_deviation = np.max(np.abs(angle_array - mean_value))

            metrics_all.append({
                'name': axis_name,
                'mean': mean_value,
                'std': std_dev,
                'max_dev': max_deviation
            })

            # Plot angle over time
            ax.plot(time_array, angle_array, color=color, linewidth=2,
                   label=f'{axis_name} Angle', alpha=0.9)

            # Add reference lines
            ax.axhline(y=mean_value, color='k', linestyle='--', linewidth=1.5,
                      label=f'Mean: {mean_value:.3f}°', alpha=0.7)
            ax.axhline(y=mean_value + std_dev, color='orange', linestyle=':',
                      linewidth=1, alpha=0.5)
            ax.axhline(y=mean_value - std_dev, color='orange', linestyle=':',
                      linewidth=1, alpha=0.5, label=f'±1σ: {std_dev:.3f}°')

            # Shaded region for ±1σ
            ax.fill_between(time_array,
                           mean_value - std_dev,
                           mean_value + std_dev,
                           color=color, alpha=0.1)

            # Labels and formatting
            ax.set_ylabel(f'{axis_name} (degrees)', fontsize=11, fontweight='bold')
            ax.set_title(f'{axis_name} Axis Stability', fontsize=12, fontweight='bold', pad=8)
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim(time_array[0], time_array[-1])

            # Only show x-label on bottom subplot
            if idx == 2:
                ax.set_xlabel('Time (s)', fontsize=11, fontweight='bold')

        # Add comprehensive metrics text box on the first subplot
        metrics_text = "Hover Stability Metrics:\n"
        metrics_text += "─────────────────────────────\n"
        for m in metrics_all:
            metrics_text += f"{m['name']:5s}: μ={m['mean']:7.3f}°  σ={m['std']:6.3f}°  max={m['max_dev']:6.3f}°\n"
        metrics_text += f"─────────────────────────────\n"
        metrics_text += f"Test Duration: {time_array[-1]:.1f}s"

        axes[0].text(0.98, 0.97, metrics_text, transform=axes[0].transAxes,
                   fontsize=9, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.85),
                   family='monospace')

        plt.tight_layout()

        # Save figure
        filename = "hover_stability_test.png"
        filepath = os.path.join(self.log_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Print summary to console
        print(f"\n[StepResponseTester] Saved hover stability test to: {filepath}")
        print("  Hover Stability Metrics:")
        for m in metrics_all:
            print(f"    {m['name']:5s}: Mean={m['mean']:7.3f}°  Std={m['std']:6.3f}°  MaxDev={m['max_dev']:6.3f}°")

    def is_testing_active(self) -> bool:
        """Check if testing is still active."""
        return self.test_phase != 'done'

    def get_current_axis_name(self) -> str:
        """Get name of current axis being tested."""
        if self.current_test_idx < len(self.test_sequence):
            return self.test_sequence[self.current_test_idx][0]
        return "none"


class ActorWrapper(torch.nn.Module):
    def __init__(self, onpolicyrunner: OnPolicyRunner, device):
        super().__init__()
        if hasattr(onpolicyrunner, "obs_normalizer"):
            self.obs_normalizer = copy.deepcopy(onpolicyrunner.obs_normalizer).to(device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(device)
        self.actor_cnn = copy.deepcopy(onpolicyrunner.alg.policy.actor_cnn).to(device)
        self.actor_rnn = copy.deepcopy(onpolicyrunner.alg.policy.actor_memory.rnn).to(device)
        self.actor = copy.deepcopy(onpolicyrunner.alg.policy.actor).to(device)
        self.hidden_states = torch.zeros(1, 1, 512, device=device)

    def forward(self, observations):
        observations = self.obs_normalizer(observations)
        obs = observations[:, :20]
        depth_obs = observations[:, 20:].reshape(-1, 1, 60, 100)
        depth_feature = self.actor_cnn(depth_obs)
        feature = torch.cat([obs, depth_feature], dim=-1)
        feature, self.hidden_states = self.actor_rnn(feature.unsqueeze(0), self.hidden_states)
        actions_mean = self.actor(feature.squeeze(0))
        actions = torch.clamp(actions_mean, -1.0, 1.0)  # 确保动作在[-1, 1]范围内
        actions[:, 3] = (actions[:, 3] + 1.0) * 0.5
        return actions
    
@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric if args_cli.disable_fabric is not None else env_cfg.sim.use_fabric
    
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # 创建CSV文件用于保存policy_action数据
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11111")
    csv_file_path = os.path.join(log_dir, "policy_actions.csv")
    csv_file = open(csv_file_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    
    # 写入CSV头部
    csv_writer.writerow(['action_0', 'action_1', 'action_2', 'thrust', 'mean_action_0', 'mean_action_1', 'mean_action_2', 'mean_thrust', 'roll', 'pitch', 'yaw'])
    print(f"[INFO] CSV file created at: {csv_file_path}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    # export_policy_as_onnx(
    #     policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    # )

    obs_input = torch.randn(1, 6020).to(agent_cfg.device)
    model = ActorWrapper(onpolicyrunner=ppo_runner, device=agent_cfg.device)
    model.eval()
    with torch.inference_mode():
        trace_model = torch.jit.script(model, obs_input)
        modelfile = log_root_path + "/actor_deploy_3.pt"
        trace_model.save(modelfile)
        print(f"模型已成功保存至 {modelfile}")

    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.get_observations()
    # Save first frame observations as npy file
    first_obs_path = os.path.join(log_dir, "first_observations.npy")
    np.save(first_obs_path, obs.cpu().numpy())
    print(f"[INFO] Saved first frame observations to: {first_obs_path}")
    
    # Save first frame observations as txt file
    first_obs_txt_path = os.path.join(log_dir, "first_observations.txt")
    np.savetxt(first_obs_txt_path, obs.cpu().numpy(), fmt='%.6f')
    print(f"[INFO] Saved first frame observations to: {first_obs_txt_path}")
    
    timestep = 0
    start_simulation_time = time.time()

    # 创建阶跃响应测试器
    step_tester = StepResponseTester(dt=dt, log_dir=log_dir)
    print("[INFO] Step response tester initialized")

    # Disable real-time plotters for step response testing
    # rate_plotter = RealTimePlotter(max_points=500)
    # attitude_plotter = AttitudePlotter(max_points=500)
    # print("[INFO] Real-time plotters initialized (Rate + Attitude windows)")

    # simulate environment
    try:
        while simulation_app.is_running() and step_tester.is_testing_active():
            start_time = time.time()
            current_time = timestep * dt

            # run everything in inference mode
            with torch.inference_mode():
                # === Step Response Testing: Get test command instead of policy action ===
                # Get command from step tester (replaces policy inference)
                test_cmd = step_tester.get_command(current_time, hover_thrust=0.6)

                # test_cmd = torch.tensor([0.0, 0.0, 0.0, 0.6])
                # Expand command to match batch size (num_envs)
                actions = test_cmd.unsqueeze(0).expand(env.unwrapped.num_envs, 4).to(env.unwrapped.device)

                # Extract commanded angles for recording (before env step)
                cmd_roll_deg = test_cmd[0].item() * 45.0    # ±45deg
                cmd_pitch_deg = test_cmd[1].item() * 45.0   # ±45deg
                cmd_yaw_deg = test_cmd[2].item() * 90.0     # ±90deg

                # env stepping
                obs, _, _, _ = env.step(actions)

                # === 提取实际姿态角度 (从四元数转换) ===
                # 获取当前无人机的姿态角度（弧度转度）
                robot_quat = env.unwrapped._robot.data.root_quat_w[0]  # 第一个无人机的四元数

                # 将四元数转换为欧拉角（roll, pitch, yaw）
                # 使用 XYZ 欧拉角约定 (与 px4_controller.py 保持一致)
                # Reference: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
                qw, qx, qy, qz = robot_quat.cpu().numpy()

                # Roll (x-axis rotation)
                sin_roll = 2.0 * (qw * qx + qy * qz)
                cos_roll = 1.0 - 2.0 * (qx * qx + qy * qy)
                roll_rad = math.atan2(sin_roll, cos_roll)

                # Pitch (y-axis rotation) with singularity handling
                sin_pitch = 2.0 * (qw * qy - qz * qx)
                if abs(sin_pitch) >= 1.0:
                    # Handle gimbal lock singularity (pitch = ±90°)
                    pitch_rad = math.copysign(math.pi / 2.0, sin_pitch)
                else:
                    pitch_rad = math.asin(sin_pitch)

                # Yaw (z-axis rotation)
                sin_yaw = 2.0 * (qw * qz + qx * qy)
                cos_yaw = 1.0 - 2.0 * (qy * qy + qz * qz)
                yaw_rad = math.atan2(sin_yaw, cos_yaw)

                # 转换为角度
                roll_deg = math.degrees(roll_rad)
                pitch_deg = math.degrees(pitch_rad)
                yaw_deg = math.degrees(yaw_rad)

                # === Record data for step response testing ===
                # Record data based on current test axis
                axis_name = step_tester.get_current_axis_name()
                if axis_name == 'hover':
                    # For hover test, record all three axes
                    step_tester.record_hover_data(current_time, roll_deg, pitch_deg, yaw_deg)
                elif axis_name == 'roll':
                    step_tester.record_data(current_time, roll_deg, cmd_roll_deg)
                elif axis_name == 'pitch':
                    step_tester.record_data(current_time, pitch_deg, cmd_pitch_deg)
                elif axis_name == 'yaw':
                    step_tester.record_data(current_time, yaw_deg, cmd_yaw_deg)

                timestep += 1

            if args_cli.video:
                # Exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    break
            # print(f"[INFO] Dilated Positions : {env.unwrapped._dilated_positions.cpu().numpy()}")
            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            # if args_cli.realtime and sleep_time > 0:
            #     time.sleep(sleep_time)
            # else:
            #     print("is too slow!!!")

    except KeyboardInterrupt:
        print("\n[INFO] Program interrupted by user (Ctrl+C)")
        print(f"[INFO] CSV file saved with current data at: {csv_file_path}")
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        print(f"[INFO] CSV file saved with current data at: {csv_file_path}")
    finally:
        # 确保CSV文件被正确关闭
        csv_file.close()
        print(f"[INFO] Policy actions saved to CSV file: {csv_file_path}")
        
        # close the simulator
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()