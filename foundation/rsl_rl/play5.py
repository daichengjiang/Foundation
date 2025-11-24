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
import random  # 添加random模块导入
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
import random

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
        self.lines['roll_rate_actual'], = self.axes[0].plot([], [], 'r-', linewidth=2,
                                                             label='Actual (ang_vel_b)', alpha=0.8)
        self.axes[0].set_title('Roll Rate (p-axis)', fontsize=13, fontweight='bold', pad=10)
        self.axes[0].set_ylabel('Angular Velocity (rad/s)', fontsize=11)
        self.axes[0].legend(loc='upper right', fontsize=10, framealpha=0.9)
        self.axes[0].grid(True, alpha=0.3, linestyle='--')
        self.axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

        # Pitch rate subplot (q - rad/s)
        self.lines['pitch_rate_sp'], = self.axes[1].plot([], [], 'b-', linewidth=2.5,
                                                          label='Target (rate_sp)', alpha=0.9)
        self.lines['pitch_rate_actual'], = self.axes[1].plot([], [], 'r-', linewidth=2,
                                                              label='Actual (ang_vel_b)', alpha=0.8)
        self.axes[1].set_title('Pitch Rate (q-axis)', fontsize=13, fontweight='bold', pad=10)
        self.axes[1].set_ylabel('Angular Velocity (rad/s)', fontsize=11)
        self.axes[1].legend(loc='upper right', fontsize=10, framealpha=0.9)
        self.axes[1].grid(True, alpha=0.3, linestyle='--')
        self.axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

        # Yaw rate subplot (r - rad/s)
        self.lines['yaw_rate_sp'], = self.axes[2].plot([], [], 'b-', linewidth=2.5,
                                                        label='Target (rate_sp)', alpha=0.9)
        self.lines['yaw_rate_actual'], = self.axes[2].plot([], [], 'r-', linewidth=2,
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
        self.lines['roll_actual'], = self.axes[0].plot([], [], 'm-', linewidth=2,
                                                        label='Actual Attitude', alpha=0.8)
        self.axes[0].set_title('Roll Angle', fontsize=13, fontweight='bold', pad=10)
        self.axes[0].set_ylabel('Angle (degrees)', fontsize=11)
        self.axes[0].legend(loc='upper right', fontsize=10, framealpha=0.9)
        self.axes[0].grid(True, alpha=0.3, linestyle='--')
        self.axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

        # Pitch angle subplot (degrees)
        self.lines['pitch_cmd'], = self.axes[1].plot([], [], 'g-', linewidth=2.5,
                                                      label='Network Output (mean_action)', alpha=0.9)
        self.lines['pitch_actual'], = self.axes[1].plot([], [], 'm-', linewidth=2,
                                                         label='Actual Attitude', alpha=0.8)
        self.axes[1].set_title('Pitch Angle', fontsize=13, fontweight='bold', pad=10)
        self.axes[1].set_ylabel('Angle (degrees)', fontsize=11)
        self.axes[1].legend(loc='upper right', fontsize=10, framealpha=0.9)
        self.axes[1].grid(True, alpha=0.3, linestyle='--')
        self.axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

        # Yaw angle subplot (degrees)
        self.lines['yaw_cmd'], = self.axes[2].plot([], [], 'g-', linewidth=2.5,
                                                    label='Network Output (mean_action)', alpha=0.9)
        self.lines['yaw_actual'], = self.axes[2].plot([], [], 'm-', linewidth=2,
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

class TorqueThrustPlotter:
    """显示力矩 (tau_des 和 torque)、推力 (thrust_cmd) 和力 (force_z) 的变化"""
    def __init__(self, max_points=500):
        self.max_points = max_points

        # 数据存储 - 力矩、推力和力
        self.time_data = deque(maxlen=max_points)
        # Desired torques (tau_des) - normalized [-1, 1]
        self.roll_torque_des_data = deque(maxlen=max_points)
        self.pitch_torque_des_data = deque(maxlen=max_points)
        self.yaw_torque_des_data = deque(maxlen=max_points)
        # Actual torques (torque) - N·m
        self.roll_torque_actual_data = deque(maxlen=max_points)
        self.pitch_torque_actual_data = deque(maxlen=max_points)
        self.yaw_torque_actual_data = deque(maxlen=max_points)
        # Thrust command - normalized [0, 1]
        self.thrust_data = deque(maxlen=max_points)
        # Actual force (force_z) - N
        self.force_z_data = deque(maxlen=max_points)

        # 设置交互式模式
        plt.ion()

        # 创建第三个窗口 (8x1 布局) - 6个力矩子图 + 1个推力子图 + 1个力子图
        self.fig, self.axes = plt.subplots(8, 1, figsize=(14, 18))
        self.fig.suptitle('Torque, Thrust and Force Commands',
                         fontsize=16, fontweight='bold')

        # 初始化线条对象
        self.lines = {}

        # Roll torque desired subplot
        self.lines['roll_torque_des'], = self.axes[0].plot([], [], 'r-', linewidth=2,
                                                           label='Desired Roll Torque (tau_des)', alpha=0.9)
        self.axes[0].set_title('Desired Roll Torque (tau_des)', fontsize=13, fontweight='bold', pad=10)
        self.axes[0].set_ylabel('Torque [-1, 1]', fontsize=11)
        self.axes[0].legend(loc='upper right', fontsize=10, framealpha=0.9)
        self.axes[0].grid(True, alpha=0.3, linestyle='--')
        self.axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

        # Roll torque actual subplot
        self.lines['roll_torque_actual'], = self.axes[1].plot([], [], 'c-', linewidth=2,
                                                              label='Actual Roll Torque', alpha=0.9)
        self.axes[1].set_title('Actual Roll Torque', fontsize=13, fontweight='bold', pad=10)
        self.axes[1].set_ylabel('Torque [N·m]', fontsize=11)
        self.axes[1].legend(loc='upper right', fontsize=10, framealpha=0.9)
        self.axes[1].grid(True, alpha=0.3, linestyle='--')
        self.axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

        # Pitch torque desired subplot
        self.lines['pitch_torque_des'], = self.axes[2].plot([], [], 'g-', linewidth=2,
                                                            label='Desired Pitch Torque (tau_des)', alpha=0.9)
        self.axes[2].set_title('Desired Pitch Torque (tau_des)', fontsize=13, fontweight='bold', pad=10)
        self.axes[2].set_ylabel('Torque [-1, 1]', fontsize=11)
        self.axes[2].legend(loc='upper right', fontsize=10, framealpha=0.9)
        self.axes[2].grid(True, alpha=0.3, linestyle='--')
        self.axes[2].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

        # Pitch torque actual subplot
        self.lines['pitch_torque_actual'], = self.axes[3].plot([], [], 'y-', linewidth=2,
                                                               label='Actual Pitch Torque', alpha=0.9)
        self.axes[3].set_title('Actual Pitch Torque', fontsize=13, fontweight='bold', pad=10)
        self.axes[3].set_ylabel('Torque [N·m]', fontsize=11)
        self.axes[3].legend(loc='upper right', fontsize=10, framealpha=0.9)
        self.axes[3].grid(True, alpha=0.3, linestyle='--')
        self.axes[3].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

        # Yaw torque desired subplot
        self.lines['yaw_torque_des'], = self.axes[4].plot([], [], 'b-', linewidth=2,
                                                          label='Desired Yaw Torque (tau_des)', alpha=0.9)
        self.axes[4].set_title('Desired Yaw Torque (tau_des)', fontsize=13, fontweight='bold', pad=10)
        self.axes[4].set_ylabel('Torque [-1, 1]', fontsize=11)
        self.axes[4].legend(loc='upper right', fontsize=10, framealpha=0.9)
        self.axes[4].grid(True, alpha=0.3, linestyle='--')
        self.axes[4].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

        # Yaw torque actual subplot
        self.lines['yaw_torque_actual'], = self.axes[5].plot([], [], 'm-', linewidth=2,
                                                             label='Actual Yaw Torque', alpha=0.9)
        self.axes[5].set_title('Actual Yaw Torque', fontsize=13, fontweight='bold', pad=10)
        self.axes[5].set_ylabel('Torque [N·m]', fontsize=11)
        self.axes[5].legend(loc='upper right', fontsize=10, framealpha=0.9)
        self.axes[5].grid(True, alpha=0.3, linestyle='--')
        self.axes[5].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

        # Thrust command subplot
        self.lines['thrust'], = self.axes[6].plot([], [], 'orange', linewidth=2,
                                                   label='Thrust Command', alpha=0.9)
        self.axes[6].set_title('Thrust Command (Normalized)', fontsize=13, fontweight='bold', pad=10)
        self.axes[6].set_ylabel('Thrust [0, 1]', fontsize=11)
        self.axes[6].legend(loc='upper right', fontsize=10, framealpha=0.9)
        self.axes[6].grid(True, alpha=0.3, linestyle='--')
        self.axes[6].axhline(y=0.5, color='k', linestyle=':', linewidth=1, alpha=0.5, label='Hover (~0.5)')

        # Force Z subplot
        self.lines['force_z'], = self.axes[7].plot([], [], 'purple', linewidth=2,
                                                   label='Actual Force Z', alpha=0.9)
        self.axes[7].set_title('Actual Force Z', fontsize=13, fontweight='bold', pad=10)
        self.axes[7].set_xlabel('Time (s)', fontsize=11)
        self.axes[7].set_ylabel('Force [N]', fontsize=11)
        self.axes[7].legend(loc='upper right', fontsize=10, framealpha=0.9)
        self.axes[7].grid(True, alpha=0.3, linestyle='--')
        self.axes[7].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

        plt.tight_layout()

    def add_data_point(self, current_time,
                       roll_torque_des, pitch_torque_des, yaw_torque_des,
                       roll_torque_actual, pitch_torque_actual, yaw_torque_actual,
                       thrust, force_z):
        """添加力矩、推力和力数据点"""
        self.time_data.append(current_time)
        # Desired torques
        self.roll_torque_des_data.append(roll_torque_des)
        self.pitch_torque_des_data.append(pitch_torque_des)
        self.yaw_torque_des_data.append(yaw_torque_des)
        # Actual torques
        self.roll_torque_actual_data.append(roll_torque_actual)
        self.pitch_torque_actual_data.append(pitch_torque_actual)
        self.yaw_torque_actual_data.append(yaw_torque_actual)
        # Thrust
        self.thrust_data.append(thrust)
        # Force
        self.force_z_data.append(force_z)

    def update_plot(self):
        if len(self.time_data) < 2:
            return

        time_list = list(self.time_data)

        # 更新Roll torque期望图
        self.lines['roll_torque_des'].set_data(time_list, list(self.roll_torque_des_data))
        # 更新Roll torque实际图
        self.lines['roll_torque_actual'].set_data(time_list, list(self.roll_torque_actual_data))

        # 更新Pitch torque期望图
        self.lines['pitch_torque_des'].set_data(time_list, list(self.pitch_torque_des_data))
        # 更新Pitch torque实际图
        self.lines['pitch_torque_actual'].set_data(time_list, list(self.pitch_torque_actual_data))

        # 更新Yaw torque期望图
        self.lines['yaw_torque_des'].set_data(time_list, list(self.yaw_torque_des_data))
        # 更新Yaw torque实际图
        self.lines['yaw_torque_actual'].set_data(time_list, list(self.yaw_torque_actual_data))

        # 更新Thrust图
        self.lines['thrust'].set_data(time_list, list(self.thrust_data))
        # 更新Force Z图
        self.lines['force_z'].set_data(time_list, list(self.force_z_data))

        # 自动调整坐标轴范围
        for ax in self.axes.flat:
            ax.relim()
            ax.autoscale_view()

        # 刷新图形
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

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

    _action_history = []
    _action_history_length = 8

    # 创建实时绘图器 (三个窗口)
    rate_plotter = RealTimePlotter(max_points=500)
    attitude_plotter = AttitudePlotter(max_points=500)
    torque_thrust_plotter = TorqueThrustPlotter(max_points=500)
    print("[INFO] Real-time plotters initialized (Rate + Attitude + Torque/Thrust windows)")

    # trace_model = torch.jit.load("/home/zjr/CrazyE2E/logs/rsl_rl/point_ctrl_direct/actor_deploy1.pt")
    # trace_model = trace_model.to("cuda")
    # simulate environment
    try:
        while simulation_app.is_running():
            start_time = time.time()
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)

                actions_copy = actions.clamp(-1.0, 1.0)
                actions_copy[:, 3] = (actions_copy[:, 3] + 1.0) * 0.5
                _action_history.append(actions_copy.clone())
                if len(_action_history) > _action_history_length:
                    _action_history.pop(0)

                # --- 新增：随机时延窗口扰动 ---
                # 只在历史足够时才扰动，否则用全部历史
                history_len = len(_action_history)
                if history_len >= _action_history_length:
                    # 随机选择窗口起点，范围为[-6, -2]，即倒数第6到倒数第2个
                    window_start = random.randint(-8, -8)
                    window_end = window_start + 7  # 窗口长度为8
                    # Python负索引切片，window_end可以为0（不包含0本身）
                    selected_actions = _action_history[window_start:window_end]
                    stacked = torch.stack(selected_actions, dim=0)
                    mean_action = stacked.mean(dim=0)
                else:
                    # 历史不足6步，直接用全部历史均值
                    stacked = torch.stack(_action_history, dim=0)
                    mean_action = stacked.mean(dim=0)

                # 提取第一个环境的mean_action数据 (用于姿态角度可视化)
                mean_action_first = mean_action[0].cpu().numpy()
                mean_action_roll_deg = mean_action_first[0] * 45.0    # ±45deg
                mean_action_pitch_deg = mean_action_first[1] * 45.0   # ±45deg
                mean_action_yaw_deg = mean_action_first[2] * 90.0     # ±90deg
                
                rate_speed = torch.tensor([0.0, 0.0, 0.0, 0.6])
                actions = rate_speed.unsqueeze(0).expand(env.unwrapped.num_envs, 4).to(env.unwrapped.device)
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

                # === 提取角速度数据用于可视化 ===
                # 获取目标角速度 (从px4info) - 单位: rad/s
                rate_sp = env.unwrapped.px4info['rate_sp'][0].cpu().numpy()  # [p, q, r] in rad/s
                roll_rate_sp = rate_sp[0]
                pitch_rate_sp = rate_sp[1]
                yaw_rate_sp = rate_sp[2]

                # 获取实际角速度 (从机体坐标系) - 单位: rad/s
                ang_vel_b = env.unwrapped._robot.data.root_ang_vel_b[0].cpu().numpy()  # [p, q, r] in rad/s
                roll_rate_actual = ang_vel_b[0]
                pitch_rate_actual = ang_vel_b[1]
                yaw_rate_actual = ang_vel_b[2]

                # === 提取力矩和推力数据用于可视化 ===
                # 获取期望力矩 (从px4info) - 单位: normalized [-1, 1]
                tau_des = env.unwrapped.px4info['tau_des'][0].cpu().numpy()  # [roll, pitch, yaw] torques
                roll_torque_des = tau_des[0]
                pitch_torque_des = tau_des[1]
                yaw_torque_des = tau_des[2]

                # 获取实际力矩 (从px4info) - 单位: N·m
                torque_actual = env.unwrapped.px4info['torque'][0].cpu().numpy()  # [roll, pitch, yaw] torques
                roll_torque_actual = torque_actual[0]
                pitch_torque_actual = torque_actual[1]
                yaw_torque_actual = torque_actual[2]

                # 获取推力命令 (从px4info) - 单位: normalized [0, 1]
                thrust_cmd = env.unwrapped.px4info['thrust_cmd'][0].item()

                # 获取实际力 (从px4info) - 单位: N
                force_z = env.unwrapped.px4info['force_z'][0].item()

                # 更新实时绘图 (三个窗口)
                # current_time = time.time() - start_simulation_time
                current_time = timestep * dt  # 使用timestep和dt计算当前时间，避免time误差累积

                # 窗口1: 角速度跟踪
                rate_plotter.add_data_point(
                    current_time,
                    roll_rate_sp, pitch_rate_sp, yaw_rate_sp,
                    roll_rate_actual, pitch_rate_actual, yaw_rate_actual
                )

                # 窗口2: 姿态角度 (mean_action vs 实际角度)
                attitude_plotter.add_data_point(
                    current_time,
                    mean_action_roll_deg, mean_action_pitch_deg, mean_action_yaw_deg,
                    roll_deg, pitch_deg, yaw_deg
                )

                # 窗口3: 力矩、推力和力
                torque_thrust_plotter.add_data_point(
                    current_time,
                    roll_torque_des, pitch_torque_des, yaw_torque_des,
                    roll_torque_actual, pitch_torque_actual, yaw_torque_actual,
                    thrust_cmd, force_z
                )

                # 每10个timestep更新一次图表以提高性能
                if timestep % 10 == 0:
                    rate_plotter.update_plot()
                    attitude_plotter.update_plot()
                    torque_thrust_plotter.update_plot()

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