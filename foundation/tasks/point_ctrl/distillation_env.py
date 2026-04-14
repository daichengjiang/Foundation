# Copyright (c) 2025 Xu Yang
# HKUST UAV Group
#
# Author: Xu Yang
# Affiliation: HKUST UAV Group
# Date: April 2025
# License: MIT License

from __future__ import annotations

import omni
import torch
import torch.nn.functional as F
import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.sim.utils import find_matching_prim_paths
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext, RenderCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import HfDiscreteObstaclesTerrainCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, matrix_from_quat
from isaaclab.utils.noise import GaussianNoiseCfg, UniformNoiseCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers import CUBOID_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG
import isaaclab.sim as sim_utils
from isaaclab_assets import CRAZYFLIE_CFG
from isaaclab.assets import ArticulationCfg
import isaacsim.core.utils.prims as prims_utils
from pxr import PhysxSchema, Sdf, UsdGeom, UsdPhysics, Gf
from isaaclab.utils.math import quat_from_euler_xyz
from collections import deque
import numpy as np
import random
import math
import time
import os
import csv

from foundation.utils.simple_controller import SimpleQuadrotorController
from foundation.utils.pid_controller import PaperPhysControllerTensor

from enum import IntEnum
import collections
import itertools

from dataclasses import dataclass

MAP_SIZE = (250, 250) 

# 手动定义球体标记配置
SPHERE_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.02,  # 默认半径，后续会通过 scale 调整
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
        ),
    }
)

# [0, 2pi] -> [-pi, pi]
def normallize_angle(angle: torch.Tensor):
    return torch.fmod(angle + math.pi, 2 * math.pi) - math.pi

@configclass
class QuadcopterDynamicsCfg:

    mass: float = 0.0282
    arm_length: float = 0.04384
    # Inertia: Ixx, Iyy, Izz
    inertia: tuple[float, float, float] = (2.44864e-5, 2.44864e-5, 3.61504e-5)
    thrust_to_weight: float = 2.25 
    motor_tau_up: float = 0.05
    motor_tau_down: float = 0.10 
    moment_scale: float = 0.016  

    multi_teacher_params: list[dict] | None = None 


class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)

@configclass
class QuadcopterSceneCfg(InteractiveSceneCfg):
    """Configuration for the Quadcopter scene."""
    num_envs: int = 512
    env_spacing: float = 64.0
    replicate_physics: bool = False

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=MAP_SIZE,
            sub_terrains={
                "obstacles": HfDiscreteObstaclesTerrainCfg(
                    num_obstacles=0,
                    obstacle_width_range=(0.1, 0.1),
                    obstacle_height_range=(0.1, 0.1)
                ),
            },
        ),
    )

@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):

    teacher_observation_space = 58

    student_observation_space = 24
    # 设置 IsaacLab 默认观测空间（通常对应学生策略维度）
    observation_space = student_observation_space 

    history_len = 5

    prob_null_trajectory = 0.0  # 50% 概率做定点控制

    # 轨迹类型选择: "langevin" 或 "figure8"
    trajectory_type = "langevin"

    train_or_play: bool = True 
    use_pid = False
    # gamma in ppo, only for logging
    gamma = 0.99

    # env
    episode_length_s = 96
    decimation = 1
    action_space = 4 
    state_space = 0
    debug_vis = False

    map_size = MAP_SIZE

    grid_rows = 80 
    grid_cols = 80 
    terrain_width = 3
    terrain_length = 3
    robots_per_env = 1

    # terrain and robot
    train = True
    robot_vis = True
    marker_size = 0.05  # Size of the markers in meters

    ui_window_class_type = QuadcopterEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100.0,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        render=RenderCfg(
            enable_dl_denoiser=True,
            dlss_mode=2,
        )
    )

    dynamics: QuadcopterDynamicsCfg = QuadcopterDynamicsCfg()

    # scene
    scene: InteractiveSceneCfg = QuadcopterSceneCfg()

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    height = 3.0
    
    # State check thresholds (for any dimension x, y, z)
    position_threshold = 15.0  # meters
    position_threshold_langevin = 14  # 根据实际需求调整

    linear_velocity_threshold = 4.0  # m/s
    angular_velocity_threshold = 35.0  # rad/s

    reward_coef_position_cost = 1.0
    reward_coef_orientation_cost = 0.2
    reward_coef_d_action_cost = 1.0
    reward_coef_termination_penalty = 100.0
    reward_constant = 1.5


class QuadcopterEnv(DirectRLEnv):
    """A quadcopter environment adapted to use the reward logic from the training code."""

    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.start_time = time.time()
        self.render_mode = "human"
        
        # 初始化张量
        self.mass_tensor = torch.zeros(self.num_envs, device=self.device)
        self.arm_l_tensor = torch.zeros(self.num_envs, device=self.device)
        self.twr_tensor = torch.zeros(self.num_envs, device=self.device)
        self.inertia_tensor = torch.zeros(self.num_envs, 3, device=self.device)
        self.motor_tau_up_tensor = torch.zeros(self.num_envs, 1, device=self.device)
        self.motor_tau_down_tensor = torch.zeros(self.num_envs, 1, device=self.device)
        self.kappa_tensor = torch.zeros(self.num_envs, device=self.device)

        if self.cfg.use_pid:
            self.wn_tensor = torch.zeros(self.num_envs, device=self.device)
            self.zeta_tensor = torch.zeros(self.num_envs, device=self.device)
            self.tc_ang_rp_tensor = torch.zeros(self.num_envs, device=self.device)
            self.tc_ang_y_tensor = torch.zeros(self.num_envs, device=self.device)
            self.tc_rate_rp_tensor = torch.zeros(self.num_envs, device=self.device)
            self.tc_rate_y_tensor = torch.zeros(self.num_envs, device=self.device)
            default_pid_params = [2, 0.7, 0.08, 0.4, 0.04, 0.2]

        self.history_len = self.cfg.history_len
        self.pos_error_b_history = torch.zeros(self.num_envs, self.history_len, 3, device=self.device)
        self.vel_error_b_history = torch.zeros(self.num_envs, self.history_len, 3, device=self.device)

        if self.cfg.dynamics.multi_teacher_params is not None:
            teacher_params_list = self.cfg.dynamics.multi_teacher_params
            num_teachers = len(teacher_params_list)
            
            # 计算每个教师分配的环境数量
            # 注意: 如果不能整除，最后一个教师将承担剩余的环境
            envs_per_teacher = self.num_envs // num_teachers
            
            print(f"[Multi-Teacher Env] Initializing {num_teachers} teachers. Base envs per teacher: {envs_per_teacher}.")
            
            for t_id, params in enumerate(teacher_params_list):
                start_idx = t_id * envs_per_teacher
                
                # 如果是最后一个教师，覆盖到末尾
                if t_id == num_teachers - 1:
                    end_idx = self.num_envs
                else:
                    end_idx = start_idx + envs_per_teacher
                
                # 生成切片索引
                indices = slice(start_idx, end_idx)
                
                # 填充参数
                self.mass_tensor[indices] = params['mass']
                self.arm_l_tensor[indices] = params['arm_length']
                self.twr_tensor[indices] = params.get('twr', params.get('thrust_to_weight')) # 兼容两种 key
                
                ixx, iyy, izz = params['inertia']
                self.inertia_tensor[indices, 0] = ixx
                self.inertia_tensor[indices, 1] = iyy
                self.inertia_tensor[indices, 2] = izz
                
                self.motor_tau_up_tensor[indices] = params['motor_tau_up']
                self.motor_tau_down_tensor[indices] = params['motor_tau_down']
                self.kappa_tensor[indices] = params['kappa']

                if self.cfg.use_pid:
                    self.wn_tensor[indices] = params.get('wn', default_pid_params[0])
                    self.zeta_tensor[indices] = params.get('zeta', default_pid_params[1])
                    self.tc_ang_rp_tensor[indices] = params.get('tc_ang_rp', default_pid_params[2])
                    self.tc_ang_y_tensor[indices] = params.get('tc_ang_y', default_pid_params[3])
                    self.tc_rate_rp_tensor[indices] = params.get('tc_rate_rp', default_pid_params[4])
                    self.tc_rate_y_tensor[indices] = params.get('tc_rate_y', default_pid_params[5])

                count = end_idx - start_idx
                print(f"  > Teacher {t_id} (ID: {params.get('id', 'N/A')}): Envs {start_idx}-{end_idx-1} (Count: {count})")

            # [添加调试打印]
            print(f"[DEBUG] Applied Multi-Teacher Params:")
            for i in range(min(5, self.num_envs)): # 打印前5个环境的质量
                print(f"  Env {i}: Mass = {self.mass_tensor[i].item():.4f}")
            mid_idx = self.num_envs // 2
            print(f"  Env {mid_idx} (Mid): Mass = {self.mass_tensor[mid_idx].item():.4f}")
        
        else:

            print(f"[WARNING] Using Default Dynamics! Check if teacher params were passed correctly.")
            # [原有逻辑] 单教师/默认参数
            print(f"[Single-Teacher Env] Using default dynamics for all {self.num_envs} envs.")
            self.mass_tensor.fill_(self.cfg.dynamics.mass)
            self.arm_l_tensor.fill_(self.cfg.dynamics.arm_length)
            self.inertia_tensor[:] = torch.tensor(self.cfg.dynamics.inertia, device=self.device)
            self.twr_tensor.fill_(self.cfg.dynamics.thrust_to_weight)
            self.motor_tau_up_tensor.fill_(self.cfg.dynamics.motor_tau_up)
            self.motor_tau_down_tensor.fill_(self.cfg.dynamics.motor_tau_down)
            self.kappa_tensor.fill_(self.cfg.dynamics.moment_scale)

            if self.cfg.use_pid:
                self.wn_tensor.fill_(default_pid_params[0])
                self.zeta_tensor.fill_(default_pid_params[1])
                self.tc_ang_rp_tensor.fill_(default_pid_params[2])
                self.tc_ang_y_tensor.fill_(default_pid_params[3])
                self.tc_rate_rp_tensor.fill_(default_pid_params[4])
                self.tc_rate_y_tensor.fill_(default_pid_params[5])


        # Store the robot mass for reference (e.g. wind force calculation if added later)
        self._robot_mass = self.mass_tensor 
        
        self.dt = self.cfg.sim.dt

        # if self.motor_tau.shape != (self.num_envs, 1):
        #      self.motor_tau = self.motor_tau.view(self.num_envs, 1)

        # self.motor_alpha = self.dt / (self.dt + self.motor_tau)
        if self.motor_tau_up_tensor.shape != (self.num_envs, 1):
             self.motor_tau_up_tensor = self.motor_tau_up_tensor.view(self.num_envs, 1)
        if self.motor_tau_down_tensor.shape != (self.num_envs, 1):
             self.motor_tau_down_tensor = self.motor_tau_down_tensor.view(self.num_envs, 1)
             
        self.motor_alpha_up = self.dt / torch.clamp(self.motor_tau_up_tensor, min=1e-6)
        self.motor_alpha_down = self.dt / torch.clamp(self.motor_tau_down_tensor, min=1e-6)

        self._current_motor_speeds = torch.zeros(self.num_envs, 4, device=self.device)

        
        self._controller = PaperPhysControllerTensor(
            num_envs=self.num_envs,
            device=self.device,
            mass=self.mass_tensor,
            arm_length=self.arm_l_tensor,
            inertia=self.inertia_tensor,
            thrust_to_weight=self.twr_tensor,
            kappa=self.kappa_tensor,
            motor_alpha_up=self.motor_alpha_up,
            motor_alpha_down=self.motor_alpha_down,
        )
        if self.cfg.use_pid:
            self._controller.update_gains(
                wn=self.wn_tensor,
                zeta=self.zeta_tensor,
                tc_ang_rp=self.tc_ang_rp_tensor,
                tc_ang_y=self.tc_ang_y_tensor,
                tc_rate_rp=self.tc_rate_rp_tensor,
                tc_rate_y=self.tc_rate_y_tensor
            )

        self._is_langevin_task = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # === 死亡原因标志位 ===
        self._numerical_is_unstable = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._died_pos_limit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)      # 飞出半径
        self._died_lin_vel_limit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # 线速度过大
        self._died_ang_vel_limit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # 角速度过大
        self._died_tilt_limit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)     # 倾角 > 90度
        self._died_nan = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)            # 数值 NaN

        # Quadcopter references
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._torques = torch.zeros(self.num_envs, 1, 3, device=self.device)
        
        # Desired states
        self.pos_des = torch.zeros(self.num_envs, 3, device=self.device)
        self.vel_des = torch.zeros(self.num_envs, 3, device=self.device)
        self.acc_des = torch.zeros(self.num_envs, 3, device=self.device) # 新增加速度项
        self.yaw_des = torch.zeros(self.num_envs, device=self.device)      # 期望偏航角
        self.yaw_rate_des = torch.zeros(self.num_envs, device=self.device) # 期望偏航角速度

        self.yaw_limit = math.pi / 2

        # Langevin parameters
        self._langevin_dt = 0.01
        self._langevin_friction = 0.5
        self._langevin_omega = 1.5
        self._langevin_sigma = 3.0
        self._langevin_alpha = 0.2
        
        # Figure-8 trajectory parameters
        self._figure8_time = torch.zeros(self.num_envs, device=self.device)
        self._figure8_frequency = 0.1
        self._figure8_scale_x = 1.0
        self._figure8_scale_y = 0.5
        self._figure8_height = 3.0
        self._figure8_warmup_duration = 5.0

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "position", "orientation", "action_smooth", "base", "terminal"
            ]
        }
        
        # Environment origins
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        self.grid_idx = None
        # Robot references
        self._body_id = self._robot.find_bodies("body")[0]

        self._last_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._spawn_pos_w = torch.zeros(self.num_envs, 3, device=self.device) 

        self._last_angular_velocity= torch.zeros(self.num_envs, 3, device=self.device)
        self._langevin_max_vel = torch.full((self.num_envs,), 1.5, device=self.device)

        # Episode tracking
        self._history_window = 100
        self._episode_outcomes = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episodes_completed = 0
        self._termination_reason_history = collections.deque(maxlen=self._history_window)
        self._vel_abs = collections.deque(maxlen=self._history_window)

        self.set_debug_vis(self.cfg.debug_vis)

        # 标志位：记录是否已经根据悬停位置重置了轨迹原点
        self._traj_origin_adjusted = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        self._calc_env_origins()

    def CHECK_NAN(self, tensor, name):
        if torch.isnan(tensor).any().item():
            print(f"[{name}] NaN detected in tensor of shape {tensor.shape}.")
            nan_env_mask = torch.any(torch.isnan(tensor), dim=1)
            
            self._died_nan = torch.logical_or(self._died_nan, nan_env_mask)
            self._numerical_is_unstable = torch.logical_or(self._numerical_is_unstable, nan_env_mask)
            
            nan_env_indices = torch.where(nan_env_mask)[0]
            print(f"NaN positions: {nan_env_indices}")
            tensor = tensor.nan_to_num(nan=0.0)
            return tensor
        else:
            return tensor

    def CHECK_state(self):
        """
        Check state thresholds and log specific failure reasons.
        """
        pos_w = self._robot.data.root_pos_w 
        lin_vel_w = self._robot.data.root_lin_vel_w 
        ang_vel_b = self._robot.data.root_ang_vel_b 
        quat_w = self._robot.data.root_quat_w
        
        # 1. Check distance from spawn point
        distance_from_spawn = torch.norm(pos_w - self._spawn_pos_w, dim=1)
        position_exceeded = distance_from_spawn > self.cfg.position_threshold
        
        # 2. Check linear velocity threshold
        linear_velocity_exceeded = torch.any(torch.abs(lin_vel_w) > self.cfg.linear_velocity_threshold, dim=1)
        
        # 3. Check angular velocity threshold
        angular_velocity_exceeded = torch.any(torch.abs(ang_vel_b) > self.cfg.angular_velocity_threshold, dim=1)

        # 4. Check Tilt > 90 degrees
        rot_matrix = matrix_from_quat(quat_w) 
        body_z_projected = rot_matrix[:, 2, 2]
        tilt_exceeded = body_z_projected < 0.0
        
        # 更新具体的死亡原因
        self._died_pos_limit = torch.logical_or(self._died_pos_limit, position_exceeded)
        self._died_lin_vel_limit = torch.logical_or(self._died_lin_vel_limit, linear_velocity_exceeded)
        self._died_ang_vel_limit = torch.logical_or(self._died_ang_vel_limit, angular_velocity_exceeded)
        self._died_tilt_limit = torch.logical_or(self._died_tilt_limit, tilt_exceeded)
        
        state_is_unstable = (
            self._died_pos_limit | 
            self._died_lin_vel_limit | 
            self._died_ang_vel_limit | 
            self._died_tilt_limit |
            self._died_nan
        )
        
        self._numerical_is_unstable = torch.logical_or(self._numerical_is_unstable, state_is_unstable)

    def _update_yaw_langevin(self, env_ids: torch.Tensor, dt: float):
        """
        受限的 Yaw 角 Langevin 动力学生成。
        范围被限制在 [-90, 90] 度之间，且带有回中趋势。
        """
        # --- 参数设置 ---
        yaw_limit = self.yaw_limit # 限制在 +/- 90 度 (1.57 rad)
        yaw_k_pos = 0.2        # 弹簧刚度 (回复力)
        yaw_k_vel = 0.5        # 阻尼
        yaw_noise_scale = 2.0  # 噪声强度
        max_yaw_rate = 2.5     # 最大角速度
            
        # 1. 获取当前状态
        current_yaw = self.yaw_des[env_ids]
        current_yaw_rate = self.yaw_rate_des[env_ids]
        
        # 2. 生成随机扰动
        noise = torch.randn(len(env_ids), device=self.device) * yaw_noise_scale
        
        # 3. 核心动力学更新 (Ornstein-Uhlenbeck Process)
        # 加速度 = 随机力 - 阻尼力 - 弹簧回复力(当前角度偏离0的程度)
        # 这里的 - yaw_k_pos * current_yaw 就是把头拉回正前方的力
        yaw_acc = noise - yaw_k_vel * current_yaw_rate - yaw_k_pos * current_yaw
        
        # 4. 积分更新角速度
        next_yaw_rate = current_yaw_rate + yaw_acc * dt
        
        # 限制角速度 (物理能力限制)
        next_yaw_rate = torch.clamp(next_yaw_rate, -max_yaw_rate, max_yaw_rate)
        
        # 5. 积分更新角度
        next_yaw = current_yaw + next_yaw_rate * dt
        
        # 6. 硬截断与边界处理 (Hard Constraints)
        # 如果超出 +/- 90 度，强制拉回，并将撞墙的速度清零
        over_max = next_yaw > yaw_limit
        under_min = next_yaw < -yaw_limit
        
        if over_max.any() or under_min.any():
            # 截断角度
            next_yaw = torch.clamp(next_yaw, -yaw_limit, yaw_limit)
            
            # 撞墙处理：如果试图冲出边界，把速度抹平（非弹性碰撞），防止粘在墙上抖动
            # 逻辑：如果超限了，把对应的 rate 设为 0 (或者反弹 -0.5 * rate)
            hit_limit_mask = over_max | under_min
            next_yaw_rate[hit_limit_mask] = 0.0
            
        # 7. 保存状态
        self.yaw_des[env_ids] = next_yaw
        self.yaw_rate_des[env_ids] = next_yaw_rate

    def _generate_desired_trajectory_langevin(self, env_ids: torch.Tensor = None):
        if env_ids is None: env_ids = torch.arange(self.num_envs, device=self.device)
        n_envs = len(env_ids)
        dt = self.dt  # 使用仿真步长，或者使用独立的轨迹步长 self._langevin_dt
        
        # --- 参数设置 (可以提取到 Config 中) ---
        # 弹簧刚度 (把无人机拉回原点，影响位置约束强弱)
        k_pos = 1.0  
        # 阻尼系数 (防止速度过大，影响最高速度)
        k_vel = 1.5   
        # 加速度平滑系数 (模拟加加速度 Jerk 的惯性，值越小加速度变化越慢)
        acc_inertia = 0.1 
        # 噪声强度 (直接决定加速度的变化幅度)
        noise_scale = 10.0 # 需要根据无人机推重比调整，通常 5.0 - 15.0 之间
        
        # 获取当前状态
        pos_current = self.pos_des[env_ids]
        vel_current = self.vel_des[env_ids]
        acc_current = self.acc_des[env_ids]
        spawn_pos = self._spawn_pos_w[env_ids]

        # --- 积分更新 ---
        # Euler 积分更新速度和位置
        vel_next = vel_current + acc_current * dt
        
        # 限制最大速度 (软限制已由阻尼 k_vel 提供，这里做硬截断以防万一)
        max_vel = 3.0 # m/s
        vel_norm = torch.norm(vel_next, dim=1, keepdim=True)
        scale = torch.clamp(max_vel / (vel_norm + 1e-6), max=1.0)
        vel_next = vel_next * scale
        
        pos_next = pos_current + vel_current * dt
        
        # 计算相对于出生点的位移
        pos_err = pos_current - spawn_pos
        
        # --- 核心物理动力学 ---
        # 目标加速度变化量 (Jerk) = 随机噪声 + 回复力(拉回中心) - 阻尼力(限制速度)
        # 这是一个受到随机力扰动的弹簧阻尼二阶系统
        noise = torch.randn(n_envs, 3, device=self.device) * noise_scale
        
        # 期望的合外力 (Target Force/Mass)
        force_total = noise - k_pos * pos_err - k_vel * vel_current
        
        # 更新加速度 (一阶低通滤波，模拟物理惯性)
        # acc_new = (1 - alpha) * acc_old + alpha * target
        acc_next = (1.0 - acc_inertia) * acc_current + acc_inertia * force_total
        
        # --- 物理限制 (Clipping) ---
        # 限制最大加速度 (基于推重比，假设最大推力约为 2G 到 3G)
        max_acc = 15.0 # m/s^2
        acc_next = torch.clamp(acc_next, -max_acc, max_acc)
 
        # --- 保存状态 ---
        self.acc_des[env_ids] = acc_next
        self.vel_des[env_ids] = vel_next
        self.pos_des[env_ids] = pos_next

        self._update_yaw_langevin(env_ids, dt)

    def _generate_desired_trajectory_figure8(self, env_ids: torch.Tensor = None):
        if env_ids is None: env_ids = torch.arange(self.num_envs, device=self.device)
        n_envs = len(env_ids)
        
        self._figure8_time[env_ids] += self.dt
        t = self._figure8_time[env_ids]
        
        # Recentering logic
        needs_recenter = (t >= self._figure8_warmup_duration) & (~self._traj_origin_adjusted[env_ids])
        if needs_recenter.any():
            recenter_idx = env_ids[needs_recenter]
            curr_pos = self._robot.data.root_pos_w[recenter_idx].clone()
            self._spawn_pos_w[recenter_idx] = curr_pos
            self.pos_des[recenter_idx] = curr_pos
            self._traj_origin_adjusted[recenter_idx] = True
            # [新增] 重置时也将 Yaw 设为 0，防止突变
            self.yaw_des[recenter_idx] = 0.0
            self.yaw_rate_des[recenter_idx] = 0.0

        in_warmup = t < self._figure8_warmup_duration
        active_mask = ~in_warmup
        active_env_ids = env_ids[active_mask]
        
        if len(active_env_ids) > 0:
            omega = 2 * math.pi * self._figure8_frequency
            spawn = self._spawn_pos_w[active_env_ids]
            t_adj = t[active_mask] - self._figure8_warmup_duration
            
            # --- 1. 位置 Position ---
            x_rel = self._figure8_scale_x * torch.sin(omega * t_adj)
            y_rel = self._figure8_scale_y * torch.sin(2 * omega * t_adj)
            z_target = spawn[:, 2] 
            pos_des_new = torch.stack([spawn[:, 0] + x_rel, spawn[:, 1] + y_rel, z_target], dim=1)
            
            # --- 2. 速度 Velocity ---
            vx = self._figure8_scale_x * omega * torch.cos(omega * t_adj)
            vy = self._figure8_scale_y * 2 * omega * torch.cos(2 * omega * t_adj)
            vz = torch.zeros_like(vx)
            vel_des_new = torch.stack([vx, vy, vz], dim=1)

            # --- 3. 加速度 Acceleration ---
            ax = -self._figure8_scale_x * (omega**2) * torch.sin(omega * t_adj)
            ay = -self._figure8_scale_y * (4 * omega**2) * torch.sin(2 * omega * t_adj)
            az = torch.zeros_like(ax)
            acc_des_new = torch.stack([ax, ay, az], dim=1)
            
            # --- 4. [新增] 正弦 Yaw 角 ---
            # 设定 Yaw 的摆动幅度，例如 90 度 (PI/2)
            yaw_amplitude = math.pi / 4   
            
            # 计算 Yaw (跟随主频率 omega 变化)
            # yaw = A * sin(omega * t)
            yaw_new = yaw_amplitude * torch.sin(omega * t_adj)
            
            # 计算 Yaw Rate (对时间求导)
            # d(yaw)/dt = A * omega * cos(omega * t)
            yaw_rate_new = yaw_amplitude * omega * torch.cos(omega * t_adj)

            # --- 赋值 ---
            self.pos_des[active_env_ids] = pos_des_new
            self.vel_des[active_env_ids] = vel_des_new
            self.acc_des[active_env_ids] = acc_des_new
            
            # [新增] 赋值 Yaw
            self.yaw_des[active_env_ids] = yaw_new
            self.yaw_rate_des[active_env_ids] = yaw_rate_new

    def _calc_env_origins(self):
        robots_per_env = self.cfg.robots_per_env
        num_groups = self.num_envs // robots_per_env + 1
        grid_rows = self.cfg.grid_rows
        grid_cols = self.cfg.grid_cols

        group_origins = torch.zeros(num_groups, 3, device=self.device)
        terrain_width = self.cfg.terrain_width
        terrain_length = self.cfg.terrain_length

        map_size_x, map_size_y = self.cfg.map_size
        offset_x = -map_size_x / 2.0
        offset_y = -map_size_y / 2.0

        for i in range(num_groups):
            row = (i // grid_cols) % grid_rows
            col = i % grid_cols
            group_origins[i, 0] = col * terrain_length + offset_x
            group_origins[i, 1] = row * terrain_width + offset_y
    
        self.env_origins = group_origins.repeat_interleave(robots_per_env, dim=0)[:self.num_envs]
        num_grids = grid_rows * grid_cols
        self.grid_idx = [[] for _ in range(num_grids)]
        for env_id in range(self.num_envs):
            group_id = env_id // robots_per_env
            row = (group_id // grid_cols) % grid_rows
            col = group_id % grid_cols
            grid_linear_idx = row * grid_cols + col
            self.grid_idx[grid_linear_idx].append(env_id)

    def _setup_scene(self):
        """Create and clone the environment scene."""
        # 1. Set up the robot articulation
        self._robot = Articulation(self.cfg.robot)
        
        # 2. Clone the scene (Create N environments)
        self.scene.clone_environments(copy_from_source=False)
        
        # 3. Register articulation to scene
        self.scene.articulations["robot"] = self._robot

        # 4. Find all robot prim paths
        robot_prims = find_matching_prim_paths("/World/envs/env_.*/Robot")

        if len(robot_prims) == 0:
            print("[ERROR] No robot prims found! Check your prim_path regex in the config.")
            return

        # ================= [DEBUG START: 应用多教师动力学参数到 PhysX] =================
        print(f"\n{'='*20} [Dynamics Setup] {'='*20}")
        print(f"Num Envs: {self.num_envs}")
        
        # 准备参数
        has_multi_teachers = self.cfg.dynamics.multi_teacher_params is not None
        if has_multi_teachers:
            teacher_params = self.cfg.dynamics.multi_teacher_params
            num_teachers = len(teacher_params)
            envs_per_teacher = self.num_envs // num_teachers
            print(f"Applying params from {num_teachers} teachers...")
        else:
            print("Applying single teacher params...")

        # [新增] 在 _setup_scene 中初始化 com_tensor，保障生命周期安全
        self.com_tensor = torch.zeros(self.num_envs, 3, device=self.device)
        com_ratios = torch.tensor([0.20, 0.20, 0.20], device=self.device)

        # 遍历所有环境，修改底层 USD/PhysX 属性
        for i, prim_path in enumerate(robot_prims):
            body_path = f"{prim_path}/body"
            
            # 确定当前环境使用哪一套参数
            if has_multi_teachers:
                # 计算 teacher id
                # 防止溢出 (e.g., if envs aren't perfectly divisible)
                t_id = min(i // envs_per_teacher, num_teachers - 1)
                params = teacher_params[t_id]
                
                mass_val = params['mass']
                inertia_val = params['inertia']
                arm_l_val = params['arm_length']
            else:
                mass_val = self.cfg.dynamics.mass
                inertia_val = self.cfg.dynamics.inertia
                arm_l_val = self.cfg.dynamics.arm_length
            
            # ==========================================
            # [修改] 1b. 实时计算该环境独立的随机重心偏移
            # ==========================================
            # 生成 [-1, 1] 的随机偏移并乘上比例和对应环境的真实臂长
            rand_offset = (torch.rand(3, device=self.device) * 2.0 - 1.0) * com_ratios * arm_l_val
            # rand_offset = 0.8 * com_ratios * arm_l_val
            self.com_tensor[i] = rand_offset  # 保存回 tensor 供后续可能的计算使用

            # --- 将配置写入底层 PhysX ---
            # 1. 修改质量
            prims_utils.set_prim_property(body_path, "physics:mass", mass_val)
            # 2. 修改惯性张量 (Isaac Sim 接受 (Ixx, Iyy, Izz) 对角形式)
            prims_utils.set_prim_property(body_path, "physics:diagonalInertia", inertia_val)
            # # 3. 强制重心
            # prims_utils.set_prim_property(body_path, "physics:centerOfMass", (0.0, 0.0, 0.0))

            # 3. 应用随机重心
            com_val = tuple(rand_offset.tolist())
            prims_utils.set_prim_property(body_path, "physics:centerOfMass", com_val)

            # --- 设置可见性 ---
            if self.cfg.robot_vis:
                prims_utils.set_prim_property(prim_path, "visibility", "visible")
            else:
                prims_utils.set_prim_property(prim_path, "visibility", "invisible")

            # --- 抽样检查 (Read-back Verification) ---
            if i == 0 or (has_multi_teachers and i % envs_per_teacher == 0 and i < self.num_envs):
                actual_mass_usd = prims_utils.get_prim_property(body_path, "physics:mass")
                actual_com_usd = prims_utils.get_prim_property(body_path, "physics:centerOfMass") # 直接从 PhysX 读回
                print(f"[Env {i}] Teacher ID: {t_id if has_multi_teachers else 0} | Set Mass: {mass_val:.4f} | PhysX Read: {actual_mass_usd:.4f}")
                print(f"[Env {i}] Tensor COM: {self.com_tensor[i].cpu().numpy()} | PhysX COM Read: {actual_com_usd}")

        print(f"{'='*60}\n")
        # ================= [DEBUG END] =================

        # 5. Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self._map_generation_timer = 0

    def _pre_physics_step(self, actions: torch.Tensor):

        # 2. Action Clamp
        raw_actions_clamped = torch.clamp(actions, -1.0, 1.0)
        action_setpoint_normalized = (raw_actions_clamped + 1.0) * 0.5
        
        self._actions = raw_actions_clamped.clone()

        target = action_setpoint_normalized
        current = self._current_motor_speeds
        alpha = torch.where(target > current, self.motor_alpha_up, self.motor_alpha_down)
        self._current_motor_speeds = current + alpha * (target - current)
        self._current_motor_speeds = torch.clamp(self._current_motor_speeds, 0.0, 1.0)

        # 计算力与力矩
        force_b, torque_b = self._controller.motor_speeds_to_wrench(self._current_motor_speeds) 
        
        # ================= [新增：风阻与空气动力学阻力 (Aerodynamic Drag)] =================
        # 1. 获取当前无人机的速度和姿态
        lin_vel_w = self._robot.data.root_lin_vel_w  # 世界系线速度 (N, 3)
        ang_vel_b = self._robot.data.root_ang_vel_b  # 机体系角速度 (N, 3)
        quat_w = self._robot.data.root_quat_w        # 世界系姿态四元数 (N, 4)

        # 2. 坐标转换矩阵 (World -> Body)
        rot_matrix_b2w = matrix_from_quat(quat_w)
        rot_matrix_w2b = rot_matrix_b2w.transpose(1, 2) 

        # 3. 设置环境风速 (世界坐标系)
        wind_vel_w = torch.zeros_like(lin_vel_w)
        # wind_vel_w[:, 0] = 1.5  # 假设有 1.5m/s 的 X 轴阵风

        # 4. 计算相对于空气的线速度，并转换到机体系
        rel_lin_vel_w = lin_vel_w - wind_vel_w
        rel_lin_vel_b = torch.bmm(rot_matrix_w2b, rel_lin_vel_w.unsqueeze(-1)).squeeze(-1)

        # 5. 定义阻力系数 (Drag Coefficients)
        c_drag_lin = torch.tensor([0.005, 0.005, 0.008], device=self.device)
        c_drag_ang = torch.tensor([0.0001, 0.0001, 0.0003], device=self.device)

        # c_drag_lin = torch.tensor([0, 0, 0], device=self.device)
        # c_drag_ang = torch.tensor([0, 0, 0], device=self.device)

        # 6. 计算气动阻力与阻力矩
        force_drag_b = -c_drag_lin * rel_lin_vel_b
        torque_drag_b = -c_drag_ang * ang_vel_b

        # -------------------------------------------------------------------------
        # [新增 DEBUG] 7. 计算并打印风阻带来的额外线加速度和角加速度
        # -------------------------------------------------------------------------
        # 线加速度 = 力 / 质量 (注意维度对齐)
        acc_drag_b = force_drag_b / self.mass_tensor.unsqueeze(-1)
        
        # 角加速度 = 力矩 / 惯性张量 (因为 inertia 是对角阵，直接元素级相除即可)
        ang_acc_drag_b = torque_drag_b / self.inertia_tensor

        # 限制打印频率 (例如每 50 步，即 0.5秒 打印一次 Env 0 的状态)
        if hasattr(self, "_figure8_time"):
            step_count = int(self._figure8_time[0].item() / self.dt)
            if step_count % 50 == 0:
                import numpy as np
                np_set_printoptions = np.get_printoptions()
                np.set_printoptions(precision=4, suppress=True) # 设置打印精度
                print(f"\n--- [Wind Drag Debug {self._figure8_time[0].item():.2f}s] Env 0 ---")
                print(f"Rel Vel (Body)      : {rel_lin_vel_b[0].cpu().numpy()} m/s")
                print(f"Force Drag (N)      : {force_drag_b[0].cpu().numpy()}")
                print(f"Added Lin Acc       : {acc_drag_b[0].cpu().numpy()} m/s^2")
                print(f"Added Ang Acc       : {ang_acc_drag_b[0].cpu().numpy()} rad/s^2")
                print(f"--------------------------------------------------")
                np.set_printoptions(**np_set_printoptions)
        # -------------------------------------------------------------------------

        # 8. 将阻力叠加到电机的原始输出上
        force_b = force_b + force_drag_b
        torque_b = torque_b + torque_drag_b
        # ========================================================================= 

        # # ================= [新增：卸桨平放/完美悬停 Debug 实验] =================
        # if self.cfg.trajectory_type == "figure8":
        #     current_t = self._figure8_time[0].item()
        #     if current_t < self._figure8_warmup_duration:
        #         # 1. 打印你想看的 motor speeds (也就是卸桨平放时，电机的期望转速)
        #         step_count = int(current_t / self.dt)
        #         if step_count % 1 == 0:
        #             print(f"[Desk Test {current_t:.2f}s] Env 0 motor speeds: {self._current_motor_speeds[0].cpu().numpy()}")
        #             print(f"[Desk Test {current_t:.2f}s] Env 0 target actions: {action_setpoint_normalized[0].cpu().numpy()}")
                
        #         # 2. 物理断开：不论策略输出什么力矩和推力，全设为 0
        #         force_b = torch.zeros_like(force_b)
        #         torque_b = torch.zeros_like(torque_b)
                
        #         # 施加一个完美的抗重力 (假设 Z 轴朝上，重力加速度为 9.81)
        #         # 这样物理引擎里它就不会往下掉
        #         force_b[:, 2] = self.mass_tensor * 9.81
                
        #         # 3. 强行锁死状态：防止物理引擎由于浮点数误差产生任何微小漂移
        #         env_ids = torch.arange(self.num_envs, device=self.device)
                
        #         # ================= [修改：构造右前方倾斜的姿态] =================
        #         # 右前倾斜：相当于同时产生 Roll (右倾) 和 Pitch (前倾) 
        #         # 假设设定倾斜角为 15 度
        #         tilt_deg = 0.0
        #         roll_angle = math.radians(-tilt_deg)   # 正值通常为右倾
        #         pitch_angle = math.radians(tilt_deg)  # 正值通常为前倾
                
        #         roll_t = torch.full((self.num_envs,), roll_angle, device=self.device)
        #         pitch_t = torch.full((self.num_envs,), pitch_angle, device=self.device)
        #         yaw_t = torch.full((self.num_envs,), math.radians(0.0), device=self.device)
                
        #         # 利用现成的工具函数生成四元数
        #         tilted_quat = quat_from_euler_xyz(roll_t, pitch_t, yaw_t)
                
        #         # 拼装固定位姿
        #         fixed_pose = torch.cat([self.pos_des, tilted_quat], dim=-1)
        #         # ==============================================================
        #         fixed_vel = torch.zeros(self.num_envs, 6, device=self.device)
                
        #         # 每一步都把位置锁死在期望悬停点，速度全部清零
        #         self._robot.write_root_pose_to_sim(fixed_pose, env_ids)
        #         self._robot.write_root_velocity_to_sim(fixed_vel, env_ids)
        # # ======================================================================

        # 5. 施加力
        self._forces.zero_()
        self._torques.zero_()
        self._forces[:, 0, :] = force_b
        self._torques[:, 0, :] = torque_b

    def _apply_action(self):
        """Apply thrust/moment to the quadcopter."""
        self._robot.set_external_force_and_torque(self._forces, self._torques, body_ids=self._body_id)

    def _get_observations(self) -> dict:

        # 更新轨迹
        if self.cfg.trajectory_type == "figure8":
            self._generate_desired_trajectory_figure8()
        elif torch.any(self._is_langevin_task):
            self._generate_desired_trajectory_langevin(env_ids=torch.where(self._is_langevin_task)[0])

        pos_w = self._robot.data.root_pos_w
        quat_w = self._robot.data.root_quat_w
        vel_w = self._robot.data.root_lin_vel_w
        ang_vel_b = self._robot.data.root_ang_vel_b
        
        # 1. 坐标系转换矩阵 (World -> Body)
        rot_matrix_b2w = matrix_from_quat(quat_w)
        rot_matrix_w2b = rot_matrix_b2w.transpose(1, 2) 

        # yaw角观测
        _, _, y = euler_xyz_from_quat(quat_w)
        yaw_error = self.yaw_des - y
        yaw_error = torch.remainder(yaw_error + math.pi, 2 * math.pi) - math.pi
        yaw_error_sin = torch.sin(yaw_error).unsqueeze(1) # (N, 1)
        yaw_error_cos = torch.cos(yaw_error).unsqueeze(1) # (N, 1)

        # 2. 计算当前帧在机体系下的误差
        curr_pos_error_b = torch.bmm(rot_matrix_w2b, (pos_w - self.pos_des).unsqueeze(-1)).squeeze(-1)
        curr_vel_error_b = torch.bmm(rot_matrix_w2b, (vel_w - self.vel_des).unsqueeze(-1)).squeeze(-1)

        # 3. 更新历史 Buffer (Rolling window)
        self.pos_error_b_history = torch.roll(self.pos_error_b_history, shifts=-1, dims=1)
        self.vel_error_b_history = torch.roll(self.vel_error_b_history, shifts=-1, dims=1)
        self.pos_error_b_history[:, -1, :] = curr_pos_error_b
        self.vel_error_b_history[:, -1, :] = curr_vel_error_b

        pos_error_flat = self.pos_error_b_history.reshape(self.num_envs, -1) # (N, 15)
        vel_error_flat = self.vel_error_b_history.reshape(self.num_envs, -1) # (N, 15)

        # 4. 展平数据
        rot_flat = rot_matrix_b2w.reshape(self.num_envs, 9)

        # 5. 处理期望量（转到机体系）
        acc_des_b = torch.bmm(rot_matrix_w2b, self.acc_des.unsqueeze(-1)).squeeze(-1)
        vel_des_b = torch.bmm(rot_matrix_w2b, self.vel_des.unsqueeze(-1)).squeeze(-1)

        # 6. 学生
        obs_student = torch.cat([
            curr_pos_error_b,             # 3
            rot_flat,                   # 9
            curr_vel_error_b,             # 3
            ang_vel_b,                  # 3
            self._last_actions,         # 4
            yaw_error_sin,              # 1
            yaw_error_cos,              # 1 
        ], dim=-1)

        # 7. 教师
        obs_teacher = torch.cat([
            pos_error_flat,             # 15
            rot_flat,                   # 9
            vel_error_flat,             # 15
            ang_vel_b,                  # 3
            self._last_actions,         # 4
            acc_des_b,                  # 3 
            vel_des_b,                  # 3 
            yaw_error_sin,              # 1
            yaw_error_cos,              # 1 
            self._current_motor_speeds, # 4
        ], dim=-1)

        obs_teacher = self.CHECK_NAN(obs_teacher, "Teacher Observation")
        obs_student = self.CHECK_NAN(obs_student, "Student Observation")
        
        # # ================= [新增：注入实物左前倾斜的真实日志数据] =================
        # if self.cfg.trajectory_type == "figure8":
        #     current_t = self._figure8_time[0].item()
        #     if current_t < self._figure8_warmup_duration:
        #         ideal_obs = torch.zeros_like(obs_student)
                
        #         # 1. Pos Err (B)
        #         ideal_obs[:, 0:3] = torch.tensor([0.0067, -0.0247, -0.0512], device=self.device)
                
        #         # 2. Rotation Matrix
        #         ideal_obs[:, 3:12] = torch.tensor([0.9993, -0.0053, 0.0363, 0.0061, 0.9998, -0.0214, -0.0362, 0.0216, 0.9991], device=self.device)
                
        #         # 3. Vel Err (B)
        #         ideal_obs[:, 12:15] = torch.tensor([0.0232, 0.2497, 0.8012], device=self.device)
                
        #         # 4. Ang Vel
        #         ideal_obs[:, 15:18] = torch.tensor([-0.0054, -0.0577, 0.0072], device=self.device)
                
        #         # 5. Last Act
        #         # 这里我们保持闭环 (使用 self._last_actions)，以便观察电机转速会收敛到什么状态去救机。
        #         # 如果你想严格测试网络面对上一帧确切动作时的"单步推理结果"，可以把下面那行解开注释：
        #         ideal_obs[:, 18:22] = self._last_actions 
        #         # ideal_obs[:, 18:22] = torch.tensor([-0.58, -1.00, -0.97, -0.81], device=self.device)
                
        #         # 6. Yaw Err Sin/Cos
        #         ideal_obs[:, 22] = 0.0007
        #         ideal_obs[:, 23] = 0.9999
                
        #         # 覆写真实观测
        #         obs_student = ideal_obs

        #         # 打印确认
        #         step_count = int(current_t / self.dt)
        #         if step_count % 50 == 0:
        #             import numpy as np
        #             np_set_printoptions = np.get_printoptions()
        #             np.set_printoptions(precision=4, suppress=True)
        #             print(f"\n--- [Desk Test {current_t:.2f}s] Env 0 FORCED REAL LOG Obs ---")
        #             print(f"Forced Obs array: {obs_student[0].cpu().numpy()}")
        #             print(f"--------------------------------------------------")
        #             np.set_printoptions(**np_set_printoptions)
        # # =========================================================================
        # print("obs_student:", obs_student[0].cpu().numpy())        
        return {"policy": obs_student, "teacher": obs_teacher, "rnd_state": obs_student}
    def _get_rewards(self) -> torch.Tensor:

        pos_error_norm = torch.norm(self._robot.data.root_pos_w - self.pos_des, dim=1)

        # yaw角姿态cost
        quat_w = self._robot.data.root_quat_w
        _, _, yaw_curr = euler_xyz_from_quat(quat_w)
        yaw_error = self.yaw_des - yaw_curr
        yaw_error = torch.remainder(yaw_error + torch.pi, 2 * torch.pi) - torch.pi
        yaw_error_mapped = torch.sin(yaw_error / 2.0)
        orientation_cost = torch.arccos(torch.clamp(1.0 - torch.abs(yaw_error_mapped), -1.0, 1.0))

        # 3. 动作平滑 Cost (保持不变)
        d_action_cost = torch.norm(self._actions - self._last_actions, dim=1)
        
        # 4. 计算各项 Reward
        r_pos = -pos_error_norm * self.cfg.reward_coef_position_cost
        r_ori = -orientation_cost * self.cfg.reward_coef_orientation_cost # 这里只包含 Yaw 误差
        r_act = -d_action_cost * self.cfg.reward_coef_d_action_cost
        r_base = self.cfg.reward_constant
        r_term = -self._numerical_is_unstable.float() * self.cfg.reward_coef_termination_penalty

        # 5. 记录日志 (Accumulate sums)
        sums = {
            "position": r_pos, 
            "orientation": r_ori, 
            "action_smooth": r_act, 
            "base": torch.full_like(r_pos, self.cfg.reward_constant), 
            "terminal": r_term
        }
        for k, v in sums.items():
            if k not in self._episode_sums: self._episode_sums[k] = torch.zeros_like(v)
            self._episode_sums[k] += v

        self._last_actions = self._actions.clone()
        return r_pos + r_ori + r_act + r_base + r_term



    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Define terminations and timeouts."""
        if self.cfg.train_or_play:
            time_out = self.episode_length_buf >= self.max_episode_length - 1
        else:
            time_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        self.CHECK_state()

        # Check distance from desired trajectory position (Langevin threshold)
        dist_traj_from_spawn = torch.norm(self.pos_des - self._spawn_pos_w, dim=1)
        position_exceeded_langevin = dist_traj_from_spawn > self.cfg.position_threshold_langevin

        conditions = [
            self._numerical_is_unstable,  # Numerical instability
            position_exceeded_langevin,  # Distance from desired trajectory exceeds threshold
        ]

        # Combine all die conditions
        died = conditions[0]
        for condition in conditions[1:]:
            died = torch.logical_or(died, condition)

        if "log" not in self.extras:
            self.extras["log"] = dict()
        completed_mask = torch.logical_or(died, time_out)
        completed_episodes = torch.sum(completed_mask == True).item()
        if completed_episodes > 0:
            died_episodes = torch.sum(died == True).item()
            timeout_episodes = torch.sum(time_out == True).item()
            self.extras["log"].update({
                    "Metrics/died_episodes_per_step": died_episodes,
                    "Metrics/completed_episodes_per_step": completed_episodes,
                    "Metrics/timeout_episodes_per_step": timeout_episodes,
            })
        else:
            self.extras["log"].update({
                "Metrics/died_episodes_per_step": 0,
                "Metrics/completed_episodes_per_step": 0,
                "Metrics/timeout_episodes_per_step": 0,
            })

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specific environment indexes."""
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        num_resets = len(env_ids)
        
        # --- 1. 日志记录与基础状态重置 ---
        if num_resets > 0:
            if "log" not in self.extras:
                self.extras["log"] = dict()
            for key in self._episode_sums.keys():
                values = self._episode_sums[key][env_ids]
                self.extras["log"][f"Episode_Reward/{key}"] = torch.mean(values).item()
                self._episode_sums[key][env_ids] = 0.0

        died_mask = self.reset_terminated[env_ids]
        timed_out_mask = self.reset_time_outs[env_ids]
        if hasattr(self, '_update_episode_outcomes_and_metrics'):
            self._update_episode_outcomes_and_metrics(env_ids, None, died_mask, timed_out_mask)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # # 清除动作和物理标志位
        self._actions[env_ids] = 0.0
        # ==========================================
        # [修改] 2. last_action 改为推重比倒数的平方根，再映射到 [-1, 1]
        # 推力与电机转速的平方成正比 (Thrust ∝ ω^2)
        # 悬停所需的电机转速比例 ω = sqrt(1.0 / TWR)
        # ==========================================
        hover_motor_speed = torch.sqrt(1.0 / self.twr_tensor[env_ids])
        
        # 将 [0, 1] 的电机转速映射到神经网络的 [-1, 1] 动作空间
        hover_action = hover_motor_speed * 2.0 - 1.0 
        self._last_actions[env_ids] = hover_action.unsqueeze(1).expand(-1, 4).clone()

        # self._last_actions[env_ids] = 0.0
        self._forces[env_ids] = 0.0
        self._torques[env_ids] = 0.0
        self._numerical_is_unstable[env_ids] = False
        self._died_pos_limit[env_ids] = False
        self._died_lin_vel_limit[env_ids] = False
        self._died_ang_vel_limit[env_ids] = False
        self._died_tilt_limit[env_ids] = False
        self._died_nan[env_ids] = False
        self._figure8_time[env_ids] = 0.0
        self._traj_origin_adjusted[env_ids] = False
        self._current_motor_speeds[env_ids] = 0.0 

        # --- 2. 确定位置与随机状态采样 (核心修正点：先定义变量) ---
        spawn_center = self.env_origins[env_ids].clone()
        spawn_center[:, 0] += self.cfg.terrain_length / 2.0
        spawn_center[:, 1] += self.cfg.terrain_width / 2.0
        spawn_center[:, 2] = self.cfg.height
        self._spawn_pos_w[env_ids] = spawn_center 

        l_arm = self.arm_l_tensor[env_ids]

        if self.cfg.train_or_play:
            # 定义采样函数
            def sample_in_sphere(radius, n_samples):
                direction = F.normalize(torch.randn(n_samples, 3, device=self.device), p=2, dim=1)
                if not isinstance(radius, (float, int)):
                    radius = radius.view(-1, 1)
                r = radius * torch.pow(torch.rand(n_samples, 1, device=self.device), 1.0 / 3.0)
                return direction * r

            pos_offset = sample_in_sphere(3.0 * l_arm, num_resets)
            lin_vel = sample_in_sphere(0.5, num_resets)
            ang_vel = sample_in_sphere(0.5, num_resets)
            
            roll = (torch.rand(num_resets, device=self.device) * 2 - 1) * (math.pi / 8)
            pitch = (torch.rand(num_resets, device=self.device) * 2 - 1) * (math.pi / 8)
            yaw = (torch.rand(num_resets, device=self.device) * 2 - 1) * (math.pi / 6)
            quat = quat_from_euler_xyz(roll, pitch, yaw)

            # 10% 完美开局
            is_perfect = torch.rand(num_resets, device=self.device) < 0.10 
            pos_offset[is_perfect] = 0.0
            lin_vel[is_perfect] = 0.0
            ang_vel[is_perfect] = 0.0
            quat[is_perfect] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

            self.yaw_des[env_ids] = (torch.rand(len(env_ids), device=self.device) * 2 - 1) * self.yaw_limit / 2.0
            self.yaw_rate_des[env_ids] = 0.0
        else:
            pos_offset = torch.zeros(num_resets, 3, device=self.device)
            lin_vel = torch.zeros(num_resets, 3, device=self.device)
            ang_vel = torch.zeros(num_resets, 3, device=self.device)
            quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(num_resets, 1)
            self.yaw_des[env_ids] = 0.0
            self.yaw_rate_des[env_ids] = 0.0
        print("pos_offset sample:", pos_offset[0].cpu().numpy())
        print("lin_vel sample:", lin_vel[0].cpu().numpy())
        print("ang_vel sample:", ang_vel[0].cpu().numpy())
        print("quat sample:", quat[0].cpu().numpy())

        # --- 3. 根据采样结果同步初始化 Buffer 和期望值 ---
        self.pos_des[env_ids] = spawn_center
        self.vel_des[env_ids] = 0.0
        self.acc_des[env_ids] = 0.0
        
        # 转换到机体系并填充 5 帧历史
        rot_w2b_init = matrix_from_quat(quat).transpose(1, 2)
        initial_pos_err_b = torch.bmm(rot_w2b_init, pos_offset.unsqueeze(-1)).squeeze(-1)
        initial_vel_err_b = torch.bmm(rot_w2b_init, lin_vel.unsqueeze(-1)).squeeze(-1)
        
        self.pos_error_b_history[env_ids] = initial_pos_err_b.unsqueeze(1).repeat(1, self.history_len, 1)
        self.vel_error_b_history[env_ids] = initial_vel_err_b.unsqueeze(1).repeat(1, self.history_len, 1)

        # --- 4. 写入物理仿真器 ---
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] = spawn_center + pos_offset
        root_state[:, 3:7] = quat
        root_state[:, 7:10] = lin_vel
        root_state[:, 10:13] = ang_vel

        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(
            self._robot.data.default_joint_pos[env_ids], 
            self._robot.data.default_joint_vel[env_ids], 
            None, env_ids
        )

        # 任务类型分配
        self._is_langevin_task[env_ids] = torch.rand(num_resets, device=self.device) > self.cfg.prob_null_trajectory

    def _set_debug_vis_impl(self, debug_vis: bool):
            """Show debug markers if debug_vis is True."""
            
            print(f"debug_vis: {self.cfg.debug_vis}")

            if debug_vis:
                # 1. Goal Position
                if not hasattr(self, "goal_pos_visualizer"):
                    marker_cfg = CUBOID_MARKER_CFG.copy()
                    marker_cfg.markers["cuboid"].size = (self.cfg.marker_size, self.cfg.marker_size, self.cfg.marker_size)
                    marker_cfg.prim_path = "/Visuals/Command/goal_position"
                    self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
                self.goal_pos_visualizer.set_visibility(True)

                # 2. Goal Yaw
                if not hasattr(self, "goal_yaw_visualizer"):
                    goal_arrow_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                    goal_arrow_cfg.markers["arrow"].scale = (self.cfg.marker_size, self.cfg.marker_size, self.cfg.marker_size*4)
                    goal_arrow_cfg.prim_path = "/Visuals/Command/goal_yaw"
                    self.goal_yaw_visualizer = VisualizationMarkers(goal_arrow_cfg)
                self.goal_yaw_visualizer.set_visibility(True)

                # 3. Current Robot Position (蓝色球)
                if not hasattr(self, "current_yaw_visualizer"):
                    print("create robot position visualizer (Sphere)")
                    current_vis_cfg = SPHERE_MARKER_CFG.copy()
                    current_vis_cfg.prim_path = "/Visuals/Command/current_yaw"
                    scale_val = self.cfg.marker_size
                    current_vis_cfg.markers["sphere"].scale = (scale_val, scale_val, scale_val)
                    current_vis_cfg.markers["sphere"].visual_material.diffuse_color = (0.0, 0.0, 1.0)
                    self.current_yaw_visualizer = VisualizationMarkers(current_vis_cfg)
                self.current_yaw_visualizer.set_visibility(True)

                # 4. Langevin Trajectory (绿色球)
                if not hasattr(self, 'traj_langevin_visualizer'):
                    print("create langevin trajectory visualizer (Sphere)")
                    langevin_cfg = SPHERE_MARKER_CFG.copy()
                    langevin_cfg.prim_path = "/Visuals/TrajectoryLangevin"
                    langevin_cfg.markers["sphere"].scale = (self.cfg.marker_size, self.cfg.marker_size, self.cfg.marker_size)
                    langevin_cfg.markers["sphere"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
                    self.traj_langevin_visualizer = VisualizationMarkers(langevin_cfg)
                self.traj_langevin_visualizer.set_visibility(True)

                # 5. Fixed/Null Trajectory (红色球)
                if not hasattr(self, 'traj_fixed_visualizer'):
                    print("create fixed trajectory visualizer (Sphere)")
                    fixed_cfg = SPHERE_MARKER_CFG.copy()
                    fixed_cfg.prim_path = "/Visuals/TrajectoryFixed"
                    fixed_cfg.markers["sphere"].scale = (self.cfg.marker_size, self.cfg.marker_size, self.cfg.marker_size)
                    fixed_cfg.markers["sphere"].visual_material.diffuse_color = (1.0, 0.0, 0.0)
                    self.traj_fixed_visualizer = VisualizationMarkers(fixed_cfg)
                self.traj_fixed_visualizer.set_visibility(True)
                    
            else:
                if hasattr(self, "goal_pos_visualizer"): self.goal_pos_visualizer.set_visibility(False)
                if hasattr(self, "goal_yaw_visualizer"): self.goal_yaw_visualizer.set_visibility(False)
                if hasattr(self, "current_yaw_visualizer"): self.current_yaw_visualizer.set_visibility(False)
                if hasattr(self, "traj_langevin_visualizer"): self.traj_langevin_visualizer.set_visibility(False)
                if hasattr(self, "traj_fixed_visualizer"): self.traj_fixed_visualizer.set_visibility(False)
                if hasattr(self, 'traj_visualizer'): self.traj_visualizer.set_visibility(False)
                
    def _debug_vis_callback(self, event):
            """Update debug markers with current robot pose."""
            
            # 1. 机器人本体的蓝色箭头
            if hasattr(self, "current_yaw_visualizer"):
                self.current_yaw_visualizer.visualize(self._robot.data.root_pos_w, self._robot.data.root_quat_w)

            # 构造一个单位四元数
            fixed_rot = torch.zeros_like(self._robot.data.root_quat_w)
            fixed_rot[:, 0] = 1.0 
            
            invisible_pos = torch.zeros_like(self.pos_des) 
            invisible_pos[:, 2] = -100.0 
            
            # 2. 更新 Langevin 绿色球
            if hasattr(self, "traj_langevin_visualizer"):
                pos_green = invisible_pos.clone()
                mask = self._is_langevin_task
                if mask.any():
                    pos_green[mask] = self.pos_des[mask]
                self.traj_langevin_visualizer.visualize(pos_green, fixed_rot)

            # 3. 更新 Fixed 红色球
            if hasattr(self, "traj_fixed_visualizer"):
                pos_red = invisible_pos.clone()
                mask = ~self._is_langevin_task
                if mask.any():
                    pos_red[mask] = self.pos_des[mask]
                self.traj_fixed_visualizer.visualize(pos_red, fixed_rot)
            
    def _update_episode_outcomes_and_metrics(self, env_ids, success_mask, died_mask, timed_out_mask):
            """
            Update episode statistics with detailed failure reasons.
            """
            completed_mask = torch.logical_or(died_mask, timed_out_mask)
            if not torch.any(completed_mask):
                return 0, 0

            completed_env_ids = env_ids[completed_mask]
            died_env_ids = env_ids[died_mask]
            
            # Process termination reasons for died episodes
            if len(died_env_ids) > 0:
                reason_pos_limit = self._died_pos_limit[died_env_ids].cpu().numpy()
                reason_lin_vel = self._died_lin_vel_limit[died_env_ids].cpu().numpy()
                reason_ang_vel = self._died_ang_vel_limit[died_env_ids].cpu().numpy()
                reason_tilt = self._died_tilt_limit[died_env_ids].cpu().numpy()
                reason_nan = self._died_nan[died_env_ids].cpu().numpy()
                
                des_w = self.pos_des[died_env_ids]
                spawn_w = self._spawn_pos_w[died_env_ids]
                dist_traj_from_spawn = torch.norm(des_w - spawn_w, dim=1)
                reason_langevin = (dist_traj_from_spawn > self.cfg.position_threshold_langevin).cpu().numpy()
                
                for i in range(len(died_env_ids)):
                    self._termination_reason_history.append({
                        "died_pos_limit": bool(reason_pos_limit[i]),
                        "died_lin_vel": bool(reason_lin_vel[i]),
                        "died_ang_vel": bool(reason_ang_vel[i]),
                        "died_tilt": bool(reason_tilt[i]),
                        "died_nan": bool(reason_nan[i]),
                        "position_exceeded_langevin": bool(reason_langevin[i])
                    })
            
            # Add empty dictionaries for timeout cases
            timeout_count_current = len(env_ids[timed_out_mask])
            self._termination_reason_history.extend([{}] * timeout_count_current)
            
            # Track average velocity
            if len(completed_env_ids) > 0:
                vel_abs = torch.linalg.norm(
                    self._robot.data.root_lin_vel_w[completed_env_ids], 
                    dim=1
                ).cpu().tolist()
                self._vel_abs.extend(vel_abs)

            # Calculate statistics
            num_termination_records = len(self._termination_reason_history)
            if num_termination_records > 0:
                reason_keys = [
                    "died_pos_limit", 
                    "died_lin_vel", 
                    "died_ang_vel", 
                    "died_tilt", 
                    "died_nan", 
                    "position_exceeded_langevin"
                ]
                reason_counts = {key: 0 for key in reason_keys}
                
                if len(self._termination_reason_history) > 0:
                    for reason in self._termination_reason_history:
                        for key in reason_keys:
                            if key in reason and reason[key]:
                                reason_counts[key] += 1
                
                died_count = sum(1 for r in self._termination_reason_history if r)
                timeout_count = num_termination_records - died_count
                
                self._episodes_completed += len(completed_env_ids)
                avg_velocity = np.mean(list(self._vel_abs)) if self._vel_abs else 0.0

                if "log" not in self.extras:
                    self.extras["log"] = {}
                
                self.extras["log"].update({
                    "Episode_Termination/died": died_count / num_termination_records * 100.0,
                    "Episode_Termination/time_out": timeout_count / num_termination_records * 100.0,
                    "Metrics/average_velocity": avg_velocity,
                    "Metrics/episodes_completed": self._episodes_completed,
                })

                for key in reason_keys:
                    self.extras["log"][f"Metrics/Died/{key}"] = reason_counts[key] / num_termination_records * 100.0

                return len(completed_env_ids), 0
    
    def close(self):
        """Clean up resources when environment is closed."""
        super().close()

    def get_pid_actions(self) -> torch.Tensor:
        """
        利用内置的 PID 控制器计算当前状态下的理想动作 (Motor Speeds [0,1])。
        用于 PID 蒸馏教学。
        """
        # 1. 获取状态
        cur_pos = self._robot.data.root_pos_w
        cur_vel = self._robot.data.root_lin_vel_w
        cur_quat = self._robot.data.root_quat_w
        cur_ang_vel = self._robot.data.root_ang_vel_b
        
        # 2. 计算期望转速 (Controller)
        target_motor_speeds = self._controller.compute_target_speeds(
            cur_pos, cur_vel, cur_quat, cur_ang_vel,
            self.pos_des, self.vel_des, self.acc_des, self.yaw_des,
            self._current_motor_speeds
        )
        
        # 映射到动作空间 [-1, 1]
        # 环境动作通常是归一化的 [-1, 1]，而 PID 输出是 [0, 1] (Motor Speeds)
        # 根据 env._pre_physics_step: 
        # action_setpoint_normalized = (raw_actions_clamped + 1.0) * 0.5
        # 所以: raw_action = setpoint * 2.0 - 1.0
        
        pid_actions = target_motor_speeds * 2.0 - 1.0
        
        return torch.clamp(pid_actions, -1.0, 1.0)
