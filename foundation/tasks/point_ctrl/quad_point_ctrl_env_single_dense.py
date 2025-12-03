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

from foundation.utils.wind_gen import WindGustGenerator

from enum import IntEnum
import collections
import itertools


MAP_SIZE = (500, 500) 

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
    replicate_physics: bool = True

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
    # custom config for the quadcopter environment

    # Updated observation space: pos_error(3) + rot_matrix(9) + vel_error(3) + ang_vel(3) + last_actions(4) + motor_speeds(4)
    frame_observation_space = 3 + 9 + 3 + 3 + 4 + 4  # 26

    # Calculate total observation space (without depth history, only current frame)
    observation_space = frame_observation_space  # 26D: pos_error(3) + rot_matrix(9) + vel_error(3) + ang_vel(3) + last_actions(4) + motor_speeds(4)


    prob_null_trajectory = 0.5  # 50% 概率做定点控制

    # 轨迹类型选择: "langevin" 或 "figure8"
    trajectory_type = "langevin"  # Default to Langevin during training

    train_or_play: bool = True  # 默认为 True (训练模式)，命令行可通过 --train_or_play=False 修改

    # gamma in ppo, only for logging
    gamma = 0.99

    # env
    episode_length_s = 96
    decimation = 1
    action_space = 4 
    state_space = 0
    debug_vis = True

    map_size = MAP_SIZE

    grid_rows = 40 # 12
    grid_cols = 40 # 1
    terrain_width = 10
    terrain_length = 10
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

    # Controller parameters
    controller_Kang = [8.0, 8.0, 8.0]  # Roll and pitch angle controller gains   #15 15 20
    controller_Kdang = [1.0, 1.0, 1.0]                                            #0.8 0.8 1.2
    controller_Kang_vel = [15.0, 15.0, 15.0]  # Roll, pitch, and yaw angular velocity controller gains

    enable_aero_drag = False
    drag_coeffs = (0.003, 0.003, 0.003)   # dx, dy, dz，机体系“转子/气动阻力系数”，可按机架调
    drag_rand_scale = 0.5              # 域随机化幅度：±50%（论文设置）
    drag_v_clip = 8.0                 # 可选：速度范数上限，避免数值爆

    # scene
    scene: InteractiveSceneCfg = QuadcopterSceneCfg()

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # thresholds
    too_low = 0.3
    too_high = 1.7
    desired_low = 0.5  
    desired_high = 1.5

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

    enable_wind_generator = False

class QuadcopterEnv(DirectRLEnv):
    """A quadcopter environment adapted to use the reward logic from the training code."""

    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.start_time = time.time()

        self.render_mode = "human"

        # Initialize wind generator
        if self.cfg.enable_wind_generator:
            self._wind_gen = WindGustGenerator(
                num_envs=self.num_envs,
                device=self.device,
                dt=self.cfg.sim.dt,
                tau=1.0,
                sigma=0.5
            )
        else:
            self._wind_gen = None

        self.motor_tau = 0.05 # 假设值，RAPTOR中是采样的
        self.dt = self.cfg.sim.dt
        self.motor_alpha = self.dt / (self.dt + self.motor_tau)
        self._current_motor_speeds = torch.zeros(self.num_envs, 4, device=self.device)
        
        # Controller

        mass_tensor = torch.full((self.num_envs,), 0.0282, device=self.device)       
        arm_l_tensor = torch.full((self.num_envs,), 0.04384, device=self.device)
        inertia_tensor = torch.tensor([2.44864e-5, 2.44864e-5, 3.61504e-5], device=self.device).repeat(self.num_envs, 1)
        
        # Store the robot mass for wind force calculation
        self._robot_mass = mass_tensor
        # --- Aerodynamic drag setup (paper model) ---
        dx, dy, dz = self.cfg.drag_coeffs
        self._drag_D = torch.tensor([dx, dy, dz], device=self.device).repeat(self.num_envs, 1)

        self._controller = SimpleQuadrotorController(
            num_envs=self.num_envs,
            device=self.device,
            attitude_p_gain=torch.tensor(self.cfg.controller_Kang, device=self.device, dtype=torch.float32),
            attitude_d_gain=torch.tensor(self.cfg.controller_Kdang, device=self.device, dtype=torch.float32),
            rate_p_gain=torch.tensor(self.cfg.controller_Kang_vel, device=self.device, dtype=torch.float32),
            # 新增参数传递
            mass=mass_tensor,
            arm_length=arm_l_tensor,
            inertia=inertia_tensor
        )

        self._is_langevin_task = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # === [修改] 细化死亡原因的标志位 ===
        self._numerical_is_unstable = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 新增具体的死亡原因记录 (Sub-reasons)
        self._died_pos_limit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)      # 飞出半径
        self._died_lin_vel_limit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # 线速度过大
        self._died_ang_vel_limit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # 角速度过大
        self._died_tilt_limit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)     # 倾角 > 90度
        self._died_nan = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)            # 数值 NaN

        # Quadcopter references
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._torques = torch.zeros(self.num_envs, 1, 3, device=self.device)
        
        # Desired states for observation and reward (maintained across steps)
        self.pos_des = torch.zeros(self.num_envs, 3, device=self.device)  # Desired position (smoothed)
        self.vel_des = torch.zeros(self.num_envs, 3, device=self.device)  # Desired velocity (smoothed)
        
        # Raw (unsmoothed) states for Langevin dynamics
        self.pos_des_raw = torch.zeros(self.num_envs, 3, device=self.device)  # Raw position
        self.vel_des_raw = torch.zeros(self.num_envs, 3, device=self.device)  # Raw velocity
        
        # Langevin trajectory generation parameters (damped harmonic oscillator with noise)
        self._langevin_dt = 0.01  # Time step for integration
        self._langevin_friction = 0.5  # Damping coefficient (gamma)
        self._langevin_omega = 1.5  # Oscillator frequency (omega)
        self._langevin_sigma = 3.0  # Noise intensity (sigma)
        self._langevin_alpha = 0.2  # Smoothing factor for exponential moving average (alpha)
        
        # Figure-8 trajectory parameters
        self._figure8_time = torch.zeros(self.num_envs, device=self.device)  # Time variable for figure-8
        self._figure8_frequency = 0.1  # Frequency of the figure-8 motion (Hz)
        self._figure8_scale_x = 1.0  # Scale of the figure-8 in x direction (meters)
        self._figure8_scale_y = 0.5  # Scale of the figure-8 in y direction (meters)
        self._figure8_height = 3.0  # Height of the figure-8 trajectory (meters)
        self._figure8_warmup_duration = 5.0  # 热身阶段时长 (秒),在此期间保持初始位置

        # Logging
        # [修改] 键名必须与 _get_rewards 中的 reward_items 字典键名完全一致
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "position",       # 原代码是 "position_penalty"
                "orientation",    # 原代码是 "orientation_penalty"
                "action_smooth",  # 原代码是 "action_smoothness_penalty"
                "base",           # 原代码是 "base_reward"
                "terminal",       # 原代码是 "terminal_penalty" <-- 这就是你要删的那个
            ]
        }
        

        # Environment origins
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        self.grid_idx = None
        # Robot references
        self._body_id = self._robot.find_bodies("body")[0]

        self._last_actions = torch.zeros(self.num_envs, 4, device=self.device) # 
        self._numerical_is_unstable = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._spawn_pos_w = torch.zeros(self.num_envs, 3, device=self.device)  # Store spawn/respawn positions

        self._last_angular_velocity= torch.zeros(self.num_envs, 3, device=self.device)

        # 默认为 3.0，后续会在 reset 中更新
        self._langevin_max_vel = torch.full((self.num_envs,), 3.0, device=self.device)

        # Episode tracking for trajectory following (no success criterion)
        self._history_window = 100
        self._episode_outcomes = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # For compatibility
        self._episodes_completed = 0
        self._termination_reason_history = collections.deque(maxlen=self._history_window)
        self._vel_abs = collections.deque(maxlen=self._history_window)

        self.set_debug_vis(self.cfg.debug_vis)


        self._calc_env_origins()

    def CHECK_NAN(self, tensor, name):
        if torch.isnan(tensor).any().item():
            print(f"[{name}] NaN detected in tensor of shape {tensor.shape}.")
            nan_env_mask = torch.any(torch.isnan(tensor), dim=1)
            
            # === [修改] 记录具体的 NaN 错误 ===
            self._died_nan = torch.logical_or(self._died_nan, nan_env_mask)
            self._numerical_is_unstable = torch.logical_or(self._numerical_is_unstable, nan_env_mask)
            # =================================
            
            nan_env_indices = torch.where(nan_env_mask)[0]
            print(f"NaN positions: {nan_env_indices}")
            tensor = tensor.nan_to_num(nan=0.0)
            # raise ValueError("observation is NAN NAN NAN") # 可以选择注释掉以免训练中断
            return tensor
        else:
            return tensor

    def CHECK_state(self):
        """
        Check state thresholds and log specific failure reasons.
        """
        # Get robot states
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
        
        # === [修改] 分别更新各个具体的死亡原因 (使用逻辑或，保留历史记录直到 reset) ===
        self._died_pos_limit = torch.logical_or(self._died_pos_limit, position_exceeded)
        self._died_lin_vel_limit = torch.logical_or(self._died_lin_vel_limit, linear_velocity_exceeded)
        self._died_ang_vel_limit = torch.logical_or(self._died_ang_vel_limit, angular_velocity_exceeded)
        self._died_tilt_limit = torch.logical_or(self._died_tilt_limit, tilt_exceeded)
        
        # 总开关：只要有一个触发，就标记为不稳定
        state_is_unstable = (
            self._died_pos_limit | 
            self._died_lin_vel_limit | 
            self._died_ang_vel_limit | 
            self._died_tilt_limit |
            self._died_nan
        )
        
        self._numerical_is_unstable = torch.logical_or(self._numerical_is_unstable, state_is_unstable)

    def _generate_desired_trajectory_langevin(self, env_ids: torch.Tensor = None):
        """
        Generate desired position and velocity using Langevin dynamics with damped harmonic oscillator.
        
        Implements the stochastic differential equation:
        dv = (-γv - ω²x) dt + σ dW
        dx = v dt
        
        where:
        - γ (gamma): damping coefficient
        - ω (omega): oscillator frequency
        - σ (sigma): noise intensity
        - dW: Wiener process (Gaussian noise)
        - α (alpha): smoothing factor for exponential moving average
        
        The trajectories are smoothed using exponential moving average with factor α.
        
        Args:
            env_ids: Environments to update. If None, updates all environments.
        """ 
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # if len(env_ids) > 0 and env_ids[0] == 0:
        #     print(f"Langevin V: {self.vel_des[0]}, Pos Error: {self.pos_des[0] - self._spawn_pos_w[0]}")

        n_envs = len(env_ids)
        
        # Get parameters from config
        gamma = self._langevin_friction
        omega = self._langevin_omega
        sigma = self._langevin_sigma
        dt = self._langevin_dt
        alpha = self._langevin_alpha
        
        sqrt_dt = torch.sqrt(torch.tensor(dt, device=self.device))
        
        # Get previous raw states
        x_prev_global = self.pos_des_raw[env_ids]
        v_prev = self.vel_des_raw[env_ids]
        
        # [核心修正] 获取对应的出生点，计算局部坐标
        spawn_pos = self._spawn_pos_w[env_ids]
        x_prev_local = x_prev_global - spawn_pos  # 这是一个相对于出生点的向量
        
        # Generate Wiener process noise
        dW = sqrt_dt * torch.randn(n_envs, 3, device=self.device)
        
        # Update velocity using damped harmonic oscillator
        # 注意：这里用 x_prev_local，确保力是把球拉回 spawn_pos，而不是世界原点
        v_next = v_prev + (-gamma * v_prev - omega * omega * x_prev_local) * dt + sigma * dW
        
        # ==================== [新增] 速度限幅 (Velocity Clamp) ====================
        max_vel_limits = self._langevin_max_vel[env_ids].unsqueeze(1)
        
        # 计算当前速度模长 (N, 1)
        v_norm = torch.norm(v_next, dim=1, keepdim=True)
        
        # 如果模长 > max_vel，则缩放；否则保持原样 (使用 min(1.0, limit/norm))
        # 加上 1e-6 防止除以零
        scale_factor = torch.clamp(max_vel_limits / (v_norm + 1e-6), max=1.0)
        
        # 应用缩放，保持方向不变
        v_next = v_next * scale_factor

        # Update position
        # 速度更新完后，计算新的局部位置，或者直接更新全局位置
        x_next_global = x_prev_global + v_next * dt
        
        # Store raw states
        self.pos_des_raw[env_ids] = x_next_global
        self.vel_des_raw[env_ids] = v_next
        
        # Apply smoothing
        v_smooth_prev = self.vel_des[env_ids]
        v_smooth = alpha * v_next + (1.0 - alpha) * v_smooth_prev
        
        x_smooth_prev = self.pos_des[env_ids]
        x_smooth = x_smooth_prev + v_smooth * dt
        
        # Store smoothed states
        self.pos_des[env_ids] = x_smooth
        self.vel_des[env_ids] = v_smooth
        

        # # ==================== [新增调试打印] ====================
        # # 检查当前更新列表中是否包含环境 0
        # # env_ids 是一个 Tensor，我们检查 0 是否在其中
        # if (env_ids == 0).any():
        #     with torch.no_grad():
        #         # 计算相对于出生点的位移，方便观察是否飘太远
        #         rel_pos = self.pos_des[0] - self._spawn_pos_w[0]
                
        #         print(f"Langevin [Env 0] | "
        #               f"Pos: {[round(x, 4) for x in self.pos_des[0].cpu().tolist()]} | "
        #               f"Rel: {[round(x, 4) for x in rel_pos.cpu().tolist()]} | "
        #               f"Vel: {[round(x, 4) for x in self.vel_des[0].cpu().tolist()]}")
        # # ======================================================

    def _generate_desired_trajectory_figure8(self, env_ids: torch.Tensor = None):
        """
        Generate desired position and velocity following a figure-8 (lemniscate) trajectory.
        
        The trajectory is parametrically defined as:
        x(t) = A * sin(ωt)
        y(t) = B * sin(2ωt)
        z(t) = constant height
        
        where:
        - A: scale in x direction
        - B: scale in y direction  
        - ω: angular frequency (2π * frequency)
        
        In the warmup phase (initial _figure8_warmup_duration seconds), pos_des and vel_des 
        remain at their initial states to allow the drone to stabilize.
        
        Args:
            env_ids: Environments to update. If None, updates all environments.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        n_envs = len(env_ids)
        
        # Update time variable
        self._figure8_time[env_ids] += self.dt
        t = self._figure8_time[env_ids]
        
        # Check if in warmup phase - if so, keep pos_des and vel_des unchanged
        in_warmup = t < self._figure8_warmup_duration
        if torch.all(in_warmup):
            return  # All environments still in warmup, no update needed
        
        # Angular frequency
        omega = 2 * math.pi * self._figure8_frequency
        
        # Get spawn positions as trajectory centers
        spawn_pos = self._spawn_pos_w[env_ids]
        
        # Calculate figure-8 position (relative to spawn point)
        # Adjust time to start from 0 after warmup
        t_adjusted = t - self._figure8_warmup_duration
        x_rel = self._figure8_scale_x * torch.sin(omega * t_adjusted)
        y_rel = self._figure8_scale_y * torch.sin(2 * omega * t_adjusted)
        z_abs = self._figure8_height
        
        # Combine into position vector (world frame)
        pos_des_new = torch.stack([
            spawn_pos[:, 0] + x_rel,
            spawn_pos[:, 1] + y_rel,
            torch.full((n_envs,), z_abs, device=self.device)
        ], dim=1)
        
        # Calculate velocity by differentiating the trajectory
        vx = self._figure8_scale_x * omega * torch.cos(omega * t_adjusted)
        vy = self._figure8_scale_y * 2 * omega * torch.cos(2 * omega * t_adjusted)
        vz = torch.zeros(n_envs, device=self.device)
        
        vel_des_new = torch.stack([vx, vy, vz], dim=1)
        
        # Only update environments that have passed warmup
        # For envs still in warmup, keep their pos_des and vel_des unchanged
        active_mask = ~in_warmup
        self.pos_des[env_ids[active_mask]] = pos_des_new[active_mask]
        self.vel_des[env_ids[active_mask]] = vel_des_new[active_mask]
        
        # Also update raw states for consistency
        self.pos_des_raw[env_ids[active_mask]] = pos_des_new[active_mask]
        self.vel_des_raw[env_ids[active_mask]] = vel_des_new[active_mask]

    def _calc_env_origins(self):
        # Generate group origins in a grid that ascends in rows and columns
        robots_per_env = self.cfg.robots_per_env
        num_groups = self.num_envs // robots_per_env + 1

        # Calculate grid dimensions for groups
        grid_rows = self.cfg.grid_rows
        grid_cols = self.cfg.grid_cols

        # Ensure the number of groups does not exceed the grid capacity
        grid_capacity = grid_rows * grid_cols
        if num_groups > grid_capacity:
            print(f"Warning: The number of groups ({num_groups}) exceeds the grid capacity ({grid_capacity}). Group origins will loop.")

        # Generate group origins
        group_origins = torch.zeros(num_groups, 3, device=self.device)
        terrain_width = self.cfg.terrain_width
        terrain_length = self.cfg.terrain_length

        map_size_x, map_size_y = self.cfg.map_size
        offset_x = -map_size_x / 2.0
        offset_y = -map_size_y / 2.0

        for i in range(num_groups):
            row = (i // grid_cols) % grid_rows  # Loop rows if exceeding grid capacity
            col = i % grid_cols  # Loop columns if exceeding grid capacity
            group_origins[i, 0] = col * terrain_length + offset_x
            group_origins[i, 1] = row * terrain_width + offset_y
    
        # Assign the same origin to all environments within a group
        self.env_origins = group_origins.repeat_interleave(robots_per_env, dim=0)[:self.num_envs]
        num_grids = grid_rows * grid_cols
        self.grid_idx = [[] for _ in range(num_grids)]
        for env_id in range(self.num_envs):
            group_id = env_id // robots_per_env
            row = (group_id // grid_cols) % grid_rows
            col = group_id % grid_cols
            grid_linear_idx = row * grid_cols + col
            self.grid_idx[grid_linear_idx].append(env_id)
        print(f"Grid indices: {self.grid_idx}")

    def _setup_scene(self):
            """Create and clone the environment scene."""
            # Set up the robot
            self._robot = Articulation(self.cfg.robot)
            
            # 1. 先克隆环境 (此时 USD 里的 /World/envs/env_X/Robot 才会被创建)
            # Clone the scene
            self.scene.clone_environments(copy_from_source=False)

            # 2. Add the robot to the scene
            self.scene.articulations["robot"] = self._robot

            # 3. 环境存在后，再去查找路径并设置可见性
            robot_prims = find_matching_prim_paths("/World/envs/env_.*/Robot")
            
            # 检查是否找到了 Prim，方便调试
            if len(robot_prims) == 0:
                print("[Warning] No robot prims found! Check your prim_path regex.")
                
            for prim_path in robot_prims:
                # 修改物理属性 (如果需要)
                prims_utils.set_prim_property(prim_path + "/body", "physics:mass", 0.0282)
                prims_utils.set_prim_property(prim_path + "/body", "physics:diagonalInertia", (2.44864e-5, 2.44864e-5, 3.61504e-5))
                prims_utils.set_prim_property(prim_path + "/body", "physics:centerOfMass", (0.0, 0.0, 0.0))
                
                # 设置可见性
                if self.cfg.robot_vis == True:
                    prims_utils.set_prim_property(prim_path, "visibility", "visible")
                else:
                    prims_utils.set_prim_property(prim_path, "visibility", "invisible")

            # Add lights
            light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
            light_cfg.func("/World/Light", light_cfg)

            self._map_generation_timer = 0

    def _pre_physics_step(self, actions: torch.Tensor):

        # 1. 更新轨迹 (根据配置的轨迹类型)
        if self.cfg.trajectory_type == "figure8":
            # 使用八字形轨迹
            self._generate_desired_trajectory_figure8()
        elif torch.any(self._is_langevin_task):
            # 使用 Langevin 轨迹 (仅针对 Langevin 任务)
            self._generate_desired_trajectory_langevin(env_ids=torch.where(self._is_langevin_task)[0])

        # 1. Action Clamp (之前讨论过的)
        raw_actions_clamped = torch.clamp(actions, -1.0, 1.0)
        action_setpoint_normalized = (raw_actions_clamped + 1.0) * 0.5

        # with torch.no_grad():
        #     # 使用 .cpu().tolist() 将 Tensor 转为纯列表，看着更清爽
        #     print(f"Action (Env 0): {action_setpoint_normalized[0].cpu().tolist()}")
        
        # 保存 Action 用于 Observation 的 "Last Action"
        self._actions = action_setpoint_normalized.clone()

        # 2. [新增] 模拟电机一阶低通滤波 (First-order Low-pass Filter)
        # 将归一化的 Setpoint 转换为 真实的 RPM/Rad/s 范围
        omega_min = self._controller.dynamics.motor_omega_min_.unsqueeze(1)
        omega_max = self._controller.dynamics.motor_omega_max_.unsqueeze(1)
        
        # 目标转速 (Target Omega)
        target_motor_speeds = omega_min + action_setpoint_normalized * (omega_max - omega_min)
        
        # 更新当前电机转速 (Current Omega)
        # formula: current = alpha * target + (1 - alpha) * previous
        # 注意：RAPTOR 论文中上升和下降的延迟可能不同 (Tm_up, Tm_down)
        self._current_motor_speeds = (self.motor_alpha * target_motor_speeds + 
                                      (1.0 - self.motor_alpha) * self._current_motor_speeds)

        # 3. 计算力 (使用带有延迟的 _current_motor_speeds)
        # 注意：这里不需要 normalized=True 了，因为我们已经反归一化了
        # 或者你修改 motor_speeds_to_wrench 让它接受非归一化输入
        force_b, torque_b, px4info = self._controller.motor_speeds_to_wrench(
            self._current_motor_speeds, 
            normalized=False # 传入的是真实 rad/s
        )

        # # [新增调试打印] 检查物理可行性
        # with torch.no_grad():
        #     # 假设 4 个电机都满速 (normalized=1.0) 时的推力
        #     # 这里我们直接看当前产生的力 force_b[0, 2] 与重力的关系
        #     gravity_force = 9.81 * 0.03  # mass = 0.8
        #     current_thrust = force_b[0, 2].item()
        #     twr = current_thrust / gravity_force
            
        #     print(f"\n=== Physics Check [Env 0] ===")
        #     print(f"Mass: 0.03 kg | Req Hover Force: {gravity_force:.2f} N")
        #     print(f"Curr Thrust: {current_thrust:.2f} N | TWR (Curr): {twr:.2f}")
        #     # 如果当前是全油门 (Action接近1)，TWR 必须 > 1.5 甚至 > 2.0 才能灵活飞行
        #     print(f"Action Mean: {self._actions[0].mean().item():.2f}") 
        #     print("=============================")

        # 4. 施加力
        self._forces.zero_()
        self._torques.zero_()
        self._forces[:, 0, :] = force_b
        self._torques[:, 0, :] = torque_b
        
        self._robot.set_external_force_and_torque(self._forces, self._torques, body_ids=self._body_id)

        # # ==================== [新增调试打印] ====================
        # # 频率控制: 每 50 步打印一次 (约 0.5秒~1秒一次，取决于仿真dt)
        # if self.common_step_counter % 1 == 0:
        #     env_id = 0  # 只监控第 0 号环境
            
        #     # 获取当前模式
        #     mode_str = "Langevin (Moving)" if self._is_langevin_task[env_id] else "Position (Fixed)"
            
        #     # 获取 Desired State
        #     p = self.pos_des[env_id].cpu().tolist()
        #     v = self.vel_des[env_id].cpu().tolist()
            
        #     # 获取当前实际位置 (用于对比)
        #     curr_p = self._robot.data.root_pos_w[env_id].cpu().tolist()
            
        #     print(f"\n[Step {self.common_step_counter}] Env {env_id} Mode: \033[1;33m{mode_str}\033[0m")
        #     print(f"  > Des Pos : [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]")
        #     print(f"  > Cur Pos : [{curr_p[0]:.3f}, {curr_p[1]:.3f}, {curr_p[2]:.3f}]")
        #     print(f"  > Des Vel : [{v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}]")
        #     print("-" * 40)
        # # ======================================================


    def _apply_action(self):
        """Apply thrust/moment to the quadcopter."""
        self._robot.set_external_force_and_torque(self._forces, self._torques, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        # 1. 获取物理状态
        pos_w = self._robot.data.root_pos_w
        quat_w = self._robot.data.root_quat_w
        vel_w = self._robot.data.root_lin_vel_w
        ang_vel_b = self._robot.data.root_ang_vel_b
        
        # 2. 计算旋转矩阵
        rot_matrix_b2w = matrix_from_quat(quat_w)  # R_b->w
        rotation_matrix_flat = rot_matrix_b2w.reshape(self.num_envs, 9)
        
        # --- [关键修改] 坐标系转换开始 ---
        # 计算世界系误差
        pos_error_w = pos_w - self.pos_des
        vel_error_w = vel_w - self.vel_des

        # 计算 R_w->b (即 R_b->w 的转置)
        rot_matrix_w2b = rot_matrix_b2w.transpose(1, 2) 

        # 投影到机体坐标系: error_body = R_w->b @ error_world
        # 使用 bmm (Batch Matrix Multiplication)
        pos_error_b = torch.bmm(rot_matrix_w2b, pos_error_w.unsqueeze(-1)).squeeze(-1)
        vel_error_b = torch.bmm(rot_matrix_w2b, vel_error_w.unsqueeze(-1)).squeeze(-1)
        # --- [关键修改] 坐标系转换结束 ---
        # [新增调试打印] 验证坐标旋转逻辑
        # with torch.no_grad():
        #     # 取出欧拉角 (Roll, Pitch, Yaw)
        #     r, p, y = euler_xyz_from_quat(quat_w[0].unsqueeze(0))
        #     yaw_deg = y.item() * 180 / 3.14159
            
        #     print(f"\n=== Coord Check [Env 0] ===")
        #     print(f"Yaw: {yaw_deg:.1f} deg")
        #     print(f"Err World: {pos_error_w[0].cpu().tolist()}")
        #     print(f"Err Body : {pos_error_b[0].cpu().tolist()}")
        #     # 简单验证逻辑：
        #     # 如果 Yaw = 0, World 和 Body 应该差不多
        #     # 如果 Yaw = 90 (机头朝左), World 的 X 应该是 Body 的 -Y
        #     print("===========================")
        # [修改] 使用手动维护的、带有物理延迟的电机速度
        # 为了让网络好训练，通常需要归一化回 [-1, 1] 或 [0, 1]
        omega_min = self._controller.dynamics.motor_omega_min_.unsqueeze(1)
        omega_max = self._controller.dynamics.motor_omega_max_.unsqueeze(1)
        
        # 归一化当前电机速度 [0, 1]
        motor_speeds_obs = (self._current_motor_speeds - omega_min) / (omega_max - omega_min)

        obs = torch.cat([
            pos_error_b,            
            rotation_matrix_flat,  
            vel_error_b,           
            ang_vel_b,              
            self._last_actions,     
            motor_speeds_obs  # <--- 使用这个替换 joint_vel
        ], dim=-1)
        
        # with torch.no_grad():
        #     o = obs[0] # 取出环境0的数据
        #     print(f"\n=== [Env 0 Observation] Total Dim: {o.shape[0]} ===")
        #     # 使用 .cpu().numpy() 并保留4位小数，方便阅读
        #     print(f"Pos Err (Body) [0:3]  : {np.array2string(o[0:3].cpu().numpy(), precision=4, suppress_small=True)}")
        #     print(f"Rot Mat (Flat) [3:12] : {np.array2string(o[3:12].cpu().numpy(), precision=4, suppress_small=True)}")
        #     print(f"Vel Err (Body) [12:15]: {np.array2string(o[12:15].cpu().numpy(), precision=4, suppress_small=True)}")
        #     print(f"Ang Vel (Body) [15:18]: {np.array2string(o[15:18].cpu().numpy(), precision=4, suppress_small=True)}")
        #     print(f"Last Actions   [18:22]: {np.array2string(o[18:22].cpu().numpy(), precision=4, suppress_small=True)}")
        #     print(f"Motor Speeds   [22:26]: {np.array2string(o[22:26].cpu().numpy(), precision=4, suppress_small=True)}")
        #     print("==============================================\n")


        # 4. 去除冗余历史堆叠，直接返回当前帧
        obs = self.CHECK_NAN(obs, "Observation")
        return {"policy": obs, "critic": obs, "rnd_state": obs}
    
    def _get_rewards(self) -> torch.Tensor:
            # --- 1. 获取状态 ---
            pos_w = self._robot.data.root_pos_w
            quat_w = self._robot.data.root_quat_w
            
            # --- 2. 计算各项原始 Cost ---
            # Position
            pos_error = pos_w - self.pos_des
            pos_error_norm = torch.norm(pos_error, dim=1)
            
            # Orientation (q_z based)
            q_z = quat_w[:, 3]
            arccos_arg = torch.clamp(1.0 - torch.abs(q_z), -1.0, 1.0)
            orientation_cost = torch.arccos(arccos_arg)
            
            # Action Smoothness
            action_diff = self._actions - self._last_actions
            d_action_cost = torch.norm(action_diff, dim=1)
            
            # Base Reward
            constant = torch.ones(self.num_envs, device=self.device)
            
            # Terminal Penalty
            terminal = (
                self._numerical_is_unstable 
                # (self._robot.data.root_pos_w[:, 2] < self.cfg.too_low) | 
                # (self._robot.data.root_pos_w[:, 2] > self.cfg.too_high)
            )
            termination_penalty = terminal.float()
            
            # --- 3. 应用权重 (计算实际 Reward) ---
            r_pos = -pos_error_norm * self.cfg.reward_coef_position_cost
            r_ori = -orientation_cost * self.cfg.reward_coef_orientation_cost
            r_act = -d_action_cost * self.cfg.reward_coef_d_action_cost
            r_base = constant * self.cfg.reward_constant
            r_term = -termination_penalty * self.cfg.reward_coef_termination_penalty

            # 总 Reward
            total_reward = r_pos + r_ori + r_act + r_base + r_term
            
            # --- 4. 累加 Episode Sums (关键步骤) ---
            # 这里的 key 名字将决定 wandb 上显示的后缀
            reward_items = {
                "position": r_pos,
                "orientation": r_ori,
                "action_smooth": r_act,
                "base": r_base,
                "terminal": r_term
            }

            for key, value in reward_items.items():
                # 确保 key 存在于 buffer 中
                if key not in self._episode_sums:
                    self._episode_sums[key] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                # 累加当前步的 reward
                self._episode_sums[key] += value
            
            # 更新历史动作
            self._last_actions = self._actions.clone()
                    
            # with torch.no_grad():
            #     print(f"\n=== Reward Breakdown [Env 0] ===")
            #     print(f"Pos Cost : {r_pos[0].item():.4f}")
            #     print(f"Ori Cost : {r_ori[0].item():.4f}")
            #     print(f"Act Cost : {r_act[0].item():.4f}")
            #     print(f"Base Rew : {r_base[0].item():.4f}")
            #     print(f"Term Pen : {r_term[0].item():.4f}")
            #     print(f"TOTAL    : {total_reward[0].item():.4f}")
            #     print("================================")

            return total_reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Define terminations and timeouts."""
        if self.cfg.train_or_play:
            time_out = self.episode_length_buf >= self.max_episode_length - 1
        else:
            # Disable timeouts: always false
            time_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        self.CHECK_state()

        # Check distance from desired trajectory position (Langevin threshold)
        dist_traj_from_spawn = torch.norm(self.pos_des - self._spawn_pos_w, dim=1)
        position_exceeded_langevin = dist_traj_from_spawn > self.cfg.position_threshold_langevin

        conditions = [
            self._numerical_is_unstable,  # Numerical instability
            # self._robot.data.root_pos_w[:, 2] < self.cfg.too_low,  # Z position too low
            # self._robot.data.root_pos_w[:, 2] > self.cfg.too_high,  # Z position too high
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
            # For trajectory tracking task: no success/failure distinction
            # Only track timeouts (max episode length) and deaths (physical limit violations)
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
            """Reset specific environment indexes with RAPTOR initialization distribution."""
            if env_ids is None:
                env_ids = self._robot._ALL_INDICES
            
            num_resets = len(env_ids)
            
            # --- 1. 日志记录逻辑 (保持不变) ---
            if num_resets > 0:
                if "log" not in self.extras:
                    self.extras["log"] = dict()
                for key in self._episode_sums.keys():
                    values = self._episode_sums[key][env_ids]
                    mean_val = torch.mean(values).item()
                    self.extras["log"][f"Episode_Reward/{key}"] = mean_val
                    self._episode_sums[key][env_ids] = 0.0

            # --- 2. 基础重置 ---
            if self._wind_gen is not None:
                self._wind_gen.reset(env_ids)

            died_mask = self.reset_terminated[env_ids]
            timed_out_mask = self.reset_time_outs[env_ids]
            if hasattr(self, '_update_episode_outcomes_and_metrics'):
                success_mask = torch.zeros(num_resets, dtype=torch.bool, device=self.device)
                self._update_episode_outcomes_and_metrics(env_ids, success_mask, died_mask, timed_out_mask)

            self._robot.reset(env_ids)
            super()._reset_idx(env_ids)

            # 状态清零
            self._actions[env_ids] = 0.0
            self._last_actions[env_ids] = 0.0
            self._forces[env_ids] = 0.0
            self._torques[env_ids] = 0.0
            self._last_angular_velocity[env_ids] = 0.0
            # === [修改] 清零所有不稳定性标志位 ===
            self._numerical_is_unstable[env_ids] = False
            self._died_pos_limit[env_ids] = False
            self._died_lin_vel_limit[env_ids] = False
            self._died_ang_vel_limit[env_ids] = False
            self._died_tilt_limit[env_ids] = False
            self._died_nan[env_ids] = False

            self._current_motor_speeds[env_ids] = 0.0

            # 重置轨迹时间 (用于八字形轨迹)
            self._figure8_time[env_ids] = 0.0

            self._langevin_max_vel[env_ids] = torch.rand(len(env_ids), device=self.device) * 2.0 + 1.0
            # --- 3. RAPTOR 初始化逻辑 ---
            
            # 定义物理参数
            l_arm = 0.04384  # 论文中的 l_arm


            if self.cfg.train_or_play:
                # ================= [TRAIN MODE] =================
                # 原有的随机化逻辑
                
                r_pos_limit = 10.0 * l_arm  # 位置采样半径
                v_lin_limit = 1.0           # 线速度限制
                v_ang_limit = 1.0           # 角速度限制

                # 辅助函数：球体内均匀采样
                def sample_in_sphere(radius, n_samples):
                    direction = torch.randn(n_samples, 3, device=self.device)
                    direction = F.normalize(direction, p=2, dim=1)
                    u = torch.rand(n_samples, 1, device=self.device)
                    r = radius * torch.pow(u, 1.0/3.0)
                    return direction * r

                # A. 生成随机状态偏移
                pos_offset = sample_in_sphere(r_pos_limit, num_resets)
                lin_vel = sample_in_sphere(v_lin_limit, num_resets)
                ang_vel = sample_in_sphere(v_ang_limit, num_resets)
                
                # 姿态随机
                roll = (torch.rand(num_resets, device=self.device) * 2 - 1) * (math.pi / 2.0)
                pitch = (torch.rand(num_resets, device=self.device) * 2 - 1) * (math.pi / 2.0)
                yaw = (torch.rand(num_resets, device=self.device) * 2 - 1) * math.pi
                quat = quat_from_euler_xyz(roll, pitch, yaw)

                # B. 10% 概率覆盖为完美初始状态
                reset_to_target_probs = torch.rand(num_resets, device=self.device)
                is_perfect_start = reset_to_target_probs < 0.10 

                pos_offset[is_perfect_start] = 0.0
                lin_vel[is_perfect_start] = 0.0
                ang_vel[is_perfect_start] = 0.0
                
                identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(num_resets, 1)
                quat[is_perfect_start] = identity_quat[is_perfect_start]
                
            else:
                # ================= [PLAY MODE] =================
                # 强制全部归零，不进行随机化
                
                pos_offset = torch.zeros(num_resets, 3, device=self.device)
                lin_vel = torch.zeros(num_resets, 3, device=self.device)
                ang_vel = torch.zeros(num_resets, 3, device=self.device)
                
                # 强制姿态水平 (Identity Quaternion: w=1, x=0, y=0, z=0)
                quat = torch.zeros(num_resets, 4, device=self.device)
                quat[:, 0] = 1.0 

            # --- 4. 设置仿真器状态 ---
            
            # 获取默认状态
            joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
            joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
            
            # 计算出生中心点 (Spawn Center)
            spawn_center = self.env_origins[env_ids].clone()
            spawn_center[:, 0] += self.cfg.terrain_length / 2.0
            spawn_center[:, 1] += self.cfg.terrain_width / 2.0
            spawn_center[:, 2] += self.cfg.height

            # 记录出生点 (Spawn Position = Center + Offset)
            # 注意：这里的 Offset 是随机扰动
            # 如果是 Perfect Start，Offset 为 0，即生在正中心
            start_pos = spawn_center + pos_offset
            
            # 更新 Spawn Pos 记录
            # 注意：通常 self._spawn_pos_w 记录的是"目标点"或"基准点"，用于计算相对距离
            # 在 RAPTOR 中，任务是 "Go to origin (relative)"，所以基准点应该是 spawn_center
            self._spawn_pos_w[env_ids] = spawn_center 

            # 构建 Root State 写入仿真
            root_state = self._robot.data.default_root_state[env_ids].clone()
            root_state[:, :3] = start_pos
            root_state[:, 3:7] = quat
            root_state[:, 7:10] = lin_vel
            root_state[:, 10:13] = ang_vel

            self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
            self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
            self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

            # --- 5. 任务与轨迹初始化 ---

            # 重新采样任务类型 (50% 概率)
            random_task_probs = torch.rand(num_resets, device=self.device)
            is_langevin = random_task_probs > self.cfg.prob_null_trajectory
            self._is_langevin_task[env_ids] = is_langevin

            # 初始化期望状态 (Desired States)
            # 无论是 Langevin 还是 Null 任务，初始期望位置都设为 spawn_center
            # 这样 Policy 一上来就会看到 pos_error = pos_offset
            self.pos_des[env_ids] = spawn_center.clone()
            self.vel_des[env_ids] = 0.0
            
            # Langevin 轨迹生成器的内部状态也重置到中心
            self.pos_des_raw[env_ids] = spawn_center.clone()
            self.vel_des_raw[env_ids] = 0.0

            # # 调试打印 (Env 0)
            # if (env_ids == 0).any():
            #     # 找到环境0在当前 batch 中的索引
            #     batch_idx = (env_ids == 0).nonzero(as_tuple=False).item()
                
            #     # 提取数据
            #     p_offset = pos_offset[batch_idx].cpu().tolist()
            #     l_v = lin_vel[batch_idx].cpu().tolist()
            #     perfect = is_perfect_start[batch_idx].item()
                
            #     # --- [修复] 从最终的四元数反算欧拉角用于显示 ---
            #     # 取出最终实际发送给物理引擎的四元数 (Root State 包含了被 Perfect Start 覆盖后的结果)
            #     final_quat = root_state[batch_idx, 3:7].unsqueeze(0)
            #     final_r, final_p, final_y = euler_xyz_from_quat(final_quat)
                
            #     r_deg = math.degrees(final_r.item())
            #     p_deg = math.degrees(final_p.item())
            #     y_deg = math.degrees(final_y.item()) # 新增：转换 Yaw
            #     # ------------------------------------------------

            #     print(f"\n\033[1;36m>>> [RAPTOR Reset Env 0]\033[0m")
            #     print(f"    Perfect Start (10%): {perfect}")
            #     print(f"    Pos Offset (m)     : [{p_offset[0]:.3f}, {p_offset[1]:.3f}, {p_offset[2]:.3f}]")
            #     print(f"    Lin Vel (m/s)      : [{l_v[0]:.3f}, {l_v[1]:.3f}, {l_v[2]:.3f}]")
            #     # 修改：同时打印 R, P, Y
            #     print(f"    Attitude (R/P/Y)   : [{r_deg:.1f}, {p_deg:.1f}, {y_deg:.1f}]")
            #     print("-" * 40)

    def _set_debug_vis_impl(self, debug_vis: bool):
            """Show debug markers if debug_vis is True."""
            
            print(f"debug_vis: {self.cfg.debug_vis}")

            if debug_vis:
                # 1. Goal Position (保持不变 - 这是一个方块)
                if not hasattr(self, "goal_pos_visualizer"):
                    marker_cfg = CUBOID_MARKER_CFG.copy()
                    marker_cfg.markers["cuboid"].size = (self.cfg.marker_size, self.cfg.marker_size, self.cfg.marker_size)
                    marker_cfg.prim_path = "/Visuals/Command/goal_position"
                    self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
                self.goal_pos_visualizer.set_visibility(True)

                # 2. Goal Yaw (保持不变 - 这是一个箭头，表示朝向目标)
                if not hasattr(self, "goal_yaw_visualizer"):
                    goal_arrow_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                    goal_arrow_cfg.markers["arrow"].scale = (self.cfg.marker_size, self.cfg.marker_size, self.cfg.marker_size*4)
                    goal_arrow_cfg.prim_path = "/Visuals/Command/goal_yaw"
                    self.goal_yaw_visualizer = VisualizationMarkers(goal_arrow_cfg)
                self.goal_yaw_visualizer.set_visibility(True)

                # 3. Current Robot Position (蓝色球)
                if not hasattr(self, "current_yaw_visualizer"):
                    print("create robot position visualizer (Sphere)")
                    # 使用球体配置
                    current_vis_cfg = SPHERE_MARKER_CFG.copy()
                    current_vis_cfg.prim_path = "/Visuals/Command/current_yaw"
                    
                    # 设置缩放 (和红绿球保持一致)
                    scale_val = self.cfg.marker_size
                    current_vis_cfg.markers["sphere"].scale = (scale_val, scale_val, scale_val)
                    
                    # 设置颜色: 蓝色 (R=0, G=0, B=1)
                    current_vis_cfg.markers["sphere"].visual_material.diffuse_color = (0.0, 0.0, 1.0)
                    
                    self.current_yaw_visualizer = VisualizationMarkers(current_vis_cfg)
                self.current_yaw_visualizer.set_visibility(True)

                # 4. Langevin Trajectory (绿色球)
                if not hasattr(self, 'traj_langevin_visualizer'):
                    print("create langevin trajectory visualizer (Sphere)")
                    # 使用球体配置
                    langevin_cfg = SPHERE_MARKER_CFG.copy()
                    langevin_cfg.prim_path = "/Visuals/TrajectoryLangevin"
                    # 设置缩放 (X, Y, Z)
                    langevin_cfg.markers["sphere"].scale = (self.cfg.marker_size, self.cfg.marker_size, self.cfg.marker_size)
                    # 设置颜色: 绿色 (R=0, G=1, B=0)
                    langevin_cfg.markers["sphere"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
                    self.traj_langevin_visualizer = VisualizationMarkers(langevin_cfg)
                self.traj_langevin_visualizer.set_visibility(True)

                # 5. Fixed/Null Trajectory (红色球)
                if not hasattr(self, 'traj_fixed_visualizer'):
                    print("create fixed trajectory visualizer (Sphere)")
                    # 使用球体配置
                    fixed_cfg = SPHERE_MARKER_CFG.copy()
                    fixed_cfg.prim_path = "/Visuals/TrajectoryFixed"
                    # 设置缩放
                    fixed_cfg.markers["sphere"].scale = (self.cfg.marker_size, self.cfg.marker_size, self.cfg.marker_size)
                    # 设置颜色: 红色 (R=1, G=0, B=0)
                    fixed_cfg.markers["sphere"].visual_material.diffuse_color = (1.0, 0.0, 0.0)
                    self.traj_fixed_visualizer = VisualizationMarkers(fixed_cfg)
                self.traj_fixed_visualizer.set_visibility(True)
                    
            else:
                # 隐藏逻辑
                if hasattr(self, "goal_pos_visualizer"): self.goal_pos_visualizer.set_visibility(False)
                if hasattr(self, "goal_yaw_visualizer"): self.goal_yaw_visualizer.set_visibility(False)
                if hasattr(self, "current_yaw_visualizer"): self.current_yaw_visualizer.set_visibility(False)
                if hasattr(self, "traj_langevin_visualizer"): self.traj_langevin_visualizer.set_visibility(False)
                if hasattr(self, "traj_fixed_visualizer"): self.traj_fixed_visualizer.set_visibility(False)
                # 清理可能的旧变量
                if hasattr(self, 'traj_visualizer'): self.traj_visualizer.set_visibility(False)
                
    def _debug_vis_callback(self, event):
            """Update debug markers with current robot pose."""
            
            # 1. 机器人本体的蓝色箭头 (这个保持不变，需要跟随机器人动)
            if hasattr(self, "current_yaw_visualizer"):
                self.current_yaw_visualizer.visualize(self._robot.data.root_pos_w, self._robot.data.root_quat_w)

            # ================= [关键修改：固定姿态为0] =================
            
            # 构造一个单位四元数 (w=1, x=0, y=0, z=0)
            # 它的形状必须和 robot.data.root_quat_w 一样: (num_envs, 4)
            fixed_rot = torch.zeros_like(self._robot.data.root_quat_w)
            fixed_rot[:, 0] = 1.0  # 设置 w 分量为 1.0
            
            # 准备一个“藏在地底”的坐标，用于隐藏不需要显示的球
            invisible_pos = torch.zeros_like(self.pos_des) 
            invisible_pos[:, 2] = -100.0 
            
            # 2. 更新 Langevin 绿色球
            if hasattr(self, "traj_langevin_visualizer"):
                pos_green = invisible_pos.clone()
                
                # 只有 Langevin 任务显示绿色
                mask = self._is_langevin_task
                if mask.any():
                    pos_green[mask] = self.pos_des[mask]
                
                # 使用 fixed_rot，球体姿态永远为0
                self.traj_langevin_visualizer.visualize(pos_green, fixed_rot)

            # 3. 更新 Fixed 红色球
            if hasattr(self, "traj_fixed_visualizer"):
                pos_red = invisible_pos.clone()
                
                # 只有 Fixed 任务显示红色
                mask = ~self._is_langevin_task
                if mask.any():
                    pos_red[mask] = self.pos_des[mask]
                    
                # 使用 fixed_rot，球体姿态永远为0
                self.traj_fixed_visualizer.visualize(pos_red, fixed_rot)

    class EpisodeOutcome(IntEnum):
        ONGOING = 0
        SUCCESS = 1
        FAILURE = 2
            
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
                # === [修改] 提取细分的死亡原因 ===
                # 将 Tensor 转为 numpy bool 数组
                reason_pos_limit = self._died_pos_limit[died_env_ids].cpu().numpy()
                reason_lin_vel = self._died_lin_vel_limit[died_env_ids].cpu().numpy()
                reason_ang_vel = self._died_ang_vel_limit[died_env_ids].cpu().numpy()
                reason_tilt = self._died_tilt_limit[died_env_ids].cpu().numpy()
                reason_nan = self._died_nan[died_env_ids].cpu().numpy()
                
                # 计算 Langevin 距离 (原来的逻辑)
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
                # ===============================
            
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
                # === [修改] 定义需要统计的键名列表 ===
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
                
                # 更新主日志
                self.extras["log"].update({
                    "Episode_Termination/died": died_count / num_termination_records * 100.0,
                    "Episode_Termination/time_out": timeout_count / num_termination_records * 100.0,
                    "Metrics/average_velocity": avg_velocity,
                    "Metrics/episodes_completed": self._episodes_completed,
                })

                # === [修改] 更新细分原因日志 ===
                for key in reason_keys:
                    # 记录名为 "Metrics/Died/died_tilt" 等等
                    self.extras["log"][f"Metrics/Died/{key}"] = reason_counts[key] / num_termination_records * 100.0
                # ===============================

                return len(completed_env_ids), 0
    
    def close(self):
        """Clean up resources when environment is closed."""
        super().close()
