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
    # 默认值 (Crazyflie 2.1 参数作为默认，用于单机或默认情况)
    mass: float = 0.0282
    arm_length: float = 0.04384
    # Inertia: Ixx, Iyy, Izz
    inertia: tuple[float, float, float] = (2.44864e-5, 2.44864e-5, 3.61504e-5)
    thrust_to_weight: float = 2.25 
    motor_tau_up: float = 0.05
    motor_tau_down: float = 0.10 # 默认值设大一点体现差异
    # [新增] 力矩系数
    moment_scale: float = 0.016  

    # [新增] 用于多教师蒸馏的参数列表 (List of dicts)
    # 格式: [{'mass': 0.03, 'arm_length': 0.04, 'inertia': (x,y,z), 'twr': 2.0, 'motor_tau': 0.05}, {...}]
    # 如果此列表不为空，将覆盖上面的单值设置，并按环境索引分段分配
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
        
        # ================= [新增] 多教师动力学参数分配逻辑 =================
        # 初始化张量
        self.mass_tensor = torch.zeros(self.num_envs, device=self.device)
        self.arm_l_tensor = torch.zeros(self.num_envs, device=self.device)
        self.twr_tensor = torch.zeros(self.num_envs, device=self.device)
        self.inertia_tensor = torch.zeros(self.num_envs, 3, device=self.device)
        # self.motor_tau = torch.zeros(self.num_envs, 1, device=self.device)
        self.motor_tau_up_tensor = torch.zeros(self.num_envs, 1, device=self.device)
        self.motor_tau_down_tensor = torch.zeros(self.num_envs, 1, device=self.device)
        self.kappa_tensor = torch.zeros(self.num_envs, device=self.device)

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
                
                # self.motor_tau[indices] = params['motor_tau']
                self.motor_tau_up_tensor[indices] = params['motor_tau_up']
                self.motor_tau_down_tensor[indices] = params['motor_tau_down']
                self.kappa_tensor[indices] = params['kappa']

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
            # self.motor_tau.fill_(self.cfg.dynamics.motor_tau)
            # [修改] 填充新参数
            self.motor_tau_up_tensor.fill_(self.cfg.dynamics.motor_tau_up)
            self.motor_tau_down_tensor.fill_(self.cfg.dynamics.motor_tau_down)
            self.kappa_tensor.fill_(self.cfg.dynamics.moment_scale)

        # =================================================================

        # Store the robot mass for reference (e.g. wind force calculation if added later)
        self._robot_mass = self.mass_tensor 

        # Controller initialization with tensors
        self._controller = SimpleQuadrotorController(
            num_envs=self.num_envs,
            device=self.device,
            mass=self.mass_tensor,        # Pass self.var
            arm_length=self.arm_l_tensor, # Pass self.var
            inertia=self.inertia_tensor,  # Pass self.var
            thrust_to_weight=self.twr_tensor  # Pass self.var
            moment_scale=self.kappa_tensor
        )

        self.dt = self.cfg.sim.dt

        # if self.motor_tau.shape != (self.num_envs, 1):
        #      self.motor_tau = self.motor_tau.view(self.num_envs, 1)

        # self.motor_alpha = self.dt / (self.dt + self.motor_tau)
        if self.motor_tau_up_tensor.shape != (self.num_envs, 1):
             self.motor_tau_up_tensor = self.motor_tau_up_tensor.view(self.num_envs, 1)
        if self.motor_tau_down_tensor.shape != (self.num_envs, 1):
             self.motor_tau_down_tensor = self.motor_tau_down_tensor.view(self.num_envs, 1)
             
        self.motor_alpha_up = self.dt / (self.dt + self.motor_tau_up_tensor)
        self.motor_alpha_down = self.dt / (self.dt + self.motor_tau_down_tensor)


        self._current_motor_speeds = torch.zeros(self.num_envs, 4, device=self.device)
        
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
        
        # Raw states for Langevin
        self.pos_des_raw = torch.zeros(self.num_envs, 3, device=self.device)
        self.vel_des_raw = torch.zeros(self.num_envs, 3, device=self.device)
        
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

    def _generate_desired_trajectory_langevin(self, env_ids: torch.Tensor = None):
        """
        Generate desired position and velocity using Langevin dynamics.
        """ 
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        n_envs = len(env_ids)
        
        gamma = self._langevin_friction
        omega = self._langevin_omega
        sigma = self._langevin_sigma
        dt = self._langevin_dt
        alpha = self._langevin_alpha
        
        sqrt_dt = torch.sqrt(torch.tensor(dt, device=self.device))
        
        # Get previous raw states
        x_prev_global = self.pos_des_raw[env_ids]
        v_prev = self.vel_des_raw[env_ids]
        
        # 获取对应的出生点，计算局部坐标
        spawn_pos = self._spawn_pos_w[env_ids]
        x_prev_local = x_prev_global - spawn_pos 
        
        dW = sqrt_dt * torch.randn(n_envs, 3, device=self.device)
        
        # Update velocity
        v_next = v_prev + (-gamma * v_prev - omega * omega * x_prev_local) * dt + sigma * dW
        
        # 速度限幅
        max_vel_limits = self._langevin_max_vel[env_ids].unsqueeze(1)
        v_norm = torch.norm(v_next, dim=1, keepdim=True)
        scale_factor = torch.clamp(max_vel_limits / (v_norm + 1e-6), max=1.0)
        v_next = v_next * scale_factor

        # Update position
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

    def _generate_desired_trajectory_figure8(self, env_ids: torch.Tensor = None):
            if env_ids is None:
                env_ids = torch.arange(self.num_envs, device=self.device)
            
            n_envs = len(env_ids)
            
            # 1. Update time variable
            self._figure8_time[env_ids] += self.dt
            t = self._figure8_time[env_ids]
            
            # 2. 重定中心逻辑
            needs_recenter = (t >= self._figure8_warmup_duration) & (~self._traj_origin_adjusted[env_ids])
            
            if needs_recenter.any():
                recenter_indices = env_ids[needs_recenter]
                current_robot_pos = self._robot.data.root_pos_w[recenter_indices].clone()
                self._spawn_pos_w[recenter_indices] = current_robot_pos
                self.pos_des[recenter_indices] = current_robot_pos
                self.pos_des_raw[recenter_indices] = current_robot_pos
                self._traj_origin_adjusted[recenter_indices] = True

            # 3. Calculate Figure-8 Path
            in_warmup = t < self._figure8_warmup_duration
            if torch.all(in_warmup):
                return 
            
            omega = 2 * math.pi * self._figure8_frequency
            spawn_pos = self._spawn_pos_w[env_ids]
            
            t_adjusted = t - self._figure8_warmup_duration
            
            x_rel = self._figure8_scale_x * torch.sin(omega * t_adjusted)
            y_rel = self._figure8_scale_y * torch.sin(2 * omega * t_adjusted)
            z_target = spawn_pos[:, 2] 
            
            pos_des_new = torch.stack([
                spawn_pos[:, 0] + x_rel,
                spawn_pos[:, 1] + y_rel,
                z_target
            ], dim=1)
            
            vx = self._figure8_scale_x * omega * torch.cos(omega * t_adjusted)
            vy = self._figure8_scale_y * 2 * omega * torch.cos(2 * omega * t_adjusted)
            vz = torch.zeros(n_envs, device=self.device)
            
            vel_des_new = torch.stack([vx, vy, vz], dim=1)
            
            active_mask = ~in_warmup
            active_env_ids = env_ids[active_mask]
            
            if len(active_env_ids) > 0:
                self.pos_des[active_env_ids] = pos_des_new[active_mask]
                self.vel_des[active_env_ids] = vel_des_new[active_mask]
                self.pos_des_raw[active_env_ids] = pos_des_new[active_mask]
                self.vel_des_raw[active_env_ids] = vel_des_new[active_mask]

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
                else:
                    mass_val = self.cfg.dynamics.mass
                    inertia_val = self.cfg.dynamics.inertia
                
                # --- 将配置写入底层 PhysX ---
                # 1. 修改质量
                prims_utils.set_prim_property(body_path, "physics:mass", mass_val)
                # 2. 修改惯性张量 (Isaac Sim 接受 (Ixx, Iyy, Izz) 对角形式)
                prims_utils.set_prim_property(body_path, "physics:diagonalInertia", inertia_val)
                # 3. 强制重心
                prims_utils.set_prim_property(body_path, "physics:centerOfMass", (0.0, 0.0, 0.0))

                # --- 设置可见性 ---
                if self.cfg.robot_vis:
                    prims_utils.set_prim_property(prim_path, "visibility", "visible")
                else:
                    prims_utils.set_prim_property(prim_path, "visibility", "invisible")

                # --- 抽样检查 (Read-back Verification) ---
                if i == 0 or (has_multi_teachers and i % envs_per_teacher == 0 and i < self.num_envs):
                    actual_mass_usd = prims_utils.get_prim_property(body_path, "physics:mass")
                    print(f"[Env {i}] Teacher ID: {t_id if has_multi_teachers else 0} | Set Mass: {mass_val:.4f} | PhysX Read: {actual_mass_usd:.4f}")

            print(f"{'='*60}\n")
            # ================= [DEBUG END] =================

            # 5. Add lights
            light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
            light_cfg.func("/World/Light", light_cfg)

            self._map_generation_timer = 0

    def _pre_physics_step(self, actions: torch.Tensor):

        # 1. 更新轨迹
        if self.cfg.trajectory_type == "figure8":
            self._generate_desired_trajectory_figure8()
        elif torch.any(self._is_langevin_task):
            self._generate_desired_trajectory_langevin(env_ids=torch.where(self._is_langevin_task)[0])

        # 2. Action Clamp
        raw_actions_clamped = torch.clamp(actions, -1.0, 1.0)
        action_setpoint_normalized = (raw_actions_clamped + 1.0) * 0.5
        
        self._actions = action_setpoint_normalized.clone()

        # 判断是加速还是减速
        # 如果 target > current, 使用 alpha_up
        # 如果 target < current, 使用 alpha_down
        target = action_setpoint_normalized
        current = self._current_motor_speeds
        
        # 构造混合 alpha
        # 这里使用了 torch.where: condition ? alpha_up : alpha_down
        alpha = torch.where(target > current, self.motor_alpha_up, self.motor_alpha_down)
        
        # 一阶低通滤波
        self._current_motor_speeds = alpha * target + (1.0 - alpha) * current

        # 计算力与力矩
        force_b, torque_b, _ = self._controller.motor_speeds_to_wrench(self._current_motor_speeds) 

        # 5. 施加力
        self._forces.zero_()
        self._torques.zero_()
        self._forces[:, 0, :] = force_b
        self._torques[:, 0, :] = torque_b
        
        self._robot.set_external_force_and_torque(self._forces, self._torques, body_ids=self._body_id)


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
        
        # 3. 坐标系转换
        # 计算世界系误差
        pos_error_w = pos_w - self.pos_des
        vel_error_w = vel_w - self.vel_des

        # 计算 R_w->b (即 R_b->w 的转置)
        rot_matrix_w2b = rot_matrix_b2w.transpose(1, 2) 

        # 投影到机体坐标系: error_body = R_w->b @ error_world
        pos_error_b = torch.bmm(rot_matrix_w2b, pos_error_w.unsqueeze(-1)).squeeze(-1)
        vel_error_b = torch.bmm(rot_matrix_w2b, vel_error_w.unsqueeze(-1)).squeeze(-1)

        motor_speeds_obs = self._current_motor_speeds

        # # 用于raptor测试
        # motor_speeds_obs = motor_speeds_obs * 2.0 - 1.0  # [0,1] → [-1,1]
        # last_actions = self._last_actions * 2.0 - 1.0  # [0,1] → [-1,1]
        # obs_teacher = torch.cat([
        #     pos_error_w,            
        #     rotation_matrix_flat,  
        #     vel_error_w,           
        #     ang_vel_b,              
        #     last_actions,    
        #     motor_speeds_obs
        # ], dim=-1)

        # obs_student = torch.cat([
        #     pos_error_w,            
        #     rotation_matrix_flat,  
        #     vel_error_w,           
        #     ang_vel_b,              
        #     last_actions,
        # ], dim=-1)

        obs_teacher = torch.cat([
            pos_error_b,            
            rotation_matrix_flat,  
            vel_error_b,           
            ang_vel_b,              
            self._last_actions,     
            motor_speeds_obs
        ], dim=-1)

        obs_student = torch.cat([
            pos_error_b,            
            rotation_matrix_flat,  
            vel_error_b,           
            ang_vel_b,              
            self._last_actions,
        ], dim=-1)

        # 4. 去除冗余历史堆叠，直接返回当前帧
        obs_teacher = self.CHECK_NAN(obs_teacher, "Observation")
        obs_student = self.CHECK_NAN(obs_student, "Observation")
        return {"policy": obs_student, "teacher": obs_teacher, "rnd_state": obs_student}
    
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
        
        # --- 4. 累加 Episode Sums ---
        reward_items = {
            "position": r_pos,
            "orientation": r_ori,
            "action_smooth": r_act,
            "base": r_base,
            "terminal": r_term
        }

        for key, value in reward_items.items():
            if key not in self._episode_sums:
                self._episode_sums[key] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self._episode_sums[key] += value
        
        # 更新历史动作
        self._last_actions = self._actions.clone()
        
        return total_reward


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
        
        # --- 1. 日志记录逻辑 ---
        if num_resets > 0:
            if "log" not in self.extras:
                self.extras["log"] = dict()
            for key in self._episode_sums.keys():
                values = self._episode_sums[key][env_ids]
                mean_val = torch.mean(values).item()
                self.extras["log"][f"Episode_Reward/{key}"] = mean_val
                self._episode_sums[key][env_ids] = 0.0


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
        # 清零所有不稳定性标志位
        self._numerical_is_unstable[env_ids] = False
        self._died_pos_limit[env_ids] = False
        self._died_lin_vel_limit[env_ids] = False
        self._died_ang_vel_limit[env_ids] = False
        self._died_tilt_limit[env_ids] = False
        self._died_nan[env_ids] = False

        # 重置轨迹时间 (用于八字形轨迹)
        self._figure8_time[env_ids] = 0.0
        self._traj_origin_adjusted[env_ids] = False

        self._langevin_max_vel[env_ids] = torch.rand(len(env_ids), device=self.device) * 1.0 + 0.5
        
        self._current_motor_speeds[env_ids] = 0.0 

        # --- 3. RAPTOR 初始化逻辑 ---
        l_arm_env = self.arm_l_tensor[env_ids]

        if self.cfg.train_or_play:
            # ================= [TRAIN MODE] =================
            r_pos_limit = 10.0 * l_arm_env   # 位置采样半径
            v_lin_limit = 1.0           # 线速度限制
            v_ang_limit = 1.0           # 角速度限制

            # 辅助函数：球体内均匀采样
            def sample_in_sphere(radius, n_samples):
                direction = torch.randn(n_samples, 3, device=self.device)
                direction = F.normalize(direction, p=2, dim=1)

                # --- 解决 float 和 [N] 张量两种情况 ---
                if isinstance(radius, float) or isinstance(radius, int):
                    # 标量 → 转成 shape=[n_samples, 1]
                    radius = torch.full((n_samples, 1), radius, device=self.device)
                else:
                    # Tensor:
                    # radius 可能是 [N] 或 [N,1]
                    if radius.dim() == 1:
                        radius = radius.unsqueeze(1)
                    # 如果是 [N,1] 直接保持

                u = torch.rand(n_samples, 1, device=self.device)
                r = radius * torch.pow(u, 1.0 / 3.0)

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
            pos_offset = torch.zeros(num_resets, 3, device=self.device)
            lin_vel = torch.zeros(num_resets, 3, device=self.device)
            ang_vel = torch.zeros(num_resets, 3, device=self.device)
            quat = torch.zeros(num_resets, 4, device=self.device)
            quat[:, 0] = 1.0 

        # --- 4. 设置仿真器状态 ---
        
        # 获取默认状态
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        
        # 计算出生中心点
        spawn_center = self.env_origins[env_ids].clone()
        spawn_center[:, 0] += self.cfg.terrain_length / 2.0
        spawn_center[:, 1] += self.cfg.terrain_width / 2.0
        spawn_center[:, 2] = self.cfg.height

        # 记录出生点
        start_pos = spawn_center + pos_offset
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

        # 初始化期望状态
        self.pos_des[env_ids] = spawn_center.clone()
        self.vel_des[env_ids] = 0.0
        self.pos_des_raw[env_ids] = spawn_center.clone()
        self.vel_des_raw[env_ids] = 0.0


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