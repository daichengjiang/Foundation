# Copyright (c) 2025 Xu Yang
# HKUST UAV Group
#
# Author: Xu Yang
# Affiliation: HKUST UAV Group
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
# Use modern Isaac Sim core imports
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
import collections
import itertools
from dataclasses import dataclass

from foundation.utils.simple_controller import SimpleQuadrotorController

MAP_SIZE = (250, 250) 

# 手动定义球体标记配置
SPHERE_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
        ),
    }
)

@configclass
class QuadcopterDynamicsCfg:
    mass: float = 0.0282
    arm_length: float = 0.04384
    inertia: tuple[float, float, float] = (2.44864e-5, 2.44864e-5, 3.61504e-5)
    thrust_to_weight: float = 2.25 
    
    # [修改] 替换单一的 motor_tau
    motor_tau_up: float = 0.05
    motor_tau_down: float = 0.10 # 默认值设大一点体现差异
    
    # [新增] 力矩系数
    moment_scale: float = 0.016 

    multi_teacher_params: list[dict] | None = None

# [0, 2pi] -> [-pi, pi]
def normallize_angle(angle: torch.Tensor):
    return torch.fmod(angle + math.pi, 2 * math.pi) - math.pi

class QuadcopterEnvWindow(BaseEnvWindow):
    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)

@configclass
class QuadcopterSceneCfg(InteractiveSceneCfg):
    num_envs: int = 512
    env_spacing: float = 64.0
    # [关键修改] 必须为 False 以支持异构物理
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
    # 15(pos_hist) + 9(rot) + 15(vel_hist) + 3(ang_vel) + 4(last_act) + 4(motor) + 3(acc_des) + 3(vel_des)
    frame_observation_space = 56
    observation_space = frame_observation_space

    history_len = 5

    prob_null_trajectory = 0.5
    trajectory_type = "langevin"
    train_or_play: bool = True
    gamma = 0.99
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

    train = True
    robot_vis = True
    marker_size = 0.05
    ui_window_class_type = QuadcopterEnvWindow

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
        render=RenderCfg(enable_dl_denoiser=True, dlss_mode=2)
    )

    dynamics: QuadcopterDynamicsCfg = QuadcopterDynamicsCfg()
    scene: InteractiveSceneCfg = QuadcopterSceneCfg()
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    height = 3.0
    position_threshold = 15.0
    position_threshold_langevin = 14
    linear_velocity_threshold = 4.0
    angular_velocity_threshold = 35.0

    reward_coef_position_cost = 1.0
    reward_coef_orientation_cost = 0.2
    reward_coef_d_action_cost = 1.0
    reward_coef_termination_penalty = 100.0
    reward_constant = 1.5


class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.start_time = time.time()
        self.render_mode = "human"
        
        # ================= [关键部分：初始化异构动力学张量] =================
        # 初始化张量容器
        self.mass_tensor = torch.zeros(self.num_envs, device=self.device)
        self.arm_l_tensor = torch.zeros(self.num_envs, device=self.device)
        self.twr_tensor = torch.zeros(self.num_envs, device=self.device)
        self.inertia_tensor = torch.zeros(self.num_envs, 3, device=self.device)
        # self.motor_tau = torch.zeros(self.num_envs, 1, device=self.device)
        self.motor_tau_up_tensor = torch.zeros(self.num_envs, 1, device=self.device)
        self.motor_tau_down_tensor = torch.zeros(self.num_envs, 1, device=self.device)
        self.kappa_tensor = torch.zeros(self.num_envs, device=self.device)

        # ================= [修改：Buffer 存储机体坐标系误差] =================
        self.history_len = self.cfg.history_len
        # 存储机体坐标系下的误差 (Body Frame)
        self.pos_error_b_history = torch.zeros(self.num_envs, self.history_len, 3, device=self.device)
        self.vel_error_b_history = torch.zeros(self.num_envs, self.history_len, 3, device=self.device)

        if self.cfg.dynamics.multi_teacher_params is not None:
            # 这里的 params 列表应该在 play_teacher_multi.py 中注入到 env_cfg
            teacher_params_list = self.cfg.dynamics.multi_teacher_params
            num_teachers = len(teacher_params_list)
            envs_per_teacher = self.num_envs // num_teachers
            
            print(f"[OffsetEnv] Initializing {num_teachers} heterogeneous teachers. Base envs per teacher: {envs_per_teacher}.")
            
            for t_id, params in enumerate(teacher_params_list):
                start_idx = t_id * envs_per_teacher
                # 最后一个教师处理剩余的所有环境
                if t_id == num_teachers - 1:
                    end_idx = self.num_envs
                else:
                    end_idx = start_idx + envs_per_teacher
                
                indices = slice(start_idx, end_idx)
                
                # 填充 Tensor (用于控制器和奖励计算)
                self.mass_tensor[indices] = params['mass']
                self.arm_l_tensor[indices] = params['arm_length']
                self.twr_tensor[indices] = params.get('twr', params.get('thrust_to_weight', 2.25))
                
                # Inertia handling
                inertia_val = params['inertia'] # Expecting tuple/list
                if isinstance(inertia_val, (list, tuple)):
                    self.inertia_tensor[indices, 0] = inertia_val[0]
                    self.inertia_tensor[indices, 1] = inertia_val[1]
                    self.inertia_tensor[indices, 2] = inertia_val[2]
                
                # self.motor_tau[indices] = params['motor_tau']
                self.motor_tau_up_tensor[indices] = params['motor_tau_up']
                self.motor_tau_down_tensor[indices] = params['motor_tau_down']
                self.kappa_tensor[indices] = params['kappa']
                
                # Debug print for first teacher
                if t_id == 0:
                    print(f"  > [T-0 Config] Mass: {params['mass']:.4f}, TWR: {self.twr_tensor[start_idx].item():.2f}")
        else:
            # Fallback to single config
            print("[OffsetEnv] No multi-teacher params found. Using default homogeneous dynamics.")
            self.mass_tensor.fill_(self.cfg.dynamics.mass)
            self.arm_l_tensor.fill_(self.cfg.dynamics.arm_length)
            self.inertia_tensor[:] = torch.tensor(self.cfg.dynamics.inertia, device=self.device)
            self.twr_tensor.fill_(self.cfg.dynamics.thrust_to_weight)
            # self.motor_tau.fill_(self.cfg.dynamics.motor_tau)
            # [修改] 填充新参数
            self.motor_tau_up_tensor.fill_(self.cfg.dynamics.motor_tau_up)
            self.motor_tau_down_tensor.fill_(self.cfg.dynamics.motor_tau_down)
            self.kappa_tensor.fill_(self.cfg.dynamics.moment_scale)
        
        # Store for reference
        self._robot_mass = self.mass_tensor 

        # 初始化控制器 (传入异构张量)
        self._controller = SimpleQuadrotorController(
            num_envs=self.num_envs,
            device=self.device,
            mass=self.mass_tensor,
            arm_length=self.arm_l_tensor,
            inertia=self.inertia_tensor,
            thrust_to_weight=self.twr_tensor,
            moment_scale=self.kappa_tensor # <--- 传入
        )
        # =================================================================

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

        # 状态标志位
        self._is_langevin_task = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._numerical_is_unstable = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._died_pos_limit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._died_lin_vel_limit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._died_ang_vel_limit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._died_tilt_limit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._died_nan = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._last_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._torques = torch.zeros(self.num_envs, 1, 3, device=self.device)
        
        # 目标状态
        self.pos_des = torch.zeros(self.num_envs, 3, device=self.device)
        self.vel_des = torch.zeros(self.num_envs, 3, device=self.device)
        self.acc_des = torch.zeros(self.num_envs, 3, device=self.device)
        
        # 轨迹参数
        self._langevin_dt = 0.01
        self._langevin_friction = 0.5
        self._langevin_omega = 1.5
        self._langevin_sigma = 3.0
        self._langevin_alpha = 0.2
        self._figure8_time = torch.zeros(self.num_envs, device=self.device)
        self._figure8_frequency = 0.1
        self._figure8_scale_x = 1.0
        self._figure8_scale_y = 0.5
        self._figure8_height = 3.0
        self._figure8_warmup_duration = 5.0

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["position", "orientation", "action_smooth", "base", "terminal"]
        }
        
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        self.grid_idx = None
        self._body_id = self._robot.find_bodies("body")[0]
        self._spawn_pos_w = torch.zeros(self.num_envs, 3, device=self.device) 
        self._last_angular_velocity= torch.zeros(self.num_envs, 3, device=self.device)
        self._langevin_max_vel = torch.full((self.num_envs,), 1.5, device=self.device)

        self._history_window = 100
        self._episode_outcomes = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episodes_completed = 0
        self._termination_reason_history = collections.deque(maxlen=self._history_window)
        self._vel_abs = collections.deque(maxlen=self._history_window)
        
        # 为了计算 Reward Mean，模拟 RSL-RL buffer
        self.reward_rolling_buffer = deque(maxlen=100)
        self.global_max_reward = -float('inf')
        self.reward_report_path = os.environ.get("TEACHER_MAX_REWARD_PATH", None)

        self.set_debug_vis(self.cfg.debug_vis)
        self._traj_origin_adjusted = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        self._calc_env_origins()

    def CHECK_NAN(self, tensor, name):
        if torch.isnan(tensor).any().item():
            print(f"[{name}] NaN detected in tensor of shape {tensor.shape}.")
            nan_env_mask = torch.any(torch.isnan(tensor), dim=1)
            self._died_nan = torch.logical_or(self._died_nan, nan_env_mask)
            self._numerical_is_unstable = torch.logical_or(self._numerical_is_unstable, nan_env_mask)
            tensor = tensor.nan_to_num(nan=0.0)
            return tensor
        else:
            return tensor

    def CHECK_state(self):
        pos_w = self._robot.data.root_pos_w 
        lin_vel_w = self._robot.data.root_lin_vel_w 
        ang_vel_b = self._robot.data.root_ang_vel_b 
        quat_w = self._robot.data.root_quat_w
        
        dist_spawn = torch.norm(pos_w - self._spawn_pos_w, dim=1)
        pos_exceeded = dist_spawn > self.cfg.position_threshold
        lin_vel_exceeded = torch.any(torch.abs(lin_vel_w) > self.cfg.linear_velocity_threshold, dim=1)
        ang_vel_exceeded = torch.any(torch.abs(ang_vel_b) > self.cfg.angular_velocity_threshold, dim=1)

        rot_matrix = matrix_from_quat(quat_w) 
        tilt_exceeded = rot_matrix[:, 2, 2] < 0.0
        
        self._died_pos_limit = torch.logical_or(self._died_pos_limit, pos_exceeded)
        self._died_lin_vel_limit = torch.logical_or(self._died_lin_vel_limit, lin_vel_exceeded)
        self._died_ang_vel_limit = torch.logical_or(self._died_ang_vel_limit, ang_vel_exceeded)
        self._died_tilt_limit = torch.logical_or(self._died_tilt_limit, tilt_exceeded)
        
        state_is_unstable = (
            self._died_pos_limit | self._died_lin_vel_limit | 
            self._died_ang_vel_limit | self._died_tilt_limit | self._died_nan
        )
        self._numerical_is_unstable = torch.logical_or(self._numerical_is_unstable, state_is_unstable)

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

            in_warmup = t < self._figure8_warmup_duration
            active_mask = ~in_warmup
            active_env_ids = env_ids[active_mask]
            
            if len(active_env_ids) > 0:
                omega = 2 * math.pi * self._figure8_frequency
                spawn = self._spawn_pos_w[active_env_ids] # 只取活跃环境的出生点
                t_adj = t[active_mask] - self._figure8_warmup_duration # 只取活跃环境的时间
                
                # --- 修正：补全 pos_des_new ---
                x_rel = self._figure8_scale_x * torch.sin(omega * t_adj)
                y_rel = self._figure8_scale_y * torch.sin(2 * omega * t_adj)
                z_target = spawn[:, 2] 
                pos_des_new = torch.stack([spawn[:, 0] + x_rel, spawn[:, 1] + y_rel, z_target], dim=1)
                
                # 速度
                vx = self._figure8_scale_x * omega * torch.cos(omega * t_adj)
                vy = self._figure8_scale_y * 2 * omega * torch.cos(2 * omega * t_adj)
                vz = torch.zeros_like(vx)
                vel_des_new = torch.stack([vx, vy, vz], dim=1)

                # 加速度
                ax = -self._figure8_scale_x * (omega**2) * torch.sin(omega * t_adj)
                ay = -self._figure8_scale_y * (4 * omega**2) * torch.sin(2 * omega * t_adj)
                az = torch.zeros_like(ax)
                acc_des_new = torch.stack([ax, ay, az], dim=1)
                
                # 赋值
                self.pos_des[active_env_ids] = pos_des_new
                self.vel_des[active_env_ids] = vel_des_new
                self.acc_des[active_env_ids] = acc_des_new

    def _calc_env_origins(self):
        robots_per_env = self.cfg.robots_per_env
        num_groups = self.num_envs // robots_per_env + 1
        grid_rows, grid_cols = self.cfg.grid_rows, self.cfg.grid_cols
        group_origins = torch.zeros(num_groups, 3, device=self.device)
        offset_x, offset_y = -MAP_SIZE[0]/2.0, -MAP_SIZE[1]/2.0

        for i in range(num_groups):
            row = (i // grid_cols) % grid_rows
            col = i % grid_cols
            group_origins[i, 0] = col * self.cfg.terrain_length + offset_x
            group_origins[i, 1] = row * self.cfg.terrain_width + offset_y
    
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
        self._robot = Articulation(self.cfg.robot)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self._robot

        robot_prims = find_matching_prim_paths("/World/envs/env_.*/Robot")
        if not robot_prims: return

        # ================= [Apply Heterogeneous Physics Properties] =================
        print(f"\n{'='*20} [OffsetEnv Physics Setup] {'='*20}")
        has_multi = self.cfg.dynamics.multi_teacher_params is not None
        
        if has_multi:
            params_list = self.cfg.dynamics.multi_teacher_params
            num_teachers = len(params_list)
            envs_per_teacher = self.num_envs // num_teachers
            print(f"Applying heterogenous properties for {num_teachers} teachers.")
        else:
            print("Applying default homogeneous properties.")

        for i, prim_path in enumerate(robot_prims):
            body_path = f"{prim_path}/body"
            
            if has_multi:
                t_id = min(i // envs_per_teacher, num_teachers - 1)
                p = params_list[t_id]
                mass = p['mass']
                inertia = p['inertia']
            else:
                mass = self.cfg.dynamics.mass
                inertia = self.cfg.dynamics.inertia
            
            # Write to USD
            prims_utils.set_prim_property(body_path, "physics:mass", mass)
            prims_utils.set_prim_property(body_path, "physics:diagonalInertia", inertia)
            prims_utils.set_prim_property(body_path, "physics:centerOfMass", (0.0, 0.0, 0.0))

            if self.cfg.robot_vis:
                prims_utils.set_prim_property(prim_path, "visibility", "visible")
            else:
                prims_utils.set_prim_property(prim_path, "visibility", "invisible")

            if i == 0 or (has_multi and i % envs_per_teacher == 0 and i < self.num_envs):
                read_mass = prims_utils.get_prim_property(body_path, "physics:mass")
                print(f"[Env {i}] Config Mass: {mass:.4f} | PhysX Read: {read_mass:.4f}")
        
        print(f"{'='*60}\n")
        
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self._map_generation_timer = 0

    def _pre_physics_step(self, actions: torch.Tensor):
        if self.cfg.trajectory_type == "figure8":
            self._generate_desired_trajectory_figure8()
        elif torch.any(self._is_langevin_task):
            self._generate_desired_trajectory_langevin(env_ids=torch.where(self._is_langevin_task)[0])

        raw_clamped = torch.clamp(actions, -1.0, 1.0)
        action_norm = (raw_clamped + 1.0) * 0.5
        self._actions = action_norm.clone()

        # 判断是加速还是减速
        # 如果 target > current, 使用 alpha_up
        # 如果 target < current, 使用 alpha_down
        target = action_norm
        current = self._current_motor_speeds
        
        # 构造混合 alpha
        # 这里使用了 torch.where: condition ? alpha_up : alpha_down
        alpha = torch.where(target > current, self.motor_alpha_up, self.motor_alpha_down)
        
        # 一阶低通滤波
        self._current_motor_speeds = alpha * target + (1.0 - alpha) * current

        # 计算力与力矩
        force_b, torque_b, _ = self._controller.motor_speeds_to_wrench(self._current_motor_speeds)    
        self._forces.zero_()
        self._torques.zero_()
        self._forces[:, 0, :] = force_b
        self._torques[:, 0, :] = torque_b
        self._robot.set_external_force_and_torque(self._forces, self._torques, body_ids=self._body_id)

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._forces, self._torques, body_ids=self._body_id)
            
    def _get_observations(self) -> dict:
        pos_w = self._robot.data.root_pos_w
        quat_w = self._robot.data.root_quat_w
        vel_w = self._robot.data.root_lin_vel_w
        ang_vel_b = self._robot.data.root_ang_vel_b
        
        # 1. 构建当前的旋转矩阵 (World -> Body)
        rot_matrix_b2w = matrix_from_quat(quat_w)
        rot_matrix_w2b = rot_matrix_b2w.transpose(1, 2) 

        # 2. 计算当前的世界坐标系误差，并立即转换到当前的 Body Frame
        curr_pos_error_w = pos_w - self.pos_des
        curr_vel_error_w = vel_w - self.vel_des
        
        # 转换当前帧到 Body Frame: (N, 3, 3) @ (N, 3, 1) -> (N, 3)
        curr_pos_error_b = torch.bmm(rot_matrix_w2b, curr_pos_error_w.unsqueeze(-1)).squeeze(-1)
        curr_vel_error_b = torch.bmm(rot_matrix_w2b, curr_vel_error_w.unsqueeze(-1)).squeeze(-1)

        # 3. 更新历史记录 (已经在 Body Frame 下的数据)
        self.pos_error_b_history = torch.roll(self.pos_error_b_history, shifts=-1, dims=1)
        self.vel_error_b_history = torch.roll(self.vel_error_b_history, shifts=-1, dims=1)
        
        self.pos_error_b_history[:, -1, :] = curr_pos_error_b
        self.vel_error_b_history[:, -1, :] = curr_vel_error_b

        # 4. 展平历史数据用于拼接 (N, T*3)
        pos_error_flat = self.pos_error_b_history.reshape(self.num_envs, -1)
        vel_error_flat = self.vel_error_b_history.reshape(self.num_envs, -1)

        # 5. 处理期望加速度和速度 (转到 Body Frame)
        acc_des_b = torch.bmm(rot_matrix_w2b, self.acc_des.unsqueeze(-1)).squeeze(-1)
        vel_des_b = torch.bmm(rot_matrix_w2b, self.vel_des.unsqueeze(-1)).squeeze(-1)

        rot_flat = rot_matrix_b2w.reshape(self.num_envs, 9)

        # 6. 拼接 Observation
        obs_teacher = torch.cat([
            pos_error_flat,             # 15
            rot_flat,                   # 9
            vel_error_flat,             # 15
            ang_vel_b,                  # 3
            self._last_actions,         # 4
            acc_des_b,                  # 3 
            vel_des_b,                  # 3 
            self._current_motor_speeds, # 4
        ], dim=-1)

        obs_teacher = self.CHECK_NAN(obs_teacher, "Observation")
        return {"policy": obs_teacher, "critic": obs_teacher, "rnd_state": obs_teacher}

    def _get_rewards(self) -> torch.Tensor:
        pos_error_norm = torch.norm(self._robot.data.root_pos_w - self.pos_des, dim=1)
        q_z = self._robot.data.root_quat_w[:, 3]
        orientation_cost = torch.arccos(torch.clamp(1.0 - torch.abs(q_z), -1.0, 1.0))
        d_action_cost = torch.norm(self._actions - self._last_actions, dim=1)
        
        r_pos = -pos_error_norm * self.cfg.reward_coef_position_cost
        r_ori = -orientation_cost * self.cfg.reward_coef_orientation_cost
        r_act = -d_action_cost * self.cfg.reward_coef_d_action_cost
        r_base = self.cfg.reward_constant
        r_term = -self._numerical_is_unstable.float() * self.cfg.reward_coef_termination_penalty

        # Accumulate sums
        sums = {
            "position": r_pos, "orientation": r_ori, "action_smooth": r_act, 
            "base": torch.full_like(r_pos, self.cfg.reward_constant), "terminal": r_term
        }
        for k, v in sums.items():
            if k not in self._episode_sums: self._episode_sums[k] = torch.zeros_like(v)
            self._episode_sums[k] += v

        self._last_actions = self._actions.clone()
        return r_pos + r_ori + r_act + r_base + r_term

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.train_or_play:
            time_out = self.episode_length_buf >= self.max_episode_length - 1
        else:
            time_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        self.CHECK_state()
        
        dist_traj = torch.norm(self.pos_des - self._spawn_pos_w, dim=1)
        pos_exceeded_langevin = dist_traj > self.cfg.position_threshold_langevin
        
        died = torch.logical_or(self._numerical_is_unstable, pos_exceeded_langevin)
        
        # Logging logic
        if "log" not in self.extras: self.extras["log"] = dict()
        comp = torch.logical_or(died, time_out)
        if torch.any(comp):
            self.extras["log"].update({
                "Metrics/died_episodes": torch.sum(died).item(),
                "Metrics/timeout_episodes": torch.sum(time_out).item()
            })
            
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None: env_ids = self._robot._ALL_INDICES
        
        # 1. --- 统计与日志处理 ---
        if len(env_ids) > 0:
            batch_rewards = torch.zeros(len(env_ids), device=self.device)
            for k in self._episode_sums.keys():
                batch_rewards += self._episode_sums[k][env_ids]
            self.reward_rolling_buffer.extend(batch_rewards.cpu().tolist())
            
            # 记录最大奖励均值
            if len(self.reward_rolling_buffer) > 0:
                cur_mean = np.mean(self.reward_rolling_buffer)
                if cur_mean > self.global_max_reward:
                    self.global_max_reward = cur_mean
                    if self.reward_report_path:
                        try:
                            with open(self.reward_report_path, "w") as f: f.write(str(cur_mean))
                        except: pass
            
            # 清理本轮奖励累计
            if "log" not in self.extras: self.extras["log"] = dict()
            for k in self._episode_sums.keys():
                values = self._episode_sums[k][env_ids]
                self.extras["log"][f"Episode_Reward/{k}"] = torch.mean(values).item()
                self._episode_sums[k][env_ids] = 0.0

        # 2. --- 状态重置与基础清理 ---
        died = self.reset_terminated[env_ids]
        tout = self.reset_time_outs[env_ids]
        self._update_episode_outcomes_and_metrics(env_ids, None, died, tout)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # 标志位与动作重置
        self._actions[env_ids] = 0.0
        self._last_actions[env_ids] = 0.0
        self._current_motor_speeds[env_ids] = 0.0
        self._numerical_is_unstable[env_ids] = False
        self._died_pos_limit[env_ids] = False
        self._died_lin_vel_limit[env_ids] = False
        self._died_ang_vel_limit[env_ids] = False
        self._died_tilt_limit[env_ids] = False
        self._died_nan[env_ids] = False
        
        self._figure8_time[env_ids] = 0.0
        self._traj_origin_adjusted[env_ids] = False
        self._langevin_max_vel[env_ids] = torch.rand(len(env_ids), device=self.device) * 1.0 + 0.5

        # 3. --- 轨迹中心点设定 ---
        spawn_center = self.env_origins[env_ids].clone()
        spawn_center[:, 0] += self.cfg.terrain_length / 2.0
        spawn_center[:, 1] += self.cfg.terrain_width / 2.0
        spawn_center[:, 2] = self.cfg.height
        
        self.pos_des[env_ids] = spawn_center
        self.vel_des[env_ids] = 0.0
        self.acc_des[env_ids] = 0.0
        self._spawn_pos_w[env_ids] = spawn_center

        # 4. --- 随机初始状态采样 (RAPTOR style) ---
        num_resets = len(env_ids)
        l_arm = self.arm_l_tensor[env_ids]
        
        if self.cfg.train_or_play:
            # 定义球体内均匀采样函数
            def sample_in_sphere(r, n):
                if isinstance(r, torch.Tensor) and r.dim()==1: r = r.unsqueeze(1)
                d = torch.randn(n, 3, device=self.device)
                d = F.normalize(d, p=2, dim=1)
                u = torch.rand(n, 1, device=self.device)
                return d * (r * torch.pow(u, 1.0/3.0))

            # 采样偏移量
            pos_offset = sample_in_sphere(10.0 * l_arm, num_resets) # 位置偏移与轴距成正比
            lin_vel = sample_in_sphere(1.0, num_resets)            # 1m/s 内的随机初速度
            ang_vel = sample_in_sphere(1.0, num_resets)            # 1rad/s 内的随机角速度
            
            # 随机旋转 (Roll, Pitch, Yaw)
            r = (torch.rand(num_resets, device=self.device)*2-1) * (math.pi/2)
            p = (torch.rand(num_resets, device=self.device)*2-1) * (math.pi/2)
            y = (torch.rand(num_resets, device=self.device)*2-1) * math.pi
            quat = quat_from_euler_xyz(r, p, y)
            
            # 10% 几率完美开局，加速初期收敛
            perfect_mask = torch.rand(num_resets, device=self.device) < 0.1
            pos_offset[perfect_mask] = 0.0
            lin_vel[perfect_mask] = 0.0
            ang_vel[perfect_mask] = 0.0
            quat[perfect_mask] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        else:
            # Play 模式固定开局
            pos_offset = torch.zeros(num_resets, 3, device=self.device)
            lin_vel = torch.zeros(num_resets, 3, device=self.device)
            ang_vel = torch.zeros(num_resets, 3, device=self.device)
            quat = torch.zeros(num_resets, 4, device=self.device)
            quat[:, 0] = 1.0

        # 5. --- [核心优化] 初始化历史 Buffer (同步随机状态) ---
        # 计算重置时刻的 Body-frame 旋转矩阵
        rot_w2b = matrix_from_quat(quat).transpose(1, 2)
        
        # 初始世界误差：机器人位置(spawn+offset) - 目标(spawn) = offset
        # 初始速度误差：机器人速度(lin_vel) - 目标速度(0) = lin_vel
        initial_pos_err_b = torch.bmm(rot_w2b, pos_offset.unsqueeze(-1)).squeeze(-1)
        initial_vel_err_b = torch.bmm(rot_w2b, lin_vel.unsqueeze(-1)).squeeze(-1)
        
        # 用当前的 Body 误差填满 5 帧历史
        self.pos_error_b_history[env_ids] = initial_pos_err_b.unsqueeze(1).repeat(1, self.history_len, 1)
        self.vel_error_b_history[env_ids] = initial_vel_err_b.unsqueeze(1).repeat(1, self.history_len, 1)

        # 6. --- 任务类型分配 ---
        # 决定该环境运行 Langevin 轨迹还是固定/其他轨迹
        self._is_langevin_task[env_ids] = torch.rand(num_resets, device=self.device) > self.cfg.prob_null_trajectory

        # 7. --- 写入物理仿真器 ---
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] = spawn_center + pos_offset # 世界坐标位置
        root_state[:, 3:7] = quat                    # 姿态
        root_state[:, 7:10] = lin_vel                 # 世界线速度
        root_state[:, 10:13] = ang_vel                # 机体角速度
        
        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        
        # 重置关节状态（如果模型有螺旋桨旋转等关节）
        self._robot.write_joint_state_to_sim(
            self._robot.data.default_joint_pos[env_ids], 
            self._robot.data.default_joint_vel[env_ids], 
            None, env_ids
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                m_cfg = CUBOID_MARKER_CFG.copy()
                m_cfg.markers["cuboid"].size = (self.cfg.marker_size,)*3
                m_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(m_cfg)
            self.goal_pos_visualizer.set_visibility(True)
            
            if not hasattr(self, "traj_langevin_visualizer"):
                l_cfg = SPHERE_MARKER_CFG.copy()
                l_cfg.prim_path = "/Visuals/TrajectoryLangevin"
                l_cfg.markers["sphere"].scale = (self.cfg.marker_size,)*3
                l_cfg.markers["sphere"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
                self.traj_langevin_visualizer = VisualizationMarkers(l_cfg)
            self.traj_langevin_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"): self.goal_pos_visualizer.set_visibility(False)
            if hasattr(self, "traj_langevin_visualizer"): self.traj_langevin_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        fixed_rot = torch.zeros_like(self._robot.data.root_quat_w); fixed_rot[:, 0] = 1.0
        inv_pos = torch.zeros_like(self.pos_des); inv_pos[:, 2] = -100.0
        
        if hasattr(self, "traj_langevin_visualizer"):
            pos = inv_pos.clone()
            mask = self._is_langevin_task
            if mask.any(): pos[mask] = self.pos_des[mask]
            self.traj_langevin_visualizer.visualize(pos, fixed_rot)

    def _update_episode_outcomes_and_metrics(self, env_ids, success_mask, died_mask, timed_out_mask):
        completed_mask = torch.logical_or(died_mask, timed_out_mask)
        if not torch.any(completed_mask): return 0, 0
        
        comp_ids = env_ids[completed_mask]
        died_ids = env_ids[died_mask]
        
        if len(died_ids) > 0:
            r_pos = self._died_pos_limit[died_ids].cpu().numpy()
            r_langevin = (torch.norm(self.pos_des[died_ids]-self._spawn_pos_w[died_ids], dim=1) > self.cfg.position_threshold_langevin).cpu().numpy()
            
            for i in range(len(died_ids)):
                self._termination_reason_history.append({
                    "died_pos_limit": bool(r_pos[i]),
                    "position_exceeded_langevin": bool(r_langevin[i])
                })
        
        self._termination_reason_history.extend([{}] * len(env_ids[timed_out_mask]))
        
        if len(comp_ids) > 0:
            vel = torch.linalg.norm(self._robot.data.root_lin_vel_w[comp_ids], dim=1).cpu().tolist()
            self._vel_abs.extend(vel)
            self._episodes_completed += len(comp_ids)

        if len(self._termination_reason_history) > 0:
            died_count = sum(1 for r in self._termination_reason_history if r)
            self.extras["log"].update({
                "Episode_Termination/died": died_count / len(self._termination_reason_history),
                "Metrics/episodes_completed": self._episodes_completed
            })
        
        return len(comp_ids), 0

    def close(self):
        super().close()