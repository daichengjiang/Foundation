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
from foundation.utils.pid_controller import PaperPhysControllerTensor
import json
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
    mass: float = 0.10667879070394651
    arm_length: float = 0.06275213191421349
    inertia: tuple[float, float, float] = (0.0005043252020762681,0.0005043252020762681,0.0009239237702037232)
    thrust_to_weight: float = 3.663525773296906
    
    # [修改] 替换单一的 motor_tau
    motor_tau_up: float = 0.04962523021404839
    motor_tau_down: float = 0.05780714703383573 # 默认值设大一点体现差异
    
    # [新增] 力矩系数
    moment_scale: float = 0.03801975614353795

    multi_teacher_params: list[dict] | None = None

    apply_disturbance: bool = False         
    max_disturbance_force: float = 0.1       # 最大力(牛顿)。如果是大飞机，可以调大到 0.5~1.0

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
    observation_space = teacher_observation_space

    history_len = 5

    prob_null_trajectory = 0.5
    trajectory_type = "langevin"
    train_or_play: bool = True
    use_pid = False
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
    reward_coef_d_action_cost = 0.5
    reward_coef_termination_penalty = 100.0
    reward_constant = 1.5

class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.start_time = time.time()
        self.render_mode = "human"
        
        # ================= [关键部分：初始化异构动力学张量] =================

        self.mass_tensor = torch.zeros(self.num_envs, device=self.device)
        self.arm_l_tensor = torch.zeros(self.num_envs, device=self.device)
        self.twr_tensor = torch.zeros(self.num_envs, device=self.device)
        self.inertia_tensor = torch.zeros(self.num_envs, 3, device=self.device)
        self.motor_tau_up_tensor = torch.zeros(self.num_envs, 1, device=self.device)
        self.motor_tau_down_tensor = torch.zeros(self.num_envs, 1, device=self.device)
        self.kappa_tensor = torch.zeros(self.num_envs, device=self.device)

        # ================= [修改：Buffer 存储机体坐标系误差] =================
        self.history_len = self.cfg.history_len
        self.pos_error_b_history = torch.zeros(self.num_envs, self.history_len, 3, device=self.device)
        self.vel_error_b_history = torch.zeros(self.num_envs, self.history_len, 3, device=self.device)

        if self.cfg.dynamics.multi_teacher_params is not None:
            teacher_params_list = self.cfg.dynamics.multi_teacher_params
            num_teachers = len(teacher_params_list)
            envs_per_teacher = self.num_envs // num_teachers
            
            print(f"[Multi-Teacher Env] Initializing {num_teachers} teachers. Base envs per teacher: {envs_per_teacher}.")
            
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
                count = end_idx - start_idx
                print(f"  > Teacher {t_id} (ID: {params.get('id', 'N/A')}): Envs {start_idx}-{end_idx-1} (Count: {count})")

            # [添加调试打印]
            print(f"[DEBUG] Applied Multi-Teacher Params:")
            for i in range(min(5, self.num_envs)): # 打印前5个环境的质量
                print(f"  Env {i}: Mass = {self.mass_tensor[i].item():.4f}")
            mid_idx = self.num_envs // 2
            print(f"  Env {mid_idx} (Mid): Mass = {self.mass_tensor[mid_idx].item():.4f}")
        
        else:
            print("[OffsetEnv] No multi-teacher params found")
            self.mass_tensor.fill_(self.cfg.dynamics.mass)
            self.arm_l_tensor.fill_(self.cfg.dynamics.arm_length)
            self.inertia_tensor[:] = torch.tensor(self.cfg.dynamics.inertia, device=self.device)
            self.twr_tensor.fill_(self.cfg.dynamics.thrust_to_weight)
            self.motor_tau_up_tensor.fill_(self.cfg.dynamics.motor_tau_up)
            self.motor_tau_down_tensor.fill_(self.cfg.dynamics.motor_tau_down)
            self.kappa_tensor.fill_(self.cfg.dynamics.moment_scale)

        # Store for reference
        self._robot_mass = self.mass_tensor 
        self.dt = self.cfg.sim.dt

        if self.motor_tau_up_tensor.shape != (self.num_envs, 1):
             self.motor_tau_up_tensor = self.motor_tau_up_tensor.view(self.num_envs, 1)
        if self.motor_tau_down_tensor.shape != (self.num_envs, 1):
             self.motor_tau_down_tensor = self.motor_tau_down_tensor.view(self.num_envs, 1)
             
        self.motor_alpha_up = self.dt / torch.clamp(self.motor_tau_up_tensor, min=1e-6)
        self.motor_alpha_down = self.dt / torch.clamp(self.motor_tau_down_tensor, min=1e-6)

        self._current_motor_speeds = torch.zeros(self.num_envs, 4, device=self.device)

        self.env_wind_force = torch.zeros((self.num_envs, 3), device=self.device)

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
        
        self.pos_des = torch.zeros(self.num_envs, 3, device=self.device)
        self.vel_des = torch.zeros(self.num_envs, 3, device=self.device)
        self.acc_des = torch.zeros(self.num_envs, 3, device=self.device)
        self.yaw_des = torch.zeros(self.num_envs, device=self.device)      # 期望偏航角
        self.yaw_rate_des = torch.zeros(self.num_envs, device=self.device) # 期望偏航角速度

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

        self._history_window = 100
        self._episode_outcomes = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episodes_completed = 0
        self._termination_reason_history = collections.deque(maxlen=self._history_window)
        self._vel_abs = collections.deque(maxlen=self._history_window)

        # ================= 全局累加器 =================
        self.eval_episodes = 0
        self.eval_pos_sum = 0.0
        self.eval_ori_sum = 0.0
        self.eval_smooth_sum = 0.0
        self.eval_base_sum = 0.0     # <--- 新增：存活分累加
        self.eval_term_sum = 0.0     # <--- 新增：死亡惩罚累加
        self.eval_total_sum = 0.0
        self.reward_report_path = os.environ.get("TEACHER_REWARD_PATH", None)
        # =================================================================


        self.set_debug_vis(self.cfg.debug_vis)
        self._traj_origin_adjusted = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        self._calc_env_origins()
        # [新增] 用于统计平均奖励的变量
        self.steps_per_iteration = self.cfg.num_steps_per_env
        self.yaw_limit = math.pi / 2

        self.delay_steps = 8  
        self._action_queue = torch.zeros(
            self.num_envs, self.delay_steps, self.cfg.action_space, device=self.device
        )

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
        
        # 1. 弹簧刚度 (回复力)
        k_pos = torch.tensor([1.0, 1.0, 2.0], device=self.device)  
        # 2. 阻尼系数 (控制目标移动速度，值越大越慢)
        k_vel = 1.5
        # 3. 加速度惯性 (控制轨迹平滑度，值越小越平滑)
        acc_inertia = 0.1
        # 4. 随机力强度 (直接决定乱窜程度)
        noise_scale = 10.0
        # 5. 最大物理限制
        max_vel = 3.0
        max_acc = 15.0
        # ==============================================================

        # 获取当前状态
        pos_current = self.pos_des[env_ids]
        vel_current = self.vel_des[env_ids]
        acc_current = self.acc_des[env_ids]
        spawn_pos = self._spawn_pos_w[env_ids]

        # --- 积分更新 ---
        # Euler 积分更新速度和位置
        vel_next = vel_current + acc_current * dt
        
        # 限制最大速度 (软限制已由阻尼 k_vel 提供，这里做硬截断以防万一)
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
        z_noise_attenuation = 0.3  
        noise[:, 2] *= z_noise_attenuation

        # 期望的合外力 (Target Force/Mass)
        force_total = noise - k_pos * pos_err - k_vel * vel_current
        
        # 更新加速度 (一阶低通滤波，模拟物理惯性)
        # acc_new = (1 - alpha) * acc_old + alpha * target
        acc_next = (1.0 - acc_inertia) * acc_current + acc_inertia * force_total
        
        # --- 物理限制 (Clipping) ---
        # 限制最大加速度 (基于推重比，假设最大推力约为 2G 到 3G)
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
        grid_rows, grid_cols = self.cfg.grid_rows, self.cfg.grid_cols
        group_origins = torch.zeros(num_groups, 3, device=self.device)
        map_size_x, map_size_y = self.cfg.map_size
        offset_x = -map_size_x / 2.0
        offset_y = -map_size_y / 2.0
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
        if len(robot_prims) == 0:
            print("[ERROR] No robot prims found! Check your prim_path regex in the config.")
            return

        # ================= [应用多教师动力学参数到 PhysX] =================
        print(f"\n{'='*20} [Dynamics Setup] {'='*20}")
        print(f"Num Envs: {self.num_envs}")
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
                arm_l_val = params['arm_length']
            else:
                mass_val = self.cfg.dynamics.mass
                inertia_val = self.cfg.dynamics.inertia
                arm_l_val = self.cfg.dynamics.arm_length

            # --- 将配置写入底层 PhysX ---
            # 1. 修改质量
            prims_utils.set_prim_property(body_path, "physics:mass", mass_val)
            # 2. 修改惯性张量 (Isaac Sim 接受 (Ixx, Iyy, Izz) 对角形式)
            prims_utils.set_prim_property(body_path, "physics:diagonalInertia", inertia_val)
            # 3. 强制重心
            prims_utils.set_prim_property(body_path, "physics:centerOfMass", (0.0, 0.0, 0.0))

            if self.cfg.robot_vis:
                prims_utils.set_prim_property(prim_path, "visibility", "visible")
            else:
                prims_utils.set_prim_property(prim_path, "visibility", "invisible")

            if i == 0 or (has_multi_teachers and i % envs_per_teacher == 0 and i < self.num_envs):
                read_mass = prims_utils.get_prim_property(body_path, "physics:mass")
                print(f"[Env {i}] Config Mass: {mass_val:.4f} | PhysX Read: {read_mass:.4f}")
        print(f"{'='*60}\n")

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self._map_generation_timer = 0


    def _pre_physics_step(self, actions: torch.Tensor):

        raw_actions_clamped = torch.clamp(actions, -1.0, 1.0)
        self._actions = raw_actions_clamped.clone()

        # === 核心：纯滞后队列操作 ===
        # 队列整体向后滚一格 (丢弃最老帧)
        self._action_queue = torch.roll(self._action_queue, shifts=1, dims=1)
        # 将当前最新动作写入队列头
        self._action_queue[:, 0, :] = raw_actions_clamped
        # 取出 N 帧之前的动作作为生效动作
        delayed_actions = self._action_queue[:, -1, :].clone()

        # 将【延迟后】的动作映射到物理占空比 [0, 1]
        action_setpoint_normalized = (delayed_actions + 1.0) * 0.5
        
        # 电机惯性 (motor_alpha) 和拟合曲线逻辑
        target = action_setpoint_normalized
        current = self._current_motor_speeds
        alpha = torch.where(target > current, self.motor_alpha_up, self.motor_alpha_down)
        self._current_motor_speeds = current + alpha * (target - current)
        self._current_motor_speeds = torch.clamp(self._current_motor_speeds, 0.0, 1.0)

        force_b, torque_b = self._controller.motor_speeds_to_wrench(self._current_motor_speeds)
        
        # 随机力扰动
        if self.cfg.dynamics.apply_disturbance:
            base_quat = self._robot.data.root_quat_w
            rot_matrix = matrix_from_quat(base_quat) # 世界到机体的正向旋转矩阵 [num_envs, 3, 3]
            rot_matrix_inv = rot_matrix.transpose(1, 2) # 正交矩阵的转置即为逆矩阵 (世界 -> 机体)
            # 3.1 外部风扰：世界系 -> 机体系
            wind_body = torch.bmm(rot_matrix_inv, self.env_wind_force.unsqueeze(-1)).squeeze(-1)
            # 3.2 叠加机体系下的受力
            force_b +=  wind_body

        self._forces.zero_()
        self._torques.zero_()
        self._forces[:, 0, :] = force_b
        self._torques[:, 0, :] = torque_b

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._forces, self._torques, body_ids=self._body_id) 

    def _get_observations(self) -> dict:

        if self.cfg.trajectory_type == "figure8":
            self._generate_desired_trajectory_figure8()
        elif torch.any(self._is_langevin_task):
            self._generate_desired_trajectory_langevin(env_ids=torch.where(self._is_langevin_task)[0])

        pos_w = self._robot.data.root_pos_w
        quat_w = self._robot.data.root_quat_w
        vel_w = self._robot.data.root_lin_vel_w
        ang_vel_b = self._robot.data.root_ang_vel_b
        
        # 1. 构建当前的旋转矩阵 (World -> Body)
        rot_matrix_b2w = matrix_from_quat(quat_w)
        rot_matrix_w2b = rot_matrix_b2w.transpose(1, 2) 

        r, p, y = euler_xyz_from_quat(quat_w)
        yaw_error = self.yaw_des - y
        yaw_error = torch.remainder(yaw_error + math.pi, 2 * math.pi) - math.pi
        yaw_error_sin = torch.sin(yaw_error).unsqueeze(1) # (N, 1)
        yaw_error_cos = torch.cos(yaw_error).unsqueeze(1) # (N, 1)

        curr_pos_error_w = pos_w - self.pos_des
        curr_vel_error_w = vel_w - self.vel_des
        
        # (N, 3, 3) @ (N, 3, 1) -> (N, 3)
        curr_pos_error_b = torch.bmm(rot_matrix_w2b, curr_pos_error_w.unsqueeze(-1)).squeeze(-1)
        curr_vel_error_b = torch.bmm(rot_matrix_w2b, curr_vel_error_w.unsqueeze(-1)).squeeze(-1)

        # 3. 更新历史记录 (已经在 Body Frame 下的数据)
        self.pos_error_b_history = torch.roll(self.pos_error_b_history, shifts=-1, dims=1)
        self.vel_error_b_history = torch.roll(self.vel_error_b_history, shifts=-1, dims=1)
        
        self.pos_error_b_history[:, -1, :] = curr_pos_error_b
        self.vel_error_b_history[:, -1, :] = curr_vel_error_b

        pos_error_flat = self.pos_error_b_history.reshape(self.num_envs, -1)
        vel_error_flat = self.vel_error_b_history.reshape(self.num_envs, -1)

        # 4. 展平数据
        rot_flat = rot_matrix_b2w.reshape(self.num_envs, 9)
        # 5. 处理期望加速度和速度 (转到 Body Frame)
        acc_des_b = torch.bmm(rot_matrix_w2b, self.acc_des.unsqueeze(-1)).squeeze(-1)
        vel_des_b = torch.bmm(rot_matrix_w2b, self.vel_des.unsqueeze(-1)).squeeze(-1)

        # 教师 58
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
        return {"policy": obs_teacher, "critic": obs_teacher, "rnd_state": obs_teacher}
    def _get_rewards(self) -> torch.Tensor:
        # 1. 位置误差 (保持不变)
        pos_error_norm = torch.norm(self._robot.data.root_pos_w - self.pos_des, dim=1)
        
        # 2. Orientation Cost (仅跟踪 Yaw)
        # 获取当前四元数
        quat_w = self._robot.data.root_quat_w
        
        # 将四元数转换为欧拉角 (Roll, Pitch, Yaw)
        # 注意：euler_xyz_from_quat 返回的是 (roll, pitch, yaw) 元组
        _, _, yaw_curr = euler_xyz_from_quat(quat_w)
        
        # 计算误差：目标 Yaw - 当前 Yaw
        yaw_error = self.yaw_des - yaw_curr
        
        # --- 关键步骤：角度归一化 (Wrap to -pi ~ pi) ---
        # 这一步是为了解决“350度”和“10度”相差只有20度，而不是340度的问题
        # 使用 torch.remainder 确保结果在 [0, 2pi] 之间，然后减去 pi 移到 [-pi, pi]
        yaw_error = torch.remainder(yaw_error + torch.pi, 2 * torch.pi) - torch.pi
        # B. 【关键修改】将角度误差映射回“四元数 Z 分量”空间
        # 原版惩罚的是 q_z，物理上 q_z = sin(yaw/2)。
        # 为了复刻原版手感，我们计算“误差角的 sin(x/2)”
        yaw_error_mapped = torch.sin(yaw_error / 2.0)
        
        # C. 套用原版非线性公式
        # 公式：arccos( clamp( 1.0 - abs(x) ) )
        # 这一步保留了“接近目标时梯度无穷大”的特性，会促使 Agent 极其精确地对齐
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
        if self.cfg.train_or_play:
            time_out = self.episode_length_buf >= self.max_episode_length - 1
        else:
            time_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        self.CHECK_state()
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

        if "log" not in self.extras: self.extras["log"] = dict()
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
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        num_resets = len(env_ids)
        
        # --- 1. 日志记录与基础状态重置 ---
        if num_resets > 0:
            if "log" not in self.extras: self.extras["log"] = dict()
            for k in self._episode_sums.keys():
                values = self._episode_sums[k][env_ids]
                self.extras["log"][f"Episode_Reward/{k}"] = torch.mean(values).item()

            current_iter = self.common_step_counter // self.steps_per_iteration
            eval_start, eval_end = 900, 1000
            # ==========================================================

            # 判断当前迭代是否在这个区间内
            if eval_start <= current_iter <= eval_end:
                # 严谨累加：不论是 1 架炸机还是 3000 架成功，权重完全按架数计算！
                batch_count = len(env_ids)
                self.eval_episodes += batch_count
                
                self.eval_pos_sum += torch.sum(self._episode_sums["position"][env_ids]).item()
                self.eval_ori_sum += torch.sum(self._episode_sums["orientation"][env_ids]).item()
                self.eval_smooth_sum += torch.sum(self._episode_sums["action_smooth"][env_ids]).item()
                self.eval_base_sum += torch.sum(self._episode_sums["base"][env_ids]).item()        # <--- 新增
                self.eval_term_sum += torch.sum(self._episode_sums["terminal"][env_ids]).item()    # <--- 新增

                # 计算这批环境的 Total 奖励总和
                batch_total = torch.zeros(batch_count, device=self.device)
                for k in self._episode_sums.keys():
                    batch_total += self._episode_sums[k][env_ids]
                self.eval_total_sum += torch.sum(batch_total).item()

                # 计算绝对的全局大平均，并覆写 JSON 供外部读取
                stats = {
                    "position": self.eval_pos_sum / self.eval_episodes,
                    "orientation": self.eval_ori_sum / self.eval_episodes,
                    "action_smooth": self.eval_smooth_sum / self.eval_episodes,
                    "base": self.eval_base_sum / self.eval_episodes,         # <--- 新增写出
                    "terminal": self.eval_term_sum / self.eval_episodes,
                    "total": self.eval_total_sum / self.eval_episodes
                }
                
                # ================= [新增] 将评估数据推送至 WandB =================
                if "log" not in self.extras: 
                    self.extras["log"] = dict()
                for key, value in stats.items():
                    # 添加统一的前缀 Eval_Metrics，方便在 WandB 面板上归类查看
                    self.extras["log"][f"Eval_Metrics/{key}"] = value
                if self.reward_report_path:
                    try:
                        import json
                        with open(self.reward_report_path, "w") as f:
                            json.dump(stats, f)
                    except Exception as e:
                        pass
            
            # C. 清理本轮奖励累计
            for k in self._episode_sums.keys():
                self._episode_sums[k][env_ids] = 0.0

        # 2. --- 状态重置与基础清理 ---
        died_mask = self.reset_terminated[env_ids]
        timed_out_mask = self.reset_time_outs[env_ids]
        if hasattr(self, '_update_episode_outcomes_and_metrics'):
            self._update_episode_outcomes_and_metrics(env_ids, None, died_mask, timed_out_mask)


        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # 标志位与动作重置
        self._actions[env_ids] = 0.0
        self._last_actions[env_ids] = 0.0
        self._current_motor_speeds[env_ids] = 0.0
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

        spawn_center = self.env_origins[env_ids].clone()
        spawn_center[:, 0] += self.cfg.terrain_length / 2.0
        spawn_center[:, 1] += self.cfg.terrain_width / 2.0
        spawn_center[:, 2] = self.cfg.height
        
        self.pos_des[env_ids] = spawn_center
        self.vel_des[env_ids] = 0.0
        self.acc_des[env_ids] = 0.0
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
            pos_offset = sample_in_sphere(10.0 * l_arm, num_resets) # 位置偏移与轴距成正比
            lin_vel = sample_in_sphere(1.0, num_resets)            # 1m/s 内的随机初速度
            ang_vel = sample_in_sphere(1.0, num_resets)            # 1rad/s 内的随机角速度
            
            # 随机旋转 (Roll, Pitch, Yaw)
            r = (torch.rand(num_resets, device=self.device)*2-1) * (math.pi / 3)
            p = (torch.rand(num_resets, device=self.device)*2-1) * (math.pi / 3)
            y = (torch.rand(num_resets, device=self.device)*2-1) * (math.pi / 6)
            quat = quat_from_euler_xyz(r, p, y)
            
            # 10% 几率完美开局，加速初期收敛
            perfect_mask = torch.rand(num_resets, device=self.device) < 0.1
            pos_offset[perfect_mask] = 0.0
            lin_vel[perfect_mask] = 0.0
            ang_vel[perfect_mask] = 0.0
            quat[perfect_mask] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            self.yaw_des[env_ids] = (torch.rand(len(env_ids), device=self.device) * 2 - 1) * self.yaw_limit / 2.0
            self.yaw_rate_des[env_ids] = 0.0
        else:
            pos_offset = torch.zeros(num_resets, 3, device=self.device)
            lin_vel = torch.zeros(num_resets, 3, device=self.device)
            ang_vel = torch.zeros(num_resets, 3, device=self.device)
            quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(num_resets, 1)
            self.yaw_des[env_ids] = 0.0
            self.yaw_rate_des[env_ids] = 0.0
        rot_w2b_init = matrix_from_quat(quat).transpose(1, 2)
        initial_pos_err_b = torch.bmm(rot_w2b_init, pos_offset.unsqueeze(-1)).squeeze(-1)
        initial_vel_err_b = torch.bmm(rot_w2b_init, lin_vel.unsqueeze(-1)).squeeze(-1)
        
        # 用当前的 Body 误差填满 5 帧历史
        self.pos_error_b_history[env_ids] = initial_pos_err_b.unsqueeze(1).repeat(1, self.history_len, 1)
        self.vel_error_b_history[env_ids] = initial_vel_err_b.unsqueeze(1).repeat(1, self.history_len, 1)

        # --- 4. 写入物理仿真器 ---
        self._is_langevin_task[env_ids] = torch.rand(num_resets, device=self.device) > self.cfg.prob_null_trajectory
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] = spawn_center + pos_offset # 世界坐标位置
        root_state[:, 3:7] = quat                    # 姿态
        root_state[:, 7:10] = lin_vel                 # 世界线速度
        root_state[:, 10:13] = ang_vel                # 机体角速度

        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(
            self._robot.data.default_joint_pos[env_ids], 
            self._robot.data.default_joint_vel[env_ids], 
            None, env_ids
        )

        # [新增] 为重置的环境重新采样外部扰动力
        if self.cfg.dynamics.apply_disturbance:
            # 1. 随机生成 3D 方向向量 (使用高斯分布以保证各个方向概率均匀)
            directions = torch.randn(len(env_ids), 3, device=self.device)
            directions = F.normalize(directions, dim=-1) # 归一化为单位向量
            
            # 2. 随机生成力的标量大小 [0, max_disturbance_force]
            magnitudes = torch.rand(len(env_ids), 1, device=self.device) * self.cfg.dynamics.max_disturbance_force
            
            # 3. 赋值给这些刚刚重置的环境
            self.env_wind_force[env_ids] = directions * magnitudes
        else:
            # 如果开关关闭，风力设为 0
            self.env_wind_force[env_ids] = 0.0

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
        super().close()