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

    # Simple flat terrain for open space trajectory tracking
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
    )

@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    # custom config for the quadcopter environment
    history_obs = 1
    # Updated observation space: pos_error(3) + rot_matrix(9) + vel_error(3) + ang_vel(3) + last_actions(4) + motor_speeds(4)
    frame_observation_space = 3 + 9 + 3 + 3 + 4 + 4  # 26

    # Calculate total observation space (without depth history, only current frame)
    observation_space = frame_observation_space  # 26D: pos_error(3) + rot_matrix(9) + vel_error(3) + ang_vel(3) + last_actions(4) + motor_speeds(4)

    # 轨迹采样配置
    prob_null_trajectory = 0.5  # 50% 概率做定点控制

    # gamma in ppo, only for logging
    gamma = 0.99

    # env
    episode_length_s = 96
    decimation = 1
    action_space = 4 
    state_space = 0
    debug_vis = True

    grid_rows = 20 # 12
    grid_cols = 20 # 1
    terrain_width = 40
    terrain_length = 40
    robots_per_env = 1

    # terrain and robot
    train = True
    robot_vis = False
    marker_size = 0.05  # Size of the markers in meters

    # Maximum velocity for Langevin trajectory generation
    max_vel = 2.0

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
    
    # State check thresholds (for any dimension x, y, z)
    position_threshold = 5.0  # meters
    position_threshold_langevin = 4.5  # 根据实际需求调整

    linear_velocity_threshold = 2.0  # m/s
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

        # Controller
        mass_tensor = torch.full((self.num_envs,), 0.800, device=self.device)
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
        )

        self._is_langevin_task = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

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
        self._langevin_dt = 0.1  # Time step for integration
        self._langevin_friction = 2.0  # Damping coefficient (gamma)
        self._langevin_omega = 1.0  # Oscillator frequency (omega)
        self._langevin_sigma = 0.5  # Noise intensity (sigma)
        self._langevin_alpha = 0.9  # Smoothing factor for exponential moving average (alpha)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "position_penalty",
                "orientation_penalty",
                "action_smoothness_penalty",
                "base_reward",
                "terminal_penalty",
            ]
        }
        

        # Environment origins
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        self.grid_idx = None
        # Robot references
        self._body_id = self._robot.find_bodies("body")[0]

        self._last_actions = torch.zeros(self.num_envs, 4, device=self.device) # [roll_rate, pitch_rate, yaw_rate, thrust]
        self._numerical_is_unstable = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._spawn_pos_w = torch.zeros(self.num_envs, 3, device=self.device)  # Store spawn/respawn positions

        self._last_angular_velocity= torch.zeros(self.num_envs, 3, device=self.device)

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
            nan_env_indices = torch.where(nan_env_mask)[0]
            print(f"NaN positions: {nan_env_indices}")
            self._numerical_is_unstable = torch.logical_or(self._numerical_is_unstable, nan_env_mask)
            tensor = tensor.nan_to_num(nan=0.0)
            raise ValueError("observation is NAN NAN NAN")
            return tensor
        else:
            return tensor

    # def CHECK_state(self):
    #         # Limit
    #         max_angular_velocity = 3.14 * 2.0 * 20.0 # rad/s

    #         # State
    #         ang_vel_b = self._robot.data.root_ang_vel_b
    #         rot_w = torch.stack(euler_xyz_from_quat(self._robot.data.root_quat_w), dim=1) # (num_envs, 3) roll, pitch, yaw
    #         rot_w = torch.stack([normallize_angle(rot_w[:, 0]), normallize_angle(rot_w[:, 1]), normallize_angle(rot_w[:, 2])], dim=1)

    #         state_is_unstable = torch.any(torch.abs(ang_vel_b) > max_angular_velocity, dim=1)

    #         self._numerical_is_unstable = torch.logical_or(self._numerical_is_unstable, state_is_unstable)

    def CHECK_state(self):
        """
        Check if any environment should terminate based on state thresholds.
        
        Episode terminates if ANY dimension (x, y, z) satisfies ANY of:
        - distance from spawn point > position_threshold
        - |linear_velocity[i]| > linear_velocity_threshold
        - |angular_velocity[i]| > angular_velocity_threshold
        
        This function updates self._numerical_is_unstable flag.
        """
        # Get robot states
        pos_w = self._robot.data.root_pos_w  # (num_envs, 3) - world position [x, y, z]
        lin_vel_w = self._robot.data.root_lin_vel_w  # (num_envs, 3) - world linear velocity
        ang_vel_b = self._robot.data.root_ang_vel_b  # (num_envs, 3) - body angular velocity
        
        # Check distance from spawn point
        distance_from_spawn = torch.norm(pos_w - self._spawn_pos_w, dim=1)  # (num_envs,)
        position_exceeded = distance_from_spawn > self.cfg.position_threshold
        
        # Check linear velocity threshold for any dimension
        linear_velocity_exceeded = torch.any(torch.abs(lin_vel_w) > self.cfg.linear_velocity_threshold, dim=1)
        
        # Check angular velocity threshold for any dimension
        angular_velocity_exceeded = torch.any(torch.abs(ang_vel_b) > self.cfg.angular_velocity_threshold, dim=1)
        
        # Combine all conditions: terminate if ANY condition is met
        state_is_unstable = position_exceeded | linear_velocity_exceeded | angular_velocity_exceeded
        
        # Update the numerical instability flag
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
        
        n_envs = len(env_ids)
        
        # Get parameters from config
        gamma = self._langevin_friction  # damping coefficient
        omega = self._langevin_omega  # oscillator frequency
        sigma = self._langevin_sigma  # noise intensity
        dt = self._langevin_dt  # integration time step
        alpha = self._langevin_alpha  # smoothing factor
        
        sqrt_dt = torch.sqrt(torch.tensor(dt, device=self.device))
        
        # Get previous raw states
        x_prev = self.pos_des_raw[env_ids]  # (n_envs, 3)
        v_prev = self.vel_des_raw[env_ids]  # (n_envs, 3)
        
        # Generate Wiener process noise: dW ~ N(0, 1) * sqrt(dt)
        dW = sqrt_dt * torch.randn(n_envs, 3, device=self.device)
        
        # Update velocity using damped harmonic oscillator with noise
        # v_next = v_prev + (-γv - ω²x) dt + σ dW
        v_next = v_prev + (-gamma * v_prev - omega * omega * x_prev) * dt + sigma * dW
        
        # Update position using velocity
        # x_next = x_prev + v_next dt
        x_next = x_prev + v_next * dt
        
        # Store raw (unsmoothed) states
        self.pos_des_raw[env_ids] = x_next
        self.vel_des_raw[env_ids] = v_next
        
        # Apply exponential moving average smoothing
        # v_smooth = α * v_next + (1 - α) * v_smooth_prev
        v_smooth_prev = self.vel_des[env_ids]
        v_smooth = alpha * v_next + (1.0 - alpha) * v_smooth_prev
        
        # x_smooth = x_smooth_prev + v_smooth * dt
        x_smooth_prev = self.pos_des[env_ids]
        x_smooth = x_smooth_prev + v_smooth * dt
        
        # Store smoothed states (these are used for control)
        self.pos_des[env_ids] = x_smooth
        self.vel_des[env_ids] = v_smooth
        
        # Keep z-coordinate within reasonable bounds (only for smoothed trajectory)
        self.pos_des[env_ids, 2] = torch.clamp(
            self.pos_des[env_ids, 2],
            self.cfg.desired_low,
            self.cfg.desired_high
        )

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

        for i in range(num_groups):
            row = (i // grid_cols) % grid_rows  # Loop rows if exceeding grid capacity
            col = i % grid_cols  # Loop columns if exceeding grid capacity
            group_origins[i, 0] = col * terrain_length
            group_origins[i, 1] = row * terrain_width
    
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
        robot_prims = find_matching_prim_paths("/World/envs/env_.*/Robot")
        for prim_path in robot_prims:
            prims_utils.set_prim_property(prim_path + "/body", "physics:mass", 0.049)
            prims_utils.set_prim_property(prim_path + "/body", "physics:diagonalInertia", (1.3615e-5, 1.3615e-5, 3.257e-5))
            if self.cfg.robot_vis == True:
                prims_utils.set_prim_property(prim_path, "visibility", "visible")
            else:
                prims_utils.set_prim_property(prim_path, "visibility", "invisible")

        # Clone the scene
        self.scene.clone_environments(copy_from_source=False)

        # Add the robot to the scene
        self.scene.articulations["robot"] = self._robot

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self._map_generation_timer = 0

    def _pre_physics_step(self, actions: torch.Tensor):
        # 1. 更新轨迹 (仅针对 Langevin 任务)
        if torch.any(self._is_langevin_task):
            self._generate_desired_trajectory_langevin(env_ids=torch.where(self._is_langevin_task)[0])

        actions = (actions + 1.0) * 0.5
        self._actions = actions.clone()

        # 1. 计算机体系下的 Wrench
        force_b, torque_b, px4info = self._controller.motor_speeds_to_wrench(self._actions, normalized=True)

        # 2. 获取姿态
        quat_w = self._robot.data.root_quat_w
        rot_b2w = matrix_from_quat(quat_w) # (num_envs, 3, 3)

        # 3. 准备机体坐标系下的力和力矩向量
        # Force: only Z component usually has value, but let's be generic
        force_vec_b = torch.zeros_like(self._forces[:, 0, :])
        force_vec_b[:, 2] = force_b[:, 2] # Thrust in Body Z

        # Torque
        torque_vec_b = torque_b # Torques are in body frame

        self.px4info = px4info

        # 4. 填充力矩 Buffer
        self._forces.zero_()
        self._torques.zero_()
        # 5. [关键] 将总力与力矩旋转到世界坐标系
        force_w = torch.bmm(rot_b2w, force_vec_b.unsqueeze(-1)).squeeze(-1)
        torque_w = torch.bmm(rot_b2w, torque_vec_b.unsqueeze(-1)).squeeze(-1)

        # 6. 赋值给 buffer
        self._forces[:, 0, :] = force_w
        self._torques[:, 0, :] = torque_w

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

        # 3. 拼接 Observation (使用转换后的 _b 变量)
        obs = torch.cat([
            pos_error_b,            # 3维 (Body Frame)
            rotation_matrix_flat,   # 9维
            vel_error_b,            # 3维 (Body Frame)
            ang_vel_b,              # 3维
            self._last_actions,     # 4维
            self._robot.data.joint_vel # 4维
        ], dim=-1)  # 总共 26维
        
        # 4. 去除冗余历史堆叠，直接返回当前帧
        obs = self.CHECK_NAN(obs, "Observation")
        return {"policy": obs, "critic": obs, "rnd_state": obs}

    def _get_rewards(self) -> torch.Tensor:
        """
        Calculate the reward for each environment based on trajectory tracking.
        
        Reward formula:
        r(s_t, a, s_{t+1}) = -c1·∥p∥ - c2·arccos(1 - |q_z|) - c3·∥a_t - a_{t-1}∥ + c4 - c5·1[terminal(s_{t+1})]
        
        Where:
        - p: position error (current_pos - pos_des)
        - q_z: z-component of quaternion (orientation error indicator)
        - a_t, a_{t-1}: current and previous actions
        - terminal: whether the episode terminates (collision, out of bounds, etc.)
        - c1, c2, c3, c4, c5: configurable reward coefficients
        """
        # Get current states
        pos_w = self._robot.data.root_pos_w  # (num_envs, 3)
        quat_w = self._robot.data.root_quat_w  # (num_envs, 4) [w, x, y, z]
        
        # 1. Position error term: -c1·∥p∥
        # p = current_pos - pos_des
        pos_error = pos_w - self.pos_des  # (num_envs, 3)
        pos_error_norm = torch.norm(pos_error, dim=1)  # (num_envs,)
        position_cost = pos_error_norm  # Raw cost (will be multiplied by coefficient)
        
        # 2. Orientation error term: -c2·arccos(1 - |q_z|)
        # q_z is the z-component of quaternion (index 3 for [w,x,y,z])
        q_z = quat_w[:, 3]  # (num_envs,)
        q_z_abs = torch.abs(q_z)
        # Clamp the argument to arccos to [-1, 1] for numerical stability
        arccos_arg = torch.clamp(1.0 - q_z_abs, -1.0, 1.0)
        orientation_cost = torch.arccos(arccos_arg)  # Raw cost
        
        # 3. Action smoothness term: -c3·∥a_t - a_{t-1}∥
        action_diff = self._actions - self._last_actions  # (num_envs, 4)
        action_diff_norm = torch.norm(action_diff, dim=1)  # (num_envs,)
        d_action_cost = action_diff_norm  # Raw cost
        
        # 4. Base reward: +c4 (reward_constant)
        constant = torch.full((self.num_envs,), 1.0, device=self.device)  # Will be scaled by reward_constant
        
        # 5. Terminal penalty: -c5·1[terminal]
        # Check for termination conditions
        terminal = (
            self._numerical_is_unstable | 
            (self._robot.data.root_pos_w[:, 2] < self.cfg.too_low) | 
            (self._robot.data.root_pos_w[:, 2] > self.cfg.too_high)
        )
        termination_penalty = terminal.float()  # 1.0 if terminal, 0.0 otherwise
        
        # Apply reward coefficients and combine all components
        reward_components = torch.stack(
            [
                -position_cost * self.cfg.reward_coef_position_cost,
                -orientation_cost * self.cfg.reward_coef_orientation_cost,
                -d_action_cost * self.cfg.reward_coef_d_action_cost,
                constant * self.cfg.reward_constant,
                -termination_penalty * self.cfg.reward_coef_termination_penalty,
            ],
            dim=-1
        )
        
        total_reward = torch.sum(reward_components, dim=1)
        
        # For logging (update episode sums)
        # Store each component for analysis
        component_names = [
            "position_penalty",
            "orientation_penalty", 
            "action_smoothness_penalty",
            "base_reward",
            "terminal_penalty"
        ]
        
        # Update episode sums for logging
        for key, idx in zip(component_names, range(reward_components.shape[1])):
            if key not in self._episode_sums:
                self._episode_sums[key] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self._episode_sums[key] = self._episode_sums[key] + reward_components[:, idx]
        
        # Update "last" values
        self._last_actions = self._actions.clone()
            
        return total_reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Define terminations and timeouts."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        self.CHECK_state()

        # Check distance from desired trajectory position (Langevin threshold)
        pos_w = self._robot.data.root_pos_w  # (num_envs, 3)
        distance_from_desired = torch.norm(pos_w - self.pos_des, dim=1)  # (num_envs,)
        position_exceeded_langevin = distance_from_desired > self.cfg.position_threshold_langevin

        conditions = [
            self._numerical_is_unstable,  # Numerical instability
            self._robot.data.root_pos_w[:, 2] < self.cfg.too_low,  # Z position too low
            self._robot.data.root_pos_w[:, 2] > self.cfg.too_high,  # Z position too high
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
        """Reset specific environment indexes."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        
        if self._wind_gen is not None:
            # Reset wind generator for the environments being reset
            self._wind_gen.reset(env_ids)

        # For trajectory tracking task: no success/failure distinction
        # Only track died (physical limit violations) and timeouts
        died_mask = self.reset_terminated[env_ids]
        timed_out_mask = self.reset_time_outs[env_ids]

        # Update episode outcomes and metrics (if this method exists)
        # For trajectory tracking: died_mask = terminated, no success_mask needed
        if hasattr(self, '_update_episode_outcomes_and_metrics'):
            # Pass empty success mask since there's no success criterion in trajectory tracking
            success_mask = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
            self._update_episode_outcomes_and_metrics(env_ids, success_mask, died_mask, timed_out_mask)

        # Update reward component logs
        extras = dict()
        for key in self._episode_sums.keys():
            extras["Episode_Reward_Avg/" + key] = torch.mean(self._episode_sums[key][env_ids])
            self._episode_sums[key][env_ids] = 0.0

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"].update(extras)

        # Reset environment states
        self._robot.reset(env_ids)
        # Parent method sets done buffers, etc.
        super()._reset_idx(env_ids)

        self._actions[env_ids] = torch.zeros(4, device=self.device)
        self._last_actions[env_ids] = torch.zeros(4, device=self.device)
        
        # Reset force and torque buffers
        self._forces[env_ids] = torch.zeros(1, 3, device=self.device)
        self._torques[env_ids] = torch.zeros(1, 3, device=self.device)
        
        # Reset angular velocity tracking
        self._last_angular_velocity[env_ids] = torch.zeros(3, device=self.device)

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()

        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] = self.env_origins[env_ids].clone()

        default_root_state[:, 0] +=  self.cfg.terrain_length / 2.0
        default_root_state[:, 1] +=  self.cfg.terrain_width / 2.0
        default_root_state[:, 2] +=  (self.cfg.too_low + self.cfg.too_high) / 2.0


        # 处理 Null Trajectory 任务 (Position Control) 的初始状态随机化
        # 论文: "random state (e.g., up to 90 deg tilt)"
        
        # 1. 随机姿态 (Roll/Pitch up to 90 deg, Yaw random)
        # 这里使用简单的欧拉角转四元数采样
        r = (torch.rand(len(env_ids), device=self.device) * 2 - 1) * (math.pi / 2) # +/- 90 deg
        p = (torch.rand(len(env_ids), device=self.device) * 2 - 1) * (math.pi / 2)
        y = (torch.rand(len(env_ids), device=self.device) * 2 - 1) * math.pi      # Full yaw
        
        # 注意：IsaacLab utils 可能需要特定的转换函数，这里示意逻辑
        # 实际上可以使用 isaaclab.utils.math.quat_from_euler_xyz(r, p, y)

        random_quat = quat_from_euler_xyz(r, p, y)
        
        # 2. 随机线速度 (e.g., +/- 2 m/s) 和角速度
        random_lin_vel = (torch.rand(len(env_ids), 3, device=self.device) * 2 - 1) * 2.0
        random_ang_vel = (torch.rand(len(env_ids), 3, device=self.device) * 2 - 1) * 5.0

        # 应用随机化 (仅对 Null Trajectory 任务更重要，Langevin 任务通常从平稳开始或者也随机)
        # 简单起见，对所有重置环境应用：
        default_root_state[:, 3:7] = random_quat
        default_root_state[:, 7:10] = random_lin_vel
        default_root_state[:, 10:13] = random_ang_vel

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Store spawn positions for distance-based termination check
        self._spawn_pos_w[env_ids] = default_root_state[:, :3]

        # Reset trajectory tracking state flags
        self._numerical_is_unstable[env_ids] = False


        # Reset episode outcome tracking for the reset environments
        self._episode_outcomes[env_ids] = 0
        self.first_reach_stamp[env_ids] = torch.inf
        
        # Initialize desired states for trajectory generation
        self.pos_des[env_ids] = default_root_state[:, :3].clone()  # Start from current position
        self.vel_des[env_ids] = default_root_state[:, 7:10].clone()  # Start with actual initial velocity
        
        # Initialize raw (unsmoothed) states
        self.pos_des_raw[env_ids] = default_root_state[:, :3].clone()
        self.vel_des_raw[env_ids] = default_root_state[:, 7:10].clone()  # Start with actual initial velocity


        # 4. 关键：在重置时决定任务类型 (50% 概率)
        random_probs = torch.rand(len(env_ids), device=self.device)
        is_langevin = random_probs > self.cfg.prob_null_trajectory
        self._is_langevin_task[env_ids] = is_langevin

        # 初始化目标状态
        # 获取重置后的初始位置
        curr_pos = self._robot.data.root_pos_w[env_ids]
        curr_vel = self._robot.data.root_lin_vel_w[env_ids]

        # 初始化 Langevin 状态变量
        self.pos_des_raw[env_ids] = curr_pos.clone()
        self.vel_des_raw[env_ids] = curr_vel.clone()
        self.pos_des[env_ids] = curr_pos.clone()
        self.vel_des[env_ids] = curr_vel.clone()

        # 如果是 Null Trajectory 任务，我们需要强制设定一个固定的目标（比如悬停在原点上方）
        # 论文: "position control, going back to the origin from any initial state"
        # 这意味着：初始位置是随机的（上面super()._reset_idx已经做了），但 pos_des 应该固定在原点
        
        null_task_ids = env_ids[~is_langevin]
        if len(null_task_ids) > 0:
            # 目标位置：环境原点 + 高度偏移 (例如 1.0m)
            target_pos = self.env_origins[null_task_ids].clone()
            target_pos[:, 2] += 1.0 # 目标高度
            
            self.pos_des[null_task_ids] = target_pos
            self.vel_des[null_task_ids] = 0.0
            
            # 也要重置 raw 变量，防止下次切换任务时跳变
            self.pos_des_raw[null_task_ids] = target_pos
            self.vel_des_raw[null_task_ids] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Show debug markers if debug_vis is True."""
        # create markers if necessary for the first tome
        
        print(f"debug_vis: {self.cfg.debug_vis}")

        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (self.cfg.marker_size, self.cfg.marker_size, self.cfg.marker_size) # 0.05, 0.05, 0.05
                # marker_cfg.markers["cuboid"].size = (5, 5, 5) # 0.05, 0.05, 0.05
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
                print("Created goal_pos_visualizer")
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)

            if not hasattr(self, "goal_yaw_visualizer"):
                goal_arrow_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                goal_arrow_cfg.markers["arrow"].scale = (self.cfg.marker_size, self.cfg.marker_size, self.cfg.marker_size*4) # 0.05, 0.05, 0.2
                # goal_arrow_cfg.markers["arrow"].scale = (5, 5, 10) #0.05, 0.05, 0.2
                # -- goal yaw
                goal_arrow_cfg.prim_path = "/Visuals/Command/goal_yaw"
                self.goal_yaw_visualizer = VisualizationMarkers(goal_arrow_cfg)
                print("Created goal_yaw_visualizer")
            # set their visibility to true
            self.goal_yaw_visualizer.set_visibility(True)

            if not hasattr(self, "current_yaw_visualizer"):
                current_arrow_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                current_arrow_cfg.markers["arrow"].scale = (self.cfg.marker_size, self.cfg.marker_size, self.cfg.marker_size*4)
                # -- current yaw
                current_arrow_cfg.prim_path = "/Visuals/Command/current_yaw"
                self.current_yaw_visualizer = VisualizationMarkers(current_arrow_cfg)
                print("Created current_yaw_visualizer")
            # set their visibility to true
            self.current_yaw_visualizer.set_visibility(True)

            # Trajectory visualizer for Langevin path
            if not hasattr(self, 'traj_visualizer'):
                print("create trajectory visualizer")
                traj_cfg = VisualizationMarkersCfg(
                    markers={
                        "sphere": sim_utils.SphereCfg(
                            radius=0.05,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(1.0, 0.0, 0.0),  # Red for trajectory
                            ),
                        ),
                    },
                    prim_path = "/Visuals/Trajectory"
                )
                self.traj_visualizer = VisualizationMarkers(traj_cfg)
                print("Created traj_visualizer")
            self.traj_visualizer.set_visibility(True)
                
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            if hasattr(self, "goal_yaw_visualizer"):
                self.goal_yaw_visualizer.set_visibility(False)
            if hasattr(self, "current_yaw_visualizer"):
                self.current_yaw_visualizer.set_visibility(False)
            if hasattr(self, 'traj_visualizer'):
                self.traj_visualizer.set_visibility(False)

        
    def _debug_vis_callback(self, event):
        """Update debug markers with current robot pose."""
        self.current_yaw_visualizer.visualize(self._robot.data.root_pos_w, self._robot.data.root_quat_w)

    class EpisodeOutcome(IntEnum):
        ONGOING = 0
        SUCCESS = 1
        FAILURE = 2
    
    def _update_episode_outcomes_and_metrics(self, env_ids, success_mask, died_mask, timed_out_mask):
        """
        Update episode statistics for trajectory tracking task.
        For trajectory tracking, there is no success criterion - episodes end by timeout or death only.
        """
        # Check for completed episodes (died or timed out)
        completed_mask = torch.logical_or(died_mask, timed_out_mask)
        if not torch.any(completed_mask):
            return 0, 0

        # Extract completion info
        completed_env_ids = env_ids[completed_mask]
        died_env_ids = env_ids[died_mask]
        
        # Process termination reasons for died episodes
        if len(died_env_ids) > 0:
            is_unstable = self._numerical_is_unstable[died_env_ids].cpu().numpy()
            pos_z = self._robot.data.root_pos_w[died_env_ids, 2].cpu().numpy()
            too_low = (pos_z < self.cfg.too_low)
            too_high = (pos_z > self.cfg.too_high)
            
            for i in range(len(died_env_ids)):
                self._termination_reason_history.append({
                    "numerical_is_unstable": bool(is_unstable[i]),
                    "too_low": bool(too_low[i]),
                    "too_high": bool(too_high[i])
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

        # Calculate statistics for trajectory tracking (no success rate needed)
        num_termination_records = len(self._termination_reason_history)
        if num_termination_records > 0:
            # Count death reasons (no collision in open space)
            reason_keys = ["numerical_is_unstable", "too_low", "too_high"]
            reason_counts = {key: 0 for key in reason_keys}
            
            if len(self._termination_reason_history) > 0:
                for reason in self._termination_reason_history:
                    for key in reason_keys:
                        if key in reason and reason[key]:
                            reason_counts[key] += 1
            
            # Calculate percentages
            died_count = sum(1 for r in self._termination_reason_history if r)  # Non-empty dict = died
            timeout_count = num_termination_records - died_count
            
            # Update episode count
            completed_count = len(completed_env_ids)
            self._episodes_completed += completed_count

            # Calculate average velocity
            avg_velocity = np.mean(list(self._vel_abs)) if self._vel_abs else 0.0

            # Prepare metrics (only meaningful ones for trajectory tracking)
            if "log" not in self.extras:
                self.extras["log"] = {}
            
            self.extras["log"].update({
                # Episode termination statistics as percentages
                "Episode_Termination/died": died_count / num_termination_records * 100.0,
                "Episode_Termination/time_out": timeout_count / num_termination_records * 100.0,

                # Death reason statistics as percentages of total episodes
                "Metrics/Died/numerical_is_unstable": reason_counts["numerical_is_unstable"] / num_termination_records * 100.0,
                "Metrics/Died/too_low": reason_counts["too_low"] / num_termination_records * 100.0,
                "Metrics/Died/too_high": reason_counts["too_high"] / num_termination_records * 100.0,

                # Progress tracking
                "Metrics/average_velocity": avg_velocity,
                "Metrics/episodes_completed": self._episodes_completed,
            })

            return completed_count, 0  # Return (completed_count, 0) since there's no success

    def close(self):
        """Clean up resources when environment is closed."""
        super().close()
