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
from isaaclab.sensors import TiledCamera, TiledCameraCfg, ContactSensorCfg, ContactSensor
from isaaclab.sim import SimulationCfg, SimulationContext, RenderCfg
from isaaclab.sim.schemas import activate_contact_sensors
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
from isaaclab.sim.schemas import CollisionPropertiesCfg, define_collision_properties, modify_collision_properties
import open3d as o3d
from scipy.spatial import KDTree
from collections import deque
import numpy as np
import random
import math
import time
import os
import csv

from e2e_drone.utils.px4_controller import PX4QuadrotorController
from e2e_drone.utils.death_replay import DeathReplay
from e2e_drone.utils.wind_gen import WindGustGenerator
from e2e_drone.utils.player import DepthViewerProcess, TerrainVisualizer
from e2e_drone.utils.rrg import TerrainRRGMap
from e2e_drone.utils.raster import TerrainRasterMap, trilinear_interpolate, check_positions_occupancy
from enum import IntEnum
import collections
import itertools
import matplotlib.pyplot as plt

def add_rounding_noise_torch(depth_map: torch.Tensor, levels: int = 128) -> torch.Tensor:
    """
    模拟传感器的量化效应 (PyTorch GPU版本)。
    将连续的深度值舍入到有限的离散级别。
    """
    # 找到深度图的范围，忽略无效的0值
    # 使用一个小的epsilon来防止在所有值都相同时出现问题
    min_depth = torch.min(depth_map[depth_map > 1e-6])
    max_depth = torch.max(depth_map)

    if max_depth <= min_depth:
        return depth_map

    # 计算每个量化步长
    step_size = (max_depth - min_depth) / levels
    if step_size <= 1e-6: # 避免除以一个极小的值
        return depth_map
        
    # 进行量化
    quantized_map = torch.round(depth_map / step_size) * step_size
    return quantized_map

def add_edge_noise_torch(depth_map: torch.Tensor, edge_threshold: float = 0.1, noise_magnitude: float = 0.3) -> torch.Tensor:
    """
    在深度图的边缘添加噪声 (PyTorch GPU版本)。
    """
    # PyTorch的卷积需要 (N, C, H, W) 格式，所以我们添加一个通道维度
    depth_map_nchw = depth_map.unsqueeze(1)

    # 定义Sobel算子核，并确保它在正确的设备上
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=depth_map.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=depth_map.device).view(1, 1, 3, 3)

    # 使用卷积计算梯度
    grad_x = F.conv2d(depth_map_nchw, sobel_x, padding=1)
    grad_y = F.conv2d(depth_map_nchw, sobel_y, padding=1)
    
    # 计算梯度幅值并移除通道维度
    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2).squeeze(1)
    
    # 创建边缘掩码
    edge_mask = gradient_magnitude > edge_threshold
    
    # 在边缘区域添加高斯噪声
    noise = torch.randn_like(depth_map) * noise_magnitude
    noisy_map = depth_map.clone()
    noisy_map[edge_mask] += noise[edge_mask]
    
    # 确保没有负深度值
    noisy_map.clamp_(min=0.0)
    
    return noisy_map

def add_filling_noise_torch(depth_map: torch.Tensor, dropout_rate: float = 0.03, kernel_size: int = 5) -> torch.Tensor:
    """
    模拟因无纹理区域导致的空洞和后续填充伪影 (PyTorch GPU版本)。
    """
    # 1. 随机制造空洞
    dropout_mask = torch.rand_like(depth_map) < dropout_rate
    holed_map = depth_map.clone()
    holed_map[dropout_mask] = 0.0 # 使用0作为无效值的标记

    # 2. 填充空洞
    # 使用平均池化（一种简单的模糊/插值方法）来模拟填充
    # 需要 (N, C, H, W) 格式
    filled_map = F.avg_pool2d(holed_map.unsqueeze(1), kernel_size=kernel_size, stride=1, padding=kernel_size//2).squeeze(1)

    # 3. 只在有空洞的地方应用填充结果，以保留原始的清晰部分
    final_map = depth_map.clone()
    final_map[dropout_mask] = filled_map[dropout_mask]

    return final_map

def add_edge_filling_noise_torch(depth_map: torch.Tensor, edge_threshold: float = 0.1, dropout_rate_on_edges: float = 0.5, kernel_size: int = 5) -> torch.Tensor:
    """
    在深度图的边缘处制造空洞，然后通过插值进行补全，以模拟边缘伪影。
    
    Args:
        depth_map (torch.Tensor): 输入的原始深度图。
        edge_threshold (float): 边缘检测的灵敏度阈值。值越小，越敏感。
        dropout_rate_on_edges (float): 在检测到的边缘上制造空洞的概率。
        kernel_size (int): 用于插值补全的邻域窗口大小。

    Returns:
        torch.Tensor: 处理后带有边缘填充伪影的深度图。
    """
    # --- 步骤 1: 检测边缘 (逻辑来自 add_edge_noise_torch) ---
    # PyTorch的卷积需要 (N, C, H, W) 格式
    depth_map_nchw = depth_map.unsqueeze(1)

    # 定义Sobel算子核
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=depth_map.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=depth_map.device).view(1, 1, 3, 3)

    # 计算梯度
    grad_x = F.conv2d(depth_map_nchw, sobel_x, padding=1)
    grad_y = F.conv2d(depth_map_nchw, sobel_y, padding=1)
    
    # 计算梯度幅值并移除通道维度
    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2).squeeze(1)
    
    # 创建边缘掩码
    edge_mask = gradient_magnitude > edge_threshold
    
    # --- 步骤 2: 在边缘处制造空洞 ---
    # 创建一个随机掩码，用于决定哪些边缘像素将被丢弃
    random_mask = torch.rand_like(depth_map) < dropout_rate_on_edges
    
    # 最终的空洞掩码是“既是边缘” AND “又被随机选中”的像素
    final_dropout_mask = edge_mask & random_mask
    
    holed_map = depth_map.clone()
    holed_map[final_dropout_mask] = 0.0  # 将选中的边缘像素设为0，制造空洞

    # --- 步骤 3: 插值补全 (逻辑来自 add_filling_noise_torch) ---
    # 使用平均池化来模拟插值填充
    filled_map = F.avg_pool2d(holed_map.unsqueeze(1), kernel_size=kernel_size, stride=1, padding=kernel_size//2).squeeze(1)

    # 只在有空洞的地方应用填充结果，以保留原始的清晰部分
    final_map = depth_map.clone()
    final_map[final_dropout_mask] = filled_map[final_dropout_mask]

    return final_map
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
    filter_collisions: bool = True

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(0.1, 0.1),
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
    history_depth = 2
    history_obs = 10
    # Updated observation space: replaced 3D euler angles with 9D rotation matrix + 3D delta pose
    frame_observation_space = 3 + 9 + 2 + 1 + 1 + 4

    # Enable/disable omnidirectional TOF sensors
    enable_omni_tof = False

    # gamma in ppo, only for logging
    gamma = 0.99

    # env
    episode_length_s = 256
    decimation = 1
    action_space = 4 # [roll, pitch, yaw,_rate thrust]
    state_space = 0
    debug_vis = True

    grid_rows = 16 # 6
    grid_cols = 1 # 8
    terrain_width = 32
    terrain_length = 74
    robots_per_env = 10
    success_threshold = 69
    distance_upper_bound = 4.0 # for kdtree query
    collision_threshold = 0.25 # for collision detection
    reaching_goal_threshold = 0.5
    success_radius = 5.0

    # stucking penalty
    stucking_timesteps = 1000
    stucking_displacement_threshold = 0.1
    reward_coef_stucking_penalty = 0.0 # 0.2

    # path penalty
    reward_coef_path_penalty = 0.0 # 1
    path_penalty_threshold = 0.5

    connectivity_vis = False  # Visualize connectivity of the raster map
    enable_dijkstra = True  # Enable Dijkstra's algorithm for pathfinding
    # RRG Map
    rrg_step_size = 0.1
    rrg_collision_search_step = 0.1
    rrg_neighbor_radius = 1.0
    rrg_goal_bias = 0.1

    # Raster Map
    raster_resolution = 0.1
    dilation_kernel_size = 2
    dijkstra_vis = False  # Visualize connectivity of the raster map
    if_raster = True
    reward_coef_dijkstra = 10.0  # 10.0
    start_x = 2.5
    load_raster_from_files = True
    distance_for_invalid = 200.0
    terrain_path = "./USD/16/16"
    grid_path = "./RASTER/16"
    enable_larger_dilation = True

    # terrain and robot
    train = True
    robot_vis = False
    marker_size = 0.05  # Size of the markers in meters
    enable_video_player = False  # Enable video player for depth visualization

    # task reward parameters
    reward_coef_task = 0.0 # 20.0
    task_spatial_scaling = 1.0
    task_reward_time = 2.0
    task_max_time = 60.0
    task_delta_check = 0.3  # 0.1

    reward_coef_cbf = 10.0
    cbf_safe_bound = 0.15
    cbf_eta = 0.1

    max_vel = 3.0

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

    scan: CollisionPropertiesCfg = CollisionPropertiesCfg(
        collision_enabled=True,
        contact_offset=0.01,
        rest_offset=0.01
    )

    # Controller parameters
    controller_Kang = [15.0, 13.0, 20]  # Roll and pitch angle controller gains   #15 15 20
    controller_Kdang = [0.8, 0.75, 1.2]                                            #0.8 0.8 1.2
    controller_Kang_vel = [5.0, 5.0, 5.0]  # Roll, pitch, and yaw angular velocity controller gains
    
    # att_p_gain = [14.0, 14.0, 20.0]  # Roll, pitch, and yaw angle P gains
    # rate_p_gain = [0.13, 0.13, 0.20]  # Roll, pitch, and yaw angular velocity P gains
    # rate_i_gain = [0.25, 0.25, 0.30]  # Roll, pitch, and yaw angular velocity I gains
    # rate_d_gain = [0.003, 0.003, 0.0]  # Roll, pitch, and yaw angular velocity D gains
    # att_yaw_weight = 0.4  # Yaw weight in attitude controller

    # att_p_gain = [6.5, 6.5, 2.8]  # Roll, pitch, and yaw angle P gains
    # rate_p_gain = [25, 20, 20]  # Roll, pitch, and yaw angular velocity P gains
    # rate_i_gain = [0.5, 0.2, 0.1]  # Roll, pitch, and yaw angular velocity I gains
    # rate_d_gain = [0.01, 0.01, 0.0]  # Roll, pitch, and yaw angular velocity D gains
    # att_rate_limit = [3.84, 3.84, 3.49]
    # att_yaw_weight = 0.4  # Yaw weight in attitude controller


    # att_p_gain = [6.5, 6.5, 2.8]  # Roll, pitch, and yaw angle P gains
    # rate_p_gain = [0.3, 0.3, 0.2]  # Roll, pitch, and yaw angular velocity P gains
    # rate_i_gain = [0.26, 0.26, 0.1]  # Roll, pitch, and yaw angular velocity I gains
    # rate_d_gain = [0.0, 0.0, 0.0]  # Roll, pitch, and yaw angular velocity D gains
    
    # att_p_gain = [4,4,2]
    # rate_p_gain = [1.0,1.0,1.5]
    # # rate_i_gain =  [0.03,0.03,0.03]
    # rate_i_gain =  [0.03,0.03,0.03]
    # rate_d_gain =  [0.0,0.0,0.0]
    # rate_k_gain = [0.001, 0.001, 0.001]
    # rate_int_limit = [0.6, 0.6, 0.6]
    # att_rate_limit = [3.84, 3.84, 3.49]
    # att_yaw_weight = 0.4  # Yaw weight in attitude controller

    att_p_gain = [40,40,20]
    rate_p_gain = [1.0,1.0,1.5]
    # rate_i_gain =  [0.03,0.03,0.03]
    rate_i_gain =  [0.03,0.03,0.03]
    rate_d_gain =  [0.0,0.0,0.0]
    rate_k_gain = [0.0001, 0.0001, 0.0001]
    rate_int_limit = [0.6, 0.6, 0.6]
    att_rate_limit = [3.84, 3.84, 3.49]
    att_yaw_weight = 0.4  # Yaw weight in attitude controller

    # Aerodynamic drag (paper model) ----------------------------
    enable_aero_drag = False
    drag_coeffs = (0.003, 0.003, 0.003)   # dx, dy, dz，机体系“转子/气动阻力系数”，可按机架调
    drag_rand_scale = 0.5              # 域随机化幅度：±50%（论文设置）
    drag_v_clip = 8.0                 # 可选：速度范数上限，避免数值爆

    # scene
    scene: InteractiveSceneCfg = QuadcopterSceneCfg()

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # contact sensor
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/body", history_length=1, update_period=0.01,
        track_air_time=False,
        debug_vis=False,
    )

    # depth camera
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/body/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.09, 0.0, -0.01), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["depth"],
        # 10->90 20->45
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=20.0, focus_distance=400.0, horizontal_aperture=27.84, clipping_range=(0.01, 7.0)
        ),
        width=100,
        height=60,
        update_period=1.0 / 30.0,
        depth_clipping_behavior="max",
    )

    # Omnidirectional TOF cameras
    if enable_omni_tof:
        left_camera: TiledCameraCfg = TiledCameraCfg(
            prim_path="/World/envs/env_.*/Robot/body/Camera_left",
            offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.05, 0.0), rot=(0.7071, 0.0, 0.0, 0.7071), convention="world"),
            data_types=["depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=20.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 4.0)
            ),
            width=8,
            height=8,
            update_period=1.0 / 15.0,
            depth_clipping_behavior="max",
        )

        right_camera: TiledCameraCfg = TiledCameraCfg(
            prim_path="/World/envs/env_.*/Robot/body/Camera_right",
            offset=TiledCameraCfg.OffsetCfg(pos=(0.0, -0.05, 0.0), rot=(0.7071, 0.0, 0.0, -0.7071), convention="world"),
            data_types=["depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=20.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 4.0)
            ),
            width=8,
            height=8,
            update_period=1.0 / 15.0,
            depth_clipping_behavior="max",
        )

        back_camera: TiledCameraCfg = TiledCameraCfg(
            prim_path="/World/envs/env_.*/Robot/body/Camera_back",
            offset=TiledCameraCfg.OffsetCfg(pos=(-0.05, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
            data_types=["depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=20.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 4.0)
            ),
            width=8,
            height=8,
            update_period=1.0 / 15.0,
            depth_clipping_behavior="max",
        )

        down_camera: TiledCameraCfg = TiledCameraCfg(
            prim_path="/World/envs/env_.*/Robot/body/Camera_down",
            offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, -0.05), rot=(0.707, 0.0, 0.707, 100), convention="world"),
            data_types=["depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=20.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 4.0)
            ),
            width=8,
            height=8,
            update_period=1.0 / 15.0,
            depth_clipping_behavior="max",
        )

    depth_size = tiled_camera.width * tiled_camera.height
    # Calculate total observation space with history
    observation_space = frame_observation_space + history_obs * frame_observation_space + history_depth * depth_size

    # thresholds
    too_low = 0.1
    too_high = 1.9
    desired_low = 0.8  
    desired_high = 1.2
    
    # Reward coefficients
    reward_coef_distance_reward: float = 0.0  # 7.0           # 10, 80
    reward_coef_direction_penalty: float = 0.0 # 0.01     # [-2, 0] -> [-0.02, 0]
    reward_coef_action_magnitude_penalty: float = 0.04   #0.8 # 0.2   # 600, 18.0
    reward_coef_action_change_penalty: float = 0.05   #1.5  #2.0 #0.5      # 400, 12.0
    reward_coef_vel_direction_alignment_penalty: float = 0.0 # 0.1  # 500, 50.0
    reward_coef_vel_speed_excess_penalty: float = 1.0 # 0.3    # 500, 25.0
    reward_coef_vel_speed_match_reward: float = 0.0 # 0.02      # [0, 2] -> [0, 0.04]
    reward_coef_z_position_penalty: float = 0.3          # 1000, 50.0
    reward_coef_obstacle_collision_penalty: float = 100.0  # 1 , 80
    reward_coef_esdf_reward: float = 1.0 #0.02                  # 48, 0.48
    reward_coef_succeed_reward: float = 100.0 # 0.1               # 100, 10.0
    reward_coef_max_ang_vel_penalty: float = 0.0          # [-30, 0] -> [-0.03, 0]
    reward_coef_max_angle_penalty: float = 0.0            # [-30, 0] -> [-0.03, 0]
    reward_coef_alive_reward: float = 0.0 # 0.1 #0.1                 # 100, 10.0    must
    reward_coef_z_vel_penalty: float = 0.0                # [-1, 0] -> [-0, 0]

    reward_coef_angular_velocity_change_penalty: float = 0.0  # 0.0001

    # # Reward coefficients
    # reward_coef_distance_reward: float = 10.0           # 10, 80
    # reward_coef_direction_penalty: float = 0.0 # 0.01     # [-2, 0] -> [-0.02, 0]
    # reward_coef_action_magnitude_penalty: float = 0.15    # 600, 18.0
    # reward_coef_action_change_penalty: float = 0.3       # 400, 12.0
    # reward_coef_vel_direction_alignment_penalty: float = 0.0 # 0.1  # 500, 50.0
    # reward_coef_vel_speed_excess_penalty: float = 0.05    # 500, 25.0
    # reward_coef_vel_speed_match_reward: float = 0.0 # 0.02      # [0, 2] -> [0, 0.04]
    # reward_coef_z_position_penalty: float = 0.05          # 1000, 50.0
    # reward_coef_obstacle_collision_penalty: float = 80.0  # 1 , 80
    # reward_coef_esdf_reward: float = 0.01                  # 48, 0.48
    # reward_coef_succeed_reward: float = 50.0 # 0.1               # 100, 10.0
    # reward_coef_max_ang_vel_penalty: float = 0.0          # [-30, 0] -> [-0.03, 0]
    # reward_coef_max_angle_penalty: float = 0.0            # [-30, 0] -> [-0.03, 0]
    # reward_coef_alive_reward: float = 0.1                 # 100, 10.0
    # reward_coef_z_vel_penalty: float = 0.0                # [-1, 0] -> [-0, 0]
    # Position control
    reward_coef_lin_vel_reward_scale: float = 0 #-0.05
    reward_coef_ang_vel_reward_scale: float = 0 #-0.01
    reward_coef_distance_to_goal_reward_scale: float = 0 # 15.0

    # DeathReplay configuration
    enable_death_replay = False
    death_replay_dir = os.path.abspath(os.path.join("logs", "death_replay"))
    death_replay_history_capacity = 5000
    death_replay_visualization_num = 10  # Number of environments to track and visualize
    death_replay_trajectory_spacing = 5.0
    death_replay_tof_frame_interval = 15

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

        # 域随机化：每个环境独立在 ±drag_rand_scale 内扰动
        if self.cfg.enable_aero_drag and self.cfg.train:
            rand = (2.0 * torch.rand_like(self._drag_D) - 1.0) * self.cfg.drag_rand_scale
            self._drag_D = self._drag_D * (1.0 + rand)

        arm_l_tensor = torch.full((self.num_envs,), 0.10, device=self.device)
        inertia_tensor = torch.tensor([2.16e-5, 2.16e-5, 4.33e-5], device=self.device).repeat(self.num_envs, 1)
        # dynamics = QuadrotorDynamics(num_envs=self.num_envs,
        #                              mass=mass_tensor,
        #                              inertia=inertia_tensor,
        #                              arm_l=arm_l_tensor,
        #                              device=self.device
        # )
        # self._controller = Quadrotor(num_envs=self.num_envs,
        #     dynamics=dynamics,
        #     Krp_ang=self.cfg.controller_Kang,
        #     Kdrp_ang=self.cfg.controller_Kdang,
        #     Kinv_ang_vel_tau=self.cfg.controller_Kang_vel,
        #     device=self.device,
        #     debug_viz=False
        # )
        self._controller = PX4QuadrotorController(
            num_envs=self.num_envs,
            device=self.device,
            att_p_gain=torch.tensor(self.cfg.att_p_gain, device=self.device, dtype=torch.float32),
            att_yaw_weight=torch.tensor(self.cfg.att_yaw_weight, device=self.device, dtype=torch.float32),
            rate_p_gain=torch.tensor(self.cfg.rate_p_gain, device=self.device, dtype=torch.float32),
            rate_i_gain=torch.tensor(self.cfg.rate_i_gain, device=self.device, dtype=torch.float32),
            rate_d_gain=torch.tensor(self.cfg.rate_d_gain, device=self.device, dtype=torch.float32),
            rate_k_gain=torch.tensor(self.cfg.rate_k_gain, device=self.device, dtype=torch.float32),
            rate_int_limit=torch.tensor(self.cfg.rate_int_limit, device=self.device, dtype=torch.float32),
            att_rate_limit=torch.tensor(self.cfg.att_rate_limit, device=self.device, dtype=torch.float32),
        )

        # Quadcopter references
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._torques = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Goal
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_yaw_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._desired_vel = torch.zeros(self.num_envs, 1, device=self.device)
        self._threshold_vel_b = torch.full((self.num_envs, 1), 0.5, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "distance_reward",
                "direction_penalty",
                "action_magnitude_penalty",
                "action_change_penalty",
                "vel_direction_alignment_penalty",
                "vel_speed_excess_penalty",
                "vel_speed_match_reward",
                "z_position_penalty",
                "obstacle_collision_penalty",
                "esdf_reward",
                "succeed_reward",
                "max_ang_vel_penalty",
                "max_angle_penalty",
                "alive_reward",
                "z_vel_penalty",
                "lin_vel_reward",
                "ang_vel_reward",
                "distance_to_goal_reward",
                # "stucking_penalty",
                # "path_penalty",
                # "task_reward",
                "dijkstra_reward",
                "angular_velocity_change_penalty",
                "cbf_reward",
            ]
        }
        
        # Environment origins
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        self.grid_idx = None
        # Robot references
        self._body_id = self._robot.find_bodies("body")[0]
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies("body")

        # Observations
        self._obs_history = torch.zeros(self.num_envs, self.cfg.history_obs, self.cfg.frame_observation_space, device=self.device)
        self._depth_history = torch.zeros(self.num_envs, self.cfg.history_depth, self.cfg.depth_size, device=self.device)

        self._action_history_length = 8
        self._action_history = torch.zeros(self.num_envs, self._action_history_length, self.cfg.action_space, device=self.device)
        self._valid_mask = torch.zeros(                  # 标记哪些历史槽位已经被真实动作填过
        self.num_envs, self._action_history_length, dtype=torch.bool, device=self.device
        )



        self._last_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._last_actions = torch.zeros(self.num_envs, 4, device=self.device) # [roll_rate, pitch_rate, yaw_rate, thrust]
        self._is_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_closest_dist = torch.zeros(self.num_envs, device=self.device)
        self._numerical_is_unstable = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._is_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.occ_kdtree = None
        self._displacement_history = torch.zeros(self.num_envs, self.cfg.stucking_timesteps, 3, device=self.device)
        self._is_stucking = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._history_distance = torch.full((self.num_envs,), float('inf'), device=self.device)
        self._dilated_positions = torch.zeros(1, 3, device=self.device)
        self._traj = torch.zeros(1, 3, device=self.device)
        self._maps = []

        self._last_angular_velocity= torch.zeros(self.num_envs, 3, device=self.device)

        # Noise config
        self._noise_5_cfg = GaussianNoiseCfg(
            mean=1.0,
            std=0.05,
            operation='scale'
        )
        self._noise_10_cfg = GaussianNoiseCfg(
            mean=1.0,
            std=0.10,
            operation='scale'
        )
        self._noise_01_cfg = UniformNoiseCfg(
            n_min=-0.1,     # 均匀分布的下限
            n_max=0.1,    # 均匀分布的上限
            operation='scale'
        )
        self._noise_20_cfg = GaussianNoiseCfg(
            mean=1.0,
            std=0.20,
            operation='scale'
        )

        # Add tracking for episode outcomes and success rate
        self._success_rate_window = 100
        self._episode_outcomes = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # 0=ongoing, 1=success, 2=died
        self._episodes_completed = 0
        self._episodes_succeeded = 0
        self._success_rate = 0.0
        self._episode_outcome_history = collections.deque(maxlen=self._success_rate_window)
        self._termination_reason_history = collections.deque(maxlen=self._success_rate_window)
        self._final_distances = collections.deque(maxlen=self._success_rate_window)
        self._vel_abs = collections.deque(maxlen=self._success_rate_window)

        self.set_debug_vis(self.cfg.debug_vis)

        # Initialize DeathReplay
        if self.cfg.enable_death_replay:
            self._death_replay = DeathReplay(
                num_envs=self.num_envs,
                tof_width=self.cfg.tiled_camera.width,
                tof_height=self.cfg.tiled_camera.height,
                history_capacity=self.cfg.death_replay_history_capacity,
                save_dir=self.cfg.death_replay_dir,
                drone_size=0.15,
                camera_fov=45,
                trajectory_spacing=self.cfg.death_replay_trajectory_spacing,
                tof_frame_interval=self.cfg.death_replay_tof_frame_interval,
                visualization_num=self.cfg.death_replay_visualization_num, 
                device=self.device
            )
        else:
            self._death_replay = None

        if self.cfg.enable_video_player:
            # img_shape = (self.cfg.tiled_camera.height, self.cfg.tiled_camera.width)
            # --- 修改部分：调整可视化窗口以支持并排显示 ---
            # 获取原始图像尺寸
            h, w = self.cfg.tiled_camera.height, self.cfg.tiled_camera.width
            # 新的图像形状是原始宽度的两倍，用于并排显示 (左:原始, 右:噪声)
            img_shape = (h, w * 2) 
            # 使用新的形状初始化共享内存和可视化进程
            self.shared_imgs = np.zeros((self.num_envs, *img_shape), dtype=np.float32)
            self.viewer = DepthViewerProcess(img_shape, self.num_envs)
            self.viewer.start()

            # --- 新增: 预计算中心加权的权重矩阵 ---
        print("[INFO] Pre-calculating center-weighted mask for open space reward.")
        h = self.cfg.tiled_camera.height
        w = self.cfg.tiled_camera.width
        center_h, center_w = h / 2.0, w / 2.0
        
        # 创建网格坐标
        y, x = torch.meshgrid(
            torch.arange(h, device=self.device, dtype=torch.float32), 
            torch.arange(w, device=self.device, dtype=torch.float32), 
            indexing='ij'
        )
        
        # 计算高斯权重。sigma 控制了中心区域的关注范围。
        # 一个经验法则是将其设置为图像短边长度的1/4到1/3。
        sigma = min(h, w) / 3.0
        
        # 计算高斯权重矩阵
        self.center_weights = torch.exp(-((x - center_w)**2 + (y - center_h)**2) / (2 * sigma**2))
        
        # (可选但推荐) 对权重进行归一化，使其总和为1，这样奖励的范围就在[0, 1]之间
        self.center_weights = self.center_weights / torch.sum(self.center_weights)
        # --- 结束新增 ---

        self._calc_env_origins()

    def _print_depth_info(self, env_id=0, show_image=True):
        """Print real-time depth information from the center pixel of the forward camera with elapsed time."""
        depth_image = self._tiled_camera.data.output["depth"]  # Shape: (num_envs, height, width)
        h, w = self.cfg.tiled_camera.height, self.cfg.tiled_camera.width
        center_h = h // 2
        center_w = w // 2
        
        pos_w = self._robot.data.root_pos_w  # Shape: (num_envs, 3)
        
        # Calculate elapsed time since training start
        elapsed_time = time.time() - self.start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        milliseconds = int((elapsed_time % 1) * 1000)
        time_str = f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
        
        center_depth = depth_image[env_id, center_h, center_w].item()
        position = pos_w[env_id].cpu().numpy()
        print(
            f"[{time_str}] Env {env_id} - "
            f"Position: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}] | "
            f"Center Depth: {center_depth:.3f}m"
        )

        if show_image:
            plt.figure(figsize=(6, 4))
            plt.imshow(depth_image[env_id].cpu().numpy(), cmap='plasma')
            plt.scatter([center_w], [center_h], c='red', s=30, label='Center')
            plt.colorbar(label='Depth (m)')
            plt.title(f"Env {env_id} Depth Image")
            plt.legend()
            plt.show()
            


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

    def CHECK_state(self):
        # Limit
        max_angular_velocity = 3.14 * 2.0 * 20.0 # rad/s

        # State
        ang_vel_b = self._robot.data.root_ang_vel_b
        rot_w = torch.stack(euler_xyz_from_quat(self._robot.data.root_quat_w), dim=1) # (num_envs, 3) roll, pitch, yaw
        rot_w = torch.stack([normallize_angle(rot_w[:, 0]), normallize_angle(rot_w[:, 1]), normallize_angle(rot_w[:, 2])], dim=1)

        state_is_unstable = torch.any(torch.abs(ang_vel_b) > max_angular_velocity, dim=1)

        self._numerical_is_unstable = torch.logical_or(self._numerical_is_unstable, state_is_unstable)

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

        # Always create the main camera
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)

        # Only create omnidirectional TOF sensors if enabled
        if self.cfg.enable_omni_tof:
            self._left_camera = TiledCamera(self.cfg.left_camera)
            self._right_camera = TiledCamera(self.cfg.right_camera)
            self._back_camera = TiledCamera(self.cfg.back_camera)
            self._down_camera = TiledCamera(self.cfg.down_camera)

        # Initialize the map generator and other components
        if self.cfg.train:
            from e2e_drone.utils.train_terrain import MapGenerator
            self._map_generator = MapGenerator(sim=self.sim, device=self.device)
        else:
            from e2e_drone.utils.eval_terrain import MapGenerator
            self._map_generator = MapGenerator(sim=self.sim, device=self.device)

        # Clone the scene
        self.scene.clone_environments(copy_from_source=False)

        # Add the robot and main camera to the scene
        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["tiled_camera"] = self._tiled_camera

        # Only add omnidirectional TOF sensors to the scene if enabled
        if self.cfg.enable_omni_tof:
            self.scene.sensors["left_camera"] = self._left_camera
            self.scene.sensors["right_camera"] = self._right_camera
            self.scene.sensors["back_camera"] = self._back_camera
            self.scene.sensors["down_camera"] = self._down_camera

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Activate contact sensors
        activate_contact_sensors("/World")
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # Counters
        self._episode_counter = 0
        self._map_generation_timer = 0

    def _regenerate_terrain(self):
        self.sim.pause()
        print("Regenerating terrain and occupancy map.")
        prims_utils.delete_prim("/World/ground")
        self._terrain = self.cfg.scene.terrain.class_type(self.cfg.scene.terrain)
        # Create new obstacles
        env_data = self._map_generator.create_environment(
            self.cfg.scene,
            self._terrain,
            num_obstacles=int(150),
            num_floaters=0,
            min_distance=0.3,
            obstacle_size_range=(0.4, 0.8),
            obstacle_height_range=(3.0, 4.0),
            floaters_size_range=(0.2, 0.6),
            floaters_height_range=(0.2, 4.0),
            terrain_length=self.cfg.terrain_length,
            terrain_width=self.cfg.terrain_width,
            grid_rows=self.cfg.grid_rows,
            grid_cols=self.cfg.grid_cols,
            plane_size = (self.cfg.terrain_length * (self.cfg.grid_cols + 2), self.cfg.terrain_width * (self.cfg.grid_rows + 2)),
            plane_translation = (self.cfg.terrain_length * self.cfg.grid_cols / 2, self.cfg.terrain_width * self.cfg.grid_rows / 2, 0.0),
            terrain_path = self.cfg.terrain_path,
        )

        self.occ_kdtree = env_data["kdtree"]
        # self.occ_kdtree = KDTree(np.zeros((1, 3)))

        if self.cfg.enable_dijkstra:
            start_time = time.time()
            all_occupied_points = []
            all_dilated_points = []
            files = self.cfg.load_raster_from_files
            for i in range(self.cfg.grid_rows):
                for j in range(self.cfg.grid_cols):
                    map_min = [j*self.cfg.terrain_length, i*self.cfg.terrain_width, self.cfg.too_low]
                    map_max = [(j+1)*self.cfg.terrain_length, (i+1)*self.cfg.terrain_width, self.cfg.too_high]
                    start = [j * self.cfg.terrain_length + self.cfg.start_x, i * self.cfg.terrain_width + self.cfg.terrain_width/2, (self.cfg.desired_low+self.cfg.desired_high)/2]
                    goal = [j * self.cfg.terrain_length + (self.cfg.terrain_length + self.cfg.success_threshold)/2, i * self.cfg.terrain_width + self.cfg.terrain_width/2, (self.cfg.desired_low+self.cfg.desired_high)/2]
                    if self.cfg.if_raster:
                        map = TerrainRasterMap(map_min, map_max, self.cfg.raster_resolution, self.cfg.raster_resolution/2.0)
                        if not files:
                            # Build occupancy grid
                            map.build_occupancy_grid_from_points(self.occ_kdtree.data)
                            map.dilate_existing_obstacles(self.cfg.dilation_kernel_size)
                            # Compute distances to goal
                            if self.cfg.enable_larger_dilation:
                                map.compute_distance_to_goal_larger_dilation(goal_world_pos=goal)
                                map.compute_esdf()
                            else:
                                map.compute_distance_to_goal(goal_world_pos=goal)
                                map.compute_esdf()
                            if os.path.isdir(self.cfg.grid_path):
                                map.save_to_file(self.cfg.grid_path + f"/raster_map_{i}_{j}.npz")
                            else:
                                os.makedirs(self.cfg.grid_path)
                                map.save_to_file(self.cfg.grid_path + f"/raster_map_{i}_{j}.npz")
                            # map.save_to_file(f"RASTER/raster_map_one")
                            # map.save_to_file(self.cfg.grid_path)
                            map.set_custom_distance_for_invalid(self.cfg.distance_for_invalid)
                            occupied_points = map.get_occupied_positions()
                            dilated_position = map.get_dilated_positions()
                            all_occupied_points.extend(occupied_points)
                            all_dilated_points.extend(dilated_position)   
                        else:
                            # map.load_from_file(f"RASTER/grid_size_5/raster_map_{i}_{j}.npz")
                            # map.load_from_file(self.cfg.grid_path+".npz")
                            map.load_from_file(self.cfg.grid_path + f"/raster_map_{i}_{j}.npz")
                            map.set_custom_distance_for_invalid(self.cfg.distance_for_invalid)
                            occupied_points = map.get_occupied_positions()
                            dilated_position = map.get_dilated_positions()
                            all_occupied_points.extend(occupied_points)
                            all_dilated_points.extend(dilated_position)
                            # map.load_from_file(f"RASTER/raster_map_test.npz")
                        # self._maps.append(map.distance_grid.copy())
                    else:
                        map = TerrainRRGMap(
                            self.occ_kdtree,
                            map_min=map_min,
                            map_max=map_max,
                            step_size=self.cfg.rrg_step_size,
                            neighbor_radius=self.cfg.rrg_neighbor_radius,
                            collision_radius=self.cfg.collision_threshold,
                            step=self.cfg.rrg_collision_search_step,
                            goal_bias=self.cfg.rrg_goal_bias,
                        )
                        map.plan(start, goal, max_iter=2000)
                    self._maps.append(map)
            self.occ_kdtree = KDTree(all_occupied_points)
            self._dilated_positions = torch.tensor(all_dilated_points, device=self.device, dtype=torch.float32)
            self._traj = torch.tensor(self._maps[0].extract_path_to_goal([2.5, 32.0, 2.0]), device=self.device, dtype=torch.float32)
            # self.occ_kdtree = KDTree(np.zeros((1, 3)))  # Placeholder, will be updated later
            print(f"Raster Map Generation Time: {time.time() - start_time:.2f} seconds")
        self.sim.play()

        # Update DeathReplay with global map and KD-tree
        if self._death_replay is not None:
            self._death_replay.set_global_map(env_data["points"], self.occ_kdtree)

    def _pre_physics_step(self, actions: torch.Tensor):
        # Clip and store actions
        
        # actions_1 = actions.clamp(-1.0, 1.0).clone()

        # --- ADDED: Print actions BEFORE processing ---
        # print(f"Env[0] - Action BEFORE processing: {actions_1[0]}")
        # # self._noise_5_cfg.func(actions, self._noise_5_cfg)
        # # self._noise_01_cfg.func(actions, self._noise_01_cfg)
        # # Custom noise function 1: Multiply by Gaussian noise (mean=1, std=0.03)
        # gaussian_noise = 1.0 + 0.03 * torch.randn_like(actions_1)
        # actions_1 = actions_1 * gaussian_noise

        # # Custom noise function 2: Multiply by uniform noise (range [0.97, 1.03])
        # uniform_noise = -0.03 + (0.03 - (-0.03)) * torch.rand_like(actions_1)
        # actions_1 = actions_1 + uniform_noise
        # print(f"Env[0] - Action AFTER processing: {actions_1[0]}")
        # --- ADDED: Print actions AFTER processing ---
        
        # actions = actions.tanh()
        actions = actions.clamp(-1.0, 1.0)
        actions[3] = (actions[3] + 1.0) * 0.5
        self._actions = actions.clone()
        
        # --- 新增：维护 action 历史 ---
        if not hasattr(self, "_action_history"):
            self._action_history_length = 8
            self._action_history = torch.zeros(
                self.num_envs, self._action_history_length, self.cfg.action_space, device=self.device
            )

        # 把历史整体向前滚动一格，再写入最新 action
        self._action_history = torch.roll(self._action_history, shifts=-1, dims=1)
        self._valid_mask = torch.roll(self._valid_mask, shifts=-1, dims=1)
        self._action_history[:, -1, :] = actions.clone()
        self._valid_mask[:, -1] = True
        # --- 新增：随机时延窗口扰动 ---
        # 在 [-8, -8] 范围里随机窗口起点
        window_start = random.randint(-8, -8)
        window_end = window_start + 7  # 窗口长度 8
        sel_actions = self._action_history[:, window_start:window_end, :]  # shape: (num_envs, 8, action_dim)
        sel_mask     = self._valid_mask[:, window_start:window_end]
        counts      = sel_mask.sum(dim=1).unsqueeze(-1)                              # [N, 1]
        sum_actions = (sel_actions * sel_mask.unsqueeze(-1)).sum(dim=1)              # [N, A]
        mean_action = torch.where(
            counts > 0,
            sum_actions / counts.clamp_min(1),                                       # 有效均值
            actions                                                            # 没有有效项 -> 用当前动作
        )
        self._actions = mean_action.clone()
        # mean_action = mean_action.clamp(-1.0, 1.0)
        # mean_action[:, 3] = (mean_action[:, 3] + 1.0) * 0.5 # [-1, 1] -> [0, 1]
        
        # self._actions = actions.clamp(-1.0, 1.0)
        # self._actions = actions.clone()
        # self._actions[:, 3] = (self._actions[:, 3] + 1.0) * 0.5 # [-1, 1] -> [0, 1]

        # --- ADDED: Print actions AFTER processing ---
        # print(f"Env[0] - Action AFTER processing: {self._actions[0]}")

        # # --- 新增：维护 action 历史 ---
        # if not hasattr(self, "_action_history"):
        #     self._action_history_length = 14
        #     self._action_history = torch.zeros(
        #         self.num_envs, self._action_history_length, self.cfg.action_space, device=self.device
        #     )
        #     self._history_ptr = 0  # 用于记录当前写入位置

        # # 把历史整体向前滚动一格，再写入最新 action
        # self._action_history = torch.roll(self._action_history, shifts=-1, dims=1)
        # self._valid_mask = torch.roll(self._valid_mask, shifts=-1, dims=1)
        # self._action_history[:, -1, :] = self._actions.clone()
        # self._valid_mask[:, -1] = True
        # # --- 新增：随机时延窗口扰动 ---
        # # 在 [-13, -11] 范围里随机窗口起点
        # window_start = random.randint(-13, -11)
        # window_end = window_start + 10  # 窗口长度 10
        # sel_actions = self._action_history[:, window_start:window_end, :]  # shape: (num_envs, 10, action_dim)
        # sel_mask     = self._valid_mask[:, window_start:window_end]
        # counts      = sel_mask.sum(dim=1).unsqueeze(-1)                              # [N, 1]
        # sum_actions = (sel_actions * sel_mask.unsqueeze(-1)).sum(dim=1)              # [N, A]
        # mean_action = torch.where(
        #     counts > 0,
        #     sum_actions / counts.clamp_min(1),                                       # 有效均值
        #     self._actions                                                            # 没有有效项 -> 用当前动作
        # )

        # Extract the current state of the robot
        # Root state [pos, quat, lin_vel, ang_vel] in simulation world frame. Shape is (num_instances, 13).
        cur_state = self._robot.data.root_state_w.clone()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        force, torque, _, px4info = self._controller.compute_control(cur_state, self._actions, self.step_dt)
        self.px4info = px4info
        # force, torque = self._controller.compute_control(cur_state, mean_action, self.step_dt)
        end.record()
        torch.cuda.synchronize()

        # Reset forces and torques to zero
        self._forces.zero_()
        self._torques.zero_()
        self._forces[:, 0, 2] = force[:, 2]
        self._torques[:, 0, :] = torque

        if self._wind_gen is not None:
            # Apply wind disturbances
            wind_acc = self._wind_gen.step()                       # (num_envs,3) m/s²
            wind_force_world = wind_acc * self._robot_mass.unsqueeze(1)  # (num_envs,3) N
            quat_w = self._robot.data.root_quat_w  # quaternion representing rotation from body to world
            rot_matrices_w2b = matrix_from_quat(quat_w).transpose(1, 2)  # shape: (num_envs, 3, 3)
            wind_force_body = torch.bmm(rot_matrices_w2b, wind_force_world.unsqueeze(2)).squeeze(2)
            wind_weight = torch.clamp(((torch.mean(self._episode_sums["alive_reward"]) / (1.0 / (1.0 - self.cfg.gamma) * self.cfg.reward_coef_alive_reward) - 0.8) / 0.2), 0.0, 1.0)
            self._forces[:, 0, :] += wind_force_body * wind_weight
            # print(f"original force: {self._forces[0, 0, :]}")
            # print(f"Wind force: {wind_force_body[0]}")

        # --- Aerodynamic drag (paper model) ---
        if self.cfg.enable_aero_drag:
            # 机体系线速度，形状 (num_envs, 3)
            v_b = self._robot.data.root_lin_vel_b

            # 可选：防数值爆，限速范数（只影响阻力，不改状态）
            if self.cfg.drag_v_clip is not None and self.cfg.drag_v_clip > 0:
                v_norm = torch.norm(v_b, dim=1, keepdim=True).clamp(max=self.cfg.drag_v_clip)
                v_dir = torch.where(v_norm > 0, v_b / (v_norm + 1e-6), torch.zeros_like(v_b))
                v_b_eff = v_dir * v_norm  # 裁剪后的 v_b
            else:
                v_b_eff = v_b
                v_norm = torch.norm(v_b_eff, dim=1, keepdim=True)

            # 机体系空气阻力：F_drag_b = - m * D * ||v_b|| * v_b
            # self._robot_mass: (num_envs,), 扩展到 (num_envs,1)

            px,py,pz = self.cfg.drag_coeffs
            env_ids = torch.arange(self.num_envs,device=self.device)
            base = torch.tensor([px,py,pz],device=self.device).repeat(len(env_ids),1)
            scale = self.cfg.drag_rand_scale
            factors = 1.0 + (2.0 * torch.rand_like(base) - 1.0) * scale
            self._drag_D = torch.clamp(base * factors,min =0.0)


            m = self._robot_mass.unsqueeze(1)
            F_drag_b = - m * self._drag_D * v_norm * v_b_eff  # 逐轴二次阻力

            a_drag_b = F_drag_b / m 
            for env_id in range(min(1, v_b_eff.shape[0])):
                vx, vy, vz = v_b_eff[env_id].tolist()
                ax, ay, az = a_drag_b[env_id].tolist()
                dx, dy, dz = self._drag_D[env_id].tolist()
                print(f"[Env {env_id}] v_b = ({vx:.3f}, {vy:.3f}, {vz:.3f}) m/s | " f"a_drag = ({ax:.5f}, {ay:.5f}, {az:.5f}) m/s^2 | " f"D = ({dx:.5f}, {dy:.5f}, {dz:.5f})")

            # 写入到你的外力缓存（机体系，和控制力同一坐标系/通道）
            self._forces[:, 0, :] += F_drag_b

        # print(f"Controller compute time: {start.elapsed_time(end)} ms")
        # print(f"Env[0] - Action: {self._actions[0]}")
        # print(f"Env[0] - Force: {self._forces[0]}, Torque: {self._torques[0]}")

    def _apply_action(self):
        """Apply thrust/moment to the quadcopter."""
        self._robot.set_external_force_and_torque(self._forces, self._torques, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        """
        Return the observations for the agent in a dictionary.
        """
        # # Get the depth image from the main camera
        # depth_image = self._tiled_camera.data.output["depth"]

        # # Save original shape and flatten to 2D for processing
        # original_shape = depth_image.shape
        # batch_size = original_shape[0]
        # flat_depth = depth_image.reshape(batch_size, -1)

        # invalid_rate = random.random() * 0.15
        # invalid_masks = torch.rand_like(flat_depth) < invalid_rate
        

        # flat_depth = torch.where(invalid_masks, 
        #                      torch.tensor(4.0, device=self.device, dtype=flat_depth.dtype),
        #                      flat_depth)

        # depth_image = flat_depth  # Already in (num_envs, -1) shape
        # self._depth_history = torch.cat([self._depth_history[:, 1:], depth_image.unsqueeze(dim=1)], dim=1)

        # --- 修改开始 ---
        # 1. 获取原始的、完美的深度图
        # 它的原始形状很可能是 (num_envs, height, width, 1)
        perfect_depth_map_nhwc = self._tiled_camera.data.output["depth"]

        # 2. 【关键修复】调整维度顺序，从 NHWC 变为 NCHW
        # (N, H, W, C) -> (N, C, H, W)
        # (300, 60, 80, 1) -> (300, 1, 60, 80)
        perfect_depth_map_nchw = perfect_depth_map_nhwc.permute(0, 3, 1, 2)
        
        # 为了后续处理和可视化，我们还需要一个没有通道维度的版本 (N, H, W)
        perfect_depth_map_nhw = perfect_depth_map_nhwc.squeeze(-1)

        # 3. 将【正确形状的】深度图送入噪声函数
        # 注意这里我们传入squeeze后的版本，因为噪声函数内部会自己处理通道
        noisy_map_temp = add_edge_noise_torch(perfect_depth_map_nhw, edge_threshold=1.0, noise_magnitude=0.05)
        noisy_map_temp1 = add_filling_noise_torch(noisy_map_temp, dropout_rate=0.03, kernel_size=5)
        final_noisy_map = add_rounding_noise_torch(noisy_map_temp1, levels=128)

        # 4. 使用【带有噪声的深度图】作为网络的输入
        depth_image_for_network = final_noisy_map 
        # --- 修改结束 ---

        # 展平为2D进行处理
        batch_size = depth_image_for_network.shape[0]
        flat_depth = depth_image_for_network.reshape(batch_size, -1)

        # 更新历史记录
        self._depth_history = torch.cat([self._depth_history[:, 1:], flat_depth.unsqueeze(dim=1)], dim=1)

        # Initialize omnidirectional TOF values with zeros
        left_depth = torch.zeros(self.num_envs, device=self.device)
        right_depth = torch.zeros(self.num_envs, device=self.device)
        back_depth = torch.zeros(self.num_envs, device=self.device)
        down_depth = torch.zeros(self.num_envs, device=self.device)
        
        # Only process omnidirectional TOF data if enabled
        if self.cfg.enable_omni_tof:
            # Get depth data from omnidirectional cameras
            left_depth_full = self._left_camera.data.output["depth"]
            right_depth_full = self._right_camera.data.output["depth"]
            back_depth_full = self._back_camera.data.output["depth"]
            down_depth_full = self._down_camera.data.output["depth"]

            # Extract center pixels for single-point measurements
            h, w = self.cfg.left_camera.height, self.cfg.left_camera.width
            center_h = [h//2 - 1, h//2]
            center_w = [w//2 - 1, w//2]

            # Extract minimum distance from center pixels for each direction
            left_depth = torch.min(left_depth_full[:, center_h[0]:center_h[1]+1, center_w[0]:center_w[1]+1].reshape(self.num_envs, -1), dim=1)[0]
            right_depth = torch.min(right_depth_full[:, center_h[0]:center_h[1]+1, center_w[0]:center_w[1]+1].reshape(self.num_envs, -1), dim=1)[0]
            back_depth = torch.min(back_depth_full[:, center_h[0]:center_h[1]+1, center_w[0]:center_w[1]+1].reshape(self.num_envs, -1), dim=1)[0]
            down_depth = torch.min(down_depth_full[:, center_h[0]:center_h[1]+1, center_w[0]:center_w[1]+1].reshape(self.num_envs, -1), dim=1)[0]

        # Get the current position, orientation, and velocity of the robot
        pos_w = self._robot.data.root_state_w[:, :3]
        quat_w = self._robot.data.root_quat_w
        
        # Directly calculate delta position in body frame by multiplying body velocity by dt
        # This is more efficient than transforming to world frame and back
        vel_b = self._robot.data.root_lin_vel_b
        dt = self.cfg.sim.dt  # Time step
        delta_pos_b = vel_b * dt  # Simple and efficient computation in body frame
        
        # Get the rotation matrix for observation
        rot_matrix_b2w = matrix_from_quat(quat_w)  # Shape: (num_envs, 3, 3)

        # Record frame for death replay
        if self._death_replay is not None:
            # For visualization, still use Euler angles
            rot_w = torch.stack(euler_xyz_from_quat(quat_w), dim=1)
            rot_w = torch.stack([normallize_angle(rot_w[:, 0]), normallize_angle(rot_w[:, 1]), normallize_angle(rot_w[:, 2])], dim=1)
            self._death_replay.record_frame(
                pos_w=pos_w,
                rot_w=rot_w,
                tof_data=depth_image.view(self.num_envs, self.cfg.tiled_camera.height, self.cfg.tiled_camera.width)
            )

        # # Get direction to goal in world frame
        # direction_to_goal_w = self._desired_pos_w - pos_w
        # direction_to_goal_xy_w = direction_to_goal_w[:, :2]
        # direction_to_goal_xy_w = direction_to_goal_xy_w / (direction_to_goal_xy_w.norm(dim=1, keepdim=True) + 1e-6)


        #Get direction to goal in body frame
        direction_to_goal_w = self._desired_pos_w - pos_w
        direction_to_goal_xy_w = direction_to_goal_w[:, :2]
        # 计算 R_b2w 的转置（即 R_w2b）
        rot_matrix_w2b = rot_matrix_b2w.transpose(1, 2)  # [num_envs, 3, 3]

        # 将 direction_to_goal_xy_w 扩展为 3D 向量（z 分量为 0）
        direction_to_goal_w_3d = torch.cat([
            direction_to_goal_xy_w,  # [num_envs, 2]
            torch.zeros(self.num_envs, 1, device=direction_to_goal_xy_w.device)  # [num_envs, 1]
        ], dim=1)  # [num_envs, 3]

        # 转换到机体坐标系：v_b = R_w2b @ v_w
        direction_to_goal_b_3d = torch.bmm(rot_matrix_w2b, direction_to_goal_w_3d.unsqueeze(-1)).squeeze(-1)  # [num_envs, 3]

        # 提取 xy 分量
        direction_to_goal_xy_b = direction_to_goal_b_3d[:, :2]  # [num_envs, 2]
        # direction_to_goal_xy_b = direction_to_goal_xy_b / (direction_to_goal_xy_b.norm(dim=1, keepdim=True) + 1e-6)

        
        direction_to_goal_xy_w = direction_to_goal_xy_w / (direction_to_goal_xy_w.norm(dim=1, keepdim=True) + 1e-6)

        # Flatten the rotation matrix to 9 elements for the observation
        rotation_matrix_flat = rot_matrix_b2w.reshape(self.num_envs, 9)
        
        # Map the observations to appropriate ranges for neural network input
        self._obs_history = torch.cat(
            [self._obs_history[:, 1:],
             torch.cat([
                # self._noise_5_cfg.func(vel_b, self._noise_5_cfg),
                vel_b,
                rotation_matrix_flat,
                direction_to_goal_xy_b,
                # delta_pos_b,  # [delta x, delta y, delta z] in body frame # 3
                (self._desired_pos_w[:, 2]).unsqueeze(dim=-1),  # [desired z]
                # (self._desired_vel),  # [desired vel]
                pos_w[:, 2].unsqueeze(dim=-1),
                self._last_actions,
            ],dim=-1).unsqueeze(dim=1)
            ],dim=1)
        
        if self.cfg.enable_video_player:
            # self.shared_imgs[:] = self._depth_history[:, -1].reshape(self.num_envs, self.cfg.tiled_camera.height, self.cfg.tiled_camera.width).cpu().numpy()
             # --- 修改开始 ---
            # 5. 创建用于并排可视化的组合图像
            h, w = self.cfg.tiled_camera.height, self.cfg.tiled_camera.width
            
            combined_vis_imgs = np.zeros((self.num_envs, h, w * 2), dtype=np.float32)
            
            # 使用我们之前 squeeze 好的 nhw 格式的原始图
            combined_vis_imgs[:, :, :w] = perfect_depth_map_nhw.cpu().numpy()
            combined_vis_imgs[:, :, w:] = final_noisy_map.cpu().numpy()
            
            self.shared_imgs[:] = combined_vis_imgs
            # --- 修改结束 ---
            self.viewer.update_images(self.shared_imgs)
            # self.viewer.update_images(self.shared_imgs)
        # obs = torch.cat([self._obs_history[:, -1].clone(), self._obs_history.view(self.num_envs,-1), self._depth_history.view(self.num_envs,-1)], dim=-1)
        # obs = torch.cat([self._obs_history[:, -2].view(self.num_envs, -1), self._depth_history[:, -2].view(self.num_envs, -1), self._obs_history[:, -1].view(self.num_envs, -1), self._depth_history[:, -1].view(self.num_envs, -1)], dim=-1)
        obs = torch.cat([self._obs_history[:, -1].view(self.num_envs, -1), self._depth_history[:, -1].view(self.num_envs, -1)], dim=-1)
        critic_obs = obs.clone()

        obs = self.CHECK_NAN(obs, "Observation")
        critic_obs = self.CHECK_NAN(critic_obs, "Privileged Observation")

        return {"policy": obs, "critic": critic_obs, "rnd_state": obs}

    def _get_rewards(self) -> torch.Tensor:
        """
        Calculate the reward for each environment.
        """
        # Current position, orientation, and velocity of the robot
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        pos_w = self._robot.data.root_state_w[:, :3]
        rot_w = torch.stack(euler_xyz_from_quat(self._robot.data.root_state_w[:, 3:7]), dim=1)
        rot_w = torch.stack([normallize_angle(rot_w[:, 0]), normallize_angle(rot_w[:, 1]), normallize_angle(rot_w[:, 2])], dim=1)
        vel_b = self._robot.data.root_lin_vel_b

        # distance to goal center [-0.01, 0.01] (1m/s / 100steps)
        distance_to_gap = (pos_w - self._desired_pos_w).norm(dim=1)
        last_distance_to_gap = (self._last_pos_w - self._desired_pos_w).norm(dim=1)
        delta_distance = last_distance_to_gap - distance_to_gap
        delta_distance = torch.clamp(delta_distance, min=-0.01 * (self.cfg.max_vel + 2.0), max=0.01 * (self.cfg.max_vel + 2.0))
        # reward_far = delta_distance * 10.0
        # distance_far = 5.0
        # distance_near = 1.0
        # reward_near = torch.where(
        #     distance_to_gap <= distance_near,
        #     0.05 * (distance_near - distance_to_gap) / distance_near,
        #     torch.zeros_like(distance_to_gap)
        # ) + 0.1
        # t = torch.clamp((distance_to_gap - distance_near) / (distance_far - distance_near), 0.0, 1.0)
        # w_smooth = 3*t*t - 2*t*t*t
        # distance_reward = w_smooth * reward_far + (1 - w_smooth) * reward_near
        distance_reward = delta_distance * 10.0

        # direction penalty [-2, 0]
        # (We define a "forward" direction as +X in world space for illustration.)
        epsilon = 1e-6
        pos_diff = pos_w - self._last_pos_w
        norm_pos_diff = pos_diff.norm(dim=-1, keepdim=True)
        pos_diff = pos_diff / (norm_pos_diff + epsilon)

        direction_to_goal = self._desired_pos_w - pos_w
        norm_direction_to_goal = direction_to_goal.norm(dim=-1, keepdim=True)
        direction_to_goal = direction_to_goal / (norm_direction_to_goal + epsilon)
        pos_diff_xy = pos_diff[..., :2]
        direction_to_goal_xy = direction_to_goal[..., :2]
        direction_penalty = (direction_to_goal_xy * pos_diff_xy).sum(dim=-1) - 1.0


        # action magnitude penalty [-6, 0]
        # shape is (num_envs, 4) -> thrust + rates
        act_abs = torch.abs(self._actions)
        # action_magnitude = torch.square(act_abs[:, 0]) + torch.square(act_abs[:, 1] )+ torch.square(2.0 * act_abs[:, 2]) + torch.square(2.0 * act_abs[:, 3])
        action_magnitude = torch.square(act_abs[:, 0]) + torch.square(act_abs[:, 1]) 
        action_magnitude_penalty = -action_magnitude

        # action change penalty (difference relative to last actions) [-4, 0]
        diff_actions = self._actions - self._last_actions

        weights = torch.tensor([1.0, 1.0, 1.0, 1.0], device=self.device)
        diff_actions_weighted = diff_actions * weights
        action_change_penalty = - (diff_actions_weighted ** 2).sum(dim=1)

        # action_change_penalty = -diff_actions.norm(dim=1)

        dt = 1/100.0
        # 计算角速度：基于前三维（角度）
        angular_velocity = (self._actions[:, 0:3] - self._last_actions[:, 0:3]) / dt
        # 保存当前角速度以供下一次计算（需要存储 last_angular_velocity）
        if not hasattr(self, '_last_angular_velocity'):
            self._last_angular_velocity = torch.zeros_like(angular_velocity, device=self.device)
        # 计算角速度变化（角加速度）
        diff_angular_velocity = angular_velocity - self._last_angular_velocity
        # 应用权重（仅对前三维）
        angular_weights = torch.tensor([1.0, 1.5, 2.0], device=self.device)  # 复用前三维权重
        diff_angular_velocity_weighted = diff_angular_velocity * angular_weights
        # 计算角速度变化惩罚
        angular_velocity_change_penalty = - (diff_angular_velocity_weighted ** 2).sum(dim=1)
        # 更新上一时间步的角速度
        self._last_angular_velocity = angular_velocity.clone()


        # Velocity-related rewards and penalties - now separated into individual components
        speed = vel_b.norm(dim=1)
        distance_to_goal = (pos_w - self._desired_pos_w).norm(dim=1)

        # Adjust desired speed based on distance to goal
        # Linearly decrease speed when within 2 meters of goal, reaching 0 at 0.25m
        original_desired_speed = self._desired_vel.squeeze(-1)
        speed_adjust_start = 5.0  # Start slowing down
        speed_adjust_end = 1.0   # Speed should be zero

        # Calculate the linear interpolation factor (0 at 5 meters, 1 at 1 meters)
        slowdown_factor = torch.clamp((speed_adjust_start - distance_to_goal) / (speed_adjust_start - speed_adjust_end), 0.0, 1.0)
        # Adjust desired speed: original speed when far, 0 when at goal
        self._desired_speed = original_desired_speed * (1.0 - slowdown_factor)

        # Velocity-orientation alignment penalty [-2, 0]
        # Convert to penalty (0 for perfect alignment, -2 for opposite direction)
        vel_direction_alignment_penalty = -torch.clamp(torch.exp(torch.abs(vel_b[:, 1]) * 5) - 1, max=5.0)

        # Gaussian bump speed reward [-1, 1], with hard penalty for speed > v_max
        # v_peak = 0.3
        # sigma = 0.1
        # v_min = 0.0
        # v_max = 0.5
        # # speed shape: (num_envs,) or (num_envs, 1)
        # # 先计算高斯bump
        # r = 2.0 * torch.exp(-((speed - v_peak) ** 2) / (2 * sigma ** 2)) - 1.0
        # # 区间外直接-1
        # vel_speed_excess_penalty = torch.where((speed < v_min) | (speed > v_max), torch.full_like(speed, -1.0), r)

        v_max = self.cfg.max_vel
        vel_norm = torch.norm(vel_b, dim=1) 
        vel_speed_excess_penalty = torch.where(
            torch.abs(vel_norm) > v_max,
            -torch.clamp(torch.exp(vel_norm - v_max) - 1.0, max=5.0),
            torch.zeros_like(vel_norm)
        )

        # Velocity matching reward [0, 2]
        # Reward for having speed close to desired speed (now using adjusted desired_speed)
        vel_speed_match_reward = torch.exp(-5.0 * torch.abs(speed - self._desired_speed)) * 2.0

        # z position penalty [-10, 0]
        z_pos = pos_w[:, 2]
        floor_dist = z_pos - (self._desired_pos_w[:, 2]) + 0.5   #0.15
        floor_penalty = torch.where(
            floor_dist < 0.0, 
            -torch.clamp(torch.exp(4.0 * (-floor_dist)) - 1.0, max=5.0),   #2.0 * (-floor_dist)
            torch.zeros_like(z_pos),
        )
        ceiling_dist = (self._desired_pos_w[:, 2]) - z_pos + 0.2
        ceiling_penalty = torch.where(
            ceiling_dist < 0.0,
            -torch.clamp(torch.exp(4.0 * (-ceiling_dist)) - 1.0, max=5.0),
            torch.zeros_like(z_pos),
        )
        z_position_penalty = floor_penalty + ceiling_penalty

        # Perform KD-tree query once to get nearest obstacle distances
        nearest_obstacle_distances = None
        if self.occ_kdtree is not None:
            d, _ = self.occ_kdtree.query(pos_w.cpu(), workers=-1, distance_upper_bound=4.0)
            nearest_obstacle_distances = torch.tensor(d, device=self.device)

        # collision penalty. [-1, 0]
        # Check for physical collisions using contact sensor
        
        # Get termination signal
        die =  self._numerical_is_unstable | self._is_contact | (self._robot.data.root_pos_w[:, 2] < self.cfg.too_low) | (self._robot.data.root_pos_w[:, 2] > self.cfg.too_high)
       
        obstacle_collision_penalty = torch.where(
            die,
            torch.ones_like(vel_b[:, 0]),
            torch.zeros_like(vel_b[:, 0]),
        )
        obstacle_collision_penalty = -obstacle_collision_penalty

        # ESDF-based reward
        esdf_reward = torch.zeros_like(vel_b[:, 0])
        if nearest_obstacle_distances is not None:
            safe_threshold = self.cfg.collision_threshold + 0.15
            esdf_reward = torch.where(
                nearest_obstacle_distances < safe_threshold,
                -(torch.exp(5.0 * (safe_threshold - nearest_obstacle_distances)) - 1.0),
                torch.zeros_like(nearest_obstacle_distances),
            )

        succeed_reward = self._is_success

        # Angular velocity penalty
        max_angular_velocity = 3.14 / 4.0 # rad/s
        ang_vel_b = self._robot.data.root_ang_vel_b.clone() # (num_envs, 3)
        max_ang_vel_penalty = torch.where(
            torch.abs(ang_vel_b) > max_angular_velocity,
            -torch.clamp(torch.exp(torch.abs(torch.abs(ang_vel_b) - max_angular_velocity)) - 1.0, max=10.0),
            torch.zeros_like(ang_vel_b),
        ) # (num_envs, 3) -> (num_envs,)
        max_ang_vel_penalty = torch.sum(max_ang_vel_penalty, dim=1)

        # Angle penalty [-20, 0]
        max_angle = 3.14 / 4.0 # rad
        max_angle_penalty = torch.where(
            torch.abs(rot_w[:, :2]) > max_angle,
            -torch.clamp(torch.exp(torch.abs(torch.abs(rot_w[:, :2]) - max_angle)) - 1.0, max=10.0),
            torch.zeros_like(rot_w[:, :2]),
        )
        max_angle_penalty = torch.sum(max_angle_penalty, dim=1)

        # z velocity penalty [-1, 0]
        z_vel_diff = torch.abs(vel_b[:, 2])
        z_vel_penalty = -torch.clamp(z_vel_diff, max=1.0)

        # Alive reward (before collision) [0, 1]
        alive_reward = torch.logical_not(die).float()

        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)

        # self._displacement_history = torch.cat(
        #     [self._displacement_history[:, :-1],
        #      (self._robot.data.root_state_w[:, :3] - self._last_pos_w).unsqueeze(1)],
        #     dim=1
        # )
        # mean_displacement = torch.norm(torch.sum(self._displacement_history, dim=1), dim=-1)
        # self._is_stucking = mean_displacement < self.cfg.stucking_displacement_threshold
        # stucking_penalty = self._is_stucking.float() * -1.0  # Penalize if stuck

        # depth = self._depth_history[:, -1].reshape(self.num_envs, self.cfg.tiled_camera.height, self.cfg.tiled_camera.width).unsqueeze(dim=1)  # (num_envs, 1, height, width)
        # downsampled_depth = F.interpolate(depth, size=(self.cfg.tiled_camera.height // 10, self.cfg.tiled_camera.width // 10), mode="bilinear", align_corners=False)
        # downsampled_depth = downsampled_depth.squeeze(dim=1)
        # exp_mean = torch.mean(downsampled_depth, dim=(1, 2))
        # path_penalty_mask = exp_mean > self.cfg.path_penalty_threshold
        # exp_mean[path_penalty_mask] = 0.0
        # path_penalty = -1.0 * exp_mean

        # task_time_mask = self.episode_length_buf > (self.cfg.task_max_time - self.cfg.task_reward_time) / (self.cfg.sim.dt * self.cfg.decimation)
        # task_random_mask = torch.rand(self.num_envs, device=self.device) < self.cfg.task_delta_check
        # task_reward = torch.logical_or(task_time_mask, task_random_mask).float()
        # direction = (self._desired_pos_w - pos_w) / self.cfg.task_spatial_scaling
        # direction = direction.norm(dim=1, keepdim=False)
        # task_reward = torch.div(task_reward, 1 + direction)

        dijkstra_reward = torch.zeros(self.num_envs, device=self.device)
        if self.cfg.enable_dijkstra:
            for i in range(self.cfg.grid_rows):
                for j in range(self.cfg.grid_cols):
                    idx = i * self.cfg.grid_cols + j
                    env_ids = self.grid_idx[idx]
                    map_min = [j*self.cfg.terrain_length, i*self.cfg.terrain_width, self.cfg.too_low]
                    if len(env_ids) == 0:
                        continue
                    # value = torch.from_numpy(trilinear_interpolate(self._maps[idx], self._robot.data.root_state_w[env_ids, :3].cpu().numpy(), map_min, self.cfg.raster_resolution)).to(dtype=torch.float32, device=self.device)
                    # last_value = torch.from_numpy(trilinear_interpolate(self._maps[idx], self._last_pos_w[env_ids, :3].cpu().numpy(), map_min, self.cfg.raster_resolution)).to(dtype=torch.float32, device=self.device)
                    # Sparse
                    # is_free, free_distances = self._maps[idx].check_positions_occupancy(self._robot.data.root_state_w[env_ids, :3].cpu().numpy())
                    # last_is_free, last_free_distances = self._maps[idx].check_positions_occupancy(self._last_pos_w[env_ids, :3].cpu().numpy())
                    # is_free = torch.tensor(is_free, dtype=torch.bool, device=self.device)
                    # free_distances = torch.tensor(free_distances, dtype=torch.float32, device=self.device)
                    # last_is_free = torch.tensor(last_is_free, dtype=torch.bool, device=self.device)
                    # last_free_distances = torch.tensor(last_free_distances, dtype=torch.float32, device=self.device)
                    # env_ids = torch.tensor(env_ids, device=self.device)
                    # both_free = is_free & last_is_free
                    # diff = (last_free_distances - free_distances)[both_free]
                    # is_free, free_distances = self._maps[idx].check_positions_occupancy(self._robot.data.root_state_w[env_ids, :3].cpu().numpy())
                    # last_is_free, last_free_distances = self._maps[idx].check_positions_occupancy(self._last_pos_w[env_ids, :3].cpu().numpy())
                    # is_free = torch.tensor(is_free, dtype=torch.bool, device=self.device)
                    # free_distances = torch.tensor(free_distances, dtype=torch.float32, device=self.device)
                    # last_is_free = torch.tensor(last_is_free, dtype=torch.bool, device=self.device)
                    # last_free_distances = torch.tensor(last_free_distances, dtype=torch.float32, device=self.device)
                    # env_ids = torch.tensor(env_ids, device=self.device)
                    # both_free = is_free & last_is_free
                    # diff = (last_free_distances - free_distances)[both_free]
                    # diff = torch.clamp(diff, min=-0.1, max=0.1)
                    # diff = 10 * diff
                    # env_ids = torch.tensor(env_ids, device=self.device)
                    # dijkstra_reward[env_ids[both_free]] = diff
                    # diff = 10 * diff
                    # env_ids = torch.tensor(env_ids, device=self.device)
                    # dijkstra_reward[env_ids[both_free]] = diff

                    # is_free, free_distances = self._maps[idx].check_positions_occupancy(self._robot.data.root_state_w[env_ids, :3].cpu().numpy())
                    # is_free = torch.tensor(is_free, dtype=torch.bool, device=self.device)
                    # next_points = self._maps[idx].get_next_step_towards_goal(self._robot.data.root_state_w[env_ids, :3][is_free].cpu().numpy())
                    # next_points = torch.tensor(next_points, dtype=torch.float32, device=self.device)
                    # directional_vec = next_points - self._robot.data.root_state_w[env_ids, :3][is_free]
                    # directional_vec = directional_vec / (directional_vec.norm(dim=1, keepdim=True) + 1e-6)  # Normalize to unit vector
                    # dijkstra_reward[env_ids[is_free]] = torch.bmm(
                    #     directional_vec.unsqueeze(1),
                    #     self._robot.data.root_state_w[env_ids[is_free], 7:10].unsqueeze(-1)
                    # ).squeeze()


                    # diff = torch.where(diff < 0, torch.zeros_like(diff), diff)
                    # dijkstra_reward[env_ids] = torch.where(diff < 0, torch.zeros_like(diff), torch.tanh(diff))
                    # dijkstra_reward[env_ids] = torch.where(diff < 0, torch.zeros_like(diff), diff)
                    
                    # 修改这两行：
                    value = torch.from_numpy(self._maps[idx].trilinear_interpolate(self._robot.data.root_state_w[env_ids, :3].cpu().numpy())).to(dtype=torch.float32, device=self.device)
                    last_value = torch.from_numpy(self._maps[idx].trilinear_interpolate(self._last_pos_w[env_ids, :3].cpu().numpy())).to(dtype=torch.float32, device=self.device)
                    # diff = (last_value - value)[both_free]
                    diff = last_value - value
                    diff = torch.clamp(diff, min=-0.01 * (self.cfg.max_vel + 2.0), max=0.01 * (self.cfg.max_vel + 2.0))
                    diff = 10 * diff
                    # dijkstra_reward[env_ids[both_free]] = diff
                    dijkstra_reward[env_ids] = diff

                    # is_free, free_distances = self._maps[idx].check_positions_occupancy(self._robot.data.root_state_w[env_ids, :3].cpu().numpy())
                    # is_free = torch.tensor(is_free, dtype=torch.bool, device=self.device)
                    # free_distances = torch.tensor(free_distances, dtype=torch.float32, device=self.device)
                    # cur_dijkstra_reward = dijkstra_reward[env_ids]
                    # cur_dijkstra_reward[~is_free] = 0.0
                    # # 用掩码进行逐元素更新
                    # mask = is_free & (self._history_distance[env_ids] > free_distances)
                    # cur_dijkstra_reward[mask] = 1.0
                    # indices = torch.tensor(env_ids, device=self.device)[mask]
                    # self._history_distance[indices] = free_distances[mask]
                    # dijkstra_reward[env_ids] = cur_dijkstra_reward

        cbf_reward = torch.zeros(self.num_envs, device=self.device)
        gamma = self.cfg.cbf_eta / self.step_dt
        safe_bound = self.cfg.cbf_safe_bound
        for i in range(self.cfg.grid_rows):
            for j in range(self.cfg.grid_cols):
                idx = i * self.cfg.grid_cols + j
                env_ids = self.grid_idx[idx]
                env_ids = torch.tensor(env_ids, device=self.device)
                map_min = [j*self.cfg.terrain_length, i*self.cfg.terrain_width, self.cfg.too_low]
                if len(env_ids) == 0:
                    continue
                esd, esd_gradient = self._maps[idx].trilinear_interpolate_esdf(self._robot.data.root_state_w[env_ids, :3].cpu().numpy())
                esd = torch.from_numpy(esd).to(dtype=torch.float32, device=self.device)
                esd_gradient = torch.from_numpy(esd_gradient).to(dtype=torch.float32, device=self.device)
                # esd_gradient = torch.from_numpy(self._maps[idx].trilinear_interpolate_esdf_gradient(self._robot.data.root_state_w[env_ids, :3].cpu().numpy())).to(dtype=torch.float32, device=self.device)
                velocity = self._robot.data.root_lin_vel_w[env_ids]  # (num_envs, 3)
        
                # 计算 h_dot = gradient · velocity (点积)
                h_dot = torch.sum(esd_gradient * velocity, dim=1)  # (num_envs,)
                # 计算CBF约束
                h = esd - safe_bound  # (num_envs,)
                cbf_reward[env_ids] = h_dot + gamma * h

        cbf_reward = torch.where(
            cbf_reward < 0.0,
            cbf_reward,
            torch.zeros_like(cbf_reward) 
        )

        cbf_reward.clamp_(min=-2.0, max=0.0)

        # Gather components - Updated to include separate velocity components
        reward_components = torch.stack(
            [
                distance_reward * self.cfg.reward_coef_distance_reward,  # 0
                direction_penalty * self.cfg.reward_coef_direction_penalty,  # 1
                action_magnitude_penalty * self.cfg.reward_coef_action_magnitude_penalty,  # 2
                action_change_penalty * self.cfg.reward_coef_action_change_penalty,  # 3
                vel_direction_alignment_penalty * self.cfg.reward_coef_vel_direction_alignment_penalty,  # 4
                vel_speed_excess_penalty * self.cfg.reward_coef_vel_speed_excess_penalty,  # 5
                vel_speed_match_reward * self.cfg.reward_coef_vel_speed_match_reward,  # 6
                z_position_penalty * self.cfg.reward_coef_z_position_penalty,  # 7
                obstacle_collision_penalty * self.cfg.reward_coef_obstacle_collision_penalty,  # 8
                esdf_reward * self.cfg.reward_coef_esdf_reward,  # 9
                succeed_reward * self.cfg.reward_coef_succeed_reward,  # 10
                max_ang_vel_penalty * self.cfg.reward_coef_max_ang_vel_penalty,  # 11
                max_angle_penalty * self.cfg.reward_coef_max_angle_penalty,  # 12
                alive_reward * self.cfg.reward_coef_alive_reward,  # 13
                z_vel_penalty * self.cfg.reward_coef_z_vel_penalty,  # 14
                lin_vel * self.cfg.reward_coef_lin_vel_reward_scale,  # 15
                ang_vel * self.cfg.reward_coef_ang_vel_reward_scale,  # 16
                distance_to_goal_mapped * self.cfg.reward_coef_distance_to_goal_reward_scale,  # 17
                # stucking_penalty * self.cfg.reward_coef_stucking_penalty,  # 18
                # path_penalty * self.cfg.reward_coef_path_penalty,  # 19
                # task_reward * self.cfg.reward_coef_task,  # 20
                dijkstra_reward * self.cfg.reward_coef_dijkstra,  # 21
                angular_velocity_change_penalty * self.cfg.reward_coef_angular_velocity_change_penalty,  # 22
                cbf_reward * self.cfg.reward_coef_cbf,  # 23
            ],
            dim=-1
        )

        total_reward = torch.sum(reward_components, dim=1)


        debug = torch.where(self._episode_sums["obstacle_collision_penalty"] < 0)
        if debug[0].numel() > 0:
            print("debug: ",debug)
            raise ValueError("debug")
 



        # For logging:
        for (key, idx) in zip(self._episode_sums.keys(), range(reward_components.shape[1])):
            # self._episode_sums[key] = self._episode_sums[key] * self.cfg.gamma + reward_components[:, idx]
            self._episode_sums[key] = self._episode_sums[key] + reward_components[:, idx]

        # Update "last" values
        self._last_pos_w = pos_w.clone()
        self._last_actions = self._actions.clone()
        end.record()
        torch.cuda.synchronize()
        # print(f"Reward compute time: {start.elapsed_time(end)} ms")

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Define terminations and timeouts."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        #
        self.CHECK_state()
        # Goal reached
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        selected_forces = torch.index_select(
            net_contact_forces,
            dim=2,
            index=torch.tensor(self._undesired_contact_body_ids, device=self.device)
        )
        max_contact = torch.max(torch.norm(selected_forces, dim=-1), dim=1)[0]
        physical_contact = torch.sum(max_contact > 0.0, dim=1) > 0
        is_contact = physical_contact
        self._is_contact = torch.logical_or(self._is_contact, is_contact.bool())

        if self.cfg.enable_dijkstra:
            for i in range(self.cfg.grid_rows):
                for j in range(self.cfg.grid_cols):
                    idx = i * self.cfg.grid_cols + j
                    env_ids = self.grid_idx[idx]
                    is_free, _ = self._maps[idx].check_positions_occupancy(self._robot.data.root_state_w[env_ids, :3].cpu().numpy())
                    is_free = torch.tensor(is_free, dtype=torch.bool, device=self.device)
                    self._is_contact[env_ids] = torch.logical_or(self._is_contact[env_ids], ~is_free)

        # distances, _ = self.occ_kdtree.query(self._robot.data.root_pos_w[:, :3].cpu(), workers=-1, distance_upper_bound=self.cfg.distance_upper_bound)
        # distances = torch.tensor(distances, device=self.device)
        # is_contact = distances < self.cfg.collision_threshold
        # self._is_contact = torch.logical_or(self._is_contact, is_contact.bool())
        # is_free = torch.tensor(is_free, dtype=torch.bool, device=self.device)
        # self._is_contact = torch.logical_or(self._is_contact, ~is_free)
        
        succeed_mask = self._robot.data.root_state_w[:, :3][:, 0] > self.env_origins[:, 0] + self.cfg.success_threshold
        # rot_matrix_b2w = matrix_from_quat(self._robot.data.root_quat_w)  # Shape: (num_envs, 3, 3)
        # direction_to_goal_w = self._desired_pos_w - self._robot.data.root_state_w[:, :3]
        # direction_to_goal_xy_w = direction_to_goal_w[:, :2]
        # # 计算 R_b2w 的转置（即 R_w2b）
        # rot_matrix_w2b = rot_matrix_b2w.transpose(1, 2)  # [num_envs, 3, 3]

        # # 将 direction_to_goal_xy_w 扩展为 3D 向量（z 分量为 0）
        # direction_to_goal_w_3d = torch.cat([
        #     direction_to_goal_xy_w,  # [num_envs, 2]
        #     torch.zeros(self.num_envs, 1, device=direction_to_goal_xy_w.device)  # [num_envs, 1]
        # ], dim=1)  # [num_envs, 3]

        # # 转换到机体坐标系：v_b = R_w2b @ v_w
        # direction_to_goal_b_3d = torch.bmm(rot_matrix_w2b, direction_to_goal_w_3d.unsqueeze(-1)).squeeze(-1)  # [num_envs, 3]

        # # 提取 xy 分量
        # direction_to_goal_xy_b = direction_to_goal_b_3d[:, :2]  # [num_envs, 2]
        # succeed_mask = torch.norm(direction_to_goal_xy_b, dim=1) < self.cfg.success_radius
        self._is_success = torch.logical_or(self._is_success, succeed_mask.bool())
        conditions = [
            self._numerical_is_unstable,  # Numerical instability
            self._is_contact,  # Collision
            self._robot.data.root_pos_w[:, 2] < self.cfg.too_low,  # Z position too low
            self._robot.data.root_pos_w[:, 2] > self.cfg.too_high,
            self._is_success,  # Goal reached
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
            success_episodes = torch.sum(self._is_success == True).item()
            timeout_episodes = torch.sum(time_out == True).item()
            outcomes = self._is_success[completed_mask]
            outcomes = (outcomes.cpu() == True).tolist()
            outcome =  [self.EpisodeOutcome.SUCCESS if success else self.EpisodeOutcome.FAILURE for success in outcomes]
            self.extras["log"].update({
                    "Metrics/success_episodes_per_step": success_episodes,
                    "Metrics/completed_episodes_per_step": completed_episodes,
                    "Metrics/timeout_episodes_per_step": timeout_episodes,
                    "Metrics/outcome_episodes_per_step": outcome,
            })
        else:
            self.extras["log"].update({
                "Metrics/success_episodes_per_step": 0,
                "Metrics/completed_episodes_per_step": 0,
                "Metrics/timeout_episodes_per_step": 0,
                "Metrics/outcome_episodes_per_step": [],
            })

        # # ----- 新增：逐步打印“新发生”的死亡原因与实时高度 -----
        # # died 当前包含了“死亡 + 成功”，我们只想要真正的“死亡”
        # death_mask = died & (~self._is_success)

        # # 只打印本 step 内“新发生”的死亡（避免一个 episode 多次刷屏）
        # newly_dead = death_mask & (self._episode_outcomes == self.EpisodeOutcome.ONGOING)
        # # Final height of the drone (z position)
        # current_heights = self._robot.data.root_pos_w[:, 2]  # Assuming root_pos_w contains the world position of the robot, with z being height
        # current_velocities = self._robot.data.root_lin_vel_b  # Velocity in body frame

        # # # 实时打印速度和高度
        # # print(f"实时速度 (m/s): {current_velocities}")
        # # print(f"实时高度 (m): {current_heights}")
        # if torch.any(newly_dead):
        #     # 当前高度（世界坐标 z）
        #     z_all = self._robot.data.root_pos_w[:, 2]

        #     # 哪些 env 新死
        #     idxs = torch.nonzero(newly_dead, as_tuple=False).squeeze(-1).tolist()
        #     logs = []
        #     for i in idxs:
        #         reasons = []
        #         if bool(self._numerical_is_unstable[i]):
        #             reasons.append("数值不稳定")
        #         if bool(self._is_contact[i]):
        #             reasons.append("碰撞")
        #         if float(self._robot.data.root_pos_w[i, 2]) < self.cfg.too_low:
        #             reasons.append("高度过低")
        #         if float(self._robot.data.root_pos_w[i, 2]) > self.cfg.too_high:
        #             reasons.append("高度过高")

        #         reason_str = "、".join(reasons) if reasons else "未知原因"
        #         logs.append(f"[Env {i}] 死亡原因：{reason_str} | 实时高度 z={z_all[i].item():.3f} m")

        #     print("\n".join(logs))
        # # ----- 新增结束 -----

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specific environment indexes."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Print depth information
        # self._print_depth_info()
        
        if self._wind_gen is not None:
            # Reset wind generator for the environments being reset
            self._wind_gen.reset(env_ids)

        # Determine episode outcomes for completed episodes
        success_mask = self._is_success[env_ids]
        died_mask = torch.logical_and(self.reset_terminated[env_ids], ~success_mask)
        timed_out_mask = self.reset_time_outs[env_ids]

        # Create environment masks for DeathReplay
        if self._death_replay is not None:
            # Create full-sized masks for all environments
            completed_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            completed_mask[env_ids] = True

            success_mask_full = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            success_mask_full[env_ids] = success_mask

            # Update DeathReplay with episode outcomes
            self._death_replay.end_episodes(completed_mask, success_mask_full)

            # Reset DeathReplay for new episodes
            self._death_replay.reset_episode(env_ids)

        # Update episode outcomes and metrics
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

        if len(env_ids) == self.num_envs:
            # Spread resets to avoid spikes
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = torch.zeros(4, device=self.device)
        self._controller.reset(env_ids)
        self._action_history[env_ids] = 0.0
        self._valid_mask[env_ids] = False

        # Sample new commands with obstacle validation
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2])

        # Sample valid positions that are at least 20cm from obstacles
        min_obstacle_distance = 0.3  # 30cm safety distance
        max_attempts = 20  # Prevent infinite loops

        if self.occ_kdtree is not None:
            # Create index mappings to track which environments still need valid positions
            original_indices = torch.arange(len(env_ids), device=self.device)
            need_valid_position = torch.ones(len(env_ids), dtype=torch.bool, device=self.device)

            attempts = 0
            while torch.any(need_valid_position) and attempts < max_attempts:
                # Get current environment IDs that need valid positions
                current_mask = need_valid_position
                current_indices = original_indices[current_mask]
                current_env_ids = env_ids[current_mask]

                if len(current_env_ids) == 0:
                    break

                # Sample new random positions for these environments
                new_positions = torch.zeros_like(self._desired_pos_w[current_env_ids])
                new_positions[:, :3] = self.env_origins[current_env_ids] 
                if self.cfg.enable_dijkstra:
                    new_positions[:, 0] += torch.zeros_like(new_positions[:, 0]) + (self.cfg.success_threshold + self.cfg.terrain_length) / 2.0
                    new_positions[:, 1] += torch.zeros_like(new_positions[:, 1]) + self.cfg.terrain_width / 2.0
                    new_positions[:, 2] = torch.zeros_like(new_positions[:, 2]) + (self.cfg.desired_low + self.cfg.desired_high) / 2.0    
                else:
                    new_positions[:, 0] += torch.zeros_like(new_positions[:, 0]).uniform_(self.cfg.success_threshold, self.cfg.terrain_length)
                    new_positions[:, 1] += torch.zeros_like(new_positions[:, 1]).uniform_(1.0, self.cfg.terrain_width - 1.0)
                    new_positions[:, 2] = torch.zeros_like(new_positions[:, 2]).uniform_(self.cfg.desired_low, self.cfg.desired_high)
                # new_positions[:, 0] += torch.zeros_like(new_positions[:, 0]).uniform_(self.cfg.terrain_length / 2.0, self.cfg.terrain_length / 2.0)
                # new_positions[:, 1] += torch.zeros_like(new_positions[:, 1]).uniform_(self.cfg.terrain_width / 2.0, self.cfg.terrain_width / 2.0)
                # new_positions[:, 2] = torch.zeros_like(new_positions[:, 2]).uniform_(self.cfg.desired_low, self.cfg.desired_high)
                # Check distances to obstacles
                distances, _ = self.occ_kdtree.query(new_positions.cpu(), workers=-1, distance_upper_bound=self.cfg.distance_upper_bound)
                distances = torch.tensor(distances, device=self.device)

                # Update positions for valid samples
                valid_mask = distances >= min_obstacle_distance

                if torch.any(valid_mask):
                    # Update goal positions for valid samples
                    self._desired_pos_w[current_env_ids[valid_mask]] = new_positions[valid_mask]

                    # Mark these positions as valid in our tracking array
                    need_valid_position[current_indices[valid_mask]] = False

                attempts += 1

            # For any remaining environments that couldn't find valid positions after max attempts,
            # use the last sampled positions (potentially unsafe)
            if torch.any(need_valid_position):
                remaining_env_ids = env_ids[need_valid_position]
                if len(remaining_env_ids) > 0:
                    new_positions = torch.zeros_like(self._desired_pos_w[remaining_env_ids])
                    new_positions = self.env_origins[remaining_env_ids]
                    if self.cfg.enable_dijkstra:
                        new_positions[:, 0] += torch.zeros_like(new_positions[:, 0]) + (self.cfg.success_threshold + self.cfg.terrain_length) / 2.0
                        new_positions[:, 1] += torch.zeros_like(new_positions[:, 1]) + self.cfg.terrain_width / 2.0
                        new_positions[:, 2] = torch.zeros_like(new_positions[:, 2]) + (self.cfg.desired_low + self.cfg.desired_high) / 2.0
                    else:
                        new_positions[:, 0] += torch.zeros_like(new_positions[:, 0]).uniform_(self.cfg.success_threshold, self.cfg.terrain_length)
                        new_positions[:, 1] += torch.zeros_like(new_positions[:, 1]).uniform_(1.0, self.cfg.terrain_width - 1.0)
                        new_positions[:, 2] = torch.zeros_like(new_positions[:, 2]).uniform_(self.cfg.desired_low, self.cfg.desired_high)
                    # new_positions[:, 0] += torch.zeros_like(new_positions[:, 0]).uniform_(self.cfg.terrain_length / 2.0, self.cfg.terrain_length / 2.0)
                    # new_positions[:, 1] += torch.zeros_like(new_positions[:, 1]).uniform_(self.cfg.terrain_width / 2.0, self.cfg.terrain_width / 2.0)
                    # new_positions[:, 2] = torch.zeros_like(new_positions[:, 2]).uniform_(self.cfg.desired_low, self.cfg.desired_high)
                    self._desired_pos_w[remaining_env_ids] = new_positions
                    # print(f"Warning: {len(remaining_env_ids)} environments could not find valid positions after {max_attempts} attempts.")
        else:
            # If KD-tree isn't available, just sample positions without validation
            self._desired_pos_w[env_ids, :3] = self.env_origins[env_ids]
            if self.cfg.enable_dijkstra:
                self._desired_pos_w[env_ids, 0] += torch.zeros_like(self._desired_pos_w[env_ids, 0]) + (self.cfg.terrain_length + self.cfg.success_threshold) / 2.0
                self._desired_pos_w[env_ids, 1] += torch.zeros_like(self._desired_pos_w[env_ids, 1]) + self.cfg.terrain_width / 2.0
                self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]) + (self.cfg.desired_low + self.cfg.desired_high) / 2.0
            else:
                self._desired_pos_w[env_ids, 0] += torch.zeros_like(self._desired_pos_w[env_ids, 0]).uniform_(self.cfg.success_threshold, self.cfg.terrain_length)
                self._desired_pos_w[env_ids, 1] += torch.zeros_like(self._desired_pos_w[env_ids, 1]).uniform_(1.0, self.cfg.terrain_width - 1.0)
                self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(self.cfg.desired_low, self.cfg.desired_high)
            # self._desired_pos_w[env_ids, 0] += torch.zeros_like(self._desired_pos_w[env_ids, 0]).uniform_(self.cfg.terrain_length / 2.0, self.cfg.terrain_length / 2.0)
            # self._desired_pos_w[env_ids, 1] += torch.zeros_like(self._desired_pos_w[env_ids, 1]).uniform_(self.cfg.terrain_width / 2.0, self.cfg.terrain_width / 2.0)
            # self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(self.cfg.desired_low, self.cfg.desired_high)

        self._desired_vel[env_ids] = torch.zeros_like(self._desired_vel[env_ids]).uniform_(0.5, 0.8)

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        # default_root_state[:, :3] += self.scene.env_origins[env_ids].clone() + torch.tensor([ (-self.cfg.scene.env_spacing / 2.0) - 1.0, 0.0, 0.0], device=self.device)
        default_root_state[:, :3] = self.env_origins[env_ids] + torch.tensor([self.cfg.start_x, 0.0, 0.0], device=self.device)
        default_root_state[:, 1] += torch.zeros_like(default_root_state[:, 1]).uniform_(1.0, self.cfg.terrain_width - 1.0)
        default_root_state[:, 2] = torch.zeros_like(default_root_state[:, 2]).uniform_(self.cfg.desired_low, self.cfg.desired_high)

        # Apply random yaw rotation to the initial root state
        # initial_random_yaw = torch.zeros_like(default_root_state[:, 0]).uniform_(-math.pi, math.pi)
        # default_root_state[:, 3] = torch.cos(initial_random_yaw * 0.5)  # w
        # default_root_state[:, 6] = torch.sin(initial_random_yaw * 0.5)  # z
        # default_root_state[:, 4] = 0.0  # x
        # default_root_state[:, 5] = 0.0  # y

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset the "last" references for rewards
        self._last_pos_w[env_ids] = default_root_state[:, :3]
        self._last_actions[env_ids] = torch.zeros(4, device=self.device)
        self._is_contact[env_ids] = False
        self._numerical_is_unstable[env_ids] = False
        self._is_success[env_ids] = False
        self._is_stucking[env_ids] = False
        self._displacement_history[env_ids] = torch.zeros(self.cfg.stucking_timesteps, 3, device=self.device)
        self._obs_history[env_ids] = torch.zeros(self.cfg.history_obs, self.cfg.frame_observation_space, device=self.device)
        self._depth_history[env_ids] = torch.zeros(self.cfg.history_depth, self.cfg.depth_size, device=self.device)
        self._history_distance = torch.full((self.num_envs,), float('inf'), device=self.device)
        # Reset episode outcome tracking for the reset environments
        self._episode_outcomes[env_ids] = 0

        # After setting the desired positions
        # Update target positions in death replay
        if self._death_replay is not None:
            self._death_replay.set_target_positions(self._desired_pos_w)

        # Without enough GPU memory, the terrain will fail to generate.
        if (time.time() - self._map_generation_timer) > 3600 * 24 * 10:
            self._calc_env_origins()
            self._regenerate_terrain()
            self._map_generation_timer = time.time()

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

            if not hasattr(self, "raster_visualizer"):
                raster_cfg = VisualizationMarkersCfg(
                    markers={
                        "raster": sim_utils.CuboidCfg(
                            size=(self.cfg.raster_resolution, self.cfg.raster_resolution, self.cfg.raster_resolution),
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.8, 0.2, 0.2),  # Red for raster
                                opacity=0.2,
                            ),
                        ),
                    }
                )
                raster_cfg.prim_path = "/Visuals/Raster"
                self.raster_visualizer = VisualizationMarkers(raster_cfg)
                print("Created raster_visualizer")

            if not hasattr(self, "rrg_point_visualizer"):
                rrg_point_cfg = VisualizationMarkersCfg(
                    markers={
                        "point": sim_utils.SphereCfg(
                            radius=self.cfg.rrg_step_size,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.1, 0.8, 0.2),  # Cyan for RRG nodes
                                opacity=0.2,
                            ),
                        ),
                    }
                )
                rrg_point_cfg.prim_path = "/Visuals/RRG/nodes"
                self.rrg_point_visualizer = VisualizationMarkers(rrg_point_cfg)
                print("Created rrg_point_visualizer")
            self.rrg_point_visualizer.set_visibility(True)

            if not hasattr(self, "rrg_edge_visualizer"):
                rrg_edge_cfg = VisualizationMarkersCfg(
                    markers={
                        "line": sim_utils.CylinderCfg(
                            radius=0.01,
                            height=1.0,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(1.0, 1.0, 0.0),  # Yellow for RRG edges
                                opacity=1.0,
                            ),
                        ),
                    }
                )
                rrg_edge_cfg.prim_path = "/Visuals/RRG/edges"
                self.rrg_edge_visualizer = VisualizationMarkers(rrg_edge_cfg)
                print("Created rrg_edge_visualizer")
            self.rrg_edge_visualizer.set_visibility(True)

            if not hasattr(self, "rrg_start_goal_visualizer"):
                start_goal_cfg = VisualizationMarkersCfg(
                    markers={
                        "start_goal": sim_utils.SphereCfg(
                            radius=self.cfg.rrg_step_size,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.2, 0.2, 0.0),  # Red for start/goal
                                opacity=0.5,
                            ),
                        ),
                    }
                )   
                start_goal_cfg.prim_path = "/Visuals/RRG/start_goal"
                self.rrg_start_goal_visualizer = VisualizationMarkers(start_goal_cfg)
                print("Created rrg_start_goal_visualizer")
            self.rrg_start_goal_visualizer.set_visibility(True)
            if not hasattr(self, 'traj_visualizer'):
                print("create trajectory visualizer")
                traj_cfg = VisualizationMarkersCfg(
                    markers={
                        "sphere": sim_utils.SphereCfg(
                            radius=0.05,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(1.0, 0.0, 0.0),  # 红色
                            ),
                        ),
                    },
                    prim_path = "/Visuals/Trajectory"
                )
                # traj_cfg.prim_path = "/Visuals/Trajectory"
                self.traj_visualizer = VisualizationMarkers(traj_cfg)
                print("Created traj_visualizer")
            self.traj_visualizer.set_visibility(True)
            if not hasattr(self, "dilation_visualizer"):
                print("create dilation visualizer")
                dilation_cfg = VisualizationMarkersCfg(
                    markers={
                        "dilation": sim_utils.CuboidCfg(
                            size=(0.1, 0.1, 0.1),  # 边长为0.1米的正方体
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.0, 0.5, 0.5),  # 青色
                                opacity=0.7,
                            ),
                        ),
                    },
                    prim_path = "/Visuals/Dilation"
                )
                self.dilation_visualizer = VisualizationMarkers(dilation_cfg)
                print("Created dilation_visualizer")
            self.dilation_visualizer.set_visibility(True)
                
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            if hasattr(self, "goal_yaw_visualizer"):
                self.goal_yaw_visualizer.set_visibility(False)
            if hasattr(self, "current_yaw_visualizer"):
                self.current_yaw_visualizer.set_visibility(False)
            if hasattr(self, "rrg_point_visualizer"):
                self.rrg_point_visualizer.set_visibility(False)
            if hasattr(self, "rrg_edge_visualizer"):
                self.rrg_edge_visualizer.set_visibility(False)
            if hasattr(self, "rrg_start_goal_visualizer"):
                self.rrg_start_goal_visualizer.set_visibility(False)
            if hasattr(self, 'traj_visualizer'):
                self.traj_visualizer.set_visibility(False)
            if hasattr(self, "dilation_visualizer"):
                self.dilation_visualizer.set_visibility(False)

        
    def _debug_vis_callback(self, event):
        """Update debug markers with new goal positions."""
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
        self.goal_yaw_visualizer.visualize(self._desired_pos_w, self._desired_yaw_quat)
        self.current_yaw_visualizer.visualize(self._robot.data.root_pos_w, self._robot.data.root_quat_w)
        
        if self.cfg.dijkstra_vis:
            if hasattr(self, 'dilation_visualizer') and hasattr(self, '_dilated_positions'):
                self.dilation_visualizer.visualize(self._dilated_positions)
            if hasattr(self, "traj_visualizer") and hasattr(self, '_traj'):
                self.traj_visualizer.visualize(self._traj)

    class EpisodeOutcome(IntEnum):
        ONGOING = 0
        SUCCESS = 1
        FAILURE = 2
    
    def _update_episode_outcomes_and_metrics(self, env_ids, success_mask, died_mask, timed_out_mask):
        # Update episode outcomes for the reset environments using vectorized operations
        self._episode_outcomes[env_ids] = torch.where(
            success_mask,
            torch.tensor(self.EpisodeOutcome.SUCCESS, device=self.device),
            torch.where(
                died_mask,
                torch.tensor(self.EpisodeOutcome.FAILURE, device=self.device),
                self._episode_outcomes[env_ids]
            )
        )

        # Check for completed episodes (success, died, or timed out)
        completed_mask = torch.logical_or(torch.logical_or(success_mask, died_mask), timed_out_mask)
        if not torch.any(completed_mask):
            return 0, 0

        # Extract completion info using efficient mask indexing
        completed_env_ids = env_ids[completed_mask]
        success_env_ids = env_ids[success_mask]
        died_env_ids = env_ids[died_mask]
        
        # Get outcomes for completed episodes in a vectorized way
        completed_success = success_mask[completed_mask]
        # Convert to list format expected by the deque in a single operation
        outcomes = (completed_success.cpu() == True).tolist()
        outcomes = [self.EpisodeOutcome.SUCCESS if success else self.EpisodeOutcome.FAILURE for success in outcomes]
        
        # Process termination reasons in a vectorized way
        if len(died_env_ids) > 0:
            # Create numpy arrays for each condition to avoid GPU transfers in loop
            is_unstable = self._numerical_is_unstable[died_env_ids].cpu().numpy()
            is_collision = self._is_contact[died_env_ids].cpu().numpy()
            pos_z = self._robot.data.root_pos_w[died_env_ids, 2].cpu().numpy()
            too_low = (pos_z < self.cfg.too_low)
            too_high = (pos_z > self.cfg.too_high)
            
            # Build termination reason dictionaries efficiently
            for i in range(len(died_env_ids)):
                self._termination_reason_history.append({
                    "numerical_is_unstable": bool(is_unstable[i]),
                    "collision": bool(is_collision[i]),
                    "too_low": bool(too_low[i]),
                    "too_high": bool(too_high[i])
                })
        
        # Add empty dictionaries for success/timeout cases in a single operation
        self._termination_reason_history.extend([{}] * (len(success_env_ids) + len(env_ids[timed_out_mask])))
        
        # Calculate final distances using vectorized operations
        if len(completed_env_ids) > 0:
            distances = torch.linalg.norm(
                self._desired_pos_w[completed_env_ids] - self._robot.data.root_pos_w[completed_env_ids], 
                dim=1
            ).cpu().tolist()
            # Add all distances at once
            self._final_distances.extend(distances)
        
        if len(completed_env_ids) > 0:
            vel_abs = torch.linalg.norm(
                self._robot.data.root_lin_vel_w[completed_env_ids], 
                dim=1
            ).cpu().tolist()
            # Add all velocities at once
            self._vel_abs.extend(vel_abs)

        # Add outcomes to history in a single operation
        self._episode_outcome_history.extend(outcomes)
        
        # Efficiently calculate statistics
        num_outcomes = len(self._episode_outcome_history)
        if num_outcomes > 0:
            # Convert deque to numpy array once for faster processing
            outcome_array = np.array(list(self._episode_outcome_history))
            success_count = np.sum(outcome_array == self.EpisodeOutcome.SUCCESS)
            died_count = np.sum(outcome_array == self.EpisodeOutcome.FAILURE)
            timeout_count = num_outcomes - success_count - died_count
            
            # Calculate success rate
            self._success_rate = success_count / num_outcomes
            
            # Calculate termination reason percentages using numpy for efficiency
            # Convert termination reasons to a structured format for faster counting
            reason_keys = ["numerical_is_unstable", "collision", "too_low", "too_high"]
            reason_counts = {key: 0 for key in reason_keys}
            
            # This is still a loop, but it's processing a summary once rather than in every call
            # For extremely large histories, a more optimized structure could be used
            if len(self._termination_reason_history) > 0:
                # Process in batches to avoid memory issues with very large histories
                batch_size = 2000
                for i in range(0, len(self._termination_reason_history), batch_size):
                    batch = list(itertools.islice(self._termination_reason_history, i, i + batch_size))
                    for key in reason_keys:
                        reason_counts[key] += sum(1 for reason in batch if key in reason and reason[key])
            
            # Update episode count tracking
            completed_count = len(outcomes)
            succeeded_count = sum(1 for o in outcomes if o == self.EpisodeOutcome.SUCCESS)
            self._episodes_completed += completed_count
            self._episodes_succeeded += succeeded_count
            cumulative_success_rate = self._episodes_succeeded / self._episodes_completed if self._episodes_completed > 0 else 0.0
            
            # Calculate average final distance efficiently
            avg_final_distance = np.mean(list(self._final_distances)) if self._final_distances else 0.0
            # Calculate average velocity efficiently
            avg_velocity = np.mean(list(self._vel_abs)) if self._vel_abs else 0.0

            # Prepare metrics
            if "log" not in self.extras:
                self.extras["log"] = {}
            
            # Add all metrics to log at once
            self.extras["log"].update({
                # Episode termination statistics as percentages
                "Episode_Termination/died": died_count / num_outcomes * 100.0,
                "Episode_Termination/time_out": timeout_count / num_outcomes * 100.0,
                "Episode_Termination/success": success_count / num_outcomes * 100.0,

                # Death reason statistics as percentages of total episodes
                "Metrics/Died/numerical_is_unstable": reason_counts["numerical_is_unstable"] / num_outcomes * 100.0,
                "Metrics/Died/collision": reason_counts["collision"] / num_outcomes * 100.0,
                "Metrics/Died/too_low": reason_counts["too_low"] / num_outcomes * 100.0,
                "Metrics/Died/too_high": reason_counts["too_high"] / num_outcomes * 100.0,

                # Success and progress tracking
                "Metrics/final_distance_to_goal": avg_final_distance,
                "Metrics/average_velocity": avg_velocity,
                "Metrics/Success/goal_reached": success_count,
                "Metrics/rolling_success_rate": self._success_rate * 100.0,
                "Metrics/cumulative_success_rate": cumulative_success_rate * 100.0,
                "Metrics/episodes_completed": self._episodes_completed,
                "Metrics/episodes_succeeded": self._episodes_succeeded,
            })

            return completed_count, succeeded_count

    def close(self):
        """Clean up resources when environment is closed."""
        super().close()
