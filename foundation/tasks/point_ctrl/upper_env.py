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

from foundation.utils.train_terrain import MapGenerator
from foundation.utils.raster import TerrainRasterMap
from foundation.utils.simple_controller import SimpleQuadrotorController
from rsl_rl.modules.student_teacher_recurrent_custom import StudentTeacherRecurrentCustom
from enum import IntEnum
import collections
import itertools
import matplotlib.pyplot as plt

def add_rounding_noise_torch(depth_map: torch.Tensor, levels: int = 128) -> torch.Tensor:
    min_depth = torch.min(depth_map[depth_map > 1e-6])
    max_depth = torch.max(depth_map)

    if max_depth <= min_depth:
        return depth_map

    step_size = (max_depth - min_depth) / levels
    if step_size <= 1e-6:
        return depth_map

    quantized_map = torch.round(depth_map / step_size) * step_size
    return quantized_map

def add_edge_noise_torch(depth_map: torch.Tensor, edge_threshold: float = 0.1, noise_magnitude: float = 0.3) -> torch.Tensor:
    depth_map_nchw = depth_map.unsqueeze(1)

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=depth_map.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=depth_map.device).view(1, 1, 3, 3)

    grad_x = F.conv2d(depth_map_nchw, sobel_x, padding=1)
    grad_y = F.conv2d(depth_map_nchw, sobel_y, padding=1)

    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2).squeeze(1)

    edge_mask = gradient_magnitude > edge_threshold
    noise = torch.randn_like(depth_map) * noise_magnitude
    noisy_map = depth_map.clone()
    noisy_map[edge_mask] += noise[edge_mask]

    noisy_map.clamp_(min=0.0)
    return noisy_map

def add_filling_noise_torch(depth_map: torch.Tensor, dropout_rate: float = 0.03, kernel_size: int = 5) -> torch.Tensor:
    dropout_mask = torch.rand_like(depth_map) < dropout_rate
    holed_map = depth_map.clone()
    holed_map[dropout_mask] = 0.0
    filled_map = F.avg_pool2d(holed_map.unsqueeze(1), kernel_size=kernel_size, stride=1, padding=kernel_size//2).squeeze(1)
    final_map = depth_map.clone()
    final_map[dropout_mask] = filled_map[dropout_mask]
    return final_map

def add_edge_filling_noise_torch(depth_map: torch.Tensor, edge_threshold: float = 0.1, dropout_rate_on_edges: float = 0.5, kernel_size: int = 5) -> torch.Tensor:
    depth_map_nchw = depth_map.unsqueeze(1)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=depth_map.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=depth_map.device).view(1, 1, 3, 3)
    grad_x = F.conv2d(depth_map_nchw, sobel_x, padding=1)
    grad_y = F.conv2d(depth_map_nchw, sobel_y, padding=1)
    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2).squeeze(1)
    edge_mask = gradient_magnitude > edge_threshold
    random_mask = torch.rand_like(depth_map) < dropout_rate_on_edges
    final_dropout_mask = edge_mask & random_mask
    holed_map = depth_map.clone()
    holed_map[final_dropout_mask] = 0.0
    filled_map = F.avg_pool2d(holed_map.unsqueeze(1), kernel_size=kernel_size, stride=1, padding=kernel_size//2).squeeze(1)
    final_map = depth_map.clone()
    final_map[final_dropout_mask] = filled_map[final_dropout_mask]
    return final_map

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

    frame_observation_space = 22  # 16状态 + 6动作
    depth_size = 6000             # 100 * 60
    
    # 【修改这里】只保留当前帧维度
    observation_space = frame_observation_space + depth_size # 结果为 6022

    gamma = 0.99

    episode_length_s = 96
    decimation = 1

    # 低层网络架构参数 (必须与 distillation 训练时一致)
    low_level_rnn_hidden_dim = 16
    low_level_rnn_num_layers = 1
    low_level_pre_rnn_dim = 16
    low_level_post_rnn_dim = 16
    low_level_student_hidden_dims = []
    low_level_rnn_type = "gru"
    low_level_activation = "elu"

    # 1.1 修改动作空间维度：从 4 (roll, pitch, yaw, thrust) 变为 6 (3 for delta_p, 3 for delta_v)
    action_space = 6 
    
    # 1.2 定义上层输出的物理缩放系数 (根据你的物理需求调整)
    delta_p_scale = 5.0  # 假设输出动作 [-1, 1] 对应 +-1.0m
    delta_v_scale = 2.0  # 假设输出动作 [-1, 1] 对应 +-2.0m/s
    
    # 1.3 预训练下层控制器的路径
    low_level_model_path = "./models/low_level_policy.pt"
    
    # 1.4 无人机动力学参数（需与低层训练时一致）
    # 如果是 Crazyflie
    mass = 0.049 
    arm_length = 0.046 # 约 46mm
    inertia = (1.3615e-5, 1.3615e-5, 3.257e-5)
    twr = 2.0
    kappa = 0.016

    state_space = 0

    grid_rows = 10
    grid_cols = 1
    terrain_width = 32
    terrain_length = 89
    robots_per_env = 10
    success_threshold = 69.0
    distance_upper_bound = 4.0

    connectivity_vis = False
    enable_dijkstra = True

    raster_resolution = 0.1
    dilation_kernel_size = 2
    dijkstra_vis = False

    start_x = 2.5
    load_raster_from_files = True
    distance_for_invalid = 200.0
    terrain_path = "./USD/flightfield"
    grid_path = "./RASTER/flightfield/"
    enable_larger_dilation = True

    debug_vis = True
    robot_vis = False
    marker_size = 0.05

    cbf_safe_bound = 0.15
    cbf_eta = 0.1

    max_vel = 1.0

    too_low = 0.0
    too_high = 2.0
    desired_low = 0.5
    desired_high = 1.5
    ceiling_height = too_high

    reward_coef_distance_reward: float = 0.0
    reward_coef_action_magnitude_penalty: float = 0.4
    reward_coef_action_change_penalty: float = 0.5
    reward_coef_vel_speed_excess_penalty: float = 1.0
    reward_coef_vel_speed_match_reward: float = 0.0
    reward_coef_z_position_penalty: float = 0.3
    reward_coef_obstacle_collision_penalty: float = 100.0
    reward_coef_succeed_reward: float = 100.0
    reward_coef_alive_reward: float = 0.0
    reward_coef_dijkstra = 10.0
    reward_coef_cbf = 10.0

    ui_window_class_type = QuadcopterEnvWindow

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100.0,
        render_interval=4,
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

    random_init_pos = False

    enable_actor_noise: bool = True

    position_noise_std: float = 0.1

    velocity_noise_std: float = 0.05

    attitude_noise_std: float = 0.070

    depth_edge_threshold: float = 1.0
    depth_edge_noise_magnitude: float = 0.05
    depth_filling_dropout_rate: float = 0.03
    depth_filling_kernel_size: int = 5
    depth_rounding_levels: int = 128

    scene: InteractiveSceneCfg = QuadcopterSceneCfg()

    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/body", history_length=1, update_period=0.01,
        track_air_time=False,
        debug_vis=False,
    )

    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/body/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.09, 0.0, -0.01), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["depth"],

        spawn=sim_utils.PinholeCameraCfg(
            focal_length=20.0, focus_distance=400.0, horizontal_aperture=27.84, clipping_range=(0.01, 7.0)
        ),
        width=100,
        height=60,
        update_period=1.0 / 30.0,
        depth_clipping_behavior="max",
    )

    depth_size = tiled_camera.width * tiled_camera.height


class QuadcopterEnv(DirectRLEnv):

    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.start_time = time.time()

        self.render_mode = "human"

        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._torques = torch.zeros(self.num_envs, 1, 3, device=self.device)

        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_vel = torch.zeros(self.num_envs, 1, device=self.device)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "distance_reward",
                "action_magnitude_penalty",
                "action_change_penalty",
                "vel_speed_excess_penalty",
                "vel_speed_match_reward",
                "z_position_penalty",
                "obstacle_collision_penalty",
                "succeed_reward",
                "alive_reward",
                "dijkstra_reward",
                "cbf_reward",
            ]
        }

        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        self.grid_idx = None

        self._body_id = self._robot.find_bodies("body")[0]
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies("body")
        self._action_history_length = 8
        self._action_history = torch.zeros(self.num_envs, self._action_history_length, self.cfg.action_space, device=self.device)
        self._valid_mask = torch.zeros(
        self.num_envs, self._action_history_length, dtype=torch.bool, device=self.device
        )
        
        self._last_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._last_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._is_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._numerical_is_unstable = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._is_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.occ_kdtree = None
        self._dilated_positions = torch.zeros(1, 3, device=self.device)
        self._traj = torch.zeros(1, 3, device=self.device)
        self._maps = []

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
            n_min=-0.1,
            n_max=0.1,
            operation='scale'
        )
        self._noise_20_cfg = GaussianNoiseCfg(
            mean=1.0,
            std=0.20,
            operation='scale'
        )

        self._success_rate_window = 100
        self._episode_outcomes = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episodes_completed = 0
        self._episodes_succeeded = 0
        self._success_rate = 0.0
        self._episode_outcome_history = collections.deque(maxlen=self._success_rate_window)
        self._termination_reason_history = collections.deque(maxlen=self._success_rate_window)
        self._final_distances = collections.deque(maxlen=self._success_rate_window)
        self._vel_abs = collections.deque(maxlen=self._success_rate_window)

        self.set_debug_vis(self.cfg.debug_vis)

        h = self.cfg.tiled_camera.height
        w = self.cfg.tiled_camera.width
        center_h, center_w = h / 2.0, w / 2.0

        y, x = torch.meshgrid(
            torch.arange(h, device=self.device, dtype=torch.float32),
            torch.arange(w, device=self.device, dtype=torch.float32),
            indexing='ij'
        )

        sigma = min(h, w) / 3.0

        self.center_weights = torch.exp(-((x - center_w)**2 + (y - center_h)**2) / (2 * sigma**2))

        self.center_weights = self.center_weights / torch.sum(self.center_weights)

        self._calc_env_origins()


        # 初始化 RAPTOR 动力学控制器
        # 构造参数张量
        mass_tensor = torch.full((self.num_envs,), self.cfg.mass, device=self.device)
        arm_l_tensor = torch.full((self.num_envs,), self.cfg.arm_length, device=self.device)
        inertia_tensor = torch.tensor(self.cfg.inertia, device=self.device).repeat(self.num_envs, 1)
        twr_tensor = torch.full((self.num_envs,), self.cfg.twr, device=self.device)
        kappa_tensor = torch.full((self.num_envs,), self.cfg.kappa, device=self.device)

        self._down_controller = SimpleQuadrotorController(
            num_envs=self.num_envs,
            device=self.device,
            mass=mass_tensor,
            arm_length=arm_l_tensor,
            inertia=inertia_tensor,
            thrust_to_weight=twr_tensor,
            moment_scale=kappa_tensor
        )

        # 2.3 初始化低层动作历史 (用于低层网络的输入)
        self._last_low_level_actions = torch.zeros(self.num_envs, 4, device=self.device)

        # 2.1 实例化低层网络
        # 注意：num_student_obs=22, num_teacher_obs 在推理阶段不重要，随便填
        self.low_level_policy = StudentTeacherRecurrentCustom(
            num_student_obs=22, 
            num_teacher_obs=56, # 对应你 teacher_observation_space
            num_actions=4,      # 电机数
            student_hidden_dims=self.cfg.low_level_student_hidden_dims,
            rnn_type=self.cfg.low_level_rnn_type,
            rnn_hidden_dim=self.cfg.low_level_rnn_hidden_dim,
            rnn_num_layers=self.cfg.low_level_rnn_num_layers,
            pre_rnn_dim=self.cfg.low_level_pre_rnn_dim,
            post_rnn_dim=self.cfg.low_level_post_rnn_dim,
            activation=self.cfg.low_level_activation
        ).to(self.device)

        # 2.2 使用 torch.load 加载权重字典
        if os.path.exists(self.cfg.low_level_model_path):
            checkpoint = torch.load(self.cfg.low_level_model_path, map_location=self.device)
            # RSL-RL 的权重通常在 'model_state_dict' 键下
            if 'model_state_dict' in checkpoint:
                self.low_level_policy.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                self.low_level_policy.load_state_dict(checkpoint) # 兼容直接保存 state_dict 的情况
            
            self.low_level_policy.eval()
            print(f"[INFO] Loaded low-level policy weights from {self.cfg.low_level_model_path}")
        else:
            raise FileNotFoundError(f"Model not found at {self.cfg.low_level_model_path}")

    def _print_depth_info(self, env_id=0, show_image=True):

        depth_image = self._tiled_camera.data.output["depth"]
        h, w = self.cfg.tiled_camera.height, self.cfg.tiled_camera.width
        center_h = h // 2
        center_w = w // 2

        pos_w = self._robot.data.root_pos_w

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

        max_angular_velocity = 3.14 * 2.0 * 20.0

        ang_vel_b = self._robot.data.root_ang_vel_b
        rot_w = torch.stack(euler_xyz_from_quat(self._robot.data.root_quat_w), dim=1)
        rot_w = torch.stack([normallize_angle(rot_w[:, 0]), normallize_angle(rot_w[:, 1]), normallize_angle(rot_w[:, 2])], dim=1)

        state_is_unstable = torch.any(torch.abs(ang_vel_b) > max_angular_velocity, dim=1)

        self._numerical_is_unstable = torch.logical_or(self._numerical_is_unstable, state_is_unstable)

    def _calc_env_origins(self):

        robots_per_env = self.cfg.robots_per_env
        num_groups = self.num_envs // robots_per_env + 1

        grid_rows = self.cfg.grid_rows
        grid_cols = self.cfg.grid_cols

        grid_capacity = grid_rows * grid_cols
        if num_groups > grid_capacity:
            print(f"Warning: The number of groups ({num_groups}) exceeds the grid capacity ({grid_capacity}). Group origins will loop.")

        group_origins = torch.zeros(num_groups, 3, device=self.device)
        terrain_width = self.cfg.terrain_width
        terrain_length = self.cfg.terrain_length

        for i in range(num_groups):
            row = (i // grid_cols) % grid_rows
            col = i % grid_cols
            group_origins[i, 0] = col * terrain_length
            group_origins[i, 1] = row * terrain_width

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

        self._robot = Articulation(self.cfg.robot)
        robot_prims = find_matching_prim_paths("/World/envs/env_.*/Robot")
        for prim_path in robot_prims:
            prims_utils.set_prim_property(prim_path + "/body", "physics:mass", 0.049)
            prims_utils.set_prim_property(prim_path + "/body", "physics:diagonalInertia", (1.3615e-5, 1.3615e-5, 3.257e-5))
            if self.cfg.robot_vis == True:
                prims_utils.set_prim_property(prim_path, "visibility", "visible")
            else:
                prims_utils.set_prim_property(prim_path, "visibility", "invisible")

        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)

        self._map_generator = MapGenerator(sim=self.sim, device=self.device)

        self.scene.clone_environments(copy_from_source=False)

        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["tiled_camera"] = self._tiled_camera

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        activate_contact_sensors("/World")
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self._map_generation_timer = 0

    def _regenerate_terrain(self):
        self.sim.pause()
        print("Regenerating terrain and occupancy map.")
        prims_utils.delete_prim("/World/ground")
        self._terrain = self.cfg.scene.terrain.class_type(self.cfg.scene.terrain)

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
            ceiling_height = self.cfg.ceiling_height,
        )

        self.occ_kdtree = env_data["kdtree"]

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
                    map = TerrainRasterMap(map_min, map_max, self.cfg.raster_resolution, self.cfg.raster_resolution/2.0)
                    if not files:

                        map.build_occupancy_grid_from_points(self.occ_kdtree.data)
                        map.dilate_existing_obstacles(self.cfg.dilation_kernel_size)

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
                        map.set_custom_distance_for_invalid(self.cfg.distance_for_invalid)
                        occupied_points = map.get_occupied_positions()
                        dilated_position = map.get_dilated_positions()
                        all_occupied_points.extend(occupied_points)
                        all_dilated_points.extend(dilated_position)
                    else:
                        map.load_from_file(self.cfg.grid_path + f"/raster_map_{i}_{j}.npz")
                        map.set_custom_distance_for_invalid(self.cfg.distance_for_invalid)
                        occupied_points = map.get_occupied_positions()
                        dilated_position = map.get_dilated_positions()
                        all_occupied_points.extend(occupied_points)
                        all_dilated_points.extend(dilated_position)
                    self._maps.append(map)
            self.occ_kdtree = KDTree(all_occupied_points)
            self._dilated_positions = torch.tensor(all_dilated_points, device=self.device, dtype=torch.float32)
            self._traj = torch.tensor(self._maps[0].extract_path_to_goal([2.5, 32.0, 2.0]), device=self.device, dtype=torch.float32)
            print(f"Raster Map Generation Time: {time.time() - start_time:.2f} seconds")
        self.sim.play()

    def _pre_physics_step(self, actions: torch.Tensor):
        # 1. 解析上层动作
        actions = actions.clamp(-1.0, 1.0)
        delta_p_b = actions[:, :3] * self.cfg.delta_p_scale
        delta_v_b = actions[:, 3:6] * self.cfg.delta_v_scale

        # 2. 获取实时状态
        quat_w = self._robot.data.root_quat_w
        rot_flat = matrix_from_quat(quat_w).reshape(self.num_envs, 9)
        ang_vel_b = self._robot.data.root_ang_vel_b

        # 3. 组装低层输入 [22 维]
        # 必须匹配 distillation_env.py 里的 obs_student 顺序
        low_level_obs = torch.cat([
            delta_p_b,                  # 3
            rot_flat,                   # 9
            delta_v_b,                  # 3
            ang_vel_b,                  # 3
            self._last_low_level_actions # 4
        ], dim=-1)

        # 4. 执行低层策略 (带有 RNN 状态更新)
        with torch.no_grad():
            # 调用 act_inference，它会自动处理 self.hidden_state 的读写
            motor_actions = self.low_level_policy.act_inference(low_level_obs)
            # 你的模型输出层带了 Tanh，所以范围是 (-1, 1)，需要映射到 (0, 1) 给 RAPTOR
            motor_actions_mapped = (motor_actions + 1.0) * 0.5
            motor_actions_mapped = motor_actions_mapped.clamp(0.0, 1.0)

        # 5. 更新历史
        self._last_low_level_actions = motor_actions.detach().clone()

        # 6. 动力学计算
        force_b, torque_b, _ = self._down_controller.motor_speeds_to_wrench(motor_actions_mapped)

        # 7. 施加力
        self._forces.zero_()
        self._torques.zero_()
        self._forces[:, 0, :] = force_b
        self._torques[:, 0, :] = torque_b

    def _apply_action(self):

        self._robot.set_external_force_and_torque(self._forces, self._torques, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        perfect_depth_map_nhwc = self._tiled_camera.data.output["depth"]
        perfect_depth_map_nhw = perfect_depth_map_nhwc.squeeze(-1)

        if self.cfg.enable_actor_noise:
            noisy_map_temp = add_edge_noise_torch(
                perfect_depth_map_nhw,
                edge_threshold=self.cfg.depth_edge_threshold,
                noise_magnitude=self.cfg.depth_edge_noise_magnitude
            )
            noisy_map_temp1 = add_filling_noise_torch(
                noisy_map_temp,
                dropout_rate=self.cfg.depth_filling_dropout_rate,
                kernel_size=self.cfg.depth_filling_kernel_size
            )
            final_noisy_map = add_rounding_noise_torch(
                noisy_map_temp1,
                levels=self.cfg.depth_rounding_levels
            )
        else:
            final_noisy_map = perfect_depth_map_nhw

        batch_size = final_noisy_map.shape[0]
        flat_noisy_depth = final_noisy_map.reshape(batch_size, -1)
        flat_perfect_depth = perfect_depth_map_nhw.reshape(batch_size, -1)

        pos_w_clean = self._robot.data.root_state_w[:, :3].clone()
        quat_w_clean = self._robot.data.root_quat_w.clone()
        vel_b_clean = self._robot.data.root_lin_vel_b.clone()

        if self.cfg.enable_actor_noise:

            pos_w_noisy = pos_w_clean + torch.randn_like(pos_w_clean) * self.cfg.position_noise_std

            vel_b_noisy = vel_b_clean * (1.0 + torch.randn_like(vel_b_clean) * self.cfg.velocity_noise_std)

            angle_noise = torch.randn(self.num_envs, 3, device=self.device) * self.cfg.attitude_noise_std
            half_angles = angle_noise * 0.5
            quat_noise = torch.zeros(self.num_envs, 4, device=self.device)
            angle_norm = torch.norm(half_angles, dim=1, keepdim=True).clamp(min=1e-6)
            quat_noise[:, 0] = torch.cos(angle_norm.squeeze())
            quat_noise[:, 1:] = torch.sin(angle_norm) * half_angles / angle_norm

            from isaaclab.utils.math import quat_mul
            quat_w_noisy = quat_mul(quat_w_clean, quat_noise)
        else:
            pos_w_noisy = pos_w_clean
            vel_b_noisy = vel_b_clean
            quat_w_noisy = quat_w_clean

        rot_matrix_b2w_clean = matrix_from_quat(quat_w_clean)
        rotation_matrix_flat_clean = rot_matrix_b2w_clean.reshape(self.num_envs, 9)

        rot_matrix_b2w_noisy = matrix_from_quat(quat_w_noisy)
        rotation_matrix_flat_noisy = rot_matrix_b2w_noisy.reshape(self.num_envs, 9)

        direction_to_goal_w_clean = self._desired_pos_w - pos_w_clean
        direction_to_goal_xy_w_clean = direction_to_goal_w_clean[:, :2]
        rot_matrix_w2b_clean = rot_matrix_b2w_clean.transpose(1, 2)
        direction_to_goal_w_3d_clean = torch.cat([
            direction_to_goal_xy_w_clean,
            torch.zeros(self.num_envs, 1, device=self.device)
        ], dim=1)
        direction_to_goal_b_3d_clean = torch.bmm(
            rot_matrix_w2b_clean,
            direction_to_goal_w_3d_clean.unsqueeze(-1)
        ).squeeze(-1)
        direction_to_goal_xy_b_clean = direction_to_goal_b_3d_clean[:, :2]

        direction_to_goal_w_noisy = self._desired_pos_w - pos_w_noisy
        direction_to_goal_xy_w_noisy = direction_to_goal_w_noisy[:, :2]
        rot_matrix_w2b_noisy = rot_matrix_b2w_noisy.transpose(1, 2)
        direction_to_goal_w_3d_noisy = torch.cat([
            direction_to_goal_xy_w_noisy,
            torch.zeros(self.num_envs, 1, device=self.device)
        ], dim=1)
        direction_to_goal_b_3d_noisy = torch.bmm(
            rot_matrix_w2b_noisy,
            direction_to_goal_w_3d_noisy.unsqueeze(-1)
        ).squeeze(-1)
        direction_to_goal_xy_b_noisy = direction_to_goal_b_3d_noisy[:, :2]

        direction_to_goal_xy_w_clean = direction_to_goal_xy_w_clean / (
            direction_to_goal_xy_w_clean.norm(dim=1, keepdim=True) + 1e-6
        )
        direction_to_goal_xy_w_noisy = direction_to_goal_xy_w_noisy / (
            direction_to_goal_xy_w_noisy.norm(dim=1, keepdim=True) + 1e-6
        )

        frame_obs_clean = torch.cat([
            vel_b_clean,
            rotation_matrix_flat_clean,
            direction_to_goal_xy_b_clean,
            (self._desired_pos_w[:, 2]).unsqueeze(dim=-1),
            pos_w_clean[:, 2].unsqueeze(dim=-1),
            self._last_actions,
        ], dim=-1)

        frame_obs_noisy = torch.cat([
            vel_b_noisy,
            rotation_matrix_flat_noisy,
            direction_to_goal_xy_b_noisy,
            (self._desired_pos_w[:, 2]).unsqueeze(dim=-1),
            pos_w_noisy[:, 2].unsqueeze(dim=-1),
            self._last_actions,
        ], dim=-1)

        # 1. 展平当前帧深度图 [6000]
        flat_noisy_depth = final_noisy_map.reshape(self.num_envs, -1)
        flat_perfect_depth = perfect_depth_map_nhw.reshape(self.num_envs, -1)

        # 2. 拼接当前帧状态 [22] + 当前帧深度 [6000] = 6022
        actor_obs = torch.cat([
            frame_obs_noisy,     # [num_envs, 22]
            flat_noisy_depth     # [num_envs, 6000]
        ], dim=-1)

        critic_obs = torch.cat([
            frame_obs_clean,     # [num_envs, 22]
            flat_perfect_depth   # [num_envs, 6000]
        ], dim=-1)

        actor_obs = self.CHECK_NAN(actor_obs, "Actor Observation")
        critic_obs = self.CHECK_NAN(critic_obs, "Critic Observation")

        return {"policy": actor_obs, "critic": critic_obs, "rnd_state": actor_obs}

    def _get_rewards(self) -> torch.Tensor:

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        pos_w = self._robot.data.root_state_w[:, :3]
        rot_w = torch.stack(euler_xyz_from_quat(self._robot.data.root_state_w[:, 3:7]), dim=1)
        rot_w = torch.stack([normallize_angle(rot_w[:, 0]), normallize_angle(rot_w[:, 1]), normallize_angle(rot_w[:, 2])], dim=1)
        vel_b = self._robot.data.root_lin_vel_b

        distance_to_gap = (pos_w - self._desired_pos_w).norm(dim=1)
        last_distance_to_gap = (self._last_pos_w - self._desired_pos_w).norm(dim=1)
        delta_distance = last_distance_to_gap - distance_to_gap
        delta_distance = torch.clamp(delta_distance, min=-0.01 * (self.cfg.max_vel + 2.0), max=0.01 * (self.cfg.max_vel + 2.0))
        distance_reward = 10.0 * delta_distance

        act_abs = torch.abs(self._actions)
        action_magnitude = torch.sum(torch.square(self._actions), dim=1)
        action_magnitude_penalty = -action_magnitude

        diff_actions = self._actions - self._last_actions

        weights = torch.tensor([1.0, 1.0, 1.0, 0.5, 0.5, 0.5], device=self.device)
        diff_actions_weighted = diff_actions * weights
        action_change_penalty = - (diff_actions_weighted ** 2).sum(dim=1)

        v_max = self.cfg.max_vel
        vel_norm = torch.norm(vel_b, dim=1)
        vel_speed_excess_penalty = torch.where(
            torch.abs(vel_norm) > v_max,
            -torch.clamp(torch.exp(vel_norm - v_max) - 1.0, max=5.0),
            torch.zeros_like(vel_norm)
        )

        speed = self._robot.data.root_lin_vel_w.norm(dim=1, keepdim=True)
        vel_speed_match_reward = torch.exp(-5.0 * torch.abs(speed - self._desired_vel)) * 2.0
        vel_speed_match_reward = vel_speed_match_reward.squeeze()

        z_pos = pos_w[:, 2]
        floor_dist = z_pos - (self._desired_pos_w[:, 2]) + 0.2
        floor_penalty = torch.where(
            floor_dist < 0.0,
            -torch.clamp(torch.exp(4.0 * (-floor_dist)) - 1.0, max=5.0),
            torch.zeros_like(z_pos),
        )
        ceiling_dist = (self._desired_pos_w[:, 2]) - z_pos + 0.2
        ceiling_penalty = torch.where(
            ceiling_dist < 0.0,
            -torch.clamp(torch.exp(4.0 * (-ceiling_dist)) - 1.0, max=5.0),
            torch.zeros_like(z_pos),
        )
        z_position_penalty = floor_penalty + ceiling_penalty

        die =  self._numerical_is_unstable | self._is_contact | (self._robot.data.root_pos_w[:, 2] < self.cfg.too_low) | (self._robot.data.root_pos_w[:, 2] > self.cfg.too_high)

        obstacle_collision_penalty = torch.where(
            die,
            torch.ones_like(vel_b[:, 0]),
            torch.zeros_like(vel_b[:, 0]),
        )
        obstacle_collision_penalty = -obstacle_collision_penalty

        succeed_reward = self._is_success

        alive_reward = torch.logical_not(die).float()

        dijkstra_reward = torch.zeros(self.num_envs, device=self.device)
        if self.cfg.enable_dijkstra:
            for i in range(self.cfg.grid_rows):
                for j in range(self.cfg.grid_cols):
                    idx = i * self.cfg.grid_cols + j
                    env_ids = self.grid_idx[idx]
                    map_min = [j*self.cfg.terrain_length, i*self.cfg.terrain_width, self.cfg.too_low]
                    if len(env_ids) == 0:
                        continue
                    value = torch.from_numpy(self._maps[idx].trilinear_interpolate(self._robot.data.root_state_w[env_ids, :3].cpu().numpy())).to(dtype=torch.float32, device=self.device)
                    last_value = torch.from_numpy(self._maps[idx].trilinear_interpolate(self._last_pos_w[env_ids, :3].cpu().numpy())).to(dtype=torch.float32, device=self.device)

                    diff = last_value - value
                    diff = torch.clamp(diff, min=-0.01 * (self.cfg.max_vel + 2.0), max=0.01 * (self.cfg.max_vel + 2.0))
                    diff = 10 * diff

                    dijkstra_reward[env_ids] = diff

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

                velocity = self._robot.data.root_lin_vel_w[env_ids]

                h_dot = torch.sum(esd_gradient * velocity, dim=1)

                h = esd - safe_bound
                cbf_reward[env_ids] = h_dot + gamma * h

        cbf_reward = torch.where(
            cbf_reward < 0.0,
            cbf_reward,
            torch.zeros_like(cbf_reward)
        )

        cbf_reward.clamp_(min=-2.0, max=0.0)

        reward_components = torch.stack(
            [
                distance_reward * self.cfg.reward_coef_distance_reward,
                action_magnitude_penalty * self.cfg.reward_coef_action_magnitude_penalty,
                action_change_penalty * self.cfg.reward_coef_action_change_penalty,
                vel_speed_excess_penalty * self.cfg.reward_coef_vel_speed_excess_penalty,
                vel_speed_match_reward * self.cfg.reward_coef_vel_speed_match_reward,
                z_position_penalty * self.cfg.reward_coef_z_position_penalty,
                obstacle_collision_penalty * self.cfg.reward_coef_obstacle_collision_penalty,
                succeed_reward * self.cfg.reward_coef_succeed_reward,
                alive_reward * self.cfg.reward_coef_alive_reward,
                dijkstra_reward * self.cfg.reward_coef_dijkstra,
                cbf_reward * self.cfg.reward_coef_cbf,
            ],
            dim=-1
        )

        total_reward = torch.sum(reward_components, dim=1)

        debug = torch.where(self._episode_sums["obstacle_collision_penalty"] < 0)
        if debug[0].numel() > 0:
            print("debug: ",debug)
            raise ValueError("debug")

        for (key, idx) in zip(self._episode_sums.keys(), range(reward_components.shape[1])):

            self._episode_sums[key] = self._episode_sums[key] + reward_components[:, idx]

        self._last_pos_w = pos_w.clone()
        self._last_actions = self._actions.clone()
        end.record()
        torch.cuda.synchronize()

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        self.CHECK_state()

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

        succeed_mask = self._robot.data.root_state_w[:, :3][:, 0] > self.env_origins[:, 0] + self.cfg.success_threshold

        self._is_success = torch.logical_or(self._is_success, succeed_mask.bool())
        conditions = [
            self._numerical_is_unstable,
            self._is_contact,
            self._robot.data.root_pos_w[:, 2] < self.cfg.too_low,
            self._robot.data.root_pos_w[:, 2] > self.cfg.too_high,
            self._is_success,
        ]

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

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        success_mask = self._is_success[env_ids]
        died_mask = torch.logical_and(self.reset_terminated[env_ids], ~success_mask)
        timed_out_mask = self.reset_time_outs[env_ids]

        self._update_episode_outcomes_and_metrics(env_ids, success_mask, died_mask, timed_out_mask)

        extras = dict()
        for key in self._episode_sums.keys():
            extras["Episode_Reward_Avg/" + key] = torch.mean(self._episode_sums[key][env_ids])
            self._episode_sums[key][env_ids] = 0.0

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)

        super()._reset_idx(env_ids)

        self._actions[env_ids] = torch.zeros(self.cfg.action_space, device=self.device)
        self._action_history[env_ids] = 0.0
        self._valid_mask[env_ids] = False

        # 重置低层动作记录
        if env_ids is None:
            self._last_low_level_actions.zero_()
        else:
            self._last_low_level_actions[env_ids] = 0.0

        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2])

        min_obstacle_distance = 0.3
        max_attempts = 20

        if self.occ_kdtree is not None:

            original_indices = torch.arange(len(env_ids), device=self.device)
            need_valid_position = torch.ones(len(env_ids), dtype=torch.bool, device=self.device)

            attempts = 0
            while torch.any(need_valid_position) and attempts < max_attempts:

                current_mask = need_valid_position
                current_indices = original_indices[current_mask]
                current_env_ids = env_ids[current_mask]

                if len(current_env_ids) == 0:
                    break

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

                distances, _ = self.occ_kdtree.query(new_positions.cpu(), workers=-1, distance_upper_bound=self.cfg.distance_upper_bound)
                distances = torch.tensor(distances, device=self.device)

                valid_mask = distances >= min_obstacle_distance

                if torch.any(valid_mask):

                    self._desired_pos_w[current_env_ids[valid_mask]] = new_positions[valid_mask]

                    need_valid_position[current_indices[valid_mask]] = False

                attempts += 1

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

                    self._desired_pos_w[remaining_env_ids] = new_positions

        else:

            self._desired_pos_w[env_ids, :3] = self.env_origins[env_ids]
            if self.cfg.enable_dijkstra:
                self._desired_pos_w[env_ids, 0] += torch.zeros_like(self._desired_pos_w[env_ids, 0]) + (self.cfg.terrain_length + self.cfg.success_threshold) / 2.0
                self._desired_pos_w[env_ids, 1] += torch.zeros_like(self._desired_pos_w[env_ids, 1]) + self.cfg.terrain_width / 2.0
                self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]) + (self.cfg.desired_low + self.cfg.desired_high) / 2.0
            else:
                self._desired_pos_w[env_ids, 0] += torch.zeros_like(self._desired_pos_w[env_ids, 0]).uniform_(self.cfg.success_threshold, self.cfg.terrain_length)
                self._desired_pos_w[env_ids, 1] += torch.zeros_like(self._desired_pos_w[env_ids, 1]).uniform_(1.0, self.cfg.terrain_width - 1.0)
                self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(self.cfg.desired_low, self.cfg.desired_high)

        self._desired_vel[env_ids] = torch.zeros_like(self._desired_vel[env_ids]).uniform_(0.5, 0.8)

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()

        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] = self.env_origins[env_ids].clone()

        if not self.cfg.random_init_pos or len(self._maps) == 0:

            default_root_state[:, :3] += torch.tensor([self.cfg.start_x, 0.0, 0.0], device=self.device)
            default_root_state[:, 1] += torch.zeros_like(default_root_state[:, 1]).uniform_(1.0, self.cfg.terrain_width - 1.0)
            default_root_state[:, 2] += torch.zeros_like(default_root_state[:, 2]).uniform_(self.cfg.desired_low, self.cfg.desired_high)

        else:

            original_indices = torch.arange(len(env_ids), device=self.device)
            need_valid_position = torch.ones(len(env_ids), dtype=torch.bool, device=self.device)

            attempts = 0
            while torch.any(need_valid_position) and attempts < max_attempts:

                current_mask = need_valid_position
                current_indices = original_indices[current_mask]

                if len(current_indices) == 0:
                    break

                new_positions = default_root_state[current_indices, :3].clone()
                new_positions[:, 0] += torch.zeros_like(new_positions[:, 0]).uniform_(self.cfg.start_x, self.cfg.success_threshold - 10.0)
                new_positions[:, 1] += torch.zeros_like(new_positions[:, 1]).uniform_(1.0, self.cfg.terrain_width - 1.0)
                new_positions[:, 2] += torch.zeros_like(new_positions[:, 2]).uniform_(self.cfg.desired_low, self.cfg.desired_high)

                if self.cfg.enable_dijkstra and self.occ_kdtree is None:
                    valid_mask = torch.zeros_like(current_indices, dtype=torch.bool, device=self.device)
                    current_env_idx = self.env_idx[env_ids[current_indices]]
                    for i, env_idx in enumerate(current_env_idx):
                        is_free, _ = self._maps[env_idx].check_positions_occupancy(new_positions[i].cpu().numpy())
                        valid_mask[i] = bool(is_free)

                else:
                    distances, _ = self.occ_kdtree.query(new_positions.cpu(), workers=-1, distance_upper_bound=self.cfg.distance_upper_bound)
                    distances = torch.tensor(distances, device=self.device)

                    valid_mask = distances >= 0.4

                if torch.any(valid_mask):

                    default_root_state[current_indices[valid_mask], :3] = new_positions[valid_mask]

                    need_valid_position[current_indices[valid_mask]] = False

                attempts += 1
            if any(need_valid_position):
                remaining_indices = original_indices[need_valid_position]
                if len(remaining_indices) > 0:
                    new_positions = torch.zeros_like(default_root_state[remaining_indices, :3])
                    new_positions[:, 0] = torch.zeros_like(new_positions[:, 0]).uniform_(self.cfg.start_x, self.cfg.start_x)
                    new_positions[:, 1] = torch.zeros_like(new_positions[:, 1]).uniform_(1.0, self.cfg.terrain_width - 1.0)
                    new_positions[:, 2] = torch.zeros_like(new_positions[:, 2]).uniform_(self.cfg.desired_low, self.cfg.desired_high)
                    default_root_state[remaining_indices, :3] += new_positions

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._last_pos_w[env_ids] = default_root_state[:, :3]
        self._last_actions[env_ids] = torch.zeros(self.cfg.action_space, device=self.device)
        self._is_contact[env_ids] = False
        self._numerical_is_unstable[env_ids] = False
        self._is_success[env_ids] = False

        self._episode_outcomes[env_ids] = 0

        if (time.time() - self._map_generation_timer) > 3600 * 24 * 10:
            self._calc_env_origins()
            self._regenerate_terrain()
            self._map_generation_timer = time.time()

        # [重要] 重置低层网络的隐状态
        # 构造一个 dones 掩码传给模型的 reset 方法
        low_level_dones = torch.zeros(self.num_envs, device=self.device)
        low_level_dones[env_ids] = 1.0
        self.low_level_policy.reset(dones=low_level_dones)

    def _set_debug_vis_impl(self, debug_vis: bool):

        print(f"debug_vis: {self.cfg.debug_vis}")

        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (self.cfg.marker_size, self.cfg.marker_size, self.cfg.marker_size)

                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
                print("Created goal_pos_visualizer")

            self.goal_pos_visualizer.set_visibility(True)

            if not hasattr(self, "current_yaw_visualizer"):
                current_arrow_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                current_arrow_cfg.markers["arrow"].scale = (self.cfg.marker_size, self.cfg.marker_size, self.cfg.marker_size*4)

                current_arrow_cfg.prim_path = "/Visuals/Command/current_yaw"
                self.current_yaw_visualizer = VisualizationMarkers(current_arrow_cfg)
                print("Created current_yaw_visualizer")

            self.current_yaw_visualizer.set_visibility(True)

            if not hasattr(self, 'traj_visualizer'):
                print("create trajectory visualizer")
                traj_cfg = VisualizationMarkersCfg(
                    markers={
                        "sphere": sim_utils.SphereCfg(
                            radius=0.05,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(1.0, 0.0, 0.0),
                            ),
                        ),
                    },
                    prim_path = "/Visuals/Trajectory"
                )

                self.traj_visualizer = VisualizationMarkers(traj_cfg)
                print("Created traj_visualizer")
            self.traj_visualizer.set_visibility(True)
            if not hasattr(self, "dilation_visualizer"):
                print("create dilation visualizer")
                dilation_cfg = VisualizationMarkersCfg(
                    markers={
                        "dilation": sim_utils.CuboidCfg(
                            size=(0.1, 0.1, 0.1),
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.0, 0.5, 0.5),
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
            if hasattr(self, "current_yaw_visualizer"):
                self.current_yaw_visualizer.set_visibility(False)
            if hasattr(self, 'traj_visualizer'):
                self.traj_visualizer.set_visibility(False)
            if hasattr(self, "dilation_visualizer"):
                self.dilation_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):

        self.goal_pos_visualizer.visualize(self._desired_pos_w)
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

        self._episode_outcomes[env_ids] = torch.where(
            success_mask,
            torch.tensor(self.EpisodeOutcome.SUCCESS, device=self.device),
            torch.where(
                died_mask,
                torch.tensor(self.EpisodeOutcome.FAILURE, device=self.device),
                self._episode_outcomes[env_ids]
            )
        )

        completed_mask = torch.logical_or(torch.logical_or(success_mask, died_mask), timed_out_mask)
        if not torch.any(completed_mask):
            return 0, 0

        completed_env_ids = env_ids[completed_mask]
        success_env_ids = env_ids[success_mask]
        died_env_ids = env_ids[died_mask]

        completed_success = success_mask[completed_mask]

        outcomes = (completed_success.cpu() == True).tolist()
        outcomes = [self.EpisodeOutcome.SUCCESS if success else self.EpisodeOutcome.FAILURE for success in outcomes]

        if len(died_env_ids) > 0:

            is_unstable = self._numerical_is_unstable[died_env_ids].cpu().numpy()
            is_collision = self._is_contact[died_env_ids].cpu().numpy()
            pos_z = self._robot.data.root_pos_w[died_env_ids, 2].cpu().numpy()
            too_low = (pos_z < self.cfg.too_low)
            too_high = (pos_z > self.cfg.too_high)

            for i in range(len(died_env_ids)):
                self._termination_reason_history.append({
                    "numerical_is_unstable": bool(is_unstable[i]),
                    "collision": bool(is_collision[i]),
                    "too_low": bool(too_low[i]),
                    "too_high": bool(too_high[i])
                })

        self._termination_reason_history.extend([{}] * (len(success_env_ids) + len(env_ids[timed_out_mask])))

        if len(completed_env_ids) > 0:
            distances = torch.linalg.norm(
                self._desired_pos_w[completed_env_ids] - self._robot.data.root_pos_w[completed_env_ids],
                dim=1
            ).cpu().tolist()

            self._final_distances.extend(distances)

        if len(completed_env_ids) > 0:
            vel_abs = torch.linalg.norm(
                self._robot.data.root_lin_vel_w[completed_env_ids],
                dim=1
            ).cpu().tolist()

            self._vel_abs.extend(vel_abs)

        self._episode_outcome_history.extend(outcomes)

        num_outcomes = len(self._episode_outcome_history)
        if num_outcomes > 0:

            outcome_array = np.array(list(self._episode_outcome_history))
            success_count = np.sum(outcome_array == self.EpisodeOutcome.SUCCESS)
            died_count = np.sum(outcome_array == self.EpisodeOutcome.FAILURE)
            timeout_count = num_outcomes - success_count - died_count

            self._success_rate = success_count / num_outcomes

            reason_keys = ["numerical_is_unstable", "collision", "too_low", "too_high"]
            reason_counts = {key: 0 for key in reason_keys}

            if len(self._termination_reason_history) > 0:

                batch_size = 2000
                for i in range(0, len(self._termination_reason_history), batch_size):
                    batch = list(itertools.islice(self._termination_reason_history, i, i + batch_size))
                    for key in reason_keys:
                        reason_counts[key] += sum(1 for reason in batch if key in reason and reason[key])

            completed_count = len(outcomes)
            succeeded_count = sum(1 for o in outcomes if o == self.EpisodeOutcome.SUCCESS)
            self._episodes_completed += completed_count
            self._episodes_succeeded += succeeded_count
            cumulative_success_rate = self._episodes_succeeded / self._episodes_completed if self._episodes_completed > 0 else 0.0

            avg_final_distance = np.mean(list(self._final_distances)) if self._final_distances else 0.0

            avg_velocity = np.mean(list(self._vel_abs)) if self._vel_abs else 0.0

            if "log" not in self.extras:
                self.extras["log"] = {}

            self.extras["log"].update({

                "Episode_Termination/died": died_count / num_outcomes * 100.0,
                "Episode_Termination/time_out": timeout_count / num_outcomes * 100.0,
                "Episode_Termination/success": success_count / num_outcomes * 100.0,

                "Metrics/Died/numerical_is_unstable": reason_counts["numerical_is_unstable"] / num_outcomes * 100.0,
                "Metrics/Died/collision": reason_counts["collision"] / num_outcomes * 100.0,
                "Metrics/Died/too_low": reason_counts["too_low"] / num_outcomes * 100.0,
                "Metrics/Died/too_high": reason_counts["too_high"] / num_outcomes * 100.0,

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

        super().close()
