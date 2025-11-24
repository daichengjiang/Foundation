# Copyright (c) 2025 Xu Yang
# HKUST UAV Group
#
# Author: Xu Yang
# Affiliation: HKUST UAV Group
# Date: April 2025
# License: MIT License

import math
import omni
import torch
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.sim.schemas import RigidBodyPropertiesCfg, CollisionPropertiesCfg
import isaacsim.core.utils.prims as prims_utils
from isaacsim.asset.gen.omap.bindings import _omap
from isaacsim.asset.gen.omap.utils import compute_coordinates, generate_image, update_location
from scipy.spatial import KDTree
import random
from pxr import UsdGeom, Sdf, Gf, Vt
import time

class MapGenerator:
    """
    Class to generate maps with walls and obstacles for drone environments.
    """
    def __init__(self, sim, device="cuda:0"):
        self.device = device
        self.sim = sim
        
    def generate_walls(self, scene, terrain_length, terrain_width, cols, rows):
        """
        Generate walls around the environment.
        
        Args:
            scene: Scene configuration with env_spacing parameter
        
        Returns:
            List of wall primitive paths
        """
        wall_prims = []

        # Walls
        wall_cfg = sim_utils.CuboidCfg(
            size=(terrain_length * cols, 0.1, 6.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.4, 0.8)),
            rigid_props=RigidBodyPropertiesCfg(rigid_body_enabled=False, kinematic_enabled=True),
            collision_props=CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.1),
            activate_contact_sensors=True,
        )
        
        for i in range(rows+1):
            wall_path = "/World/ground/wall_{}".format(i)
            wall_cfg.func(wall_path, wall_cfg, translation=(terrain_length * cols / 2.0, terrain_width * i, 3.0))
            wall_prims.append(wall_path)

        # Forward wall
        # forward_wall_path = "/World/ground/forward_wall"
        # wall_cfg.func(forward_wall_path, wall_cfg, translation=(scene.env_spacing / 2.0 + 2.0, 0.0, 2.0), orientation=(0.707, 0.0, 0.0, 0.707))
        # wall_prims.append(forward_wall_path)
        
        # Backward wall
        backward_wall_cfg = sim_utils.CuboidCfg(
            size=((terrain_width * rows) + 1, 0.1, 6.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.4, 0.8)),
            rigid_props=RigidBodyPropertiesCfg(rigid_body_enabled=False, kinematic_enabled=True),
            collision_props=CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.1),
            activate_contact_sensors=True,
        )

        for i in range(cols+1):
            wall_path = "/World/ground/backward_wall_{}".format(i)
            backward_wall_cfg.func(wall_path, backward_wall_cfg, translation=(terrain_length * i , terrain_width * rows / 2.0, 3.0), orientation=(0.707, 0.0, 0.0, 0.707))
            wall_prims.append(wall_path)
        
        # Ceiling
        # ceiling_path = "/World/ground/ceiling"
        # wall_cfg.func(ceiling_path, wall_cfg.replace(size=(scene.env_spacing * 1.5, scene.env_spacing * 1.5, 0.1)), translation=(0.0, 0.0, 2.5))
        # wall_prims.append(ceiling_path)

        # Floor
        # floor_path = "/World/ground/floor"
        # wall_cfg.func(floor_path, wall_cfg.replace(size=(scene.env_spacing * 1.5, scene.env_spacing * 1.5, 0.1), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.33, 0.66))), translation=(0.0, 0.0, -0.05))
        # wall_prims.append(floor_path)
        
        return wall_prims

    def generate_ground_plane(self, scene, size=(10, 10), translation=(0, 0, 0)):
        """
        Generate a ground plane for the environment.
        
        Args:
            scene: Scene configuration with env_spacing parameter
            
        Returns:
            Ground plane primitive path
        """
        ground_path = "/World/ground/defaultGroundPlane"
        cfg_ground = sim_utils.GroundPlaneCfg(
            color=(0.1, 0.1, 0.1),
            size=size
        )
        cfg_ground.func(ground_path, cfg_ground, translation=translation)

        return ground_path

    def generate_lights(self, scene):
        light_cfg = sim_utils.LightCfg(
            intensity=4000.0,
            color=(0.8, 0.8, 0.8),
            translation=(65*8/2.0-5, 20*6/2.0, 0),
            orientation=(0.8939967, 0.0, 0.0, -0.4480736),
            scale=(65*10, 20*8, 6.0),
        )
        light_cfg.func("/World/ground/light", light_cfg)

    def create_environment(self, scene, **kwargs):
        self.sim.pause()
        print("Generating environment...")
        start_time = time.time()

        # Generate ground plane
        ground_time = time.time()
        size = kwargs.get("plane_size", (65*10, 20*8))
        translation = kwargs.get("plane_translation", (65*8/2.0, 20*6/2.0, 0))
        ground_plane = self.generate_ground_plane(scene, size=size, translation=translation)
        print(f"Ground plane generation: {time.time() - ground_time:.3f} seconds")
        
        # Generate walls
        walls_time = time.time()
        terrain_length = kwargs.get("terrain_length", 65)
        terrain_width = kwargs.get("terrain_width", 20)
        rows = kwargs.get("rows", 6)
        cols = kwargs.get("cols", 8)
        walls = self.generate_walls(scene, terrain_length, terrain_width, cols, rows)
        print(f"Walls generation: {time.time() - walls_time:.3f} seconds")

        # light_time = time.time()
        # light = self.generate_lights(scene)
        # print(f"Light generation: {time.time() - light_time:.3f} seconds")
        
        self.sim.play()
        
        total_time = time.time() - start_time
        print(f"Total environment generation time: {total_time:.3f} seconds")

        return {
            "ground_plane": ground_plane,
            "walls": walls,
            "kdtree": None,
            "generation_time": total_time
        }