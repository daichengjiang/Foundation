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
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.sim.schemas import RigidBodyPropertiesCfg, CollisionPropertiesCfg
import isaacsim.core.utils.prims as prims_utils
import isaacsim.core.utils.stage as stage_utils
from isaacsim.asset.gen.omap.bindings import _omap
from isaacsim.asset.gen.omap.utils import compute_coordinates, generate_image, update_location
from scipy.spatial import KDTree
import random
from pxr import UsdGeom, Sdf, Gf, Vt
import open3d as o3d
from scipy.spatial import KDTree
import time
import os
from .pcd2usd import load_pcd_as_points

class MapGenerator:
    """
    Class to generate maps with walls and obstacles for drone environments.
    """
    def __init__(self, sim, device="cuda:0"):
        self.device = device
        self.sim = sim

    def generate_occupancy_map(self, scene, terrain_length, terrain_width, cols, rows):
        """
        Generate an occupancy map for the environment.
        
        Args:
            scene: Scene configuration with env_spacing parameter
            
        Returns:
            KDTree of occupied points
        """
        # Occupancy map
        physx = omni.physx.acquire_physx_interface()
        stage_id = omni.usd.get_context().get_stage_id()
        occ_generator = _omap.Generator(physx, stage_id)
        
        # Use a larger voxel size for better performance (0.2m → 0.3m)
        # Adjust these parameters based on your environment size and required precision
        voxel_size = 0.3  # Larger voxel size for better performance
        padding = 4      # Extra padding around the volume
        max_depth = 5    # Max tree depth 
        
        occ_generator.update_settings(voxel_size, padding, max_depth, 6)
        
        # Set location to map from and the min and max bounds to map to
        occ_generator.set_transform(
            (0, 0, 0), 
            (0, 0, -0.1),
            (terrain_length*cols, terrain_width*rows, 6.0)
        )
        
        # Generate the 3D occupancy map
        occ_generator.generate3d()
        points = occ_generator.get_occupied_positions()
        print(f"Raw occupied points: {len(points)}")
        
        # Handle empty points case
        if len(points) == 0:
            print("No points found in the occupancy map.")
            points = np.array([[0, 0, 0]])
            return KDTree(points), points
        
        # Convert to numpy array for faster processing
        points_np = np.array(points)
        
        # Create KDTree for efficient nearest neighbor queries
        kdtree = KDTree(points_np)
        
        return kdtree, points_np
        
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

        # Backward wall
        backward_wall_cfg = sim_utils.CuboidCfg(
            size=((terrain_width * rows) + 1, 0.1, 6.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.4, 0.8)),
            rigid_props=RigidBodyPropertiesCfg(rigid_body_enabled=False, kinematic_enabled=True),
            collision_props=CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.1),
            activate_contact_sensors=True,
        )

        ceiling_size = (terrain_length * cols, terrain_width * rows, 0.1)
        ceiling_path = "/World/ground/ceiling"
        ceiling_cfg = sim_utils.CuboidCfg(
            size=ceiling_size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0), opacity=0.0),
            rigid_props=RigidBodyPropertiesCfg(rigid_body_enabled=False, kinematic_enabled=True),
            collision_props=CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.1),
            activate_contact_sensors=True,
        )
        ceiling_cfg.func(ceiling_path, ceiling_cfg, translation=(terrain_length * cols / 2.0, terrain_width * rows / 2.0, 2.0))
        wall_prims.append(ceiling_path)

        for i in range(cols+1):
            wall_path = "/World/ground/backward_wall_{}".format(i)
            backward_wall_cfg.func(wall_path, backward_wall_cfg, translation=(terrain_length * i , terrain_width * rows / 2.0, 3.0), orientation=(0.707, 0.0, 0.0, 0.707))
            wall_prims.append(wall_path)
        
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

    def generate_lights(self, scene, terrain_length, terrain_width, cols, rows):
        light_cfg = sim_utils.LightCfg(
            intensity=4000.0,
            color=(0.8, 0.8, 0.8),
            translation=(terrain_length * cols / 2.0, terrain_width * rows / 2.0, 6.0),
            orientation=(0.8939967, 0.0, 0.0, -0.4480736),
            scale=(terrain_length * cols, terrain_width * rows, 6.0)
        )
        light_cfg.func("/World/ground/light", light_cfg)

    def create_environment(self, scene, terrain: TerrainImporterCfg, **kwargs):
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
        rows = kwargs.get("grid_rows", 6)
        cols = kwargs.get("grid_cols", 8)
        walls = self.generate_walls(scene, terrain_length, terrain_width, cols, rows)
        print(f"Walls generation: {time.time() - walls_time:.3f} seconds")

        # light_time = time.time()
        # light = self.generate_lights(scene, terrain_length, terrain_width, cols, rows)
        # print(f"Light generation: {time.time() - light_time:.3f} seconds")

        # terrain.import_usd(name="terrains", usd_path="./USD/combined_output_grid.usd")
        terrain_path = kwargs.get("terrain_path", "./USD/terrain")
        usd_path = terrain_path+".usd"
        pcd_path = terrain_path+".pcd"
        # terrain.import_usd(name="terrains", usd_path=usd_path)
        # from isaaclab.sim.spawners.from_files import UsdFileCfg
        # from isaaclab.sim.spawners.from_files import spawn_from_usd
        # usd_cfg = UsdFileCfg(
        #     usd_path= os.path.join("./USD", "combined_output_grid.usd"),
        #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.4, 0.4)),
        #     rigid_props=RigidBodyPropertiesCfg(rigid_body_enabled=False, kinematic_enabled=True),
        #     collision_props=CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.1),
        #     activate_contact_sensors=True,
        # )
        # spawn_from_usd(prim_path="/World/ground", cfg=usd_cfg, translation=(0, 0, 0), orientation=(1, 0, 0, 0))
        
        # pcd = o3d.io.read_point_cloud(os.path.join("./USD", "combined_output_grid.pcd"))
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)
        occ_kdtree = KDTree(points)

        stage = stage_utils.get_current_stage()
        load_pcd_as_points(stage, "/World/PCDPoints", pcd_path,
                    point_size=math.sqrt(3) * 0.1, res=0.05, tx=0.0, ty=0.0, tz=0.0,
                    use_height_color=True, colormap='rainbow')
        # occ_kdtree = None  # Placeholder for occupancy map, to be generated later
        
        # # Generate occupancy map
        # occ_time = time.time()
        # occ_kdtree, points = self.generate_occupancy_map(scene, terrain_length, terrain_width, cols, rows)
        # print(f"Occupancy map generation: {time.time() - occ_time:.3f} seconds")

        # # Create an Open3D point cloud object
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)

        # # Save to PCD file
        # o3d.io.write_point_cloud("./USD/output.pcd", pcd)
        self.sim.play()
        
        total_time = time.time() - start_time
        print(f"Total environment generation time: {total_time:.3f} seconds")

        return {
            "ground_plane": ground_plane,
            "walls": walls,
            "kdtree": occ_kdtree,
            "generation_time": total_time
        }