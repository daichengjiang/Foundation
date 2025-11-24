# Copyright (c) 2025 Xu Yang
# HKUST UAV Group
#
# Author: Xu Yang
# Affiliation: HKUST UAV Group
# Date: April 2025
# License: MIT License

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.colors import Normalize
import time
from collections import deque
import math
import random
import threading
import concurrent.futures
from scipy.spatial import KDTree
import textwrap

class DeathReplay:
    """
    Records and visualizes drone flight data, including trajectories and sensor readings.
    Used for analyzing both successful flights and failures.
    """
    
    def __init__(self, 
                 num_envs,
                 tof_width=8,
                 tof_height=8,
                 history_capacity=1000,
                 save_dir="/workspace/isaaclab/logs/death_replay",
                 drone_size=0.15,
                 camera_fov=45,
                 trajectory_spacing=1.0,
                 final_frames_to_show=5,
                 tof_frame_interval=5,
                 visualization_num=10,
                 stats_window=100,
                 device="cuda"):
        """
        Initialize the death replay recorder.
        
        Args:
            num_envs: Number of environments to track
            tof_width: Width of ToF depth image
            tof_height: Height of ToF depth image
            history_capacity: Maximum number of frames to store per environment
            save_dir: Directory to save visualizations
            drone_size: Size of drone for visualization (meters)
            camera_fov: Field of view angle of the drone camera (degrees)
            trajectory_spacing: Minimum distance between visualized drone positions (meters)
            final_frames_to_show: Number of final frames to always show
            tof_frame_interval: Interval between ToF frames to visualize
            visualization_num: Number of environments to randomly select for recording and visualization
            stats_window: Number of recent episodes to track for success statistics
            device: Device for tensor computations
        """
        # Parameters
        self.num_envs = num_envs
        self.tof_width = tof_width
        self.tof_height = tof_height
        self.history_capacity = history_capacity
        self.save_dir = save_dir
        self.drone_size = drone_size
        self.camera_fov = camera_fov
        self.trajectory_spacing = trajectory_spacing
        self.final_frames_to_show = final_frames_to_show
        self.tof_frame_interval = tof_frame_interval
        self.visualization_num = min(visualization_num, num_envs)  # Ensure we don't exceed num_envs
        self.device = device
        
        # Statistics tracking
        self.stats_window = stats_window
        self.recent_episode_outcomes = deque(maxlen=stats_window)  # 1 for success, 0 for failure
        self.total_episodes = 0
        self.total_successes = 0
        
        # Create directories
        self.success_dir = os.path.join(save_dir, "success")
        self.failure_dir = os.path.join(save_dir, "failure")
        os.makedirs(self.success_dir, exist_ok=True)
        os.makedirs(self.failure_dir, exist_ok=True)
        
        # Select environments to track (randomly)
        self.selected_envs = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._select_environments()
        
        # State tracking
        self.is_recording = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.episode_count = 0
        self.success_count = 0
        self.failure_count = 0
        
        # Data storage - use deques for efficient append/pop operations
        # Only allocate memory for selected environments
        self.history = {
            "pos": [deque(maxlen=history_capacity) if i else None for i in self.selected_envs.cpu().tolist()],
            "rot": [deque(maxlen=history_capacity) if i else None for i in self.selected_envs.cpu().tolist()],
            "tof": [deque(maxlen=history_capacity) if i else None for i in self.selected_envs.cpu().tolist()],
        }
        self.episode_outcome = torch.zeros(num_envs, dtype=torch.long, device=device)  # 0=ongoing, 1=success, 2=failure
        
        # Map data
        self.global_map = None
        self.map_2d = None
        
        # Create thread pool for parallel visualization processing
        self.visualization_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.visualization_futures = []
        
        # Store target positions for visualization
        self.target_positions = None
        
        # Store KD-tree for collision detection
        self.obstacle_kdtree = None
        
        print(f"Death replay initialized with {self.visualization_num} selected environments. " 
              f"Visualizations will be saved to {save_dir} (using {4} worker threads)")
        print(f"Tracking success statistics over the last {stats_window} episodes")
    
    def _select_environments(self):
        """Randomly select environments to track"""
        # Reset selection
        self.selected_envs.fill_(False)
        
        # Randomly select visualization_num environments
        selected_indices = random.sample(range(self.num_envs), self.visualization_num)
        self.selected_envs[selected_indices] = True
    
    def reset_episode(self, env_ids):
        """
        Reset history for the given environment IDs and start recording.
        Also prints success statistics.
        
        Args:
            env_ids: Tensor of environment IDs to reset
        """
        # Print success statistics if we have data
        if len(self.recent_episode_outcomes) > 0:
            recent_success_count = sum(self.recent_episode_outcomes)
            recent_success_rate = recent_success_count / len(self.recent_episode_outcomes) * 100
            overall_success_rate = (self.total_successes / self.total_episodes) * 100 if self.total_episodes > 0 else 0
            
            print(f"Success statistics: {recent_success_rate:.2f}% over last {len(self.recent_episode_outcomes)} episodes "
                  f"({recent_success_count}/{len(self.recent_episode_outcomes)}), "
                  f"{overall_success_rate:.2f}% overall ({self.total_successes}/{self.total_episodes} episodes)")
        
        # Only process selected environments that are being reset
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[env_ids] = True
        selected_reset_mask = torch.logical_and(mask, self.selected_envs)
        selected_reset_indices = torch.nonzero(selected_reset_mask).flatten().cpu().tolist()
        
        for env_id in selected_reset_indices:
            # Clear history for this environment
            self.history["pos"][env_id].clear()
            self.history["rot"][env_id].clear()
            self.history["tof"][env_id].clear()
            self.episode_outcome[env_id] = 0  # Mark as ongoing
        
        # Set recording flag for all reset environments
        self.is_recording[env_ids] = True
    
    def record_frame(self, pos_w, rot_w, tof_data):
        """
        Record a frame of data only for selected active environments.
        
        Args:
            pos_w: Position tensor of shape (num_envs, 3) in world frame
            rot_w: Rotation tensor of shape (num_envs, 3) as roll, pitch, yaw in world frame
            tof_data: ToF depth data tensor of shape (num_envs, H, W)
        """
        # Combine selection mask with recording mask to avoid unnecessary processing
        active_selection_mask = torch.logical_and(self.selected_envs, self.is_recording)
        active_env_indices = torch.nonzero(active_selection_mask).flatten().cpu().tolist()
        
        # Only process selected and active environments
        for env_id in active_env_indices:
            # Check for NaN values to avoid storing corrupt data
            if (torch.isnan(pos_w[env_id]).any() or 
                torch.isnan(rot_w[env_id]).any() or 
                torch.isnan(tof_data[env_id]).any()):
                print(f"Warning: NaN values detected for env {env_id}, skipping this frame")
                continue
                
            self.history["pos"][env_id].append(pos_w[env_id].clone().cpu())
            self.history["rot"][env_id].append(rot_w[env_id].clone().cpu())
            self.history["tof"][env_id].append(tof_data[env_id].clone().cpu())
    
    def end_episodes(self, completed_mask, success_mask):
        """
        End recording for completed episodes and mark their outcomes.
        Submit visualizations to the thread pool and wait for completion.
        
        Args:
            completed_mask: Boolean tensor indicating which environments completed an episode
            success_mask: Boolean tensor indicating which environments completed successfully
        """
        # Identify completed environments that we're tracking
        selected_completed_mask = torch.logical_and(completed_mask, self.selected_envs)
        selected_completed_indices = torch.nonzero(selected_completed_mask).flatten().cpu().tolist()
        
        # Create a list to hold futures for this batch of visualization tasks
        current_futures = []
        
        # Count all completed episodes for statistics
        all_completed_indices = torch.nonzero(completed_mask).flatten().cpu().tolist()
        
        # Update global statistics
        for env_id in all_completed_indices:
            self.total_episodes += 1
            if success_mask[env_id]:
                self.total_successes += 1
                self.recent_episode_outcomes.append(1)
            else:
                self.recent_episode_outcomes.append(0)
        
        # Process selected environments for visualization
        for env_id in selected_completed_indices:
            self.is_recording[env_id] = False
            
            if success_mask[env_id]:
                self.episode_outcome[env_id] = 1  # Success
                self.success_count += 1
                # Submit visualization task to thread pool
                future = self.visualization_pool.submit(self._visualize_episode_wrapper, env_id, True)
                current_futures.append(future)
            else:
                self.episode_outcome[env_id] = 2  # Failure
                self.failure_count += 1
                # Submit visualization task to thread pool
                future = self.visualization_pool.submit(self._visualize_episode_wrapper, env_id, False)
                current_futures.append(future)
        
        # Wait for all visualizations to complete
        for future in current_futures:
            future.result()  # This blocks until the future completes
        
        # Update episode count
        self.episode_count += len(selected_completed_indices)
    
    def set_target_positions(self, positions):
        """
        Set target positions for visualization.
        
        Args:
            positions: Tensor of shape (num_envs, 3) containing target x,y,z positions
        """
        if positions is None:
            self.target_positions = None
            return
            
        if isinstance(positions, torch.Tensor):
            positions = positions.clone().cpu()
        self.target_positions = positions
    
    def set_global_map(self, point_cloud, kdtree=None):
        """
        Set the global map from a point cloud and optionally a pre-built KD-tree.
        
        Args:
            point_cloud: Tensor, numpy array, or list of points of shape (N, 3) containing x, y, z coordinates
            kdtree: Optional pre-built KD-tree for the point cloud
        """
        # Convert list to numpy array if needed
        if isinstance(point_cloud, list):
            point_cloud = np.array(point_cloud)
            
        # Convert numpy array to tensor if needed
        if isinstance(point_cloud, np.ndarray):
            point_cloud = torch.tensor(point_cloud, device=self.device)
            
        self.global_map = point_cloud.clone().cpu()
        self.map_2d = self._convert_to_2d_map(point_cloud)
        
        # Store the KD-tree if provided, otherwise build it
        if kdtree is not None:
            self.obstacle_kdtree = kdtree
        else:
            # Build KD-tree from point cloud for collision detection
            if len(point_cloud) > 0:
                self.obstacle_kdtree = KDTree(point_cloud.cpu().numpy())
    
    def _save_points_to_pcd(self, points, filename):
        if os.path.exists(filename):
            os.remove(filename)
        num_points = len(points)
        header = textwrap.dedent(f"""\
            # .PCD v0.7 - Point Cloud Data file format
            VERSION 0.7
            FIELDS x y z
            SIZE 4 4 4
            TYPE F F F
            COUNT 1 1 1
            WIDTH {num_points}
            HEIGHT 1
            VIEWPOINT 0 0 0 1 0 0 0
            POINTS {num_points}
            DATA ascii
            """)
        with open(filename, 'w') as file:
            file.write(header)
            for point in points:
                file.write(f"{point[0]} {point[1]} {point[2]}\n")
    
    def _convert_to_2d_map(self, point_cloud):
        """
        Convert 3D point cloud to 2.5D top-down map.
        
        Args:
            point_cloud: Tensor of shape (N, 3) containing x, y, z coordinates
            
        Returns:
            2.5D map as a dictionary with 'positions' and 'heights'
        """
        # Filter points below 0.1m
        mask = point_cloud[:, 2] >= 0.2
        filtered_points = point_cloud[mask]
        
        if len(filtered_points) == 0:
            return {"positions": torch.zeros((0, 2)), "heights": torch.zeros(0)}
        
        # Extract x,y positions and z heights
        positions = filtered_points[:, :2]  # x,y
        heights = torch.clamp(filtered_points[:, 2], 0.2, 2.8)  # z
        
        # Save points to PCD file for debugging
        # pcd_filename = os.path.join(self.save_dir, "map.pcd")
        # self._save_points_to_pcd(filtered_points.cpu().numpy(), pcd_filename)
        
        return {"positions": positions.cpu(), "heights": heights.cpu()}
    
    def visualize_episode(self, env_id, is_success):
        """
        Create and save a visualization of the episode with performance optimizations.
        
        Args:
            env_id: Environment ID to visualize
            is_success: Whether this was a successful episode
        """
        if not self.selected_envs[env_id] or len(self.history["pos"][env_id]) == 0:
            print(f"No data to visualize for environment {env_id}")
            return
            
        # Check if we have enough data to visualize
        if (len(self.history["pos"][env_id]) == 0 or 
            len(self.history["rot"][env_id]) == 0):
            print(f"Insufficient trajectory data for environment {env_id}")
            return
        
        # Create figure with optimized size for faster rendering
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 14), 
                                               gridspec_kw={'height_ratios': [3, 2]},
                                               dpi=100)  # Lower DPI for faster rendering
        
        # 1. Top subplot: 2.5D map with trajectory
        if self.map_2d is not None:
            # Plot the 2.5D map
            cmap = plt.cm.viridis
            norm = Normalize(0, 4.0)  # Normalize heights to [0,4]
            
            positions = self.map_2d["positions"].numpy()
            heights = self.map_2d["heights"].numpy()
            
            if len(positions) > 0:
                # Downsample map points for faster plotting if there are too many
                # if len(positions) > 80000:
                #     downsample_rate = len(positions) // 10000 + 1
                #     indices = np.random.choice(len(positions), len(positions) // downsample_rate, replace=False)
                #     positions = positions[indices]
                #     heights = heights[indices]

                # Use smaller point size and simplified colorbar
                scatter = ax_top.scatter(
                    positions[:, 0], positions[:, 1],
                    c=heights, cmap=cmap, norm=norm, s=0.3, alpha=0.4
                )
                plt.colorbar(scatter, ax=ax_top, label="Height (m)", shrink=0.8)
        
        # Extract trajectory data
        try:
            # Safely convert deque to list before stacking
            pos_list = list(self.history["pos"][env_id])
            rot_list = list(self.history["rot"][env_id])
            
            if len(pos_list) == 0 or len(rot_list) == 0:
                plt.close(fig)
                return
                
            positions = torch.stack(pos_list).numpy()
            rotations = torch.stack(rot_list).numpy()
            
            # Plot the trajectory
            trajectory_x = positions[:, 0]
            trajectory_y = positions[:, 1]
            trajectory_z = positions[:, 2]
            
            # Downsample trajectory points if there are too many
            if len(trajectory_x) > 1000:
                downsample_rate = len(trajectory_x) // 1000 + 1
                idx = list(range(0, len(trajectory_x), downsample_rate))
                # Always include start and end points
                if idx[-1] != len(trajectory_x) - 1:
                    idx.append(len(trajectory_x) - 1)
                
                trajectory_x = trajectory_x[idx]
                trajectory_y = trajectory_y[idx]
                trajectory_z = trajectory_z[idx]
                positions = positions[idx]
                rotations = rotations[idx]
            
            # Color trajectory by height with smaller point size
            ax_top.scatter(
                trajectory_x, trajectory_y, 
                c=trajectory_z, cmap=cmap, norm=norm, s=0.5, marker='.'
            )
            
            # Connect trajectory points with a line (thinner line for performance)
            ax_top.plot(trajectory_x, trajectory_y, 'k-', linewidth=0.1, alpha=0.5)
            
            # Draw start and end points (smaller markers)
            ax_top.plot(trajectory_x[0], trajectory_y[0], 'go', markersize=4, label='Start')
            
            # Draw target position as a star if available
            if self.target_positions is not None and env_id < len(self.target_positions):
                target_pos = self.target_positions[env_id]
                if target_pos is not None:
                    ax_top.plot(target_pos[0], target_pos[1], marker='*', 
                               markersize=4, color='green', label='Target')
            
            # For failure cases, identify and mark the closest obstacle point
            if not is_success and self.obstacle_kdtree is not None:
                # Get the final position
                final_pos = positions[-1]
                
                # Query the KD-tree to find the closest point
                dist, idx = self.obstacle_kdtree.query(final_pos, k=1)
                
                if dist < 0.5:  # Only show if reasonably close (within 0.5m)
                    closest_point = self.obstacle_kdtree.data[idx]
                    # Mark the collision point with a red dot
                    ax_top.plot(closest_point[0], closest_point[1], 'ro', 
                               markersize=0.2, label='Collision Point')
            
            # Determine which frames to visualize the drone (limit for performance)
            frames_to_visualize = self._get_frames_to_visualize(positions, rotations)
            if len(frames_to_visualize) > 10:  # Limit to 10 drone positions for performance
                stride = len(frames_to_visualize) // 10 + 1
                frames_to_visualize = frames_to_visualize[::stride]
                if 0 not in frames_to_visualize:
                    frames_to_visualize = [0] + frames_to_visualize
                if (len(positions) - 1) not in frames_to_visualize:
                    frames_to_visualize.append(len(positions) - 1)
            
            # Draw drone representations at selected frames
            for frame_idx in frames_to_visualize:
                if frame_idx >= len(positions):
                    continue
                    
                pos = positions[frame_idx]
                rot = rotations[frame_idx]
                
                # Draw a square for the drone
                half_size = self.drone_size / 2
                rect = Rectangle(
                    (pos[0] - half_size, pos[1] - half_size),
                    self.drone_size, self.drone_size,
                    angle=np.degrees(rot[2]),  # Convert yaw to degrees
                    edgecolor='black', facecolor='none', linewidth=0.1  # Thinner line
                )
                ax_top.add_patch(rect)
                
                # Draw the camera FOV as a triangle with simplified style
                self._draw_camera_fov(ax_top, pos, rot)
                
            outcome_text = "SUCCESS" if is_success else "FAILURE"
            ax_top.set_title(f"Trajectory - {outcome_text}", fontsize=12)
            ax_top.set_xlabel("X (m)", fontsize=10)
            ax_top.set_ylabel("Y (m)", fontsize=10)
            ax_top.set_aspect('equal')
            ax_top.grid(True, alpha=0.3)  # Lighter grid
            ax_top.legend(fontsize=8, loc='upper right')
            
        except Exception as e:
            print(f"Error processing trajectory data for env {env_id}: {e}")
        
        # 2. Bottom subplot: ToF observations (limited to 20 frames for performance)
        try:
            # Check if we have ToF data
            if len(self.history["tof"][env_id]) == 0:
                ax_bottom.text(0.5, 0.5, "No ToF data", ha='center', va='center')
            else:
                tof_frames = self._get_tof_frames_to_visualize(len(self.history["tof"][env_id]))
                
                if len(tof_frames) > 0:
                    # Get ToF data
                    tof_list = list(self.history["tof"][env_id])
                    if len(tof_list) > 0:
                        # Only stack the frames we need rather than all frames
                        tof_data = torch.stack([tof_list[i] for i in tof_frames if i < len(tof_list)])
                        
                        # Get ToF data shape
                        if len(tof_data.shape) == 3:  # If shape is [frames, height, width]
                            tof_height, tof_width = tof_data.shape[1], tof_data.shape[2]
                        else:  # If shape is [frames, pixels]
                            tof_height, tof_width = self.tof_height, self.tof_width
                            tof_data = tof_data.reshape(-1, tof_height, tof_width)
                        
                        # Determine grid layout for ToF observations
                        grid_cols = min(5, len(tof_data))  # Max 5 columns
                        grid_rows = math.ceil(len(tof_data) / grid_cols)
                        
                        # Create a grid of ToF visualization images
                        tof_grid = np.zeros((grid_rows * tof_height, grid_cols * tof_width))
                        
                        for i, frame_data in enumerate(tof_data):
                            if i >= grid_rows * grid_cols:
                                break
                                
                            row = i // grid_cols
                            col = i % grid_cols
                            
                            tof_data_frame = frame_data.numpy()
                            
                            # Place in the grid
                            tof_grid[
                                row * tof_height:(row + 1) * tof_height,
                                col * tof_width:(col + 1) * tof_width
                            ] = tof_data_frame
                        
                        # Display the grid with simplified settings
                        tof_img = ax_bottom.imshow(
                            tof_grid, cmap='jet', vmin=0, vmax=4.0, 
                            interpolation='nearest'
                        )
                        # Simplified colorbar
                        plt.colorbar(tof_img, ax=ax_bottom, label="Depth (m)", shrink=0.8)
                        
                        # Add simplified frame numbers
                        for i in range(len(tof_data)):
                            if i >= grid_rows * grid_cols:
                                break
                            
                            row = i // grid_cols
                            col = i % grid_cols
                            
                            # Add simpler text
                            ax_bottom.text(
                                col * tof_width + tof_width // 2,
                                row * tof_height + 2,
                                f"{tof_frames[i]}",
                                color='white', fontsize=6, ha='center'
                            )
                    else:
                        ax_bottom.text(0.5, 0.5, "Empty ToF data", ha='center', va='center')
                else:
                    ax_bottom.text(0.5, 0.5, "No ToF frames", ha='center', va='center')
        except Exception as e:
            print(f"Error processing ToF data for env {env_id}: {e}")
            ax_bottom.text(0.5, 0.5, "Error processing ToF data", ha='center', va='center')
        
        ax_bottom.set_title("ToF Observations", fontsize=12)
        
        # Save the visualization with optimized settings
        try:
            outcome_dir = self.success_dir if is_success else self.failure_dir
            filename = f"ep_{self.episode_count}_env{env_id}_{outcome_text.lower()}_{time.strftime('%Y%m%d_%H%M%S')}.png"
            filepath = os.path.join(outcome_dir, filename)
            
            # Use optimized save settings for performance
            plt.savefig(filepath, 
                       dpi=500,
                       bbox_inches='tight',
                       pad_inches=0.1,     # Less padding
                       transparent=False,  # Don't use transparency
                    )
            
            print(f"Saved visualization to {filepath}")
        except Exception as e:
            print(f"Error saving visualization for env {env_id}: {e}")
        finally:
            plt.close(fig)
    
    def _get_frames_to_visualize(self, positions, rotations):
        """
        Determine which frames to visualize based on trajectory spacing and final frames.
        
        Args:
            positions: NumPy array of drone positions
            rotations: NumPy array of drone rotations
            
        Returns:
            List of frame indices to visualize
        """
        n_frames = len(positions)
        frames_to_visualize = []
        
        # Always include the final frames
        final_frames = min(self.final_frames_to_show, n_frames)
        frames_to_visualize.extend(range(n_frames - final_frames, n_frames))
        
        # Always include the first frame
        if n_frames > 0:
            frames_to_visualize.append(0)
        
        # Add frames spaced by distance
        last_pos = None
        accumulated_distance = 0.0
        
        for i in range(1, n_frames - final_frames):
            current_pos = positions[i]
            
            if last_pos is None:
                last_pos = positions[0]
            
            distance = np.linalg.norm(current_pos[:2] - last_pos[:2])  # XY distance
            accumulated_distance += distance
            
            if accumulated_distance >= self.trajectory_spacing:
                frames_to_visualize.append(i)
                last_pos = current_pos
                accumulated_distance = 0.0
        
        return sorted(list(set(frames_to_visualize)))
    
    def _get_tof_frames_to_visualize(self, n_tof_frames):
        """
        Determine which ToF frames to visualize, including recent frames and limiting history.
        
        Args:
            n_tof_frames: Total number of ToF frames
            
        Returns:
            List of frame indices to visualize (at most 20)
        """
        frames_to_visualize = []
        
        # Special case for when we have very few frames
        if n_tof_frames <= 20:
            return list(range(n_tof_frames))
        
        # Always include the 10 most recent frames
        recent_frame_count = min(10, n_tof_frames)
        frames_to_visualize.extend(range(n_tof_frames - recent_frame_count, n_tof_frames))
        
        # Define how far back we go (maximum 200 frames from the end)
        max_history_frame = max(0, n_tof_frames - 200)
        
        # Calculate how many more frames we need to reach 20 total
        remaining_slots = 20 - len(frames_to_visualize)
        
        if remaining_slots > 0 and max_history_frame < (n_tof_frames - recent_frame_count):
            # Calculate the range we're sampling from
            sample_range = n_tof_frames - recent_frame_count - max_history_frame
            
            # If the range is small, just take all frames
            if sample_range <= remaining_slots:
                frames_to_visualize.extend(range(max_history_frame, n_tof_frames - recent_frame_count))
            else:
                # Calculate stride to distribute the remaining slots evenly across the history
                stride = max(1, sample_range // remaining_slots)
                
                # Add historical frames at regular intervals
                for i in range(max_history_frame, n_tof_frames - recent_frame_count, stride):
                    if len(frames_to_visualize) < 20:  # Safety check to not exceed 20
                        frames_to_visualize.append(i)
        
        # Sort frames and ensure the count is at most 20
        return sorted(frames_to_visualize)[:20]
    
    def _draw_camera_fov(self, ax, pos, rot):
        """
        Draw a simplified camera field of view as a triangle.
        
        Args:
            ax: Matplotlib axis
            pos: Drone position (x, y, z)
            rot: Drone rotation (roll, pitch, yaw)
        """
        yaw = rot[2]  # Yaw angle in radians
        
        # Convert FOV from degrees to radians
        fov_rad = np.radians(self.camera_fov)
        
        # Create the triangle points with shorter projection
        triangle_points = np.array([
            [pos[0], pos[1]],  # Drone position
            [pos[0] + 1.5 * np.cos(yaw - fov_rad/2), pos[1] + 1.5 * np.sin(yaw - fov_rad/2)],  # Left edge of FOV
            [pos[0] + 1.5 * np.cos(yaw + fov_rad/2), pos[1] + 1.5 * np.sin(yaw + fov_rad/2)],  # Right edge of FOV
        ])
        
        # Draw the triangle with simplified style
        polygon = Polygon(
            triangle_points, closed=True,
            edgecolor='blue', facecolor='none', linewidth=0.5,  # Thinner line
            alpha=0.7  # More transparent
        )
        ax.add_patch(polygon)
    
    def _visualize_episode_wrapper(self, env_id, is_success):
        """Wrapper to handle exceptions in thread pool."""
        try:
            self.visualize_episode(env_id, is_success)
        except Exception as e:
            print(f"Error visualizing episode for env {env_id}: {e}")
