#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to visualize trajectory tracking results like RAPTOR Figure 5."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import os

def plot_yaw_tracking(timestamps, desired_yaw, actual_yaw, save_path=None):
    """Plot Yaw tracking over time."""
    if len(desired_yaw) == 0:
        return

    # 转换为角度 (Degrees)
    des_deg = np.degrees(desired_yaw)
    act_deg = np.degrees(actual_yaw)
    
    # 简单的角度 Wrap 处理 (为了绘图好看，如果数据跳变，图上会有竖线)
    # 如果希望图更好看，可以对 act_deg 做 unwrap，或者仅绘制 raw data
    
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, des_deg, label='Desired Yaw', color='red', linestyle='--', linewidth=1.5)
    plt.plot(timestamps, act_deg, label='Actual Yaw', color='blue', alpha=0.8, linewidth=1.5)
    
    plt.title('Yaw Angle Tracking', fontsize=14)
    plt.xlabel('Time [s]', fontsize=12)
    plt.ylabel('Yaw Angle [deg]', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 限制 Y 轴范围 (根据你的限制 ±90度，稍微多给点空间)
    plt.ylim(-110, 110) 
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved yaw plot to: {save_path}")
    plt.close()

def get_colored_segments(x, y, z, color_vals, cmap_name='plasma'):
    """Prepare segments and colors for plotting."""
    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Normalize color values (speed)
    norm = plt.Normalize(color_vals.min(), color_vals.max())
    return segments, norm

def plot_paper_style_2d(desired_pos, actual_pos, actual_vel, save_path=None):
    """Plot 2D projections with velocity color mapping (Like Fig 5)."""
    if len(actual_pos) == 0:
        print("No data to plot.")
        return

    speed = np.linalg.norm(actual_vel, axis=1)
    max_speed = np.max(speed)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    norm = plt.Normalize(0, max_speed)
    cmap = plt.get_cmap('plasma')
    
    planes = [
        (0, 1, 'X (m)', 'Y (m)', 'XY Plane (Top)'),
        (0, 2, 'X (m)', 'Z (m)', 'XZ Plane (Side)'),
        (1, 2, 'Y (m)', 'Z (m)', 'YZ Plane (Front)')
    ]
    
    line = None
    
    for i, (idx1, idx2, xlabel, ylabel, title) in enumerate(planes):
        ax = axes[i]
        
        # 1. Plot Desired Trajectory (Black thin line)
        ax.plot(desired_pos[:, idx1], desired_pos[:, idx2], 'k--', linewidth=1.0, alpha=0.5, label='Reference')
        
        # 2. Plot Actual Trajectory with Color mapping (Velocity)
        # Create segments (x, y)
        points = np.array([actual_pos[:, idx1], actual_pos[:, idx2]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(speed[:-1]) # use speed for color
        lc.set_linewidth(2.5)
        
        line = ax.add_collection(lc)
        
        # Adjust limits
        all_x = np.concatenate([desired_pos[:, idx1], actual_pos[:, idx1]])
        all_y = np.concatenate([desired_pos[:, idx2], actual_pos[:, idx2]])
        margin = 0.2
        ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
        ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.axis('equal')

    # Add Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7]) # Position [left, bottom, width, height]
    if line:
        cbar = fig.colorbar(line, cax=cbar_ax)
        cbar.set_label('Speed [m/s]', fontsize=12)
    
    plt.subplots_adjust(wspace=0.3, right=0.9)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"2D paper-style plot saved to: {save_path}")
    plt.show()

def plot_paper_style_3d(desired_pos, actual_pos, actual_vel, save_path=None):
    """Plot 3D trajectory with velocity color mapping."""
    if len(actual_pos) == 0:
        print("No data to plot.")
        return

    speed = np.linalg.norm(actual_vel, axis=1)
    max_speed = np.max(speed)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Plot Desired (Thin dashed line)
    ax.plot(desired_pos[:, 0], desired_pos[:, 1], desired_pos[:, 2], 
            'k--', linewidth=0.8, alpha=0.4, label='Reference')
    
    # 2. Plot Actual (Colored by speed)
    points = np.array([actual_pos[:, 0], actual_pos[:, 1], actual_pos[:, 2]]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    norm = plt.Normalize(0, max_speed)
    cmap = plt.get_cmap('plasma')
    
    lc = Line3DCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(speed[:-1])
    lc.set_linewidth(2.0)
    
    ax.add_collection(lc)
    
    # Set limits
    max_range = np.array([
        actual_pos[:, 0].max() - actual_pos[:, 0].min(),
        actual_pos[:, 1].max() - actual_pos[:, 1].min(),
        actual_pos[:, 2].max() - actual_pos[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (actual_pos[:, 0].max() + actual_pos[:, 0].min()) * 0.5
    mid_y = (actual_pos[:, 1].max() + actual_pos[:, 1].min()) * 0.5
    mid_z = (actual_pos[:, 2].max() + actual_pos[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory (Colored by Speed)', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.1)
    cbar.set_label('Speed [m/s]', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"3D paper-style plot saved to: {save_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize trajectory tracking results (Paper Style)')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Directory containing trajectory_data.npz')
    parser.add_argument('--save_plots', action='store_true', default=True,
                        help='Save plots to files')
    parser.add_argument('--start_step', type=int, default=None,
                        help='Manually override start step for plotting (default: uses saved metrics start step or 3000)')
    args = parser.parse_args()
    
    # Load trajectory data
    data_file = os.path.join(args.data_dir, 'trajectory_data.npz')
    if not os.path.exists(data_file):
        print(f"Error: Data file not found: {data_file}")
        return
    
    print(f"Loading trajectory data from: {data_file}")
    data = np.load(data_file)
    
    desired_pos = data['desired_pos']
    actual_pos = data['actual_pos']
    actual_vel = data['actual_vel']
    desired_yaw = data['desired_yaw']
    actual_yaw = data['actual_yaw']
    
    # Determine start step for stats and plotting
    stats_start = 3000 # Default fallback
    
    if args.start_step is not None:
        stats_start = args.start_step
        print(f"[INFO] Using manual start step: {stats_start}")
    elif 'metrics' in data and len(data['metrics']) > 3:
        stats_start = int(data['metrics'][3])
        print(f"[INFO] Using start step from saved metrics: {stats_start}")
    else:
        print(f"[INFO] Using default start step: {stats_start}")

    # --- Print Metrics ---
    # Recalculate based on slicing to be safe (or use saved if preferred, but recalculating ensures alignment with plot)
    valid_actual = actual_pos[stats_start:]
    valid_desired = desired_pos[stats_start:]
    valid_vel = actual_vel[stats_start:]
    
    if len(valid_actual) == 0:
        print(f"Error: Start step {stats_start} is larger than data length {len(actual_pos)}")
        return

    pos_error_sq = np.sum((valid_actual - valid_desired)**2, axis=1)
    rmse = np.sqrt(np.mean(pos_error_sq))
    
    pos_error_xy_sq = np.sum((valid_actual[:, :2] - valid_desired[:, :2])**2, axis=1)
    rmse_no_z = np.sqrt(np.mean(pos_error_xy_sq))
    
    speed = np.linalg.norm(valid_vel, axis=1)
    max_vel = np.max(speed)
    
    print(f"\nMetrics (Step {stats_start} -> End):")
    print(f"  RMSE:          {rmse:.4f} m")
    print(f"  RMSE w/o z:    {rmse_no_z:.4f} m")
    print(f"  Max Velocity:  {max_vel:.4f} m/s")
    
    # Create plots directory
    plots_dir = os.path.join(args.data_dir, 'plots')
    if args.save_plots:
        os.makedirs(plots_dir, exist_ok=True)
        print(f"\nSaving plots to: {plots_dir}")
    
    # Generate plots
    print(f"\nGenerating paper-style plots (Data Sliced: {stats_start} -> End)...")
    
    # Use SLICED data for plotting
    plot_desired = desired_pos[stats_start:]
    plot_actual = actual_pos[stats_start:]
    plot_vel = actual_vel[stats_start:]
    
    # 1. 2D Projections with Velocity Coloring
    save_path_2d = os.path.join(plots_dir, '2d_velocity_trajectory.png') if args.save_plots else None
    plot_paper_style_2d(plot_desired, plot_actual, plot_vel, save_path_2d)
    
    # 2. 3D Trajectory with Velocity Coloring
    save_path_3d = os.path.join(plots_dir, '3d_velocity_trajectory.png') if args.save_plots else None
    plot_paper_style_3d(plot_desired, plot_actual, plot_vel, save_path_3d)

    plot_yaw_tracking(
        timestamps, 
        desired_yaw, 
        actual_yaw, 
        save_path=os.path.join(plots_dir, 'yaw_tracking.png') if args.save_plots else None
    )
    
    print("\nVisualization complete!")

if __name__ == '__main__':
    main()