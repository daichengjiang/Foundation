#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to visualize trajectory tracking results."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def plot_3d_trajectory(desired_pos, actual_pos, save_path=None):
    """Plot 3D trajectory comparison."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories
    ax.plot(desired_pos[:, 0], desired_pos[:, 1], desired_pos[:, 2], 
            'g-', linewidth=2, label='Desired Trajectory', alpha=0.7)
    ax.plot(actual_pos[:, 0], actual_pos[:, 1], actual_pos[:, 2], 
            'b-', linewidth=1.5, label='Actual Trajectory', alpha=0.8)
    
    # Mark start and end points
    ax.scatter(desired_pos[0, 0], desired_pos[0, 1], desired_pos[0, 2], 
               c='green', marker='o', s=100, label='Start (Desired)')
    ax.scatter(actual_pos[0, 0], actual_pos[0, 1], actual_pos[0, 2], 
               c='blue', marker='o', s=100, label='Start (Actual)')
    ax.scatter(desired_pos[-1, 0], desired_pos[-1, 1], desired_pos[-1, 2], 
               c='red', marker='s', s=100, label='End')
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title('3D Trajectory Tracking', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Equal aspect ratio
    max_range = np.array([
        desired_pos[:, 0].max() - desired_pos[:, 0].min(),
        desired_pos[:, 1].max() - desired_pos[:, 1].min(),
        desired_pos[:, 2].max() - desired_pos[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (desired_pos[:, 0].max() + desired_pos[:, 0].min()) * 0.5
    mid_y = (desired_pos[:, 1].max() + desired_pos[:, 1].min()) * 0.5
    mid_z = (desired_pos[:, 2].max() + desired_pos[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"3D trajectory plot saved to: {save_path}")
    plt.show()

def plot_2d_projections(desired_pos, actual_pos, save_path=None):
    """Plot 2D projections of the trajectory."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # XY plane
    axes[0].plot(desired_pos[:, 0], desired_pos[:, 1], 'g-', linewidth=2, label='Desired', alpha=0.7)
    axes[0].plot(actual_pos[:, 0], actual_pos[:, 1], 'b-', linewidth=1.5, label='Actual', alpha=0.8)
    axes[0].scatter(desired_pos[0, 0], desired_pos[0, 1], c='green', marker='o', s=100)
    axes[0].scatter(actual_pos[0, 0], actual_pos[0, 1], c='blue', marker='o', s=100)
    axes[0].set_xlabel('X (m)', fontsize=12)
    axes[0].set_ylabel('Y (m)', fontsize=12)
    axes[0].set_title('XY Plane (Top View)', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')
    
    # XZ plane
    axes[1].plot(desired_pos[:, 0], desired_pos[:, 2], 'g-', linewidth=2, label='Desired', alpha=0.7)
    axes[1].plot(actual_pos[:, 0], actual_pos[:, 2], 'b-', linewidth=1.5, label='Actual', alpha=0.8)
    axes[1].scatter(desired_pos[0, 0], desired_pos[0, 2], c='green', marker='o', s=100)
    axes[1].scatter(actual_pos[0, 0], actual_pos[0, 2], c='blue', marker='o', s=100)
    axes[1].set_xlabel('X (m)', fontsize=12)
    axes[1].set_ylabel('Z (m)', fontsize=12)
    axes[1].set_title('XZ Plane (Side View)', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # YZ plane
    axes[2].plot(desired_pos[:, 1], desired_pos[:, 2], 'g-', linewidth=2, label='Desired', alpha=0.7)
    axes[2].plot(actual_pos[:, 1], actual_pos[:, 2], 'b-', linewidth=1.5, label='Actual', alpha=0.8)
    axes[2].scatter(desired_pos[0, 1], desired_pos[0, 2], c='green', marker='o', s=100)
    axes[2].scatter(actual_pos[0, 1], actual_pos[0, 2], c='blue', marker='o', s=100)
    axes[2].set_xlabel('Y (m)', fontsize=12)
    axes[2].set_ylabel('Z (m)', fontsize=12)
    axes[2].set_title('YZ Plane (Front View)', fontsize=12, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"2D projections plot saved to: {save_path}")
    plt.show()

def plot_position_errors(timestamps, position_error, save_path=None):
    """Plot position tracking errors over time."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Error over time
    axes[0].plot(timestamps, position_error, 'r-', linewidth=1, alpha=0.7)
    axes[0].axhline(y=np.mean(position_error), color='b', linestyle='--', 
                    linewidth=2, label=f'Mean: {np.mean(position_error):.4f}m')
    axes[0].fill_between(timestamps, 0, position_error, alpha=0.3, color='red')
    axes[0].set_xlabel('Time (s)', fontsize=12)
    axes[0].set_ylabel('Position Error (m)', fontsize=12)
    axes[0].set_title('Position Tracking Error Over Time', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Error histogram
    axes[1].hist(position_error, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[1].axvline(x=np.mean(position_error), color='r', linestyle='--', 
                    linewidth=2, label=f'Mean: {np.mean(position_error):.4f}m')
    axes[1].axvline(x=np.median(position_error), color='g', linestyle='--', 
                    linewidth=2, label=f'Median: {np.median(position_error):.4f}m')
    axes[1].set_xlabel('Position Error (m)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Position Error Distribution', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Position error plot saved to: {save_path}")
    plt.show()

def plot_velocity_comparison(timestamps, desired_vel, actual_vel, save_path=None):
    """Plot velocity comparison over time."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    axes[0].plot(timestamps, desired_vel[:, 0], 'g-', linewidth=2, label='Desired', alpha=0.7)
    axes[0].plot(timestamps, actual_vel[:, 0], 'b-', linewidth=1.5, label='Actual', alpha=0.8)
    axes[0].set_ylabel('Vx (m/s)', fontsize=11)
    axes[0].set_title('Velocity Tracking', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(timestamps, desired_vel[:, 1], 'g-', linewidth=2, label='Desired', alpha=0.7)
    axes[1].plot(timestamps, actual_vel[:, 1], 'b-', linewidth=1.5, label='Actual', alpha=0.8)
    axes[1].set_ylabel('Vy (m/s)', fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(timestamps, desired_vel[:, 2], 'g-', linewidth=2, label='Desired', alpha=0.7)
    axes[2].plot(timestamps, actual_vel[:, 2], 'b-', linewidth=1.5, label='Actual', alpha=0.8)
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].set_ylabel('Vz (m/s)', fontsize=11)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Velocity comparison plot saved to: {save_path}")
    plt.show()

def plot_actions(timestamps, actions, save_path=None):
    """Plot control actions over time."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    for i in range(4):
        axes[i].plot(timestamps, actions[:, i], linewidth=1.5, alpha=0.8)
        axes[i].set_ylabel(f'Motor {i+1}', fontsize=11)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim([0, 1])
    
    axes[0].set_title('Control Actions (Motor Commands)', fontsize=12, fontweight='bold')
    axes[3].set_xlabel('Time (s)', fontsize=12)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Actions plot saved to: {save_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize trajectory tracking results')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Directory containing trajectory_data.npz')
    parser.add_argument('--save_plots', action='store_true', default=True,
                        help='Save plots to files')
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
    desired_vel = data['desired_vel']
    actual_vel = data['actual_vel']
    position_error = data['position_error']
    velocity_error = data['velocity_error']
    actions = data['actions']
    timestamps = data['timestamps']
    
    print(f"\nTrajectory data loaded:")
    print(f"  Duration: {timestamps[-1]:.2f} seconds")
    print(f"  Time steps: {len(timestamps)}")
    print(f"  Mean position error: {np.mean(position_error):.4f} m")
    print(f"  Mean velocity error: {np.mean(velocity_error):.4f} m/s")
    
    # Create plots directory
    plots_dir = os.path.join(args.data_dir, 'plots')
    if args.save_plots:
        os.makedirs(plots_dir, exist_ok=True)
        print(f"\nSaving plots to: {plots_dir}")
    
    # Generate plots
    print("\nGenerating plots...")
    
    # 1. 3D trajectory
    save_path = os.path.join(plots_dir, '3d_trajectory.png') if args.save_plots else None
    plot_3d_trajectory(desired_pos, actual_pos, save_path)
    
    # 2. 2D projections
    save_path = os.path.join(plots_dir, '2d_projections.png') if args.save_plots else None
    plot_2d_projections(desired_pos, actual_pos, save_path)
    
    # 3. Position errors
    save_path = os.path.join(plots_dir, 'position_errors.png') if args.save_plots else None
    plot_position_errors(timestamps, position_error, save_path)
    
    # 4. Velocity comparison
    save_path = os.path.join(plots_dir, 'velocity_comparison.png') if args.save_plots else None
    plot_velocity_comparison(timestamps, desired_vel, actual_vel, save_path)
    
    # 5. Actions
    save_path = os.path.join(plots_dir, 'actions.png') if args.save_plots else None
    plot_actions(timestamps, actions, save_path)
    
    print("\nVisualization complete!")

if __name__ == '__main__':
    main()
