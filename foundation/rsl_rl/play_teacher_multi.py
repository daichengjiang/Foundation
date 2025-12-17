# Copyright (c) 2025 Xu Yang
# HKUST UAV Group
#
# Author: Xu Yang
# Affiliation: HKUST UAV Group
# License: MIT License

"""Script to play and evaluate multiple teachers with heterogeneous dynamics AND models."""

import argparse
import sys
import os
import glob
import pandas as pd
import torch
import numpy as np
import time
import copy
from datetime import datetime

# [Headless Config must be before importing plt]
from isaaclab.app import AppLauncher
# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play multiple teachers with specific dynamics and models.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playing.")
parser.add_argument("--video_length", type=int, default=2000, help="Length of the recorded video (in steps).")
parser.add_argument("--task", type=str, default="offset", help="Name of the task (use offset for heterogeneous physics).")

# ==============================================================================
# [Unified Path Arguments]
# ==============================================================================
parser.add_argument("--teacher_dir", type=str, default=None, help="Path to the teacher experiment directory (containing csv and teacher_xxxx folders).")
parser.add_argument("--teacher_ids", type=str, default="0", help="Comma-separated list or range (e.g., '0-4') of teacher IDs.")

parser.add_argument("--envs_per_teacher", type=int, default=80, help="Number of environments per teacher ID.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_steps", type=int, default=8000, help="Maximum steps to run.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# ==============================================================================
# [Headless Matplotlib Configuration]
# ==============================================================================
import matplotlib
if args_cli.headless:
    print("[INFO] Headless mode detected. Setting Matplotlib backend to 'Agg'.")
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==============================================================================
# [Manual Validation]
# ==============================================================================
if __name__ == "__main__":
    if args_cli.teacher_dir is None:
        parser.error("the following arguments are required: --teacher_dir")

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ==============================================================================
# [CRITICAL] Imports that depend on Isaac Sim must happen AFTER app launch
# ==============================================================================
import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, multi_agent_to_single_agent
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.utils.math import matrix_from_quat  # [新增] 用于坐标系转换

# Import your foundation tasks so 'offset' is registered in Gym!
from foundation import tasks

# Statistics Start Step
STATS_START_STEP = 5000

def parse_teacher_config(teacher_dir, teacher_ids_str):
    """
    Loads dynamics and finds model paths based on teacher_distillation.py logic.
    """
    # 1. Parse IDs
    teacher_ids = []
    if '-' in teacher_ids_str:
        start, end = map(int, teacher_ids_str.split('-'))
        teacher_ids = list(range(start, end + 1))
    else:
        teacher_ids = [int(x) for x in teacher_ids_str.split(',')]
    
    # 2. Load CSV
    csv_path = os.path.join(teacher_dir, "teacher_dynamics.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dynamics CSV not found at: {csv_path}")
        
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded dynamics from {csv_path}")
    
    teachers_data = []

    for t_id in teacher_ids:
        row = df[df['id'] == t_id]
        if row.empty:
            raise ValueError(f"Teacher ID {t_id} not found in CSV.")
        row = row.iloc[0]

        params = {
            "id": int(t_id),
            "mass": float(row['mass']),
            "arm_length": float(row['arm_length']),
            "inertia": (float(row['Ixx']), float(row['Iyy']), float(row['Izz'])),
            "twr": float(row['twr']) if 'twr' in row else float(row['thrust_to_weight']),
            "motor_tau": float(row['motor_tau'])
        }
        
        teacher_run_name = f"teacher_{t_id:04d}"
        folder_path = os.path.join(teacher_dir, teacher_run_name)
        model_path = os.path.join(folder_path, "best_model.pt")
        if not os.path.exists(model_path):
            search_pattern = os.path.join(folder_path, "model_*.pt")
            models = glob.glob(search_pattern)
            if not models:
                print(f"[WARNING] No model found for teacher {t_id}. Will use random init.")
                model_path = None
            else:
                model_path = max(models, key=os.path.getctime)
        
        params['model_path'] = model_path
        teachers_data.append(params)
        
        if model_path:
            print(f"  > [T-{t_id}] Mass={params['mass']:.3f} | Model: {os.path.basename(model_path)}")
        else:
            print(f"  > [T-{t_id}] Mass={params['mass']:.3f} | Model: NONE (Random)")
            
    return teachers_data

def plot_error_distribution(teacher_id, mass, error_data, save_dir):
    """
    绘制并保存单个 Teacher 的误差分布图 (Body Frame)。
    error_data shape: (N_samples, 3) where columns are Body X, Body Y, Body Z error
    """
    if len(error_data) > 10000:
        indices = np.random.choice(len(error_data), 10000, replace=False)
        data = error_data[indices]
    else:
        data = error_data

    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(f"Teacher ID: {teacher_id} (Mass: {mass:.3f} kg)\nBody-Frame Position Error (m)", fontsize=16)

    # 1. 3D Scatter Plot (Top Left)
    ax_3d = fig.add_subplot(2, 2, 1, projection='3d')
    ax_3d.scatter(data[:, 0], data[:, 1], data[:, 2], s=1, alpha=0.3, c='blue')
    ax_3d.set_xlabel('Body X (Forward) Error')
    ax_3d.set_ylabel('Body Y (Left) Error')
    ax_3d.set_zlabel('Body Z (Up) Error')
    ax_3d.set_title('3D Error Cloud (Body Frame)')
    ax_3d.scatter([0], [0], [0], c='red', marker='x', s=100, label='Target')

    # 2. XY Plane - Top View (Top Right)
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xy.scatter(data[:, 0], data[:, 1], s=1, alpha=0.3, c='green')
    ax_xy.set_xlabel('Body X (Forward) Error')
    ax_xy.set_ylabel('Body Y (Left) Error')
    ax_xy.set_title('Top View (XY)')
    ax_xy.grid(True, linestyle='--', alpha=0.6)
    ax_xy.scatter([0], [0], c='red', marker='x', s=100)
    ax_xy.set_aspect('equal', 'box')

    # 3. XZ Plane - Side View (Bottom Left)
    ax_xz = fig.add_subplot(2, 2, 3)
    ax_xz.scatter(data[:, 0], data[:, 2], s=1, alpha=0.3, c='purple')
    ax_xz.set_xlabel('Body X (Forward) Error')
    ax_xz.set_ylabel('Body Z (Up) Error')
    ax_xz.set_title('Side View (XZ)')
    ax_xz.grid(True, linestyle='--', alpha=0.6)
    ax_xz.scatter([0], [0], c='red', marker='x', s=100)
    ax_xz.set_aspect('equal', 'box')

    # 4. YZ Plane - Front View (Bottom Right)
    ax_yz = fig.add_subplot(2, 2, 4)
    ax_yz.scatter(data[:, 1], data[:, 2], s=1, alpha=0.3, c='orange')
    ax_yz.set_xlabel('Body Y (Left) Error')
    ax_yz.set_ylabel('Body Z (Up) Error')
    ax_yz.set_title('Front View (YZ)')
    ax_yz.grid(True, linestyle='--', alpha=0.6)
    ax_yz.scatter([0], [0], c='red', marker='x', s=100)
    ax_yz.set_aspect('equal', 'box')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    filename = f"teacher_{teacher_id:04d}_body_error.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=150)
    plt.close(fig)

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with multiple teachers, each having its own dynamics and model."""
    
    # 1. 路径准备
    csv_path = os.path.join(args_cli.teacher_dir, "teacher_dynamics.csv")
    
    # 2. Parse Configs and Find Models
    print(f"\n{'='*60}")
    print(f"Heterogeneous Multi-Teacher Evaluation Setup")
    print(f"Teacher Dir: {args_cli.teacher_dir}")
    
    teachers_data = parse_teacher_config(args_cli.teacher_dir, args_cli.teacher_ids)
    num_teachers = len(teachers_data)
    envs_per_teacher = args_cli.envs_per_teacher
    total_envs = num_teachers * envs_per_teacher
    
    print(f"Teachers Selected: {num_teachers}")
    print(f"Envs per Teacher: {envs_per_teacher}")
    print(f"Total Environments: {total_envs}")
    print(f"{'='*60}\n")

    # 3. Configure Environment
    env_cfg.scene.num_envs = total_envs
    env_cfg.dynamics.multi_teacher_params = teachers_data
    
    env_cfg.train_or_play = False
    env_cfg.prob_null_trajectory = 1.0   
    env_cfg.trajectory_type = "fixed"    
    env_cfg.debug_vis = True            
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    
    # 4. Create Environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env_wrapper = RslRlVecEnvWrapper(env)
    
    base_env = env.unwrapped
    device = base_env.device

    # 5. Load Policies
    print("[INFO] Loading individual policies for each teacher...")
    runner = OnPolicyRunner(env_wrapper, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    policies = []
    for idx, params in enumerate(teachers_data):
        model_path = params['model_path']
        if model_path is None or not os.path.exists(model_path):
            policy_copy = copy.deepcopy(runner.get_inference_policy(device=agent_cfg.device))
        else:
            runner.load(model_path, load_optimizer=False)
            runner.eval_mode()
            policy_copy = copy.deepcopy(runner.get_inference_policy(device=agent_cfg.device))
        policies.append(policy_copy)
    print(f"[INFO] All {len(policies)} policies loaded.")

    # 6. 运行仿真
    obs, _ = env_wrapper.get_observations()
    
    squared_error_sum = torch.zeros((num_teachers, envs_per_teacher), device=device)
    pos_error_sum = torch.zeros((num_teachers, envs_per_teacher, 3), device=device)
    raw_error_history = [] 
    
    sample_count = 0
    
    print(f"[INFO] Starting simulation for {args_cli.max_steps} steps...")
    
    for timestep in range(args_cli.max_steps):
        with torch.inference_mode():
            # 分段推理
            action_list = []
            for idx in range(num_teachers):
                start = idx * envs_per_teacher
                end = (idx + 1) * envs_per_teacher
                action_list.append(policies[idx](obs[start:end]))
            actions = torch.cat(action_list, dim=0)

            obs, rewards, dones, extras = env_wrapper.step(actions)
            
            if timestep >= STATS_START_STEP:
                # 获取世界系坐标
                current_pos_w = base_env._robot.data.root_pos_w
                desired_pos_w = base_env.pos_des
                current_quat_w = base_env._robot.data.root_quat_w
                
                # 计算世界系误差: P_err_w = P_drone - P_target
                raw_error_w = current_pos_w - desired_pos_w
                
                # --- [Coordinate Transformation] ---
                # 1. Quat -> Rot Matrix (Body to World)
                rot_b2w = matrix_from_quat(current_quat_w)
                # 2. Transpose -> Rot Matrix (World to Body)
                rot_w2b = rot_b2w.transpose(1, 2)
                
                # 3. Transform Error: P_err_b = R_w2b @ P_err_w
                # unsqueeze(-1) makes it (N, 3, 1) for multiplication
                raw_error_b = torch.bmm(rot_w2b, raw_error_w.unsqueeze(-1)).squeeze(-1)
                
                # 统计和记录都使用 Body Frame Error
                error_sq = torch.norm(raw_error_b, dim=1) ** 2
                squared_error_sum += error_sq.view(num_teachers, envs_per_teacher)
                pos_error_sum += raw_error_b.view(num_teachers, envs_per_teacher, 3)
                
                raw_error_history.append(raw_error_b.view(num_teachers, envs_per_teacher, 3).clone())
                
                sample_count += 1
        
        if timestep % 100 == 0:
            print(f"Step {timestep}/{args_cli.max_steps}")

    # 7. 计算统计结果
    mean_mse_per_teacher = squared_error_sum.sum(dim=1) / (sample_count * envs_per_teacher)
    rmse_per_teacher = torch.sqrt(mean_mse_per_teacher).cpu().numpy()
    
    mean_pos_error = pos_error_sum.sum(dim=1) / (sample_count * envs_per_teacher)
    mean_pos_error_cpu = mean_pos_error.cpu().numpy()

    # 8. 更新 CSV 文件 (Offsets Only - Now in Body Frame)
    print(f"\n[INFO] Updating CSV file with BODY FRAME offsets: {csv_path}")
    df = pd.read_csv(csv_path)
    update_data = {'x_off_mean': {}, 'y_off_mean': {}, 'z_off_mean': {}}

    for idx, params in enumerate(teachers_data):
        t_id = params['id']
        update_data['x_off_mean'][t_id] = float(mean_pos_error_cpu[idx, 0])
        update_data['y_off_mean'][t_id] = float(mean_pos_error_cpu[idx, 1])
        update_data['z_off_mean'][t_id] = float(mean_pos_error_cpu[idx, 2])

    for col in ['x_off_mean', 'y_off_mean', 'z_off_mean']:
        if col in df.columns:
            df[col] = df['id'].map(update_data[col]).fillna(df[col])
        else:
            df[col] = df['id'].map(update_data[col]).fillna(0.0)

    df.to_csv(csv_path, index=False)
    print(f"[SUCCESS] CSV updated successfully.")

    # 9. 生成点云分布图 (Body Frame)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_dir = os.path.join(args_cli.teacher_dir, f"error_plots_body_{timestamp}")
    os.makedirs(plot_dir, exist_ok=True)
    print(f"\n[INFO] Generating BODY FRAME error plots in: {plot_dir}")

    print("  -> Processing plotting data...")
    all_errors_gpu = torch.stack(raw_error_history, dim=0) 
    
    for idx, params in enumerate(teachers_data):
        t_id = params['id']
        mass = params['mass']
        teacher_errors = all_errors_gpu[:, idx, :, :].reshape(-1, 3).cpu().numpy()
        
        print(f"  -> Plotting Teacher {t_id} (Samples: {len(teacher_errors)})...")
        plot_error_distribution(t_id, mass, teacher_errors, plot_dir)

    print(f"[SUCCESS] All plots saved to {plot_dir}")

    # 10. 打印报告
    print(f"\n{'='*100}")
    print(f"{'ID':<5} | {'Mass':<8} | {'RMSE':<10} | {'Body_X':<8} | {'Body_Y':<8} | {'Body_Z':<8}")
    print(f"{'-'*100}")
    for idx, params in enumerate(teachers_data):
        t_id = params['id']
        print(f"{t_id:<5} | {params['mass']:<8.3f} | {rmse_per_teacher[idx]:<10.4f} | "
              f"{mean_pos_error_cpu[idx,0]:<8.4f} | {mean_pos_error_cpu[idx,1]:<8.4f} | {mean_pos_error_cpu[idx,2]:<8.4f}")
    print(f"{'='*100}\n")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()