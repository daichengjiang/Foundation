# Copyright (c) 2025
# RAPTOR Implementation - Play Student Policy with Dynamic Switching

import argparse
import sys
import os
import pandas as pd
import numpy as np

# [1] 确保能够导入 foundation 包
project_root = "/home/nv/Foundation"
if project_root not in sys.path:
    sys.path.append(project_root)

from isaaclab.app import AppLauncher

# [2] 定义参数
parser = argparse.ArgumentParser(description="Play RAPTOR Student Policy")
parser.add_argument("--task", type=str, default="point_ctrl_single_dense", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the student_policy.pt file.")
parser.add_argument("--video", action="store_true", default=False, help="Record video.")
parser.add_argument("--max_steps", type=int, default=2000, help="Max steps to play.")
parser.add_argument("--student_hidden", type=int, default=16, help="Hidden dimension of GRU.")

# Dynamics Arguments
parser.add_argument("--dynamics_mode", type=str, default="random", choices=["csv", "random"], help="Source of dynamics parameters.")
parser.add_argument("--csv_path", type=str, default=None, help="Path to teacher_dynamics.csv.")
parser.add_argument("--teacher_ids", type=int, nargs="+", default=[0], help="List of Teacher IDs to load from CSV.")

# Add standard Isaac Lab args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# [3] 启动仿真器
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Imports (Must be after AppLauncher)
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from datetime import datetime

# Isaac Lab Imports
from isaaclab.envs import DirectRLEnvCfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry # 用于手动加载配置

# [4] 注册环境 (处理 NameNotFound 问题)
try:
    from foundation import tasks
except ImportError:
    print("[WARN] Could not import 'foundation.tasks'. Attempting manual registration...")
    from foundation.tasks.point_ctrl.quad_point_ctrl_env_single_dense import QuadcopterEnv, QuadcopterEnvCfg
    
    if args_cli.task not in gym.registry:
        gym.register(
            id=args_cli.task,
            entry_point="foundation.tasks.point_ctrl.quad_point_ctrl_env_single_dense:QuadcopterEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": QuadcopterEnvCfg,
            },
        )
        print(f"[INFO] Manually registered task: {args_cli.task}")

# ---------------------------------------------------------------------------
# Student Policy
# ---------------------------------------------------------------------------
class StudentPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=16):
        super().__init__()
        self.input_layer = nn.Linear(obs_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x, hidden_state=None):
        if x.dim() == 2:
            x = x.unsqueeze(1) 
        x = F.tanh(self.input_layer(x))
        x, new_hidden = self.gru(x, hidden_state)
        x = self.output_layer(x)
        return x, new_hidden

# ---------------------------------------------------------------------------
# Dynamics Helpers
# ---------------------------------------------------------------------------
def sample_raptor_dynamics(num_envs, device):
    """Replicates random distribution logic."""
    print(f"[Dynamics] Sampling RANDOM dynamics for {num_envs} environments...")
    twr = torch.rand(num_envs, device=device) * (5.0 - 1.5) + 1.5
    s_min = 0.02**(1/3)
    s_max = 5.0**(1/3)
    s = torch.rand(num_envs, device=device) * (s_max - s_min) + s_min
    mass = s ** 3
    
    base_ratio = 0.04384 / (0.032**(1/3))
    u = torch.randn(num_envs, device=device) * 0.1
    u = torch.clamp(u, -0.3, 0.3)
    s_ms = torch.where(u < 0, 1.0 / (1.0 - u), 1.0 + u)
    arm_length = base_ratio * (mass**(1/3)) * s_ms
    
    r_t2i = torch.rand(num_envs, device=device) * (1200 - 40) + 40
    total_thrust = twr * 9.81 * mass
    tau_max = total_thrust * 1.414 * arm_length
    ixx = tau_max / r_t2i
    iyy = ixx.clone()
    izz = ixx * 1.832 
    inertia = torch.stack([ixx, iyy, izz], dim=1) 
    
    motor_tau = torch.rand(num_envs, device=device) * (0.12 - 0.02) + 0.02
    
    return {
        "mass": mass, "arm_length": arm_length, "inertia": inertia,
        "thrust_to_weight": twr, "motor_tau": motor_tau
    }

def load_csv_dynamics(csv_path, teacher_ids, num_envs_needed, device):
    """Loads specific teacher dynamics from CSV."""
    print(f"[Dynamics] Loading from CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    mass_list, arm_list, i_list, twr_list, tau_list = [], [], [], [], []
    
    for i in range(num_envs_needed):
        target_id = teacher_ids[i % len(teacher_ids)]
        row = df[df['id'] == target_id]
        if row.empty:
            raise ValueError(f"Teacher ID {target_id} not found in CSV!")
        row = row.iloc[0]
        
        mass_list.append(row['mass'])
        arm_list.append(row['arm_length'])
        i_list.append([row['Ixx'], row['Iyy'], row['Izz']])
        twr_list.append(row['twr'])
        tau_list.append(row['motor_tau'])
        
        print(f"  Env {i} -> Teacher {target_id} | Mass: {row['mass']:.3f}, TWR: {row['twr']:.2f}")

    return {
        "mass": torch.tensor(mass_list, device=device, dtype=torch.float32),
        "arm_length": torch.tensor(arm_list, device=device, dtype=torch.float32),
        "inertia": torch.tensor(i_list, device=device, dtype=torch.float32),
        "thrust_to_weight": torch.tensor(twr_list, device=device, dtype=torch.float32),
        "motor_tau": torch.tensor(tau_list, device=device, dtype=torch.float32)
    }

def inject_dynamics(env, params):
    base_env = env.unwrapped
    
    # 1. Update Env Tensors
    base_env.mass_tensor = params["mass"]
    base_env.arm_l_tensor = params["arm_length"]
    base_env.inertia_tensor = params["inertia"]
    base_env.twr_tensor = params["thrust_to_weight"]
    
    if params["motor_tau"].dim() == 1:
        base_env.motor_tau = params["motor_tau"].unsqueeze(1)
    else:
        base_env.motor_tau = params["motor_tau"]
    
    # Recalculate alpha
    base_env.motor_alpha = base_env.dt / (base_env.dt + base_env.motor_tau)
    
    # 2. Update Controller
    base_env._controller.mass = params["mass"]
    base_env._controller.arm_length = params["arm_length"]
    base_env._controller.inertia = params["inertia"]
    base_env._controller.thrust_to_weight = params["thrust_to_weight"]
    
    print("[Dynamics] Injection Complete.")

# ---------------------------------------------------------------------------
# Main Evaluation Logic
# ---------------------------------------------------------------------------
def main():
    # [5] 手动加载配置
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    
    # [6] 手动应用配置覆盖
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.train_or_play = False 
    env_cfg.trajectory_type = "figure8" 
    env_cfg.prob_null_trajectory = 0.0
    env_cfg.debug_vis = True 
    
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    env_cfg.sim.device = device

    # Create Environment
    print(f"[INFO] Creating environment: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # 2. Inject Dynamics
    if args_cli.dynamics_mode == "csv":
        if not args_cli.csv_path or not os.path.exists(args_cli.csv_path):
            raise ValueError(f"CSV path not found: {args_cli.csv_path}")
        
        dyn_params = load_csv_dynamics(
            args_cli.csv_path, 
            args_cli.teacher_ids, 
            args_cli.num_envs, 
            device
        )
    else:
        # Random
        dyn_params = sample_raptor_dynamics(args_cli.num_envs, device)
        
    inject_dynamics(env, dyn_params)

    # Optional: Video Recording
    if args_cli.video:
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_student_play"
        video_folder = os.path.join("logs", "videos", log_dir)
        env = gym.wrappers.RecordVideo(env, video_folder=video_folder, step_trigger=lambda step: step % 1000 == 0)

    # 3. Initialize Student Model
    # [修正] 使用 .shape[-1] 获取 Feature Dimension，避免被 Batch Dimension 干扰
    full_obs_dim = env.observation_space.shape[-1]
    action_dim = env.action_space.shape[-1]
    
    student_obs_dim = full_obs_dim - 4 
    
    print(f"[INFO] Full Obs Dim: {full_obs_dim}, Student Obs Dim: {student_obs_dim}, Action Dim: {action_dim}")
    
    model = StudentPolicy(student_obs_dim, action_dim, hidden_dim=args_cli.student_hidden).to(device)
    
    # Load Weights
    print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
    checkpoint = torch.load(args_cli.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 4. Simulation Loop
    obs, _ = env.reset() # obs is (N, 26)
    
    # Init Hidden State: (1, N, Hidden)
    hidden_state = torch.zeros(1, env.unwrapped.num_envs, args_cli.student_hidden, device=device)

    timestep = 0
    with torch.no_grad():
        while simulation_app.is_running() and timestep < args_cli.max_steps:
            
            # A. Prepare Observation (Mask Motor Speeds)
            if isinstance(obs, dict): obs = obs['policy'] 
            
            # obs shape is (Batch, 26)
            student_obs = obs[:, :-4]
            
            # B. Inference
            actions_seq, hidden_state = model(student_obs, hidden_state)
            actions = actions_seq.squeeze(1)
            
            # C. Step
            obs, rewards, dones, truncated, extras = env.step(actions)
            
            # D. Handle Resets
            combined_dones = dones | truncated
            if combined_dones.any():
                hidden_state[:, combined_dones, :] = 0.0
            
            timestep += 1
            if timestep % 100 == 0:
                print(f"Step {timestep}")

    print("[INFO] Finished.")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()