# Copyright (c) 2025
# RAPTOR Implementation - Play Student Policy

"""
Usage:
    python foundation/rsl_rl/play_student.py \
        --task point_ctrl_single_dense \
        --checkpoint logs/rsl_rl/experiment/student/student_policy_best.pt \
        --dynamics_csv logs/rsl_rl/experiment/teacher_dynamics.csv \
        --num_envs 4
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play and evaluate Student (RAPTOR) policy.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playing.")
parser.add_argument("--video_length", type=int, default=2000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=10000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, required=True, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_steps", type=int, default=2000, help="Maximum steps to run for trajectory tracking.")
parser.add_argument("--save_trajectory", action="store_true", default=True, help="Save trajectory data for analysis.")

parser.add_argument("--checkpoint", type=str, required=True, help="Path to student .pt checkpoint.")
parser.add_argument("--student_hidden", type=int, default=16, help="Hidden dimension of the student GRU.")

# [动力学参数 CSV 路径]
parser.add_argument("--dynamics_csv", type=str, default=None, help="Path to teacher_dynamics.csv. If provided, env dynamics will be sampled from here.")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from datetime import datetime

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_tasks.utils.hydra import hydra_task_config
from foundation import tasks

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# Student Policy Architecture
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

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play and evaluate trajectory tracking with Student model."""
    
    # 1. Configure Environment
    env_cfg.scene.num_envs = args_cli.num_envs
    # env_cfg.trajectory_type = "figure8"
    env_cfg.trajectory_type = "langevin"
    env_cfg.prob_null_trajectory = 0.0
    env_cfg.train_or_play = False
    env_cfg.debug_vis = True
    
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # 2. Logging Setup
    ckpt_root = os.path.dirname(os.path.dirname(args_cli.checkpoint))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_play_student"
    log_path = os.path.join(ckpt_root, log_dir)
    os.makedirs(log_path, exist_ok=True)
    print(f"[INFO] Logging evaluation to: {log_path}")

    # 3. Create Environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_path, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env)
    device = torch.device(env.unwrapped.device)

    # =======================================================================
    # [关键修改] 覆盖动力学参数 (Dynamics Overwrite) - 循环分配模式
    # =======================================================================
    if args_cli.dynamics_csv and os.path.exists(args_cli.dynamics_csv):
        print(f"\n[INFO] Loading dynamics from CSV: {args_cli.dynamics_csv}")
        df = pd.read_csv(args_cli.dynamics_csv)
        total_rows = len(df)
        
        # [修改] 使用取模运算进行循环分配 (Round-Robin / Cyclic)
        # 例如: CSV有2行, num_envs=4 -> indices=[0, 1, 0, 1]
        sampled_indices = [i % total_rows for i in range(args_cli.num_envs)]
        sampled_df = df.iloc[sampled_indices]
        
        print(f"[INFO] Assigned dynamics cyclically.")
        print(f"       CSV Rows: {total_rows}, Num Envs: {args_cli.num_envs}")
        print(f"       Indices assigned: {sampled_indices}")
        
        # 提取参数并转为 Tensor
        mass_t = torch.tensor(sampled_df['mass'].values, device=device, dtype=torch.float32)
        arm_t = torch.tensor(sampled_df['arm_length'].values, device=device, dtype=torch.float32)
        twr_t = torch.tensor(sampled_df['twr'].values, device=device, dtype=torch.float32)
        tau_t = torch.tensor(sampled_df['motor_tau'].values, device=device, dtype=torch.float32)
        
        ixx = torch.tensor(sampled_df['Ixx'].values, device=device, dtype=torch.float32)
        iyy = torch.tensor(sampled_df['Iyy'].values, device=device, dtype=torch.float32)
        izz = torch.tensor(sampled_df['Izz'].values, device=device, dtype=torch.float32)
        inertia_t = torch.stack([ixx, iyy, izz], dim=1) # (N, 3)
        
        # 打印部分参数确认
        print(f"  > Env 0 Dynamics (Row {sampled_indices[0]}): Mass={mass_t[0]:.4f}, TWR={twr_t[0]:.2f}")
        if args_cli.num_envs > 1:
            print(f"  > Env 1 Dynamics (Row {sampled_indices[1]}): Mass={mass_t[1]:.4f}, TWR={twr_t[1]:.2f}")

        # 强制覆盖 Environment 内部变量
        unwrapped_env = env.unwrapped
        
        unwrapped_env.mass_tensor = mass_t
        unwrapped_env.arm_l_tensor = arm_t
        unwrapped_env.inertia_tensor = inertia_t
        unwrapped_env.twr_tensor = twr_t
        unwrapped_env.motor_tau = tau_t.view(-1, 1) # (N, 1)
        
        # 重新计算 Derived Parameters
        unwrapped_env._controller.mass = mass_t
        unwrapped_env._controller.arm_length = arm_t
        unwrapped_env._controller.inertia = inertia_t
        unwrapped_env._controller.thrust_to_weight = twr_t
        
        unwrapped_env.motor_alpha = unwrapped_env.dt / (unwrapped_env.dt + unwrapped_env.motor_tau)
        
    else:
        print("\n[WARNING] No dynamics CSV provided. Using DEFAULT Crazyflie dynamics for all envs!")
        print("          Results may be poor if Student expects diverse dynamics.")

    # 4. Load Student Model
    full_obs_dim = env.unwrapped.observation_space.shape[-1]
    student_obs_dim = full_obs_dim - 4
    action_dim = env.unwrapped.action_space.shape[-1]
    
    student = StudentPolicy(
        obs_dim=student_obs_dim, 
        action_dim=action_dim, 
        hidden_dim=args_cli.student_hidden
    ).to(device)
    
    checkpoint = torch.load(args_cli.checkpoint, map_location=device)
    student.load_state_dict(checkpoint)
    student.eval()

    # 5. Initialize Hidden State
    hidden_state = torch.zeros(env.num_envs, 1, args_cli.student_hidden, device=device).transpose(0, 1)

    # 6. Data Storage
    trajectory_data = {
        'desired_pos': [], 'actual_pos': [],
        'desired_vel': [], 'actual_vel': [],
        'position_error': [], 'velocity_error': [],
        'actions': [], 'timestamps': []
    }
    pos_errors_all = [[] for _ in range(env.num_envs)]
    vel_errors_all = [[] for _ in range(env.num_envs)]

    # 7. Simulation Loop
    obs, _ = env.get_observations()
    dt = env.unwrapped.step_dt
    timestep = 0
    start_time = time.time()

    print(f"\n{'=' * 80}")
    print(f"Student Policy Evaluation (Env Dynamics Randomized: {args_cli.dynamics_csv is not None})")
    print(f"{'=' * 80}\n")

    while simulation_app.is_running() and timestep < args_cli.max_steps:
        with torch.inference_mode():
            student_obs = obs[:, :-4]
            actions_seq, hidden_state = student(student_obs, hidden_state)
            actions = actions_seq.squeeze(1)
            actions = torch.clamp(actions, -1.0, 1.0)

            des_pos = env.unwrapped.pos_des.clone()
            des_vel = env.unwrapped.vel_des.clone()

            obs, rewards, dones, extras = env.step(actions)

            if torch.any(dones):
                reset_indices = torch.where(dones)[0]
                hidden_state[:, reset_indices, :] = 0.0

            cur_pos = env.unwrapped._robot.data.root_pos_w.clone()
            cur_vel = env.unwrapped._robot.data.root_lin_vel_w.clone()

            p_err = torch.norm(cur_pos - des_pos, dim=1)
            v_err = torch.norm(cur_vel - des_vel, dim=1)

            for i in range(env.num_envs):
                pos_errors_all[i].append(p_err[i].item())
                vel_errors_all[i].append(v_err[i].item())

            if args_cli.save_trajectory:
                trajectory_data['desired_pos'].append(des_pos[0].cpu().numpy())
                trajectory_data['actual_pos'].append(cur_pos[0].cpu().numpy())
                trajectory_data['desired_vel'].append(des_vel[0].cpu().numpy())
                trajectory_data['actual_vel'].append(cur_vel[0].cpu().numpy())
                trajectory_data['position_error'].append(p_err[0].item())
                trajectory_data['velocity_error'].append(v_err[0].item())
                trajectory_data['actions'].append(actions[0].cpu().numpy())
                trajectory_data['timestamps'].append(timestep * dt)

            timestep += 1
            if timestep % 1000 == 0:
                print(f"Step {timestep:5d} | Pos Err: {p_err.mean().item():.4f}m | Vel Err: {v_err.mean().item():.4f}m/s")

    # 8. Post-Processing
    total_time = time.time() - start_time
    all_p_err = np.concatenate([np.array(x) for x in pos_errors_all])
    all_v_err = np.concatenate([np.array(x) for x in vel_errors_all])

    print(f"\n{'=' * 80}")
    print(f"Student Evaluation Results")
    print(f"{'-' * 80}")
    print(f"Overall Pos Error: {np.mean(all_p_err):.4f} ± {np.std(all_p_err):.4f} m")
    print(f"Overall Vel Error: {np.mean(all_v_err):.4f} ± {np.std(all_v_err):.4f} m/s")

    np.savez(os.path.join(log_path, "tracking_errors.npz"), pos_errors=all_p_err, vel_errors=all_v_err)
    if args_cli.save_trajectory:
        np.savez(os.path.join(log_path, "trajectory_data.npz"), **trajectory_data)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()