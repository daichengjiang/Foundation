# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play and evaluate trajectory tracking with the best trained model."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Play and evaluate trajectory tracking with the best trained model.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playing.")
parser.add_argument("--video_length", type=int, default=2000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=10000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, required=True, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_steps", type=int, default=10000, help="Maximum steps to run for trajectory tracking.")
parser.add_argument("--save_trajectory", action="store_true", default=True, help="Save trajectory data for analysis.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--realtime", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments (this includes --checkpoint)
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
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
import os
import time
import torch
import numpy as np
from datetime import datetime

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.assets import retrieve_file_path

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from foundation import tasks

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# ==========================================
# CONFIGURATION: Statistics Start Step
# ==========================================
STATS_START_STEP = 3000
# ==========================================

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play and evaluate trajectory tracking with best model."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    
    # Force figure-8 trajectory for testing
    env_cfg.trajectory_type = "figure8"
    env_cfg.prob_null_trajectory = 0.0  # Disable null trajectory

    env_cfg.train_or_play = False  # Set to Play mode
    
    # Enable debug visualization for trajectory tracking
    env_cfg.debug_vis = True
    
    # set the environment seed
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric if args_cli.disable_fabric is not None else env_cfg.sim.use_fabric

    # Example dynamics parameters (Modify as needed or keep commented to use defaults)
    # crazyfile
    env_cfg.dynamics.mass = 0.027
    env_cfg.dynamics.arm_length = 0.046
    env_cfg.dynamics.inertia = (1.657e-5,1.665e-5,2.926e-5)
    env_cfg.dynamics.thrust_to_weight = 2
    env_cfg.dynamics.motor_tau_up = 0.06
    env_cfg.dynamics.motor_tau_down = 0.15
    env_cfg.dynamics.moment_scale = 0.025

    # # x500
    # env_cfg.dynamics.mass = 2
    # env_cfg.dynamics.arm_length = 0.25
    # env_cfg.dynamics.inertia = (0.022,0.022,0.04)
    # env_cfg.dynamics.thrust_to_weight = 2.2
    # env_cfg.dynamics.motor_tau_up = 0.06
    # env_cfg.dynamics.motor_tau_down = 0.15
    # env_cfg.dynamics.moment_scale = 0.025

    # # Flightmare
    # env_cfg.dynamics.mass = 0.73
    # env_cfg.dynamics.arm_length = 0.17
    # env_cfg.dynamics.inertia = (7.911e-3,7.911e-3,1.231e-2)
    # env_cfg.dynamics.thrust_to_weight = 4.5
    # env_cfg.dynamics.motor_tau_up = 0.06
    # env_cfg.dynamics.motor_tau_down = 0.15
    # env_cfg.dynamics.moment_scale = 0.025

    # env_cfg.dynamics.mass = 0.6863819667949842
    # env_cfg.dynamics.arm_length = 0.12728135908262386
    # env_cfg.dynamics.inertia = (0.0058040111630157975,0.0058040111630157975,0.010632948450644941)
    # env_cfg.dynamics.thrust_to_weight = 2.6983768014823015
    # env_cfg.dynamics.motor_tau_up = 0.06495171311000257
    # env_cfg.dynamics.motor_tau_down = 0.2601621073237004
    # env_cfg.dynamics.moment_scale = 0.04343799444626214

    # get checkpoint path
    checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO]: Loading best model checkpoint from: {checkpoint_path}")
    
    # specify directory for logging this play session
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_trajectory_tracking"
    log_dir = os.path.join(log_root_path, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging trajectory tracking evaluation in directory: {log_dir}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during trajectory tracking.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    
    print(f"[INFO]: Loading model checkpoint from: {checkpoint_path}")

    # 1. Load checkpoint manually
    loaded_dict = torch.load(checkpoint_path, map_location=agent_cfg.device)
    
    # 2. Filter weights: Keep Student, remove Teachers
    full_state_dict = loaded_dict['model_state_dict']
    student_only_state_dict = {}
    
    for k, v in full_state_dict.items():
        if "teachers_list" in k:
            continue
        if "teacher" in k and "student" not in k:
            continue
        student_only_state_dict[k] = v
            
    # 3. Load weights
    runner.alg.policy.load_state_dict(student_only_state_dict, strict=False)
    print("[INFO] Model weights loaded (Student only, Teachers ignored).")

    # 4. Load Normalizer
    if agent_cfg.empirical_normalization:
        if 'obs_norm_state_dict' in loaded_dict:
            runner.obs_normalizer.load_state_dict(loaded_dict['obs_norm_state_dict'])
            print("[INFO] Observation Normalizer loaded.")
        else:
            print("[WARNING] Empirical normalization is enabled but no 'obs_norm_state_dict' found!")
    
    runner.eval_mode()
    policy = runner.get_inference_policy(device=agent_cfg.device)
    policy_model = runner.alg.policy
    dt = env.unwrapped.step_dt
    obs, _ = env.get_observations()
    
    # Data storage
    trajectory_data = {
        'desired_pos': [],
        'actual_pos': [],
        'desired_vel': [],
        'actual_vel': [],
        'actions': [],
        'timestamps': []
    }
    
    print(f"\n{'=' * 80}")
    print(f"Trajectory Tracking Evaluation (Student)")
    print(f"Number of environments: {env.num_envs}")
    print(f"Maximum steps: {args_cli.max_steps}")
    print(f"Statistics start step: {STATS_START_STEP}")
    print(f"{'=' * 80}\n")
    
    timestep = 0
    start_time = time.time()
    
    # Storage for calculating overall metrics across all steps and envs
    total_squared_error_pos = 0.0
    total_squared_error_pos_xy = 0.0
    max_velocity_observed = 0.0
    total_samples = 0

    while simulation_app.is_running() and timestep < args_cli.max_steps:
        step_start_time = time.time()
        
        with torch.inference_mode():
            desired_pos = env.unwrapped.pos_des.clone()
            desired_vel = env.unwrapped.vel_des.clone()
            
            actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions)

            if hasattr(policy_model, "reset"):
                policy_model.reset(dones)
            
            current_pos = env.unwrapped._robot.data.root_pos_w.clone()
            current_vel = env.unwrapped._robot.data.root_lin_vel_w.clone()
            
            # --- Calculation for Metrics ---
            pos_error_vec = current_pos - desired_pos
            squared_error = torch.sum(pos_error_vec**2, dim=1) 
            squared_error_xy = torch.sum(pos_error_vec[:, :2]**2, dim=1) 
            vel_mag = torch.norm(current_vel, dim=1)
            
            # === ONLY ACCUMULATE STATISTICS IF TIMESTEP >= STATS_START_STEP ===
            if timestep >= STATS_START_STEP:
                # 1. Total Squared Error (for RMSE)
                total_squared_error_pos += torch.sum(squared_error).item()
                
                # 2. XY Squared Error (for RMSE w/o z)
                total_squared_error_pos_xy += torch.sum(squared_error_xy).item()
                
                # 3. Max Velocity
                current_max_vel = torch.max(vel_mag).item()
                if current_max_vel > max_velocity_observed:
                    max_velocity_observed = current_max_vel
                
                total_samples += env.num_envs
            # =================================================================

            # Save trajectory data (Env 0 only) - We save ALL steps for visualization
            if args_cli.save_trajectory:
                trajectory_data['desired_pos'].append(desired_pos[0].cpu().numpy())
                trajectory_data['actual_pos'].append(current_pos[0].cpu().numpy())
                trajectory_data['desired_vel'].append(desired_vel[0].cpu().numpy())
                trajectory_data['actual_vel'].append(current_vel[0].cpu().numpy())
                trajectory_data['actions'].append(actions[0].cpu().numpy())
                trajectory_data['timestamps'].append(timestep * dt)
                        
            timestep += 1
            
            if timestep % 200 == 0:
                # Print instantaneous RMSE for monitoring
                cur_rmse = np.sqrt(torch.mean(squared_error).item())
                cur_rmse_xy = np.sqrt(torch.mean(squared_error_xy).item())
                status = " (Collecting Stats)" if timestep >= STATS_START_STEP else " (Warmup)"
                print(f"Step {timestep:5d}{status} | RMSE: {cur_rmse:.4f}m | RMSE w/o z: {cur_rmse_xy:.4f}m")
                
        if args_cli.realtime:
            sleep_time = dt - (time.time() - step_start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    total_time = time.time() - start_time
    
    # --- Final Metric Calculation ---
    if total_samples > 0:
        rmse_final = np.sqrt(total_squared_error_pos / total_samples)
        rmse_xy_final = np.sqrt(total_squared_error_pos_xy / total_samples)
    else:
        rmse_final = 0.0
        rmse_xy_final = 0.0
    
    print(f"\n{'=' * 80}")
    print(f"Paper Metrics Results (Calculated from step {STATS_START_STEP} onwards):")
    print(f"{'-' * 80}")
    print(f"  RMSE [m]:             {rmse_final:.4f}")
    print(f"  RMSE w/o z [m]:       {rmse_xy_final:.4f}")
    print(f"  Max velocity [m/s]:   {max_velocity_observed:.4f}")
    print(f"{'-' * 80}")
    print(f"  Total Steps:          {timestep}")
    print(f"  Valid Stat Steps:     {timestep - STATS_START_STEP}")
    
    # Save statistics to file
    stats_file = os.path.join(log_dir, "tracking_statistics.txt")
    with open(stats_file, 'w') as f:
        f.write(f"RAPTOR Paper Metrics Evaluation\n")
        f.write(f"{'=' * 80}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Task: {args_cli.task}\n")
        f.write(f"Stats Start Step: {STATS_START_STEP}\n")
        f.write(f"Total Steps: {timestep}\n")
        f.write(f"\nKEY METRICS (Steps >= {STATS_START_STEP}):\n")
        f.write(f"  RMSE [m]:             {rmse_final:.4f}\n")
        f.write(f"  RMSE w/o z [m]:       {rmse_xy_final:.4f}\n")
        f.write(f"  Max velocity [m/s]:   {max_velocity_observed:.4f}\n")
        
    print(f"\nStatistics saved to: {stats_file}")
    
    if args_cli.save_trajectory and len(trajectory_data['timestamps']) > 0:
        traj_file = os.path.join(log_dir, "trajectory_data.npz")
        np.savez(traj_file,
                 desired_pos=np.array(trajectory_data['desired_pos']),
                 actual_pos=np.array(trajectory_data['actual_pos']),
                 desired_vel=np.array(trajectory_data['desired_vel']),
                 actual_vel=np.array(trajectory_data['actual_vel']),
                 actions=np.array(trajectory_data['actions']),
                 timestamps=np.array(trajectory_data['timestamps']),
                 # Save calculated metrics and the start step used
                 metrics=np.array([rmse_final, rmse_xy_final, max_velocity_observed, STATS_START_STEP]))
        print(f"Trajectory data saved to: {traj_file}")

    print(f"{'=' * 80}\n")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()