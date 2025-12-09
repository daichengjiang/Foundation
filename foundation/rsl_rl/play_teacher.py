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

# [新增] 动力学 CSV 参数
parser.add_argument("--dynamics_csv", type=str, default=None, help="Path to teacher_dynamics.csv to overwrite env physics.")

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
import pandas as pd  # [新增]
from datetime import datetime

# [修正] 使用正确的 Isaac Sim Core 路径
import isaacsim.core.utils.prims as prims_utils

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

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
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

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # =======================================================================
    # [关键修改] 覆盖动力学参数 (Dynamics Overwrite)
    # 逻辑与 play_student.py 保持一致，确保 Teacher 在熟悉的物理参数下飞行
    # =======================================================================
    if args_cli.dynamics_csv and os.path.exists(args_cli.dynamics_csv):
        print(f"\n[INFO] Loading dynamics from CSV: {args_cli.dynamics_csv}")
        df = pd.read_csv(args_cli.dynamics_csv)
        total_rows = len(df)
        
        # 使用取模运算进行循环分配
        sampled_indices = [i % total_rows for i in range(env.num_envs)]
        sampled_df = df.iloc[sampled_indices]
        
        print(f"[INFO] Assigned dynamics cyclically to {env.num_envs} envs.")
        
        device = torch.device(env.unwrapped.device)

        # 提取参数并转为 Tensor
        mass_t = torch.tensor(sampled_df['mass'].values, device=device, dtype=torch.float32)
        arm_t = torch.tensor(sampled_df['arm_length'].values, device=device, dtype=torch.float32)
        twr_t = torch.tensor(sampled_df['twr'].values, device=device, dtype=torch.float32)
        tau_t = torch.tensor(sampled_df['motor_tau'].values, device=device, dtype=torch.float32)
        
        ixx = torch.tensor(sampled_df['Ixx'].values, device=device, dtype=torch.float32)
        iyy = torch.tensor(sampled_df['Iyy'].values, device=device, dtype=torch.float32)
        izz = torch.tensor(sampled_df['Izz'].values, device=device, dtype=torch.float32)
        inertia_t = torch.stack([ixx, iyy, izz], dim=1) # (N, 3)
        
        print(f"  > Env 0 Dynamics (Row {sampled_indices[0]}): Mass={mass_t[0]:.4f}, TWR={twr_t[0]:.2f}")

        # 强制覆盖 Environment 内部变量
        unwrapped_env = env.unwrapped
        
        # 1. 覆盖环境 Tensor
        unwrapped_env.mass_tensor = mass_t
        unwrapped_env.arm_l_tensor = arm_t
        unwrapped_env.inertia_tensor = inertia_t
        unwrapped_env.twr_tensor = twr_t
        unwrapped_env.motor_tau = tau_t.view(-1, 1)
        
        # 2. 覆盖控制器 Controller 参数
        # (Teacher 策略通常依赖控制器参数来做归一化或重力补偿，这步非常关键)
        # 注意：这里我们手动赋值给 controller 的内部变量
        unwrapped_env._controller.mass_ = mass_t
        if not hasattr(unwrapped_env._controller, 'thrust_to_weight_'):
             # 如果 Controller 之前没存 TWR，这里手动加上
             unwrapped_env._controller.thrust_to_weight_ = twr_t
        else:
             unwrapped_env._controller.thrust_to_weight_ = twr_t

        # [关键] 调用 Controller 的更新函数，重新计算推力系数！
        # 请确保你在 simple_controller.py 中实现了 update_dependent_params 方法
        if hasattr(unwrapped_env._controller, 'update_dependent_params'):
            unwrapped_env._controller.update_dependent_params()
        else:
            print("\n[WARNING] Controller missing 'update_dependent_params' method!")
            print("          Thrust curves WILL NOT be updated, likely causing Z-axis drift.")

        # 3. 覆盖 Physics (USD/PhysX)
        print(f"[Override Check] Syncing physics properties to USD/PhysX...")
        for i in range(env.num_envs):
            # 获取 CSV 中的具体数值
            m_val = mass_t[i].item()
            Ixx_val = inertia_t[i, 0].item()
            Iyy_val = inertia_t[i, 1].item()
            Izz_val = inertia_t[i, 2].item()
            
            # 找到对应的 Prim 路径 (假设标准路径结构)
            # 可以在 Isaac Sim GUI 的 Stage 树里确认一下 "body" 的名字
            prim_path = f"/World/envs/env_{i}/Robot/body"
            
            # 修改质量
            prims_utils.set_prim_property(prim_path, "physics:mass", m_val)
            
            # 修改惯性张量 (Isaac Sim 需要 (Ixx, Iyy, Izz))
            prims_utils.set_prim_property(prim_path, "physics:diagonalInertia", (Ixx_val, Iyy_val, Izz_val))
            
            if i == 0:
                print(f"  > PhysX Env 0 Updated: Mass={m_val:.4f}, Inertia={Ixx_val:.2e}, {Iyy_val:.2e}, {Izz_val:.2e}")

        # 4. 重新计算派生参数
        unwrapped_env.motor_alpha = unwrapped_env.dt / (unwrapped_env.dt + unwrapped_env.motor_tau)
        
    else:
        print("\n[WARNING] No dynamics CSV provided. Using DEFAULT Crazyflie dynamics!")
        print("          If your Teacher was trained on specific dynamics, expect biases (Z-offset).")

    # =======================================================================
    # [修改结束]
    # =======================================================================

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    
    # load the best model checkpoint
    print(f"[INFO]: Loading model checkpoint from: {checkpoint_path}")
    runner.load(checkpoint_path, load_optimizer=False)
    
    # set policy to evaluation mode
    runner.eval_mode()

    # get inference policy
    policy = runner.get_inference_policy(device=agent_cfg.device)

    # simulation timestep
    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.get_observations()
    
    # Trajectory tracking data storage
    trajectory_data = {
        'desired_pos': [],      # Desired trajectory positions
        'actual_pos': [],       # Actual drone positions
        'desired_vel': [],      # Desired velocities
        'actual_vel': [],       # Actual velocities
        'position_error': [],   # Position tracking errors
        'velocity_error': [],   # Velocity tracking errors
        'actions': [],          # Control actions
        'timestamps': []        # Time stamps
    }
    
    # Tracking error statistics (per environment)
    position_errors_all = [[] for _ in range(env.num_envs)]
    velocity_errors_all = [[] for _ in range(env.num_envs)]
    
    print(f"\n{'=' * 80}")
    print(f"Trajectory Tracking Evaluation")
    print(f"Number of environments: {env.num_envs}")
    print(f"Maximum steps: {args_cli.max_steps}")
    print(f"{'=' * 80}\n")
    
    timestep = 0
    start_time = time.time()
    
    # simulate environment
    while simulation_app.is_running() and timestep < args_cli.max_steps:
        step_start_time = time.time()
        
        # run everything in inference mode
        with torch.inference_mode():
            
            # -----------------------------------------------------------------
            # --- 1. 获取 t 步的期望状态，并缓存下来 ---
            # -----------------------------------------------------------------
            desired_pos = env.unwrapped.pos_des.clone()
            desired_vel = env.unwrapped.vel_des.clone()
            
            # -----------------------------------------------------------------
            # --- 2. 获取 t 步的动作 a_t ---
            # -----------------------------------------------------------------
            actions = policy(obs) # obs 来自上一个时间步的 step() 结果
            
            # -----------------------------------------------------------------
            # --- 3. 执行动作，环境从 t-1 转移到 t ---
            # -----------------------------------------------------------------
            # 在 env.step() 内部，机器人实际位置变为 current_pos_t
            # 且环境的期望位置 pos_des/vel_des 可能会更新为下一时刻 t+1 的值
            obs, rewards, dones, extras = env.step(actions)
            
            # -----------------------------------------------------------------
            # --- 4. 获取 t 步的实际状态 ---
            # -----------------------------------------------------------------
            # 这是动作执行后的新位置/速度
            current_pos = env.unwrapped._robot.data.root_pos_w.clone()
            current_vel = env.unwrapped._robot.data.root_lin_vel_w.clone()
            
            # -----------------------------------------------------------------
            # --- 5. 计算跟踪误差 (使用 t 步的实际状态 和 t 步缓存的期望状态) ---
            # -----------------------------------------------------------------
            pos_error = torch.norm(current_pos - desired_pos, dim=1) 
            vel_error = torch.norm(current_vel - desired_vel, dim=1) 

            # Store tracking errors for each environment
            for env_id in range(env.num_envs):
                # 使用修正后的 pos_error/vel_error
                position_errors_all[env_id].append(pos_error[env_id].item())
                velocity_errors_all[env_id].append(vel_error[env_id].item())
            
            # Save trajectory data (only for environment 0 to reduce storage)
            if args_cli.save_trajectory:
                # 记录 t 步的数据
                trajectory_data['desired_pos'].append(desired_pos[0].cpu().numpy())
                trajectory_data['actual_pos'].append(current_pos[0].cpu().numpy())
                trajectory_data['desired_vel'].append(desired_vel[0].cpu().numpy())
                trajectory_data['actual_vel'].append(current_vel[0].cpu().numpy())
                trajectory_data['position_error'].append(pos_error[0].item())
                trajectory_data['velocity_error'].append(vel_error[0].item())
                trajectory_data['actions'].append(actions[0].cpu().numpy()) # actions 是 t 步使用的动作
                trajectory_data['timestamps'].append(timestep * dt)
                        
            timestep += 1
            
            # Print periodic statistics
            if timestep % 1000 == 0:
                avg_pos_error = torch.mean(pos_error).item()
                avg_vel_error = torch.mean(vel_error).item()
                print(f"Step {timestep:5d} | "
                    f"Avg Pos Error: {avg_pos_error:.4f}m | "
                    f"Avg Vel Error: {avg_vel_error:.4f}m/s")
                
        # time delay for real-time evaluation
        if args_cli.realtime:
            sleep_time = dt - (time.time() - step_start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    total_time = time.time() - start_time
    
    # Calculate overall statistics
    print(f"\n{'=' * 80}")
    print(f"Trajectory Tracking Results:")
    print(f"{'-' * 80}")
    print(f"  Total Steps:          {timestep}")
    print(f"  Total Time:           {total_time:.2f}s")
    print(f"  Average FPS:          {timestep / total_time:.2f}")
    
    # Calculate statistics for each environment
    all_env_pos_errors = []
    all_env_vel_errors = []
    
    print(f"\n{'Per-Environment Statistics:':^80}")
    print(f"{'-' * 80}")
    for env_id in range(env.num_envs):
        if len(position_errors_all[env_id]) > 0:
            pos_errors = np.array(position_errors_all[env_id])
            vel_errors = np.array(velocity_errors_all[env_id])
            
            all_env_pos_errors.extend(position_errors_all[env_id])
            all_env_vel_errors.extend(velocity_errors_all[env_id])
            
            print(f"  Env {env_id:2d} | "
                  f"Pos Error: {np.mean(pos_errors):.4f}±{np.std(pos_errors):.4f}m | "
                  f"Vel Error: {np.mean(vel_errors):.4f}±{np.std(vel_errors):.4f}m/s")
    
    # Overall statistics
    if len(all_env_pos_errors) > 0:
        all_pos_errors = np.array(all_env_pos_errors)
        all_vel_errors = np.array(all_env_vel_errors)
        
        print(f"\n{'Overall Statistics:':^80}")
        print(f"{'-' * 80}")
        print(f"  Position Error:")
        print(f"    Mean:     {np.mean(all_pos_errors):.4f} m")
        print(f"    Std:      {np.std(all_pos_errors):.4f} m")
        print(f"    Median:   {np.median(all_pos_errors):.4f} m")
        print(f"    Max:      {np.max(all_pos_errors):.4f} m")
        print(f"    95th %ile: {np.percentile(all_pos_errors, 95):.4f} m")
        
        print(f"\n  Velocity Error:")
        print(f"    Mean:     {np.mean(all_vel_errors):.4f} m/s")
        print(f"    Std:      {np.std(all_vel_errors):.4f} m/s")
        print(f"    Median:   {np.median(all_vel_errors):.4f} m/s")
        print(f"    Max:      {np.max(all_vel_errors):.4f} m/s")
        print(f"    95th %ile: {np.percentile(all_vel_errors, 95):.4f} m/s")
        
        # Save statistics to file
        stats_file = os.path.join(log_dir, "tracking_statistics.txt")
        with open(stats_file, 'w') as f:
            f.write(f"Trajectory Tracking Evaluation Results\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write(f"Task: {args_cli.task}\n")
            f.write(f"trajectory_type: {env_cfg.trajectory_type}\n")
            f.write(f"Num Envs: {env_cfg.scene.num_envs}\n")
            f.write(f"Seed: {env_cfg.seed}\n")
            f.write(f"Total Steps: {timestep}\n")
            f.write(f"Total Time: {total_time:.2f}s\n")
            f.write(f"Average FPS: {timestep / total_time:.2f}\n")
            
            f.write(f"\n{'Overall Statistics':^80}\n")
            f.write(f"{'-' * 80}\n")
            f.write(f"Position Error:\n")
            f.write(f"  Mean:     {np.mean(all_pos_errors):.4f} m\n")
            f.write(f"  Std:      {np.std(all_pos_errors):.4f} m\n")
            f.write(f"  Median:   {np.median(all_pos_errors):.4f} m\n")
            f.write(f"  Max:      {np.max(all_pos_errors):.4f} m\n")
            f.write(f"  95th %%:   {np.percentile(all_pos_errors, 95):.4f} m\n")
            
            f.write(f"\nVelocity Error:\n")
            f.write(f"  Mean:     {np.mean(all_vel_errors):.4f} m/s\n")
            f.write(f"  Std:      {np.std(all_vel_errors):.4f} m/s\n")
            f.write(f"  Median:   {np.median(all_vel_errors):.4f} m/s\n")
            f.write(f"  Max:      {np.max(all_vel_errors):.4f} m/s\n")
            f.write(f"  95th %%:   {np.percentile(all_vel_errors, 95):.4f} m/s\n")
            
            f.write(f"\n{'Per-Environment Statistics':^80}\n")
            f.write(f"{'-' * 80}\n")
            for env_id in range(env.num_envs):
                if len(position_errors_all[env_id]) > 0:
                    pos_errors = np.array(position_errors_all[env_id])
                    vel_errors = np.array(velocity_errors_all[env_id])
                    f.write(f"Env {env_id:2d}:\n")
                    f.write(f"  Pos Error: {np.mean(pos_errors):.4f} ± {np.std(pos_errors):.4f} m\n")
                    f.write(f"  Vel Error: {np.mean(vel_errors):.4f} ± {np.std(vel_errors):.4f} m/s\n")
        
        print(f"\nStatistics saved to: {stats_file}")
        
        # Save numpy arrays for detailed analysis
        error_data_file = os.path.join(log_dir, "tracking_errors.npz")
        np.savez(error_data_file,
                 position_errors=all_pos_errors,
                 velocity_errors=all_vel_errors)
        print(f"Error data saved to: {error_data_file}")
        
        # Save trajectory data if enabled
        if args_cli.save_trajectory and len(trajectory_data['timestamps']) > 0:
            traj_file = os.path.join(log_dir, "trajectory_data.npz")
            np.savez(traj_file,
                     desired_pos=np.array(trajectory_data['desired_pos']),
                     actual_pos=np.array(trajectory_data['actual_pos']),
                     desired_vel=np.array(trajectory_data['desired_vel']),
                     actual_vel=np.array(trajectory_data['actual_vel']),
                     position_error=np.array(trajectory_data['position_error']),
                     velocity_error=np.array(trajectory_data['velocity_error']),
                     actions=np.array(trajectory_data['actions']),
                     timestamps=np.array(trajectory_data['timestamps']))
            print(f"Trajectory data (Env 0) saved to: {traj_file}")
            print(f"  - Use this data to visualize desired vs actual trajectories")
            print(f"  - Data contains {len(trajectory_data['timestamps'])} time steps")
    
    print(f"{'=' * 80}\n")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()