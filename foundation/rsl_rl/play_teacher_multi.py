# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play and evaluate trajectory tracking with the best trained model (Supports PPO & SAC)."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import glob

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play and evaluate trajectory tracking with the best trained model.")
parser.add_argument("--experiment_dir", type=str, required=True, help="Path to the directory containing teacher_dynamics.csv and teacher_xxxx model folders.")
parser.add_argument("--algorithm", type=str, default="sac", choices=["ppo", "sac"], help="Choose RL algorithm: ppo or sac.")

parser.add_argument("--video", action="store_true", default=False, help="Record videos during playing.")
parser.add_argument("--video_length", type=int, default=2000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=10000, help="Interval between video recordings (in steps).")
parser.add_argument("--task", type=str, default=None, required=True, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_steps", type=int, default=10000, help="Maximum steps to run for trajectory tracking.")
parser.add_argument("--save_trajectory", action="store_true", default=True, help="Save trajectory data for analysis.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--realtime", action="store_true", default=False, help="Run in real-time, if possible.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.algorithm.lower() == "sac":
    os.environ["RSL_RL_SAC"] = "1"
    print("[INFO] Selected Algorithm: SAC")
else:
    os.environ["RSL_RL_SAC"] = "0"
    print("[INFO] Selected Algorithm: PPO")

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import time
import torch
import pandas as pd
from isaaclab.utils.math import euler_xyz_from_quat
import numpy as np
from datetime import datetime

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

try:
    from rsl_rl.runners import OffPolicyRunner
except ImportError:
    OffPolicyRunner = None

try:
    from rsl_rl.runners import OnPolicyRunner
except ImportError:
    OnPolicyRunner = None

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from foundation import tasks

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

STATS_START_STEP = 3000
ENVS_PER_TEACHER = 10

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    
    env_cfg.trajectory_type = "figure8"
    env_cfg.prob_null_trajectory = 0.0
    env_cfg.train_or_play = False
    env_cfg.debug_vis = True
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric if args_cli.disable_fabric is not None else env_cfg.sim.use_fabric

    csv_path = os.path.join(args_cli.experiment_dir, "teacher_dynamics.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find teacher_dynamics.csv in {args_cli.experiment_dir}")
        
    df = pd.read_csv(csv_path)
    multi_teacher_params = []
    
    for i, row in df.iterrows():
        inertia = (float(row['Ixx']), float(row['Iyy']), float(row['Izz']))
        params = {
            'id': int(row['id']),
            'mass': float(row['mass']),
            'arm_length': float(row['arm_length']),
            'inertia': inertia,
            'thrust_to_weight': float(row['twr']),
            'motor_tau_up': float(row['motor_tau_up']),
            'motor_tau_down': float(row['motor_tau_down']),
            'kappa': float(row['kappa'])
        }
        multi_teacher_params.append(params)
        
    num_teachers = len(multi_teacher_params)
    print(f"[INFO] Loaded dynamics for {num_teachers} teachers from CSV.")

    args_cli.num_envs = num_teachers * ENVS_PER_TEACHER
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.dynamics.multi_teacher_params = multi_teacher_params

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_multi_teacher_tracking"
    os.makedirs(log_dir, exist_ok=True)

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
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env)

    runners = []
    policies = []
    policy_models = []
    
    for i in range(num_teachers):
        t_id = multi_teacher_params[i]['id']
        pattern = os.path.join(args_cli.experiment_dir, f"teacher_{t_id:04d}", "best_model.pt")
        match = glob.glob(pattern)
        if not match:
            pattern_alt = os.path.join(args_cli.experiment_dir, f"teacher_{t_id}", "best_model.pt")
            match = glob.glob(pattern_alt)
        if not match:
            raise FileNotFoundError(f"Model for teacher {t_id} not found")
            
        model_path = match[0]
        print(f"[INFO] Loading Model for Teacher ID {t_id} (Env Group {i}): {model_path}")
        # 缩减 Replay Buffer 内存占用
        eval_agent_cfg_dict = agent_cfg.to_dict()
        if "algorithm" in eval_agent_cfg_dict:
            # 对于 SAC：将 1000000 步的默认缓存砍到 10 步，彻底释放显存
            eval_agent_cfg_dict["algorithm"]["replay_buffer_size"] = 10
        # 对于 PPO/SAC：缩减 Rollout 临时缓存
        eval_agent_cfg_dict["num_steps_per_env"] = 2 
        if args_cli.algorithm.lower() == "sac":
            runner = OffPolicyRunner(env, eval_agent_cfg_dict, log_dir=None, device=agent_cfg.device)
            runner.load(model_path)
            runner.eval_mode()
            policies.append(runner.alg.act)
            policy_models.append(runner.alg.actor)
        else:
            runner = OnPolicyRunner(env, eval_agent_cfg_dict, log_dir=None, device=agent_cfg.device)
            runner.load(model_path)
            runner.eval_mode()
            policies.append(runner.get_inference_policy(device=agent_cfg.device))
            policy_models.append(runner.alg.policy)
        runners.append(runner)

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    
    # =======================================================
    # 【新增】使用 Numpy 数组向量化记录所有环境的数据
    # =======================================================
    env_sq_err_pos = np.zeros(args_cli.num_envs)
    env_sq_err_pos_xy = np.zeros(args_cli.num_envs)
    env_sq_err_yaw = np.zeros(args_cli.num_envs)
    env_max_vel = np.zeros(args_cli.num_envs)
    stat_steps = 0
    
    print(f"\n{'=' * 80}")
    print(f"Multi-Teacher Evaluation with Disturbance Testing")
    print(f"Total environments: {env.num_envs} ({num_teachers} teachers x {ENVS_PER_TEACHER} envs)")
    print(f"{'=' * 80}\n")
    
    timestep = 0
    start_time = time.time()
    
    import omni.timeline 
    timeline = omni.timeline.get_timeline_interface()
    timeline.pause()
    
    while simulation_app.is_running() and timestep < args_cli.max_steps:
        step_start_time = time.time()
        
        with torch.inference_mode():
            desired_pos = env.unwrapped.pos_des.clone()
            
            actions = torch.zeros((env.num_envs, 4), device=env.device)
            for t in range(num_teachers):
                start_idx = t * ENVS_PER_TEACHER
                end_idx = start_idx + ENVS_PER_TEACHER
                obs_slice = obs[start_idx:end_idx]
                actions[start_idx:end_idx] = policies[t](obs_slice)

            obs, rewards, dones, extras = env.step(actions)

            for t, model in enumerate(policy_models):
                if hasattr(model, "reset"):
                    start_idx = t * ENVS_PER_TEACHER
                    end_idx = start_idx + ENVS_PER_TEACHER
                    global_dones_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
                    global_dones_mask[start_idx:end_idx] = dones[start_idx:end_idx]
                    model.reset(global_dones_mask)
            
            current_pos = env.unwrapped._robot.data.root_pos_w.clone()
            current_vel = env.unwrapped._robot.data.root_lin_vel_w.clone()
            quat_w = env.unwrapped._robot.data.root_quat_w
            _, _, yaw_curr = euler_xyz_from_quat(quat_w)
            
            pos_error_vec = current_pos - desired_pos
            squared_error = torch.sum(pos_error_vec**2, dim=1) 
            squared_error_xy = torch.sum(pos_error_vec[:, :2]**2, dim=1) 
            vel_mag = torch.norm(current_vel, dim=1)
            
            if timestep >= STATS_START_STEP:
                batch_yaw_err = env.unwrapped.yaw_des - yaw_curr
                batch_yaw_err = torch.remainder(batch_yaw_err + torch.pi, 2 * torch.pi) - torch.pi
                
                # 【优化】极速向量化累加，告别 for 循环
                env_sq_err_pos += squared_error.cpu().numpy()
                env_sq_err_pos_xy += squared_error_xy.cpu().numpy()
                env_sq_err_yaw += (batch_yaw_err**2).cpu().numpy()
                env_max_vel = np.maximum(env_max_vel, vel_mag.cpu().numpy())
                
                stat_steps += 1
            
            timestep += 1
            
            if timestep % 200 == 0:
                cur_rmse = np.sqrt(torch.mean(squared_error).item())
                status = " (Collecting Stats)" if timestep >= STATS_START_STEP else " (Warmup)"
                print(f"Step {timestep:5d}{status} | Global Avg RMSE: {cur_rmse:.4f}m")
        
        if args_cli.realtime:
            sleep_time = dt - (time.time() - step_start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    # =======================================================
    # 最终结算与矩阵变换分析
    # =======================================================
    if stat_steps > 0:
        # 计算每台飞机的 RMSE
        env_rmse_pos = np.sqrt(env_sq_err_pos / stat_steps)
        env_rmse_pos_xy = np.sqrt(env_sq_err_pos_xy / stat_steps)
        env_rmse_yaw_deg = np.degrees(np.sqrt(env_sq_err_yaw / stat_steps))
        
        # 将 1D 数组重塑为 2D 矩阵：(num_teachers, ENVS_PER_TEACHER)
        # 矩阵的行代表不同的 Teacher，列代表不同的 Disturbance Level (0.0 -> 1.0)
        matrix_rmse_pos = env_rmse_pos.reshape(num_teachers, ENVS_PER_TEACHER)
        matrix_rmse_pos_xy = env_rmse_pos_xy.reshape(num_teachers, ENVS_PER_TEACHER)
        matrix_rmse_yaw = env_rmse_yaw_deg.reshape(num_teachers, ENVS_PER_TEACHER)
        matrix_max_vel = env_max_vel.reshape(num_teachers, ENVS_PER_TEACHER)
        
        dist_levels = np.linspace(0.0, 1.0, ENVS_PER_TEACHER)

        # --- 报表 1: 按 Teacher 统计平均误差 ---
        teacher_results = []
        for t in range(num_teachers):
            teacher_results.append({
                'Teacher_ID': multi_teacher_params[t]['id'],
                'Avg_RMSE_m': matrix_rmse_pos[t].mean(),
                'Avg_Yaw_RMSE_deg': matrix_rmse_yaw[t].mean(),
            })
            
        # --- 报表 2: 按 Disturbance 强度跨 Teacher 统计误差 ---
        dist_results = []
        for d in range(ENVS_PER_TEACHER):
            dist_results.append({
                # 强制保留 16 位小数，确保 0.0 和 1.0 对齐格式
                'Disturbance_Ratio': f"{dist_levels[d]:.16f}",
                'Avg_RMSE_m': matrix_rmse_pos[:, d].mean(),
                'Avg_Yaw_RMSE_deg': matrix_rmse_yaw[:, d].mean(),
            })

        # --- 报表 3: 详细矩阵 (每个 Teacher 在每个 Disturbance 下的表现，带分隔符) ---
        detailed_results = []
        for t in range(num_teachers):
            for d in range(ENVS_PER_TEACHER):
                detailed_results.append({
                    'Teacher_ID': multi_teacher_params[t]['id'],
                    # 强制保留 16 位小数
                    'Disturbance_Ratio': f"{dist_levels[d]:.16f}",
                    'RMSE_m': matrix_rmse_pos[t, d],
                    'RMSE_xy_m': matrix_rmse_pos_xy[t, d],
                    'Yaw_RMSE_deg': matrix_rmse_yaw[t, d],
                    'Max_Vel_ms': matrix_max_vel[t, d]
                })
            
            # 【新增】如果不是最后一个教师，在末尾插入一行分隔符
            if t < num_teachers - 1:
                detailed_results.append({
                    'Teacher_ID': '---',
                    'Disturbance_Ratio': '---',
                    'RMSE_m': '---',
                    'RMSE_xy_m': '---',
                    'Yaw_RMSE_deg': '---',
                    'Max_Vel_ms': '---'
                })
            
        print(f"\n{'=' * 80}")
        print("1. Performance by Teacher (Averaged across disturbances):")
        for res in teacher_results:
            print(f"Teacher {res['Teacher_ID']:03d} | RMSE: {res['Avg_RMSE_m']:.4f}m | Yaw: {res['Avg_Yaw_RMSE_deg']:.4f}°")
            
        print(f"\n{'-' * 80}")
        print("2. Performance by Disturbance Intensity (Averaged across all teachers):")
        for res in dist_results:
            print(f"Dist Ratio: {res['Disturbance_Ratio']} | Avg RMSE: {res['Avg_RMSE_m']:.4f}m | Yaw: {res['Avg_Yaw_RMSE_deg']:.4f}°")
            
        # ==========================================
        # 导出三个维度的统计结果到输入的 experiment_dir 目录下
        # ==========================================
        out_dir = args_cli.experiment_dir
        
        pd.DataFrame(teacher_results).to_csv(os.path.join(out_dir, "stats_by_teacher_avg.csv"), index=False)
        pd.DataFrame(dist_results).to_csv(os.path.join(out_dir, "stats_by_disturbance_avg.csv"), index=False)
        
        detailed_df = pd.DataFrame(detailed_results)
        detailed_csv_path = os.path.join(out_dir, "stats_detailed_matrix.csv")
        detailed_df.to_csv(detailed_csv_path, index=False)
        
        print(f"\n[INFO] 报表已保存至输入目录: {out_dir}")
        print(f"[INFO] 详细数据请查看: {detailed_csv_path}")
        
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()