# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play and evaluate multiple teacher models in parallel with heterogeneous dynamics and global statistics."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import glob
import time
import copy
import csv
import numpy as np
import torch
import multiprocessing as mp
import matplotlib
# 开启非交互式后端，极大提升多进程无头画图速度，并防止子进程崩溃
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from datetime import datetime

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play and evaluate multiple teachers in parallel.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playing.")
parser.add_argument("--video_length", type=int, default=2000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=10000, help="Interval between video recordings (in steps).")
parser.add_argument("--task", type=str, default=None, required=True, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_steps", type=int, default=2000, help="Maximum steps to run for trajectory tracking.")
parser.add_argument("--save_trajectory", action="store_true", default=True, help="Save trajectory data for analysis.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--realtime", action="store_true", default=False, help="Run in real-time, if possible.")

# --- Multi-Teacher Arguments ---
parser.add_argument("--teachers_dir", type=str, required=True, help="Path to the directory containing teacher_xxxx folders.")
parser.add_argument("--envs_per_teacher", type=int, default=1, help="Number of environments to run per teacher.")

# append RSL-RL cli arguments
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
from isaaclab.utils.math import euler_xyz_from_quat
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
STATS_START_STEP = 1000
# ==========================================

# =========================================================================
# 多模型策略聚合器 (Multi-Teacher Policy Wrapper)
# =========================================================================
class MultiTeacherPolicy(torch.nn.Module):
    def __init__(self, checkpoints, runner, device, envs_per_teacher):
        super().__init__()
        self.policies = []
        self.envs_per_teacher = envs_per_teacher
        self.device = device
        
        total_ckpts = len(checkpoints)
        print(f"[INFO] 正在并行加载 {total_ckpts} 个 Teacher 模型...")
        for i, ckpt in enumerate(checkpoints):
            runner.load(ckpt, load_optimizer=False)
            
            ac = copy.deepcopy(runner.alg.policy).to(device)
            ac.eval()
            
            if runner.obs_normalizer is not None:
                norm = copy.deepcopy(runner.obs_normalizer).to(device)
                norm.eval()
            else:
                norm = None
                
            self.policies.append((ac, norm))
            
            if (i + 1) % 10 == 0 or i == total_ckpts - 1:
                print(f"       已加载模型: {i + 1}/{total_ckpts}")
                
        print(f"[INFO] 所有 Teacher 模型加载完毕！")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        dummy_action = self.policies[0][0].act_inference(obs[0:1])
        action_dim = dummy_action.shape[1]
        
        actions = torch.zeros((obs.shape[0], action_dim), device=self.device)
        
        for i, (ac, norm) in enumerate(self.policies):
            start_idx = i * self.envs_per_teacher
            end_idx = start_idx + self.envs_per_teacher
            
            env_obs = obs[start_idx:end_idx]
            if norm is not None:
                env_obs = norm(env_obs)
                
            actions[start_idx:end_idx] = ac.act_inference(env_obs)
            
        return actions

    def reset(self, dones):
        pass

# =========================================================================
# Paper Style 轨迹画图函数
# =========================================================================
def plot_paper_style_2d(desired_pos, actual_pos, actual_vel, save_path, title_suffix=""):
    speed = np.linalg.norm(actual_vel, axis=1)
    max_speed = np.max(speed) if np.max(speed) > 0 else 1.0
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    norm = plt.Normalize(0, max_speed)
    cmap = plt.get_cmap('plasma')
    
    planes = [
        (0, 1, 'X (m)', 'Y (m)', f'XY Plane {title_suffix}'),
        (0, 2, 'X (m)', 'Z (m)', f'XZ Plane {title_suffix}'),
        (1, 2, 'Y (m)', 'Z (m)', f'YZ Plane {title_suffix}')
    ]
    
    for i, (idx1, idx2, xlabel, ylabel, title) in enumerate(planes):
        ax = axes[i]
        ax.plot(desired_pos[:, idx1], desired_pos[:, idx2], 'k--', linewidth=1.0, alpha=0.5, label='Reference')
        
        points = np.array([actual_pos[:, idx1], actual_pos[:, idx2]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(speed[:-1])
        lc.set_linewidth(2.5)
        line = ax.add_collection(lc)
        
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

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(line, cax=cbar_ax)
    cbar.set_label('Speed [m/s]', fontsize=12)
    plt.subplots_adjust(wspace=0.3, right=0.9)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_paper_style_3d(desired_pos, actual_pos, actual_vel, save_path, title_suffix=""):
    speed = np.linalg.norm(actual_vel, axis=1)
    max_speed = np.max(speed) if np.max(speed) > 0 else 1.0
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(desired_pos[:, 0], desired_pos[:, 1], desired_pos[:, 2], 
            'k--', linewidth=0.8, alpha=0.4, label='Reference')
    
    points = np.array([actual_pos[:, 0], actual_pos[:, 1], actual_pos[:, 2]]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    norm = plt.Normalize(0, max_speed)
    cmap = plt.get_cmap('plasma')
    
    lc = Line3DCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(speed[:-1])
    lc.set_linewidth(2.0)
    ax.add_collection(lc)
    
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
    ax.set_title(f'3D Trajectory {title_suffix}', fontsize=14, fontweight='bold')
    
    cbar = fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.1)
    cbar.set_label('Speed [m/s]', fontsize=12)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

# =========================================================================
# 多进程工作节点函数 (Worker Function)
# =========================================================================
def worker_save_and_plot(args):
    """供子进程独立调用的画图与数据保存函数"""
    (teacher_id, teacher_folder, checkpoint, assigned_param,
     dp, ap, dv, av, dy, ay, acts, t_arr,
     rmse, rmse_xy, yaw_rmse, max_vel, stats_start_step) = args

    stats_txt_path = os.path.join(teacher_folder, "tracking_statistics.txt")
    with open(stats_txt_path, 'w') as f:
        f.write(f"Teacher {teacher_id:04d} Tracking Statistics\n")
        f.write(f"{'=' * 45}\n")
        f.write(f"RMSE [m]:        {rmse:.4f}\n")
        f.write(f"RMSE w/o z [m]:  {rmse_xy:.4f}\n")
        f.write(f"Yaw RMSE [deg]:  {yaw_rmse:.4f}\n")
        f.write(f"Max Vel [m/s]:   {max_vel:.4f}\n")
        f.write(f"{'-' * 45}\n")
        f.write(f"Source Checkpoint: {checkpoint}\n")
        f.write(f"\n[Environment Dynamics Parameters]\n")
        f.write(f"  Mass:          {assigned_param['mass']:.4f} kg\n")
        f.write(f"  Arm Length:    {assigned_param['arm_length']:.4f} m\n")
        f.write(f"  TWR:           {assigned_param['twr']:.4f}\n")
        f.write(f"  Ixx:           {assigned_param['inertia'][0]:.4e}\n")
        f.write(f"  Iyy:           {assigned_param['inertia'][1]:.4e}\n")
        f.write(f"  Izz:           {assigned_param['inertia'][2]:.4e}\n")

    path_2d = os.path.join(teacher_folder, '2d_velocity_trajectory.png')
    plot_paper_style_2d(dp, ap, av, save_path=path_2d, title_suffix=f"- Teacher {teacher_id:04d}")
    
    path_3d = os.path.join(teacher_folder, '3d_velocity_trajectory.png')
    plot_paper_style_3d(dp, ap, av, save_path=path_3d, title_suffix=f"- Teacher {teacher_id:04d}")

    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f"Tracking Performance - Teacher {teacher_id:04d}", fontsize=16, fontweight='bold')
    
    axs[0].plot(t_arr, dp[:, 0], 'r--', label='Desired X', linewidth=2)
    axs[0].plot(t_arr, ap[:, 0], 'b-', label='Actual X', alpha=0.8)
    axs[0].set_ylabel('Pos X (m)')
    axs[0].grid(True, linestyle='--', alpha=0.6)

    axs[1].plot(t_arr, dp[:, 1], 'r--', label='Desired Y', linewidth=2)
    axs[1].plot(t_arr, ap[:, 1], 'b-', label='Actual Y', alpha=0.8)
    axs[1].set_ylabel('Pos Y (m)')
    axs[1].grid(True, linestyle='--', alpha=0.6)

    axs[2].plot(t_arr, dp[:, 2], 'r--', label='Desired Z', linewidth=2)
    axs[2].plot(t_arr, ap[:, 2], 'b-', label='Actual Z', alpha=0.8)
    axs[2].set_ylabel('Pos Z (m)')
    axs[2].grid(True, linestyle='--', alpha=0.6)

    axs[3].plot(t_arr, np.degrees(dy), 'r--', label='Desired Yaw', linewidth=2)
    axs[3].plot(t_arr, np.degrees(ay), 'b-', label='Actual Yaw', alpha=0.8)
    axs[3].set_ylabel('Yaw (deg)')
    axs[3].set_xlabel('Time (s)')
    axs[3].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(teacher_folder, "tracking_curves_vs_time.png"), dpi=150)
    plt.close(fig)

    np.savez_compressed(
        os.path.join(teacher_folder, "trajectory_data.npz"),
        timestamps=t_arr,
        desired_pos=dp, actual_pos=ap, 
        desired_vel=dv, actual_vel=av,
        desired_yaw=dy, actual_yaw=ay,
        actions=acts,
        metrics=np.array([rmse, rmse_xy, max_vel, stats_start_step]),
        dynamics=assigned_param
    )
    
    return teacher_id

# =========================================================================
# 主函数入口
# =========================================================================
@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play and evaluate multi-teacher trajectory tracking."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    
    teacher_dirs = sorted(glob.glob(os.path.join(args_cli.teachers_dir, "teacher_*")))
    checkpoints = [os.path.join(d, "best_model.pt") for d in teacher_dirs if os.path.exists(os.path.join(d, "best_model.pt"))]
    
    num_teachers = len(checkpoints)
    if num_teachers == 0:
        raise ValueError(f"未在 {args_cli.teachers_dir} 找到任何包含 best_model.pt 的 teacher 文件夹！")
        
    num_envs_total = num_teachers * args_cli.envs_per_teacher
    env_cfg.scene.num_envs = num_envs_total

    env_cfg.trajectory_type = "figure8"
    env_cfg.prob_null_trajectory = 0.0
    env_cfg.train_or_play = True
    env_cfg.debug_vis = True
    
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric if args_cli.disable_fabric is not None else env_cfg.sim.use_fabric

    # --- 读取 CSV 并赋予异构动力学 ---
    csv_path = os.path.join(args_cli.teachers_dir, "teacher_dynamics.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[ERROR] 找不到对应的动力学参数文件: {csv_path}")

    dynamics_map = {}
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dynamics_map[int(row['id'])] = {
                'mass': float(row['mass']),
                'arm_length': float(row['arm_length']),
                'inertia': (float(row['Ixx']), float(row['Iyy']), float(row['Izz'])),
                'twr': float(row['twr']),
                'motor_tau_up': float(row['motor_tau_up']),
                'motor_tau_down': float(row['motor_tau_down']),
                'kappa': float(row['kappa'])
            }

    multi_env_params = []
    for i in range(num_teachers):
        teacher_id = int(os.path.basename(teacher_dirs[i]).split('_')[1])
        if teacher_id not in dynamics_map:
            raise ValueError(f"[ERROR] CSV 中缺失 id={teacher_id} 的动力学参数！")
        base_param = dynamics_map[teacher_id]
        
        for j in range(args_cli.envs_per_teacher):
            global_env_idx = i * args_cli.envs_per_teacher + j
            multi_env_params.append({
                'id': global_env_idx,
                'mass': base_param['mass'],
                'arm_length': base_param['arm_length'],
                'inertia': base_param['inertia'],
                'twr': base_param['twr'],
                'motor_tau_up': base_param['motor_tau_up'],
                'motor_tau_down': base_param['motor_tau_down'],
                'kappa': base_param['kappa']
            })

    env_cfg.dynamics.multi_teacher_params = multi_env_params

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_multiteacher_eval"
    log_dir = os.path.join(log_root_path, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"\n{'=' * 80}")
    print(f" Multi-Teacher Evaluation Initialization ")
    print(f" Found Teachers: {num_teachers}")
    print(f" Envs Per Teacher: {args_cli.envs_per_teacher}")
    print(f" Total Envs: {num_envs_total}")
    print(f" Logging to: {log_dir}")
    print(f"{'=' * 80}\n")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env)
    
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    policy = MultiTeacherPolicy(checkpoints, runner, agent_cfg.device, args_cli.envs_per_teacher)

    dt = env.unwrapped.step_dt
    obs, _ = env.get_observations()
    
    time_history, des_pos_history, act_pos_history = [], [], []
    des_vel_history, act_vel_history = [], []
    des_yaw_history, act_yaw_history, actions_history = [], [], []
    
    total_squared_error_per_env = np.zeros(num_envs_total)
    total_squared_error_xy_per_env = np.zeros(num_envs_total)
    total_squared_yaw_error_per_env = np.zeros(num_envs_total)
    max_velocity_per_env = np.zeros(num_envs_total)
    total_samples_per_env = np.zeros(num_envs_total)
    
    import omni.timeline
    timeline = omni.timeline.get_timeline_interface()
    print("[INFO] 环境加载完毕。已强制暂停仿真。")
    print("[INFO] 👉 请在 Isaac Sim 窗口中调整视角，准备好后按下【空格键】开始运行！")
    timeline.pause()
    
    timestep = 0
    start_time = time.time()

    # --- 仿真主循环 ---
    while simulation_app.is_running() and timestep < args_cli.max_steps:
        step_start_time = time.time()
        
        with torch.inference_mode():
            desired_pos = env.unwrapped.pos_des.clone()
            desired_vel = env.unwrapped.vel_des.clone()
            
            actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions)

            current_pos = env.unwrapped._robot.data.root_pos_w.clone()
            current_vel = env.unwrapped._robot.data.root_lin_vel_w.clone()
            
            pos_error_vec = current_pos - desired_pos
            squared_error = torch.sum(pos_error_vec**2, dim=1) 
            squared_error_xy = torch.sum(pos_error_vec[:, :2]**2, dim=1) 
            vel_mag = torch.norm(current_vel, dim=1)

            quat_w = env.unwrapped._robot.data.root_quat_w
            _, _, yaw_curr = euler_xyz_from_quat(quat_w)
            
            if args_cli.save_trajectory:
                time_history.append(timestep * dt)
                des_pos_history.append(desired_pos.cpu().numpy())
                act_pos_history.append(current_pos.cpu().numpy())
                des_vel_history.append(desired_vel.cpu().numpy())
                act_vel_history.append(current_vel.cpu().numpy())
                des_yaw_history.append(env.unwrapped.yaw_des.cpu().numpy())
                act_yaw_history.append(yaw_curr.cpu().numpy())
                actions_history.append(actions.cpu().numpy())

            if timestep >= STATS_START_STEP:
                total_squared_error_per_env += squared_error.cpu().numpy()
                total_squared_error_xy_per_env += squared_error_xy.cpu().numpy()
                
                batch_yaw_err = env.unwrapped.yaw_des - yaw_curr
                batch_yaw_err = torch.remainder(batch_yaw_err + torch.pi, 2 * torch.pi) - torch.pi
                total_squared_yaw_error_per_env += (batch_yaw_err**2).cpu().numpy()

                current_vels_np = vel_mag.cpu().numpy()
                max_velocity_per_env = np.maximum(max_velocity_per_env, current_vels_np)
                total_samples_per_env += 1
                        
            timestep += 1
            
            if timestep % 200 == 0:
                cur_mean_rmse = np.sqrt(torch.mean(squared_error).item())
                status = " (Collecting Stats)" if timestep >= STATS_START_STEP else " (Warmup)"
                print(f"Step {timestep:5d}/{args_cli.max_steps}{status} | Batch Mean RMSE: {cur_mean_rmse:.4f}m")
                
        if args_cli.realtime:
            sleep_time = dt - (time.time() - step_start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ==========================================================
    # 全局数据统计与最差 10% 筛选 (新增模块)
    # ==========================================================
    print(f"\n[INFO] 仿真结束，正在计算全局统计指标并筛查劣质 Teacher...")
    
    valid_samples = np.maximum(total_samples_per_env, 1)
    rmse_per_env = np.sqrt(total_squared_error_per_env / valid_samples)
    rmse_xy_per_env = np.sqrt(total_squared_error_xy_per_env / valid_samples)
    yaw_rmse_per_env = np.degrees(np.sqrt(total_squared_yaw_error_per_env / valid_samples))

    # 聚合每个 Teacher 的平均表现 (应对 envs_per_teacher > 1 的情况)
    teacher_metrics_list = []
    for i in range(num_teachers):
        start_idx = i * args_cli.envs_per_teacher
        end_idx = start_idx + args_cli.envs_per_teacher
        teacher_id = int(os.path.basename(teacher_dirs[i]).split('_')[1])
        
        t_rmse = np.mean(rmse_per_env[start_idx:end_idx])
        t_rmse_xy = np.mean(rmse_xy_per_env[start_idx:end_idx])
        t_yaw = np.mean(yaw_rmse_per_env[start_idx:end_idx])
        
        teacher_metrics_list.append({
            'id': teacher_id,
            'rmse': t_rmse,
            'rmse_xy': t_rmse_xy,
            'yaw': t_yaw,
            'param': dynamics_map[teacher_id],
            'ckpt': checkpoints[i],
            'env_start_idx': start_idx # 留给后面画图传参用
        })

    # 计算全局分布统计量
    all_rmses = np.array([m['rmse'] for m in teacher_metrics_list])
    all_rmse_xys = np.array([m['rmse_xy'] for m in teacher_metrics_list])
    all_yaws = np.array([m['yaw'] for m in teacher_metrics_list])

    # 绘制并保存全局分布直方图
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].hist(all_rmses, bins=20, color='skyblue', edgecolor='black')
    axs[0].set_title('Global RMSE Distribution (m)')
    axs[0].set_xlabel('RMSE')
    axs[0].set_ylabel('Frequency')
    
    axs[1].hist(all_rmse_xys, bins=20, color='lightgreen', edgecolor='black')
    axs[1].set_title('Global RMSE w/o Z Distribution (m)')
    axs[1].set_xlabel('RMSE XY')
    
    axs[2].hist(all_yaws, bins=20, color='salmon', edgecolor='black')
    axs[2].set_title('Global Yaw RMSE Distribution (deg)')
    axs[2].set_xlabel('Yaw RMSE')
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'global_metrics_distribution.png'), dpi=150)
    plt.close(fig)

    # 筛选最差的 10% (以 3D RMSE 为主指标进行降序排列)
    num_worst = max(1, int(num_teachers * 0.1))
    sorted_teachers = sorted(teacher_metrics_list, key=lambda x: x['rmse'], reverse=True)
    worst_10_percent = sorted_teachers[:num_worst]

    # 将全局统计报告写入文件
    global_report_path = os.path.join(log_dir, "global_evaluation_report.txt")
    with open(global_report_path, 'w') as f:
        f.write(f"RAPTOR Multi-Teacher Global Evaluation Report\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Total Teachers Evaluated: {num_teachers}\n")
        f.write(f"Envs per Teacher:         {args_cli.envs_per_teacher}\n")
        f.write(f"Total Simulation Steps:   {args_cli.max_steps}\n")
        f.write(f"Stats Calculation Start:  Step {STATS_START_STEP}\n\n")
        
        f.write(f"--- Global Performance Statistics ---\n")
        f.write(f"{'Metric':<15} | {'Mean':<8} | {'Median':<8} | {'Std Dev':<8} | {'Min':<8} | {'Max':<8}\n")
        f.write(f"{'-' * 60}\n")
        f.write(f"{'RMSE (m)':<15} | {np.mean(all_rmses):.4f}   | {np.median(all_rmses):.4f}   | {np.std(all_rmses):.4f}   | {np.min(all_rmses):.4f}   | {np.max(all_rmses):.4f}\n")
        f.write(f"{'RMSE XY (m)':<15} | {np.mean(all_rmse_xys):.4f}   | {np.median(all_rmse_xys):.4f}   | {np.std(all_rmse_xys):.4f}   | {np.min(all_rmse_xys):.4f}   | {np.max(all_rmse_xys):.4f}\n")
        f.write(f"{'Yaw RMSE (deg)':<15} | {np.mean(all_yaws):.4f}   | {np.median(all_yaws):.4f}   | {np.std(all_yaws):.4f}   | {np.min(all_yaws):.4f}   | {np.max(all_yaws):.4f}\n\n")

        f.write(f"--- ⚠️ WARNING: Bottom 10% Teachers (Worst Tracking Accuracy) ---\n")
        for i, bad_t in enumerate(worst_10_percent):
            f.write(f"\n[Rank {i+1} Worst] Teacher ID: {bad_t['id']:04d}\n")
            f.write(f"  -> RMSE: {bad_t['rmse']:>6.4f}m | RMSE XY: {bad_t['rmse_xy']:>6.4f}m | Yaw RMSE: {bad_t['yaw']:>6.4f}deg\n")
            f.write(f"  -> Target Mass: {bad_t['param']['mass']:.4f}kg | Arm: {bad_t['param']['arm_length']:.4f}m | TWR: {bad_t['param']['twr']:.2f}\n")
            f.write(f"  -> Checkpoint: {bad_t['ckpt']}\n")

    print(f"[INFO] 全局统计完毕！报告已生成: {global_report_path}")

    # ==========================================================
    # 进入原有的多进程并行出图环节
    # ==========================================================
    if args_cli.save_trajectory and len(time_history) > 0:
        t_arr = np.array(time_history)
        des_pos_arr = np.array(des_pos_history)
        act_pos_arr = np.array(act_pos_history)
        des_vel_arr = np.array(des_vel_history)
        act_vel_arr = np.array(act_vel_history)
        des_yaw_arr = np.array(des_yaw_history)
        act_yaw_arr = np.array(act_yaw_history)
        actions_arr = np.array(actions_history)

        tasks_list = []
        for t_metric in teacher_metrics_list:
            teacher_id = t_metric['id']
            env_idx = t_metric['env_start_idx'] # 提取该教师的第一个环境用于出图
            assigned_param = t_metric['param']
            
            teacher_folder = os.path.join(log_dir, f"teacher_{teacher_id:04d}")
            os.makedirs(teacher_folder, exist_ok=True)
            
            task_args = (
                teacher_id, teacher_folder, t_metric['ckpt'], assigned_param,
                des_pos_arr[:, env_idx, :], act_pos_arr[:, env_idx, :], 
                des_vel_arr[:, env_idx, :], act_vel_arr[:, env_idx, :],
                des_yaw_arr[:, env_idx], act_yaw_arr[:, env_idx], actions_arr[:, env_idx, :],
                t_arr,
                rmse_per_env[env_idx], rmse_xy_per_env[env_idx], yaw_rmse_per_env[env_idx],
                max_velocity_per_env[env_idx], STATS_START_STEP
            )
            tasks_list.append(task_args)

        num_cores = max(1, mp.cpu_count() - 2) 
        print(f"\n[INFO] 正在启动多进程加速池生成子文件夹图表 (使用 {num_cores} 个核心)...")
        
        with mp.Pool(processes=num_cores) as pool:
            for count, completed_teacher_id in enumerate(pool.imap_unordered(worker_save_and_plot, tasks_list), 1):
                print(f"       [{count}/{num_teachers}] 已成功生成 Teacher {completed_teacher_id:04d} 的图表与数据包")
            
            # (可选) 确保子进程被干净利落回收
            pool.terminate()
            pool.join()

        print(f"\n[SUCCESS] 全部并行渲染完毕！所有分析结果存放于: {log_dir}")
        
        print("[INFO] 评估任务已全部彻底完成！触发强退指令。")
        os._exit(0)  # 直接物理断电，规避 Isaac Sim 退出时的假死

    env.close()

if __name__ == "__main__":
    main()
    print("[INFO] 正在关闭 Isaac Sim 底层引擎...")
    try:
        simulation_app.close()
    except Exception as e:
        print(f"[WARNING] 引擎关闭异常: {e}")
        
    print("[INFO] 评估任务已全部彻底完成！")
    
    # 直接调用即可，注意要和上方的 print 保持同样的缩进（或者顶格写）
    os._exit(0)