# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play and evaluate trajectory tracking with actual drone parameters."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from datetime import datetime
import os
import time
import csv

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play and evaluate trajectory tracking with actual drone parameters.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playing.")
parser.add_argument("--video_length", type=int, default=2000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=10000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=100, help="Number of environments/samples to simulate (default: 100).")
parser.add_argument("--task", type=str, default=None, required=True, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_steps", type=int, default=10000, help="Maximum steps to run for trajectory tracking.")
parser.add_argument("--save_trajectory", action="store_true", default=True, help="Save trajectory data for analysis.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--realtime", action="store_true", default=False, help="Run in real-time, if possible.")

# 实机物理参数输入
parser.add_argument("--target_mass", type=float, default=1.0, help="实机目标质量 (kg)")
parser.add_argument("--target_arm", type=float, default=0.15, help="实机目标轴距 (m)")
parser.add_argument("--target_twr", type=float, default=2.2, help="实机目标推重比")

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
import torch
import torch.nn as nn      
import copy                
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

class LowerActorDeployWrapper(nn.Module):
    def __init__(self, policy, obs_normalizer, device):
        super().__init__()
        if obs_normalizer is not None:
            self.obs_normalizer = copy.deepcopy(obs_normalizer).to(device)
        else:
            self.obs_normalizer = nn.Identity().to(device)

        self.pre_rnn_mlp = copy.deepcopy(policy.pre_rnn_mlp).to(device)
        self.rnn = copy.deepcopy(policy.rnn).to(device)
        self.post_rnn_mlp = copy.deepcopy(policy.post_rnn_mlp).to(device)
        self.student = copy.deepcopy(policy.student).to(device)

        num_layers = self.rnn.num_layers
        hidden_size = self.rnn.hidden_size
        self.hidden_state = torch.zeros(num_layers, 1, hidden_size, device=device)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(observations)
        x = self.pre_rnn_mlp(x)
        x = x.unsqueeze(0)
        x, self.hidden_state = self.rnn(x, self.hidden_state)
        x = x.squeeze(0)
        x = self.post_rnn_mlp(x)
        actions_mean = self.student(x)
        actions = torch.clamp(actions_mean, -1.0, 1.0)
        return actions

    @torch.jit.export
    def reset(self):
        self.hidden_state = torch.zeros_like(self.hidden_state)

def generate_target_drone_params(target_mass, target_arm, target_twr, num_samples):
    params_list = []
    print(f"\n{'=' * 80}")
    print(f"[INFO] 正在生成 {num_samples} 组衍生实机参数...")
    print(f"       目标基准: Mass={target_mass}kg, Arm={target_arm}m, TWR={target_twr}")
    print(f"{'=' * 80}")
    
    for i in range(num_samples):
        # r_t2i = np.random.uniform(40, 1200)
        r_t2i = np.random.uniform(150, 500)
        total_thrust = target_twr * 9.81 * target_mass
        tau = total_thrust * np.sqrt(2) * target_arm
        Ixx = tau / r_t2i
        Iyy = Ixx
        Izz = Ixx * 1.832

        motor_tau_up = np.random.uniform(0.03, 0.1)
        motor_tau_down = np.random.uniform(0.03, 0.3)
        kappa = np.random.uniform(0.005, 0.05)

        params_list.append({
            'id': i,
            'mass': target_mass,
            'arm_length': target_arm,
            'inertia': (Ixx, Iyy, Izz),
            'twr': target_twr,
            'motor_tau_up': motor_tau_up,
            'motor_tau_down': motor_tau_down,
            'kappa': kappa,
        })
        
        print(f"[Env {i:03d}] Ixx/Iyy: {Ixx:.2e} | Izz: {Izz:.2e} | "
              f"Tau Up: {motor_tau_up:.4f}s | Tau Down: {motor_tau_down:.4f}s | "
              f"Kappa: {kappa:.4f}")
              
    print(f"{'=' * 80}\n")
    return params_list

# =========================================================================
# Paper Style 轨迹画图函数
# =========================================================================
def plot_paper_style_2d(desired_pos, actual_pos, actual_vel, save_path=None, title_suffix=""):
    speed = np.linalg.norm(actual_vel, axis=1)
    max_speed = np.max(speed)
    
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
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_paper_style_3d(desired_pos, actual_pos, actual_vel, save_path=None, title_suffix=""):
    speed = np.linalg.norm(actual_vel, axis=1)
    max_speed = np.max(speed)
    
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
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
# =========================================================================

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    
    env_cfg.trajectory_type = "figure8"
    env_cfg.prob_null_trajectory = 0.0 
    env_cfg.train_or_play = False  
    env_cfg.debug_vis = True
    
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric if args_cli.disable_fabric is not None else env_cfg.sim.use_fabric

    generated_params = generate_target_drone_params(
        target_mass=args_cli.target_mass,
        target_arm=args_cli.target_arm,
        target_twr=args_cli.target_twr,
        num_samples=env_cfg.scene.num_envs
    )
    env_cfg.dynamics.multi_teacher_params = generated_params

    checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO]: Loading best model checkpoint from: {checkpoint_path}")
    
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_actual_drone_eval"
    log_dir = os.path.join(log_root_path, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging actual drone evaluation in directory: {log_dir}")

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
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    
    loaded_dict = torch.load(checkpoint_path, map_location=agent_cfg.device)
    full_state_dict = loaded_dict['model_state_dict']
    student_only_state_dict = {}
    
    for k, v in full_state_dict.items():
        if "teachers_list" in k: continue
        if "teacher" in k and "student" not in k: continue
        student_only_state_dict[k] = v
            
    runner.alg.policy.load_state_dict(student_only_state_dict, strict=False)
    print("[INFO] Model weights loaded (Student only).")

    if agent_cfg.empirical_normalization:
        if 'obs_norm_state_dict' in loaded_dict:
            runner.obs_normalizer.load_state_dict(loaded_dict['obs_norm_state_dict'])
    
    runner.eval_mode()
    policy = runner.get_inference_policy(device=agent_cfg.device)
    policy_model = runner.alg.policy
    dt = env.unwrapped.step_dt
    obs, _ = env.get_observations()
    
    print(f"\n{'=' * 80}")
    print(f"Actual Drone Batch Evaluation (Robustness Test)")
    print(f"Number of target variations: {env.num_envs}")
    print(f"Maximum steps: {args_cli.max_steps}")
    print(f"{'=' * 80}\n")
    
    timestep = 0
    start_time = time.time()
    
    num_envs = env.num_envs
    total_squared_error_per_env = np.zeros(num_envs)
    total_squared_error_xy_per_env = np.zeros(num_envs)
    total_squared_yaw_error_per_env = np.zeros(num_envs)
    max_velocity_per_env = np.zeros(num_envs)
    total_samples_per_env = np.zeros(num_envs)
    has_crashed_per_env = np.zeros(num_envs, dtype=bool)

    time_history = []
    des_pos_history = []
    act_pos_history = []
    des_vel_history = []
    act_vel_history = []
    des_yaw_history = []
    act_yaw_history = []

    import omni.timeline 
    timeline = omni.timeline.get_timeline_interface()
    print("[INFO] 环境加载完毕。已强制暂停仿真。")
    print("[INFO] 👉 请在 Isaac Sim 窗口中调整视角，准备好后按下【空格键】开始运行（如果使用 --headless 则自动继续）！")
    timeline.pause()

    while simulation_app.is_running() and timestep < args_cli.max_steps:
        step_start_time = time.time()
        
        with torch.inference_mode():
            desired_pos = env.unwrapped.pos_des.clone()
            desired_vel = env.unwrapped.vel_des.clone()
            
            actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions)

            died_this_step = env.unwrapped.reset_terminated.cpu().numpy()
            has_crashed_per_env |= died_this_step

            if hasattr(policy_model, "reset"):
                policy_model.reset(dones)
            
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
                print(f"Step {timestep:5d}{status} | Current Batch Mean RMSE: {cur_mean_rmse:.4f}m")
                
        if args_cli.realtime:
            sleep_time = dt - (time.time() - step_start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    survived_mask = ~has_crashed_per_env
    num_survived = np.sum(survived_mask)
    survival_rate = (num_survived / num_envs) * 100.0

    valid_samples = np.maximum(total_samples_per_env, 1)
    rmse_per_env = np.sqrt(total_squared_error_per_env / valid_samples)
    rmse_xy_per_env = np.sqrt(total_squared_error_xy_per_env / valid_samples)
    yaw_rmse_per_env = np.degrees(np.sqrt(total_squared_yaw_error_per_env / valid_samples))

    if num_survived > 0:
        clean_rmse = rmse_per_env[survived_mask]
        clean_yaw = yaw_rmse_per_env[survived_mask]
        
        stat_mean = np.mean(clean_rmse)
        stat_std = np.std(clean_rmse)
        stat_min = np.min(clean_rmse)
        stat_max = np.max(clean_rmse)
        stat_median = np.median(clean_rmse)
        stat_p90 = np.percentile(clean_rmse, 90)
        stat_yaw_mean = np.mean(clean_yaw)
    else:
        stat_mean = stat_std = stat_min = stat_max = stat_median = stat_p90 = stat_yaw_mean = 0.0

    stat_max_vel = np.max(max_velocity_per_env)
    
    print(f"\n{'=' * 80}")
    print(f"Actual Drone Simulation Results (Calculated from step {STATS_START_STEP} onwards):")
    print(f"{'-' * 80}")
    print(f"  Target Params:      Mass={args_cli.target_mass}kg, Arm={args_cli.target_arm}m, TWR={args_cli.target_twr}")
    print(f"  Num Envs (Samples): {env.num_envs}")
    print(f"{'-' * 80}")
    print(f"  ⭐ Survival Rate:   {survival_rate:.1f}% ({num_survived}/{num_envs} survived full trajectory)")
    print(f"{'-' * 80}")
    if num_survived > 0:
        print(f"  Clean RMSE Distribution (across {num_survived} surviving variations):")
        print(f"    Mean   : {stat_mean:.4f} m  (± {stat_std:.4f})")
        print(f"    Median : {stat_median:.4f} m")
        print(f"    Min    : {stat_min:.4f} m")
        print(f"    Max    : {stat_max:.4f} m  <-- 存活飞机里的最差情况")
        print(f"    90th % : {stat_p90:.4f} m")
        print(f"  Clean Mean Yaw RMSE: {stat_yaw_mean:.4f} deg")
    else:
        print(f"  [CRITICAL] 所有飞机均发生坠毁/失控，无法计算纯净追踪精度！")
    print(f"{'-' * 80}")
    print(f"  Absolute Max Vel:   {stat_max_vel:.4f} m/s (包含坠毁前的挣扎)")
    print(f"  Total Steps:        {timestep}")
    
    # 导出详细参数与结果到全局 CSV
    csv_file_path = os.path.join(log_dir, "detailed_tracking_results.csv")
    with open(csv_file_path, mode='w', newline='') as csv_file:
        fieldnames = [
            'env_id', 'mass', 'arm_length', 'twr', 
            'Ixx', 'Iyy', 'Izz', 'motor_tau_up', 'motor_tau_down', 'kappa',
            'survived', 'rmse_m', 'rmse_xy_m', 'yaw_rmse_deg', 'max_vel_m_s'
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(num_envs):
            params = generated_params[i]
            survived = not has_crashed_per_env[i]
            writer.writerow({
                'env_id': i,
                'mass': params['mass'],
                'arm_length': params['arm_length'],
                'twr': params['twr'],
                'Ixx': params['inertia'][0],
                'Iyy': params['inertia'][1],
                'Izz': params['inertia'][2],
                'motor_tau_up': params['motor_tau_up'],
                'motor_tau_down': params['motor_tau_down'],
                'kappa': params['kappa'],
                'survived': survived,
                'rmse_m': rmse_per_env[i],
                'rmse_xy_m': rmse_xy_per_env[i],
                'yaw_rmse_deg': yaw_rmse_per_env[i],
                'max_vel_m_s': max_velocity_per_env[i]
            })

    print(f"\n[INFO] Detailed parameter and tracking results saved to CSV: {csv_file_path}")

    # ==========================================================
    # 遍历所有环境，生成专属文件夹并绘制 Paper-Style 轨迹图 + 保存独立精度数据
    # ==========================================================
    if args_cli.save_trajectory and len(time_history) > 0:
        print(f"\n[INFO] 正在为 {num_envs} 个环境生成轨迹图像与精度报告，请耐心等待...")
        
        t_arr = np.array(time_history)
        des_pos_arr = np.array(des_pos_history)
        act_pos_arr = np.array(act_pos_history)
        act_vel_arr = np.array(act_vel_history)
        des_yaw_arr = np.array(des_yaw_history)
        act_yaw_arr = np.array(act_yaw_history)

        for i in range(num_envs):
            # 进度提示
            if (i + 1) % 10 == 0 or i == num_envs - 1:
                print(f"       已处理: {i + 1}/{num_envs} 个环境...")

            env_folder = os.path.join(log_dir, f"env_{i:03d}")
            os.makedirs(env_folder, exist_ok=True)
            
            dp = des_pos_arr[:, i, :]
            ap = act_pos_arr[:, i, :]
            av = act_vel_arr[:, i, :]
            dy = des_yaw_arr[:, i]
            ay = act_yaw_arr[:, i]
            is_survived = not has_crashed_per_env[i]
            status_text = "[SURVIVED]" if is_survived else "[CRASHED]"

            # ---------------------------------------------------------
            # 1. 写入独立的 tracking_stats.txt
            # ---------------------------------------------------------
            stats_txt_path = os.path.join(env_folder, "tracking_stats.txt")
            with open(stats_txt_path, 'w') as f:
                f.write(f"Environment {i:03d} Tracking Statistics\n")
                f.write(f"{'=' * 45}\n")
                f.write(f"Status:          {status_text}\n")
                f.write(f"{'-' * 45}\n")
                f.write(f"RMSE [m]:        {rmse_per_env[i]:.4f}\n")
                f.write(f"RMSE w/o z [m]:  {rmse_xy_per_env[i]:.4f}\n")
                f.write(f"Yaw RMSE [deg]:  {yaw_rmse_per_env[i]:.4f}\n")
                f.write(f"Max Vel [m/s]:   {max_velocity_per_env[i]:.4f}\n")
                f.write(f"{'-' * 45}\n")
                f.write(f"Generated Physical Parameters:\n")
                p = generated_params[i]
                f.write(f"  Mass:          {p['mass']:.4f} kg\n")
                f.write(f"  Arm Length:    {p['arm_length']:.4f} m\n")
                f.write(f"  TWR:           {p['twr']:.4f}\n")
                f.write(f"  Ixx:           {p['inertia'][0]:.4e}\n")
                f.write(f"  Iyy:           {p['inertia'][1]:.4e}\n")
                f.write(f"  Izz:           {p['inertia'][2]:.4e}\n")
                f.write(f"  Motor Tau Up:  {p['motor_tau_up']:.4f} s\n")
                f.write(f"  Motor Tau Down:{p['motor_tau_down']:.4f} s\n")
                f.write(f"  Kappa:         {p['kappa']:.4f}\n")

            # ---------------------------------------------------------
            # 2. 生成 2D 速度投影图 (XY, XZ, YZ)
            # ---------------------------------------------------------
            path_2d = os.path.join(env_folder, '2d_velocity_trajectory.png')
            plot_paper_style_2d(dp, ap, av, save_path=path_2d, title_suffix=status_text)
            
            # ---------------------------------------------------------
            # 3. 生成 3D 速度轨迹图
            # ---------------------------------------------------------
            path_3d = os.path.join(env_folder, '3d_velocity_trajectory.png')
            plot_paper_style_3d(dp, ap, av, save_path=path_3d, title_suffix=status_text)

            # ---------------------------------------------------------
            # 4. 基础的时间跟踪曲线图 (X, Y, Z, Yaw)
            # ---------------------------------------------------------
            fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
            color_theme = 'green' if is_survived else 'red'
            fig.suptitle(f"Tracking Performance - Env {i:03d} {status_text}", fontsize=16, color=color_theme, fontweight='bold')
            
            axs[0].plot(t_arr, dp[:, 0], 'r--', label='Desired X', linewidth=2)
            axs[0].plot(t_arr, ap[:, 0], 'b-', label='Actual X', alpha=0.8)
            axs[0].set_ylabel('Position X (m)')
            axs[0].legend(loc='upper right')
            axs[0].grid(True, linestyle='--', alpha=0.6)

            axs[1].plot(t_arr, dp[:, 1], 'r--', label='Desired Y', linewidth=2)
            axs[1].plot(t_arr, ap[:, 1], 'b-', label='Actual Y', alpha=0.8)
            axs[1].set_ylabel('Position Y (m)')
            axs[1].legend(loc='upper right')
            axs[1].grid(True, linestyle='--', alpha=0.6)

            axs[2].plot(t_arr, dp[:, 2], 'r--', label='Desired Z', linewidth=2)
            axs[2].plot(t_arr, ap[:, 2], 'b-', label='Actual Z', alpha=0.8)
            axs[2].set_ylabel('Position Z (m)')
            axs[2].legend(loc='upper right')
            axs[2].grid(True, linestyle='--', alpha=0.6)

            axs[3].plot(t_arr, np.degrees(dy), 'r--', label='Desired Yaw', linewidth=2)
            axs[3].plot(t_arr, np.degrees(ay), 'b-', label='Actual Yaw', alpha=0.8)
            axs[3].set_ylabel('Yaw (deg)')
            axs[3].set_xlabel('Time (s)')
            axs[3].legend(loc='upper right')
            axs[3].grid(True, linestyle='--', alpha=0.6)

            plt.tight_layout()
            plt.savefig(os.path.join(env_folder, "tracking_curves_vs_time.png"), dpi=150)
            plt.close(fig)

            # ---------------------------------------------------------
            # 5. 导出数据包
            # ---------------------------------------------------------
            np.savez_compressed(
                os.path.join(env_folder, "flight_data.npz"),
                time=t_arr,
                des_pos=dp, act_pos=ap, act_vel=av,
                des_yaw=dy, act_yaw=ay,
                params=generated_params[i]
            )

        print("[INFO] 所有独立环境曲线图生成完毕！")

        # ==========================================================
        # [新增] 绘制全局的动力学生存分布对比图 (Pairplot & Parallel)
        # ==========================================================
        print(f"\n[INFO] 正在生成【存活/坠毁】的参数分布边界对比图，保存于根目录...")
        try:
            import pandas as pd
            import seaborn as sns
            from pandas.plotting import parallel_coordinates

            # 构造 DataFrame
            records = []
            for i in range(num_envs):
                p = generated_params[i]
                survived = not has_crashed_per_env[i]
                records.append({
                    'Status': 'Survived' if survived else 'Crashed',
                    'Ixx': p['inertia'][0],  # 仅用 Ixx 代表整体惯量
                    'Tau_Up': p['motor_tau_up'],
                    'Tau_Down': p['motor_tau_down'],
                    'Kappa': p['kappa']
                })
            
            df_plot = pd.DataFrame(records)
            cols_to_plot = ['Ixx', 'Tau_Up', 'Tau_Down', 'Kappa']
            
            # --- 1. 散点矩阵分布图 (Pairplot) ---
            sns.set_theme(style="whitegrid")
            palette = {'Survived': '#2ca02c', 'Crashed': '#d62728'}  # 绿/红
            
            g = sns.pairplot(
                df_plot,
                vars=cols_to_plot,
                hue='Status',
                palette=palette,
                diag_kind='kde',
                plot_kws={'alpha': 0.7, 's': 60, 'edgecolor': 'w'},
                corner=True
            )
            g.fig.suptitle("Dynamics Parameters Survival Boundaries", y=1.02, fontsize=16, fontweight='bold')
            
            save_path_pair = os.path.join(log_dir, "dynamics_survival_pairplot.png")
            plt.savefig(save_path_pair, dpi=200, bbox_inches='tight')
            plt.close()

            # --- 2. 平行坐标图 (Parallel Coordinates) ---
            plt.figure(figsize=(10, 6))
            df_norm = df_plot.copy()
            # [0, 1] 归一化
            for col in cols_to_plot:
                min_v = df_norm[col].min()
                max_v = df_norm[col].max()
                if max_v > min_v:
                    df_norm[col] = (df_norm[col] - min_v) / (max_v - min_v)
            
            # 绘制：先把 Survived 垫在下面(透明度高点)，再把 Crashed 盖在上面(透明度低点)
            df_surv = df_norm[df_norm['Status'] == 'Survived']
            df_cras = df_norm[df_norm['Status'] == 'Crashed']
            
            if not df_surv.empty:
                parallel_coordinates(df_surv, 'Status', color=['#2ca02c'], alpha=0.3)
            if not df_cras.empty:
                parallel_coordinates(df_cras, 'Status', color=['#d62728'], alpha=0.7, linewidth=2.5)
                
            plt.title("Parallel Coordinates: Survived vs Crashed (Normalized 0-1)", fontsize=14, fontweight='bold')
            plt.ylabel("Normalized Value")
            plt.xticks(rotation=0, fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.5)
            
            # 整理图例防止重复
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc='upper right')

            save_path_para = os.path.join(log_dir, "dynamics_survival_parallel.png")
            plt.savefig(save_path_para, dpi=200, bbox_inches='tight')
            plt.close()

            print(f"[SUCCESS] 生存分布对比图已成功生成: ")
            print(f"  -> {save_path_pair}")
            print(f"  -> {save_path_para}")

        except ImportError:
            print("[WARNING] 缺少 pandas 或 seaborn，跳过生成参数生存分布图。可通过 'pip install pandas seaborn' 安装。")
        except Exception as e:
            print(f"[WARNING] 生成生存分布图失败: {e}")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()