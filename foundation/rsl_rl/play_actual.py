# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play and evaluate trajectory tracking with actual drone parameters."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # 需要终端执行 pip install PyQt5
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
parser.add_argument("--sample_raptor", action="store_true", default=False, help="If True, sample dynamics strictly matching train_teacher_multi.py")

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
def generate_raptor_drone_params(num_samples):
    params_list = []
    print(f"\n{'=' * 80}")
    print(f"[INFO] 正在生成 {num_samples} 组 Raptor Teacher 动力学参数...")
    print(f"{'=' * 80}")
    
    for i in range(num_samples):
        # Teacher multi sampling logic
        twr = np.random.uniform(1.5, 5.0)
        m_min = 0.02
        m_max = 5.0
        s = np.random.uniform(np.cbrt(m_min), np.cbrt(m_max))
        mass = s ** 3
        
        m_cf = 0.032 
        l_cf = 0.04384 
        base_ratio = l_cf / (m_cf**(1/3)) 
        u = np.random.normal(-0.1, 0.1) 
        if u < 0: 
            s_ms = 1.0 / (1.0 - u)
        else: 
            s_ms = 1.0 + u
        arm_length = base_ratio * (mass**(1/3)) / s_ms
        
        r_t2i = np.random.uniform(40, 1200)
        total_thrust = twr * 9.81 * mass
        tau = total_thrust * np.sqrt(2) * arm_length
        Ixx = tau / r_t2i
        Iyy = Ixx 
        Izz = Ixx * 1.832 
        
        motor_tau_up = np.random.uniform(0.03, 0.1)
        motor_tau_down = np.random.uniform(0.03, 0.3)
        kappa = np.random.uniform(0.005, 0.05)

        params_list.append({
            'id': i,
            'mass': mass,
            'arm_length': arm_length,
            'inertia': (Ixx, Iyy, Izz),
            'twr': twr,
            'motor_tau_up': motor_tau_up,
            'motor_tau_down': motor_tau_down,
            'kappa': kappa,
        })
        
        print(f"[Env {i:03d}] Mass: {mass:.3f}kg | Arm: {arm_length:.3f}m | TWR: {twr:.2f} | "
              f"Ixx/Iyy: {Ixx:.2e} | Izz: {Izz:.2e} | "
              f"Tau Up: {motor_tau_up:.4f}s | Tau Down: {motor_tau_down:.4f}s")
              
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

    # 替换原本的 generated_params 赋值部分
    if args_cli.sample_raptor:
        generated_params = generate_raptor_drone_params(
            num_samples=env_cfg.scene.num_envs
        )
    else:
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
    # obs, _ = env.get_observations()
    obs, _ = env.reset()

    # =========================================================================
    # [新增] 导出实物部署模型 (TorchScript) - Student/Lower Network
    # =========================================================================
    export_device = agent_cfg.device
    print(f"\n[INFO] 正在构建 Lower (Student) 实物部署模型...")
    try:
        # 获取最新的 normalizer（如果有的话）
        obs_normalizer = getattr(runner, "obs_normalizer", None)
        
        # 1. 实例化 Wrapper 并置为 eval 模式
        deploy_model = LowerActorDeployWrapper(
            policy=policy_model, 
            obs_normalizer=obs_normalizer, 
            device=export_device
        )
        deploy_model.eval()
        
        # 2. 根据环境推断单帧观测维度
        total_obs_dim = obs.shape[1]
        print(f"[INFO] 预期实物端输入的单帧观测向量维度 (obs_dim): {total_obs_dim}")
        
        # 3. 构造 dummy_input 追踪 JIT 编译
        dummy_obs_input = torch.randn(1, total_obs_dim, device=export_device)
        
        with torch.inference_mode():
            trace_model = torch.jit.script(deploy_model, dummy_obs_input)
            
            # 保存在当前 checkpoint 目录下
            export_dir = os.path.dirname(checkpoint_path)
            export_path = os.path.join(export_dir, "down_actor_deploy.pt")
            trace_model.save(export_path)
            
            print(f"[SUCCESS] Lower 部署模型已成功保存至: {export_path}")
            print(f"        -> 实体机(LibTorch)直接加载此文件即可运行")
            print(f"        -> 实物端调用输入形状需严格为: (1, {total_obs_dim})\n")
    except Exception as e:
        print(f"[ERROR] 导出部署模型失败: {e}\n")
    # =========================================================================
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
    # [新增] 记录 Env 0 的感知误差
    env0_perceived_pos_err_history = []
    env0_perceived_vel_err_history = []
    
    # [新增] 记录 Env 0 前 20 秒的每一帧观测和动作 (用于 CSV)
    env0_csv_data = []

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
            
            # [新增] 仅收集前20秒（基于当前步长与总步数转换），写入到内存中
            if timestep * dt <= 20.0:
                env0_csv_data.append((
                    timestep * dt,
                    obs[0].cpu().numpy().copy(),
                    actions[0].cpu().numpy().copy()
                ))

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
                # [新增] 提取 Env 0 的感知误差 (位置 0:3, 速度 12:15)
                env0_perceived_pos_err_history.append(obs[0, 0:3].cpu().numpy())
                env0_perceived_vel_err_history.append(obs[0, 12:15].cpu().numpy())

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
    rmse_xy_per_env = np.sqrt(total_squared_error_xy_per_env / valid_samples)  # XY 平面误差
    yaw_rmse_per_env = np.degrees(np.sqrt(total_squared_yaw_error_per_env / valid_samples))

    if num_survived > 0:
        clean_rmse = rmse_per_env[survived_mask]
        clean_xy_rmse = rmse_xy_per_env[survived_mask]  # 存活飞机的 XY 误差
        clean_yaw = yaw_rmse_per_env[survived_mask]
        
        # 3D RMSE 统计
        stat_mean = np.mean(clean_rmse)
        stat_std = np.std(clean_rmse)
        stat_min = np.min(clean_rmse)
        stat_max = np.max(clean_rmse)
        stat_median = np.median(clean_rmse)
        stat_p90 = np.percentile(clean_rmse, 90)

        # XY 平面 RMSE 统计
        stat_xy_mean = np.mean(clean_xy_rmse)
        stat_xy_std = np.std(clean_xy_rmse)
        stat_xy_median = np.median(clean_xy_rmse)
        stat_xy_max = np.max(clean_xy_rmse)

        stat_yaw_mean = np.mean(clean_yaw)
    else:
        stat_mean = stat_std = stat_min = stat_max = stat_median = stat_p90 = 0.0
        stat_xy_mean = stat_xy_std = stat_xy_median = stat_xy_max = 0.0
        stat_yaw_mean = 0.0

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
        print(f"  Clean 3D RMSE Distribution (across {num_survived} surviving variations):")
        print(f"    Mean   : {stat_mean:.4f} m  (± {stat_std:.4f})")
        print(f"    Median : {stat_median:.4f} m")
        print(f"    Min    : {stat_min:.4f} m")
        print(f"    Max    : {stat_max:.4f} m  <-- 3D 存活飞机里的最差情况")
        print(f"    90th % : {stat_p90:.4f} m")
        print(f"{'-' * 40}")
        print(f"  Clean XY-Plane RMSE Distribution (Without Z axis):")
        print(f"    Mean   : {stat_xy_mean:.4f} m  (± {stat_xy_std:.4f})")
        print(f"    Median : {stat_xy_median:.4f} m")
        print(f"    Max    : {stat_xy_max:.4f} m")
        print(f"{'-' * 40}")
        print(f"  Clean Mean Yaw RMSE: {stat_yaw_mean:.4f} deg")
    else:
        print(f"  [CRITICAL] 所有飞机均发生坠毁/失控，无法计算纯净追踪精度！")
    print(f"{'-' * 80}")
    print(f"  Absolute Max Vel:   {stat_max_vel:.4f} m/s (包含坠毁前的挣扎)")
    print(f"  Total Steps:        {timestep}")

    # [新增] 导出 Env 0 前20秒的 obs 和 actions 到 csv
    if env0_csv_data:
        csv_file_path = os.path.join(log_dir, "env0_obs_act_20s.csv")
        with open(csv_file_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            obs_dim = len(env0_csv_data[0][1])
            act_dim = len(env0_csv_data[0][2])
            header = ["time"] + [f"obs_{i}" for i in range(obs_dim)] + [f"act_{i}" for i in range(act_dim)]
            writer.writerow(header)
            for row in env0_csv_data:
                t, o, a = row
                writer.writerow([t] + o.tolist() + a.tolist())
        print(f"\n[INFO] Env 0 前20秒的观测和动作已保存至: {csv_file_path}")

        # ==========================================================
        # [新增] 绘制 Env 0 前 20 秒各观测物理量与动作的随时间变化曲线
        # ==========================================================
        print(f"[INFO] 正在生成 Env 0 前 20 秒观测物理量变化曲线图...")
        try:
            # 提取时间、观测和动作数据进行绘图
            t_data = np.array([row[0] for row in env0_csv_data])
            obs_data = np.array([row[1] for row in env0_csv_data])
            act_data = np.array([row[2] for row in env0_csv_data])
            
            fig_obs, axs_obs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
            
            # 1. 绘制位置误差 (obs_0:3)
            axs_obs[0].plot(t_data, obs_data[:, 0], label='X Pos Error', alpha=0.8)
            axs_obs[0].plot(t_data, obs_data[:, 1], label='Y Pos Error', alpha=0.8)
            axs_obs[0].plot(t_data, obs_data[:, 2], label='Z Pos Error (Drift)', color='red', linewidth=2)
            axs_obs[0].set_ylabel('Pos Error (m)')
            axs_obs[0].set_title('Env 0: Position Error (Body Frame)', fontweight='bold')
            axs_obs[0].legend(loc='upper right')
            axs_obs[0].grid(True, linestyle='--', alpha=0.6)
            
            # 2. 绘制速度误差 (obs_12:15)
            axs_obs[1].plot(t_data, obs_data[:, 12], label='X Vel Error', alpha=0.8)
            axs_obs[1].plot(t_data, obs_data[:, 13], label='Y Vel Error', alpha=0.8)
            axs_obs[1].plot(t_data, obs_data[:, 14], label='Z Vel Error (Drift)', color='red', linewidth=2)
            axs_obs[1].set_ylabel('Vel Error (m/s)')
            axs_obs[1].set_title('Env 0: Velocity Error (Body Frame)', fontweight='bold')
            axs_obs[1].legend(loc='upper right')
            axs_obs[1].grid(True, linestyle='--', alpha=0.6)
            
            # 3. 绘制机体角速度 (obs_15:18)
            axs_obs[2].plot(t_data, obs_data[:, 15], label='Roll Rate (X)', alpha=0.8)
            axs_obs[2].plot(t_data, obs_data[:, 16], label='Pitch Rate (Y)', alpha=0.8)
            axs_obs[2].plot(t_data, obs_data[:, 17], label='Yaw Rate (Z)', alpha=0.8)
            axs_obs[2].set_ylabel('Ang Vel (rad/s)')
            axs_obs[2].set_title('Env 0: Angular Velocity (Body Frame)', fontweight='bold')
            axs_obs[2].legend(loc='upper right')
            axs_obs[2].grid(True, linestyle='--', alpha=0.6)

            # 4. 绘制动作指令 (act_0:4)
            axs_obs[3].plot(t_data, act_data[:, 0], label='Motor 1 Cmd', alpha=0.8)
            axs_obs[3].plot(t_data, act_data[:, 1], label='Motor 2 Cmd', alpha=0.8)
            axs_obs[3].plot(t_data, act_data[:, 2], label='Motor 3 Cmd', alpha=0.8)
            axs_obs[3].plot(t_data, act_data[:, 3], label='Motor 4 Cmd', alpha=0.8)
            axs_obs[3].set_ylabel('Action [-1, 1]')
            axs_obs[3].set_xlabel('Time (s)')
            axs_obs[3].set_title('Env 0: Policy Actions', fontweight='bold')
            axs_obs[3].legend(loc='upper right')
            axs_obs[3].grid(True, linestyle='--', alpha=0.6)

            # 防止科学计数法导致坐标轴显示重叠
            for ax in axs_obs:
                ax.ticklabel_format(useOffset=False, style='plain')
            
            plt.tight_layout()
            obs_plot_path = os.path.join(log_dir, "env0_obs_act_20s_curves.png")
            plt.savefig(obs_plot_path, dpi=200)
            plt.close(fig_obs)
            print(f"[SUCCESS] Env 0 观测物理量变化曲线已成功保存至: {obs_plot_path}")
        except Exception as e:
            print(f"[ERROR] 绘制 Env 0 观测物理量变化曲线失败: {e}")
    # # 导出详细参数与结果到全局 CSV
    # csv_file_path = os.path.join(log_dir, "detailed_tracking_results.csv")
    # with open(csv_file_path, mode='w', newline='') as csv_file:
    #     fieldnames = [
    #         'env_id', 'mass', 'arm_length', 'twr', 
    #         'Ixx', 'Iyy', 'Izz', 'motor_tau_up', 'motor_tau_down', 'kappa',
    #         'survived', 'rmse_m', 'rmse_xy_m', 'yaw_rmse_deg', 'max_vel_m_s'
    #     ]
    #     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    #     writer.writeheader()
        
    #     for i in range(num_envs):
    #         params = generated_params[i]
    #         survived = not has_crashed_per_env[i]
    #         writer.writerow({
    #             'env_id': i,
    #             'mass': params['mass'],
    #             'arm_length': params['arm_length'],
    #             'twr': params['twr'],
    #             'Ixx': params['inertia'][0],
    #             'Iyy': params['inertia'][1],
    #             'Izz': params['inertia'][2],
    #             'motor_tau_up': params['motor_tau_up'],
    #             'motor_tau_down': params['motor_tau_down'],
    #             'kappa': params['kappa'],
    #             'survived': survived,
    #             'rmse_m': rmse_per_env[i],
    #             'rmse_xy_m': rmse_xy_per_env[i],
    #             'yaw_rmse_deg': yaw_rmse_per_env[i],
    #             'max_vel_m_s': max_velocity_per_env[i]
    #         })

    # print(f"\n[INFO] Detailed parameter and tracking results saved to CSV: {csv_file_path}")

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

            # # ---------------------------------------------------------
            # # 1. 写入独立的 tracking_stats.txt
            # # ---------------------------------------------------------
            # stats_txt_path = os.path.join(env_folder, "tracking_stats.txt")
            # with open(stats_txt_path, 'w') as f:
            #     f.write(f"Environment {i:03d} Tracking Statistics\n")
            #     f.write(f"{'=' * 45}\n")
            #     f.write(f"Status:          {status_text}\n")
            #     f.write(f"{'-' * 45}\n")
            #     f.write(f"RMSE [m]:        {rmse_per_env[i]:.4f}\n")
            #     f.write(f"RMSE w/o z [m]:  {rmse_xy_per_env[i]:.4f}\n")
            #     f.write(f"Yaw RMSE [deg]:  {yaw_rmse_per_env[i]:.4f}\n")
            #     f.write(f"Max Vel [m/s]:   {max_velocity_per_env[i]:.4f}\n")
            #     f.write(f"{'-' * 45}\n")
            #     f.write(f"Generated Physical Parameters:\n")
            #     p = generated_params[i]
            #     f.write(f"  Mass:          {p['mass']:.4f} kg\n")
            #     f.write(f"  Arm Length:    {p['arm_length']:.4f} m\n")
            #     f.write(f"  TWR:           {p['twr']:.4f}\n")
            #     f.write(f"  Ixx:           {p['inertia'][0]:.4e}\n")
            #     f.write(f"  Iyy:           {p['inertia'][1]:.4e}\n")
            #     f.write(f"  Izz:           {p['inertia'][2]:.4e}\n")
            #     f.write(f"  Motor Tau Up:  {p['motor_tau_up']:.4f} s\n")
            #     f.write(f"  Motor Tau Down:{p['motor_tau_down']:.4f} s\n")
            #     f.write(f"  Kappa:         {p['kappa']:.4f}\n")

            # ---------------------------------------------------------
            # 2. 生成 2D 速度投影图 (XY, XZ, YZ)
            # ---------------------------------------------------------
            path_2d = os.path.join(env_folder, '2d_velocity_trajectory.png')
            plot_paper_style_2d(dp, ap, av, save_path=path_2d, title_suffix=status_text)
            
            # # ---------------------------------------------------------
            # # 3. 生成 3D 速度轨迹图
            # # ---------------------------------------------------------
            # path_3d = os.path.join(env_folder, '3d_velocity_trajectory.png')
            # plot_paper_style_3d(dp, ap, av, save_path=path_3d, title_suffix=status_text)

            # # ---------------------------------------------------------
            # # 4. 基础的时间跟踪曲线图 (X, Y, Z, Yaw)
            # # ---------------------------------------------------------
            # fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
            # color_theme = 'green' if is_survived else 'red'
            # fig.suptitle(f"Tracking Performance - Env {i:03d} {status_text}", fontsize=16, color=color_theme, fontweight='bold')
            
            # axs[0].plot(t_arr, dp[:, 0], 'r--', label='Desired X', linewidth=2)
            # axs[0].plot(t_arr, ap[:, 0], 'b-', label='Actual X', alpha=0.8)
            # axs[0].set_ylabel('Position X (m)')
            # axs[0].legend(loc='upper right')
            # axs[0].grid(True, linestyle='--', alpha=0.6)

            # axs[1].plot(t_arr, dp[:, 1], 'r--', label='Desired Y', linewidth=2)
            # axs[1].plot(t_arr, ap[:, 1], 'b-', label='Actual Y', alpha=0.8)
            # axs[1].set_ylabel('Position Y (m)')
            # axs[1].legend(loc='upper right')
            # axs[1].grid(True, linestyle='--', alpha=0.6)

            # axs[2].plot(t_arr, dp[:, 2], 'r--', label='Desired Z', linewidth=2)
            # axs[2].plot(t_arr, ap[:, 2], 'b-', label='Actual Z', alpha=0.8)
            # axs[2].set_ylabel('Position Z (m)')
            # axs[2].legend(loc='upper right')
            # axs[2].grid(True, linestyle='--', alpha=0.6)

            # axs[3].plot(t_arr, np.degrees(dy), 'r--', label='Desired Yaw', linewidth=2)
            # axs[3].plot(t_arr, np.degrees(ay), 'b-', label='Actual Yaw', alpha=0.8)
            # axs[3].set_ylabel('Yaw (deg)')
            # axs[3].set_xlabel('Time (s)')
            # axs[3].legend(loc='upper right')
            # axs[3].grid(True, linestyle='--', alpha=0.6)

            # plt.tight_layout()
            # plt.savefig(os.path.join(env_folder, "tracking_curves_vs_time.png"), dpi=150)
            # plt.close(fig)

            # ---------------------------------------------------------
            # 5. 导出数据包
            # ---------------------------------------------------------
            # np.savez_compressed(
            #     os.path.join(env_folder, "flight_data.npz"),
            #     time=t_arr,
            #     des_pos=dp, act_pos=ap, act_vel=av,
            #     des_yaw=dy, act_yaw=ay,
            #     params=generated_params[i]
            # )

        print("[INFO] 所有独立环境曲线图生成完毕！")

        # ==========================================================
        # [新增] 绘制 Env 0 的 Perceived Errors (锯齿形漂移验证)
        # ==========================================================
        print(f"\n[INFO] 正在生成 Env 0 感知误差 (EKF 漂移) 曲线图...")
        try:
            env0_pos_err_arr = np.array(env0_perceived_pos_err_history)
            env0_vel_err_arr = np.array(env0_perceived_vel_err_history)
            
            fig_err, axs_err = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            # 绘制位置感知误差
            axs_err[0].plot(t_arr, env0_pos_err_arr[:, 0], label='X Error', alpha=0.8)
            axs_err[0].plot(t_arr, env0_pos_err_arr[:, 1], label='Y Error', alpha=0.8)
            axs_err[0].plot(t_arr, env0_pos_err_arr[:, 2], label='Z Error (Drift)', color='red', linewidth=2)
            axs_err[0].set_ylabel('Perceived Pos Error (m)')
            axs_err[0].set_title('Env 0: Perceived Position Error (Body Frame)', fontweight='bold')
            axs_err[0].legend(loc='upper right')
            axs_err[0].grid(True, linestyle='--', alpha=0.6)
            
            # 绘制线速度感知误差
            axs_err[1].plot(t_arr, env0_vel_err_arr[:, 0], label='X Error', alpha=0.8)
            axs_err[1].plot(t_arr, env0_vel_err_arr[:, 1], label='Y Error', alpha=0.8)
            axs_err[1].plot(t_arr, env0_vel_err_arr[:, 2], label='Z Error (Drift)', color='red', linewidth=2)
            axs_err[1].set_ylabel('Perceived Vel Error (m/s)')
            axs_err[1].set_xlabel('Time (s)')
            axs_err[1].set_title('Env 0: Perceived Velocity Error (Body Frame)', fontweight='bold')
            axs_err[1].legend(loc='upper right')
            axs_err[1].grid(True, linestyle='--', alpha=0.6)
            # 禁用科学计数法和偏移计算，防止无限放大时的字体渲染崩溃
            axs_err[0].ticklabel_format(useOffset=False, style='plain')
            axs_err[1].ticklabel_format(useOffset=False, style='plain')
            
            plt.tight_layout()
            err_plot_path = os.path.join(log_dir, "env0_perceived_errors_drift.png")
            plt.savefig(err_plot_path, dpi=200)
            print("[INFO] 正在显示 Env 0 感知误差图，请使用窗口上的放大镜工具查看细节。关闭窗口即可结束程序...")
            plt.show()
            plt.close(fig_err)
            print(f"[SUCCESS] Env 0 感知误差曲线已保存至: {err_plot_path}")
        except Exception as e:
            print(f"[ERROR] 绘制 Env 0 感知误差曲线失败: {e}")

        # ==========================================================
        # [修改] 绘制 Env 0 前 20 秒实际位置和实际速度 (打点图，X/Y/Z分离绘制)
        # ==========================================================
        print(f"\n[INFO] 正在生成 Env 0 前 20 秒实际位置与速度打点图(X/Y/Z 分离)...")
        try:
            mask_20s = t_arr <= 20.0
            t_20s = t_arr[mask_20s]
            act_pos_20s = act_pos_arr[mask_20s, 0, :]
            act_vel_20s = act_vel_arr[mask_20s, 0, :]

            # 创建 2x3 网格 (2 行: 位置、速度; 3 列: X, Y, Z)
            fig_act, axs_act = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
            
            titles_pos = ['Actual X Position', 'Actual Y Position', 'Actual Z Position']
            ylabels_pos = ['X (m)', 'Y (m)', 'Z (m)']
            colors = ['#1f77b4', '#ff7f0e', '#d62728'] # 蓝，橙，红
            
            for dim in range(3):
                # 第一行：位置 (X, Y, Z)
                axs_act[0, dim].plot(t_20s, act_pos_20s[:, dim], marker='.', linestyle='none', markersize=4, color=colors[dim], alpha=0.8)
                axs_act[0, dim].set_title(f'Env 0: {titles_pos[dim]}', fontweight='bold')
                axs_act[0, dim].set_ylabel(ylabels_pos[dim])
                axs_act[0, dim].grid(True, linestyle='--', alpha=0.6)
                axs_act[0, dim].ticklabel_format(useOffset=False, style='plain')

                # 第二行：速度 (Vx, Vy, Vz)
                v_axis = ["x", "y", "z"][dim]
                axs_act[1, dim].plot(t_20s, act_vel_20s[:, dim], marker='.', linestyle='none', markersize=4, color=colors[dim], alpha=0.8)
                axs_act[1, dim].set_title(f'Env 0: Actual V{v_axis}', fontweight='bold')
                axs_act[1, dim].set_ylabel(f'V{v_axis} (m/s)')
                axs_act[1, dim].set_xlabel('Time (s)')
                axs_act[1, dim].grid(True, linestyle='--', alpha=0.6)
                axs_act[1, dim].ticklabel_format(useOffset=False, style='plain')

            plt.tight_layout()
            act_plot_path = os.path.join(log_dir, "env0_actual_pos_vel_20s.png")
            plt.savefig(act_plot_path, dpi=200)
            print("[INFO] 正在显示 Env 0 实际位置与速度打点图，关闭窗口即可继续程序...")
            plt.show()
            plt.close(fig_act)
            print(f"[SUCCESS] Env 0 实际位置与速度打点图 (X/Y/Z分离) 已保存至: {act_plot_path}")
        except Exception as e:
            print(f"[ERROR] 绘制 Env 0 实际位置与速度打点图失败: {e}")

        # # ==========================================================
        # # [新增] 绘制全局的动力学生存分布对比图 (Pairplot & Parallel)
        # # ==========================================================
        # print(f"\n[INFO] 正在生成【存活/坠毁】的参数分布边界对比图，保存于根目录...")
        # try:
        #     import pandas as pd
        #     import seaborn as sns
        #     from pandas.plotting import parallel_coordinates

        #     # 构造 DataFrame
        #     records = []
        #     for i in range(num_envs):
        #         p = generated_params[i]
        #         survived = not has_crashed_per_env[i]
        #         records.append({
        #             'Status': 'Survived' if survived else 'Crashed',
        #             'Ixx': p['inertia'][0],  # 仅用 Ixx 代表整体惯量
        #             'Tau_Up': p['motor_tau_up'],
        #             'Tau_Down': p['motor_tau_down'],
        #             'Kappa': p['kappa']
        #         })
            
        #     df_plot = pd.DataFrame(records)
        #     cols_to_plot = ['Ixx', 'Tau_Up', 'Tau_Down', 'Kappa']
            
        #     # --- 1. 散点矩阵分布图 (Pairplot) ---
        #     sns.set_theme(style="whitegrid")
        #     palette = {'Survived': '#2ca02c', 'Crashed': '#d62728'}  # 绿/红
            
        #     g = sns.pairplot(
        #         df_plot,
        #         vars=cols_to_plot,
        #         hue='Status',
        #         palette=palette,
        #         diag_kind='kde',
        #         plot_kws={'alpha': 0.7, 's': 60, 'edgecolor': 'w'},
        #         corner=True
        #     )
        #     g.fig.suptitle("Dynamics Parameters Survival Boundaries", y=1.02, fontsize=16, fontweight='bold')
            
        #     save_path_pair = os.path.join(log_dir, "dynamics_survival_pairplot.png")
        #     plt.savefig(save_path_pair, dpi=200, bbox_inches='tight')
        #     plt.close()

        #     # --- 2. 平行坐标图 (Parallel Coordinates) ---
        #     plt.figure(figsize=(10, 6))
        #     df_norm = df_plot.copy()
        #     # [0, 1] 归一化
        #     for col in cols_to_plot:
        #         min_v = df_norm[col].min()
        #         max_v = df_norm[col].max()
        #         if max_v > min_v:
        #             df_norm[col] = (df_norm[col] - min_v) / (max_v - min_v)
            
        #     # 绘制：先把 Survived 垫在下面(透明度高点)，再把 Crashed 盖在上面(透明度低点)
        #     df_surv = df_norm[df_norm['Status'] == 'Survived']
        #     df_cras = df_norm[df_norm['Status'] == 'Crashed']
            
        #     if not df_surv.empty:
        #         parallel_coordinates(df_surv, 'Status', color=['#2ca02c'], alpha=0.3)
        #     if not df_cras.empty:
        #         parallel_coordinates(df_cras, 'Status', color=['#d62728'], alpha=0.7, linewidth=2.5)
                
        #     plt.title("Parallel Coordinates: Survived vs Crashed (Normalized 0-1)", fontsize=14, fontweight='bold')
        #     plt.ylabel("Normalized Value")
        #     plt.xticks(rotation=0, fontsize=12)
        #     plt.grid(True, linestyle='--', alpha=0.5)
            
        #     # 整理图例防止重复
        #     handles, labels = plt.gca().get_legend_handles_labels()
        #     by_label = dict(zip(labels, handles))
        #     plt.legend(by_label.values(), by_label.keys(), loc='upper right')

        #     save_path_para = os.path.join(log_dir, "dynamics_survival_parallel.png")
        #     plt.savefig(save_path_para, dpi=200, bbox_inches='tight')
        #     plt.close()

        #     print(f"[SUCCESS] 生存分布对比图已成功生成: ")
        #     print(f"  -> {save_path_pair}")
        #     print(f"  -> {save_path_para}")

        # except ImportError:
        #     print("[WARNING] 缺少 pandas 或 seaborn，跳过生成参数生存分布图。可通过 'pip install pandas seaborn' 安装。")
        # except Exception as e:
        #     print(f"[WARNING] 生成生存分布图失败: {e}")

        # ==========================================================
        # [新增] 绘制每个动力学参数与跟踪误差(RMSE)的关系散点图
        # ==========================================================
        print(f"\n[INFO] 正在生成【动力学参数 vs 跟踪误差】的散点趋势图...")
        try:
            import pandas as pd
            import seaborn as sns

            # 构造用于误差分析的 DataFrame
            error_records = []
            for i in range(num_envs):
                p = generated_params[i]
                survived = not has_crashed_per_env[i]
                error_records.append({
                    'Status': 'Survived' if survived else 'Crashed',
                    'Mass (kg)': p['mass'],
                    'Arm Length (m)': p['arm_length'],
                    'TWR': p['twr'],
                    'Ixx (kg·m²)': p['inertia'][0],
                    'Izz (kg·m²)': p['inertia'][2],
                    'Tau Up (s)': p['motor_tau_up'],
                    'Tau Down (s)': p['motor_tau_down'],
                    'Kappa': p['kappa'],
                    'RMSE (m)': rmse_per_env[i]
                })
            
            df_error = pd.DataFrame(error_records)
            params_to_compare = [
                'Mass (kg)', 'Arm Length (m)', 'TWR', 
                'Ixx (kg·m²)', 'Izz (kg·m²)', 
                'Tau Up (s)', 'Tau Down (s)', 'Kappa'
            ]

            # 创建一个多子图画布 (2行 4列)
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            axes = axes.flatten()
            palette = {'Survived': '#2ca02c', 'Crashed': '#d62728'}  # 绿/红区分存活与坠毁

            for idx, param in enumerate(params_to_compare):
                ax = axes[idx]
                sns.scatterplot(
                    data=df_error, 
                    x=param, 
                    y='RMSE (m)', 
                    hue='Status', 
                    palette=palette, 
                    alpha=0.75, 
                    s=55, 
                    ax=ax,
                    edgecolor='w'
                )
                ax.set_title(f'RMSE vs {param}', fontsize=12, fontweight='bold')
                ax.set_xlabel(param, fontsize=10)
                ax.set_ylabel('RMSE (m)', fontsize=10)
                ax.grid(True, linestyle='--', alpha=0.5)
                
                # 仅在第一个子图保留图例，其余子图移除以保持整洁
                if idx > 0:
                    if ax.legend_ is not None:
                        ax.legend_.remove()
                else:
                    ax.legend(loc='upper right', frameon=True)

            plt.suptitle("Tracking Error (RMSE) vs Dynamics Parameters", fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            save_path_scatter = os.path.join(log_dir, "dynamics_vs_error_scatter.png")
            plt.savefig(save_path_scatter, dpi=200, bbox_inches='tight')
            plt.close()

            print(f"[SUCCESS] 动力学参数与误差散点图已成功生成: {save_path_scatter}")

        except Exception as e:
            print(f"[WARNING] 生成动力学参数与误差散点图失败: {e}")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()