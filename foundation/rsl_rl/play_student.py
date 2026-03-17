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
import torch.nn as nn      # [新增] 
import copy                # [新增]
from isaaclab.utils.math import euler_xyz_from_quat
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

# =========================================================================
# [新增] 下层学生网络实物部署 Wrapper
# =========================================================================
class LowerActorDeployWrapper(nn.Module):
    def __init__(self, policy, obs_normalizer, device):
        super().__init__()
        # 1. 拷贝归一化器
        if obs_normalizer is not None:
            self.obs_normalizer = copy.deepcopy(obs_normalizer).to(device)
        else:
            self.obs_normalizer = nn.Identity().to(device)

        # 2. 拷贝学生网络的独立组件 (跳过 Teacher 部分)
        self.pre_rnn_mlp = copy.deepcopy(policy.pre_rnn_mlp).to(device)
        self.rnn = copy.deepcopy(policy.rnn).to(device)
        self.post_rnn_mlp = copy.deepcopy(policy.post_rnn_mlp).to(device)
        self.student = copy.deepcopy(policy.student).to(device)

        # 3. 初始化 GRU 隐状态
        # 针对实物端: num_layers=1, batch_size=1, hidden_size=16
        num_layers = self.rnn.num_layers
        hidden_size = self.rnn.hidden_size
        self.hidden_state = torch.zeros(num_layers, 1, hidden_size, device=device)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 1. 归一化
        x = self.obs_normalizer(observations)
        
        # 2. 前置 MLP
        x = self.pre_rnn_mlp(x)
        
        # 3. GRU 推理 (需扩展 Seq=1 维度)
        x = x.unsqueeze(0)
        x, self.hidden_state = self.rnn(x, self.hidden_state)
        x = x.squeeze(0)
        
        # 4. 后置 MLP
        x = self.post_rnn_mlp(x)
        
        # 5. 输出动作层
        actions_mean = self.student(x)
        
        # 6. 安全限幅
        actions = torch.clamp(actions_mean, -1.0, 1.0)
        return actions
# =========================================================================

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

    # x500
    # env_cfg.dynamics.mass = 2
    # env_cfg.dynamics.arm_length = 0.25
    # env_cfg.dynamics.inertia = (0.022,0.022,0.04)
    # env_cfg.dynamics.thrust_to_weight = 2.2
    # env_cfg.dynamics.motor_tau_up = 0.06
    # env_cfg.dynamics.motor_tau_down = 0.15
    # env_cfg.dynamics.moment_scale = 0.025

    # Flightmare
    # env_cfg.dynamics.mass = 0.73
    # env_cfg.dynamics.arm_length = 0.085
    # env_cfg.dynamics.inertia = (7.911e-3,7.911e-3,1.231e-2)
    # env_cfg.dynamics.thrust_to_weight = 4.5
    # env_cfg.dynamics.motor_tau_up = 0.06
    # env_cfg.dynamics.motor_tau_down = 0.15
    # env_cfg.dynamics.moment_scale = 0.025

    # dynamics_dict = [0.153643254847717,0.075243797123877,0.000409381561426928,0.000409381561426928,0.000749987020534131,2.97714411561979,0.0637168711122302,0.062396508781855,0.0190941516379497]
    # env_cfg.dynamics.mass = dynamics_dict[0]
    # env_cfg.dynamics.arm_length = dynamics_dict[1]
    # env_cfg.dynamics.inertia = (dynamics_dict[2], dynamics_dict[3], dynamics_dict[4])
    # env_cfg.dynamics.thrust_to_weight = dynamics_dict[5]
    # env_cfg.dynamics.motor_tau_up = dynamics_dict[6]
    # env_cfg.dynamics.motor_tau_down = dynamics_dict[7]
    # env_cfg.dynamics.moment_scale = dynamics_dict[8]

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
    
    # Data storage
    trajectory_data = {
        'desired_pos': [],
        'actual_pos': [],
        'desired_vel': [],
        'actual_vel': [],
        'desired_yaw': [], 
        'actual_yaw': [],
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

            quat_w = env.unwrapped._robot.data.root_quat_w
            _, _, yaw_curr = euler_xyz_from_quat(quat_w)
            
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
                trajectory_data['desired_yaw'].append(env.unwrapped.yaw_des[0].cpu().numpy())
                trajectory_data['actual_yaw'].append(yaw_curr[0].cpu().numpy())
                        
            timestep += 1
            
            if timestep % 200 == 0:
                # Print instantaneous RMSE for monitoring
                cur_rmse = np.sqrt(torch.mean(squared_error).item())
                cur_rmse_xy = np.sqrt(torch.mean(squared_error_xy).item())

                batch_yaw_err = env.unwrapped.yaw_des - yaw_curr
                batch_yaw_err = torch.remainder(batch_yaw_err + torch.pi, 2 * torch.pi) - torch.pi
                cur_yaw_rmse = np.degrees(torch.sqrt(torch.mean(batch_yaw_err**2)).item())

                status = " (Collecting Stats)" if timestep >= STATS_START_STEP else " (Warmup)"
                print(f"Step {timestep:5d}{status} | RMSE: {cur_rmse:.4f}m | RMSE w/o z: {cur_rmse_xy:.4f}m | Yaw RMSE: {cur_yaw_rmse:.4f} deg")
                
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
    
    desired_yaw = np.array(trajectory_data['desired_yaw'])
    actual_yaw = np.array(trajectory_data['actual_yaw'])
    
    valid_des_yaw = desired_yaw[STATS_START_STEP:]
    valid_act_yaw = actual_yaw[STATS_START_STEP:]
    
    yaw_error = valid_des_yaw - valid_act_yaw
    yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi
    yaw_rmse = np.sqrt(np.mean(yaw_error**2))
    yaw_rmse_deg = np.degrees(yaw_rmse)
    
    print(f"\n{'=' * 80}")
    print(f"Paper Metrics Results (Calculated from step {STATS_START_STEP} onwards):")
    print(f"{'-' * 80}")
    print(f"  RMSE [m]:             {rmse_final:.4f}")
    print(f"  RMSE w/o z [m]:       {rmse_xy_final:.4f}")
    print(f"  Yaw RMSE [deg]:       {yaw_rmse_deg:.4f}") # [新增]
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
        f.write(f"  Yaw RMSE [deg]:       {yaw_rmse_deg:.4f}\n")  # [新增]
        f.write(f"  Max velocity [m/s]:   {max_velocity_observed:.4f}\n")
        
    print(f"\nStatistics saved to: {stats_file}")
    
    if args_cli.save_trajectory and len(trajectory_data['timestamps']) > 0:
        traj_file = os.path.join(log_dir, "trajectory_data.npz")
        np.savez(traj_file,
                 desired_pos=np.array(trajectory_data['desired_pos']),
                 actual_pos=np.array(trajectory_data['actual_pos']),
                 desired_vel=np.array(trajectory_data['desired_vel']),
                 actual_vel=np.array(trajectory_data['actual_vel']),
                 desired_yaw=np.array(trajectory_data['desired_yaw']), # [新增]
                 actual_yaw=np.array(trajectory_data['actual_yaw']),   # [新增]
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