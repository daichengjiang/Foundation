# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL (Multi-Teacher Distillation Mode)."""

import argparse
import sys
import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from isaaclab.app import AppLauncher

import cli_args 

parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")

parser.add_argument("--teacher_dir", type=str, default=None, required=True, help="Path to the teacher experiment directory.")
parser.add_argument("--teacher_ids", type=str, default="0", help="Comma-separated list of teacher IDs.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from datetime import datetime

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import ActorCritic 

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

import isaaclab_tasks  
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from foundation import tasks

try:
    from multi_teacher_policy import MultiTeacherPolicy
except ImportError:
    raise ImportError("Could not import 'MultiTeacherPolicy'. Please create 'multi_teacher_policy.py' first.")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ==================================================================================
    # [新增] 多教师配置解析与加载
    # ==================================================================================
    print(f"\n{'='*20} [Multi-Teacher Distillation Setup] {'='*20}")
    
    teacher_ids = []
    if '-' in args_cli.teacher_ids:
        start, end = map(int, args_cli.teacher_ids.split('-'))
        teacher_ids = list(range(start, end + 1))
    else:
        teacher_ids = [int(x) for x in args_cli.teacher_ids.split(',')]
    
    num_teachers = len(teacher_ids)
    print(f"Target Teachers ({num_teachers}): {teacher_ids}")
    
    if env_cfg.scene.num_envs % num_teachers != 0:
        old_num = env_cfg.scene.num_envs
        new_num = (old_num // num_teachers) * num_teachers
        if new_num == 0: new_num = num_teachers 
        
        print(f"[WARNING] num_envs ({old_num}) is not divisible by num_teachers ({num_teachers}).")
        print(f"          Adjusting num_envs to {new_num}.")
        env_cfg.scene.num_envs = new_num

    teacher_params_list = []
    loaded_teachers_state_dicts = []
    teacher_offsets_list = []  # [新增] 存储每个教师的稳态误差
    
    csv_path = os.path.join(args_cli.teacher_dir, "teacher_dynamics.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dynamics CSV not found at: {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    print(f"Loading dynamics and models from {args_cli.teacher_dir}...")
    
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
            "motor_tau_up": float(row['motor_tau_up']) if 'motor_tau_up' in row else 0.05,
            "motor_tau_down": float(row['motor_tau_down']) if 'motor_tau_down' in row else 0.07,
            "kappa": float(row['kappa']) if 'kappa' in row else 0.016,
        }
        teacher_params_list.append(params)
        
        # [新增] 读取稳态误差
        x_offset = float(row['x_off_mean']) if 'x_off_mean' in row else 0.0
        y_offset = float(row['y_off_mean']) if 'y_off_mean' in row else 0.0
        z_offset = float(row['z_off_mean']) if 'z_off_mean' in row else 0.0
        teacher_offsets_list.append((x_offset, y_offset, z_offset))
        
        teacher_run_name = f"teacher_{t_id:04d}"
        folder_path = os.path.join(args_cli.teacher_dir, teacher_run_name)
        
        model_path = os.path.join(folder_path, "best_model.pt")
        if not os.path.exists(model_path):
            search_pattern = os.path.join(folder_path, "model_*.pt")
            models = glob.glob(search_pattern)
            if not models:
                raise FileNotFoundError(f"No model found for teacher {t_id} in {folder_path}")
            model_path = max(models, key=os.path.getctime)
            
        print(f"  > [T-{t_id}] Dynamics: Mass={params['mass']:.3f} | Offset: ({x_offset:.4f}, {y_offset:.4f}, {z_offset:.4f}) | Model: {os.path.basename(model_path)}")
        
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        loaded_teachers_state_dicts.append(ckpt)

    try:
        env_cfg.dynamics.multi_teacher_params = teacher_params_list
    except AttributeError:
        print("[ERROR] env_cfg.dynamics does not have 'multi_teacher_params'.")
        raise

    print(f"{'='*60}\n")
    # ==================================================================================

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    
    if len(teacher_ids) > 1:
        log_dir += f"_MultiT_{min(teacher_ids)}-{max(teacher_ids)}"
    else:
        log_dir += f"_T{teacher_ids[0]}"
        
    log_dir = os.path.join(log_root_path, log_dir)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)
    
    # ==================================================================================
    # [核心] 构建 MultiTeacherPolicy 并替换 Runner 中的策略
    # ==================================================================================
    print("\n[Distillation] Constructing Multi-Teacher Policy...")
    
    obs, _ = env.get_observations() 
    real_student_obs_dim = obs.shape[1] 
    
    first_ckpt = loaded_teachers_state_dicts[0]
    teacher_input_weight = None
    for key in ['actor.0.weight', 'actor.layers.0.weight', 'actor.actor_mlp.0.weight']:
        if key in first_ckpt['model_state_dict']:
            teacher_input_weight = first_ckpt['model_state_dict'][key]
            break
            
    if teacher_input_weight is None:
        raise ValueError("Could not infer Teacher input dimension from checkpoint.")
    
    real_teacher_obs_dim = teacher_input_weight.shape[1]
    
    teacher_modules = []
    teacher_norm_dicts = []
    for i, ckpt in enumerate(loaded_teachers_state_dicts):
        teacher_p = ActorCritic(
            num_actor_obs=real_teacher_obs_dim,
            num_critic_obs=real_teacher_obs_dim,
            num_actions=env.num_actions,
            actor_hidden_dims=agent_cfg.policy.teacher_hidden_dims,
            critic_hidden_dims=agent_cfg.policy.teacher_hidden_dims, 
            activation="elu", 
            init_noise_std=1.0,  # Note: 推理时使用 act_inference()，不使用此参数；将被 checkpoint 覆盖
        ).to(agent_cfg.device)
        
        teacher_p.load_state_dict(ckpt['model_state_dict'])
        teacher_p.eval() 
        teacher_modules.append(teacher_p)

        if 'obs_norm_state_dict' in ckpt:
            teacher_norm_dicts.append(ckpt['obs_norm_state_dict'])
        else:
            teacher_norm_dicts.append(None) 

    multi_policy = MultiTeacherPolicy(
        num_student_obs=real_student_obs_dim,
        num_teacher_obs=real_teacher_obs_dim,
        num_actions=env.num_actions,
        student_hidden_dims=agent_cfg.policy.student_hidden_dims,
        teacher_hidden_dims=agent_cfg.policy.teacher_hidden_dims,
        activation=agent_cfg.policy.activation,
        rnn_type=agent_cfg.policy.rnn_type,
        rnn_hidden_dim=agent_cfg.policy.rnn_hidden_dim,
        rnn_num_layers=agent_cfg.policy.rnn_num_layers,
        pre_rnn_dim=agent_cfg.policy.pre_rnn_dim,
        post_rnn_dim=agent_cfg.policy.post_rnn_dim,
        init_noise_std=agent_cfg.policy.init_noise_std,
        teacher_models=teacher_modules,
        teacher_norm_state_dicts=teacher_norm_dicts,
        teacher_offsets=teacher_offsets_list,  # [新增] 传递稳态误差给 Policy
    ).to(agent_cfg.device)
    
    resume_path = None
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        
    if resume_path:
        print(f"[INFO] Loading Student checkpoint from: {resume_path}")
        loaded_dict = torch.load(resume_path, map_location=agent_cfg.device)
        multi_policy.load_state_dict(loaded_dict['model_state_dict'], strict=False)

    runner.alg.policy = multi_policy
    runner.alg.optimizer = torch.optim.Adam(runner.alg.policy.parameters(), lr=runner.alg.learning_rate)
    runner.alg.policy.loaded_teacher = True
    
    if agent_cfg.empirical_normalization:
        print("[INFO] Disabling global privileged_obs_normalizer in Runner.")
        print("       (Normalization is now handled internally by MultiTeacherPolicy per teacher)")
        runner.privileged_obs_normalizer = torch.nn.Identity().to(agent_cfg.device)
    # ==================================================================================

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()