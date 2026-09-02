# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause



import argparse
import sys
import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
# 1. 导入 Isaac Lab 仿真应用启动器
from isaaclab.app import AppLauncher

import cli_args 

# 2. 定义命令行参数解析器
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")

# 教师相关的核心控制参数
parser.add_argument("--teacher_dir", type=str, default=None, required=True, help="Path to the teacher experiment directory.")
parser.add_argument("--teacher_ids", type=str, default="0", help="Comma-separated list of teacher IDs.")
parser.add_argument("--exclude_teacher_ids", type=str, default="", help="Comma-separated list of teacher IDs to exclude (e.g., '141,14,78').")

# 注册 RSL-RL 与 AppLauncher 专属参数
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

# 3. 初始化并启动 Isaac Sim 后端引擎（必须在导入 Gym/Torch 仿真模块前执行）
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

# 导入多教师策略网络架构
try:
    from multi_teacher_policy import MultiTeacherPolicy
except ImportError:
    raise ImportError("Could not import 'MultiTeacherPolicy'. Please create 'multi_teacher_policy.py' first.")

# 4. 开启 PyTorch 性能优化开关（TF32 矩阵乘法加速）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # 5. 更新并对齐环境与算法配置
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # 6. 解析 Teacher IDs（支持范围如 "0-99" 或列表如 "1,2,5"）
    teacher_ids = []
    if '-' in args_cli.teacher_ids:
        start, end = map(int, args_cli.teacher_ids.split('-'))
        teacher_ids = list(range(start, end + 1))
    else:
        teacher_ids = [int(x) for x in args_cli.teacher_ids.split(',') if x.strip()]
    
    # 7. 剔除指定的劣质 Teacher
    if args_cli.exclude_teacher_ids:
        exclude_ids = [int(x.strip()) for x in args_cli.exclude_teacher_ids.split(',') if x.strip()]
        original_count = len(teacher_ids)
        teacher_ids = [t_id for t_id in teacher_ids if t_id not in exclude_ids]
        print(f"\n[INFO] 发现过滤指令！已成功剔除 {original_count - len(teacher_ids)} 个劣质 Teacher。")
        print(f"       剔除名单: {exclude_ids}\n")

    num_teachers = len(teacher_ids)
    if num_teachers == 0:
        raise ValueError("[ERROR] 剔除后没有任何 Teacher 剩余，请检查参数！")
        
    print(f"Target Teachers ({num_teachers}): {teacher_ids}")
    
    # 8. 环境数量对齐检查（确保并行环境总数能被教师总数整除）
    if env_cfg.scene.num_envs % num_teachers != 0:
        old_num = env_cfg.scene.num_envs
        new_num = (old_num // num_teachers) * num_teachers
        if new_num == 0: new_num = num_teachers 
        
        print(f"[WARNING] num_envs ({old_num}) is not divisible by num_teachers ({num_teachers}).")
        print(f"          Adjusting num_envs to {new_num}.")
        env_cfg.scene.num_envs = new_num

    # 9. 加载教师的动力学参数 (CSV) 与神经网络模型权重 (.pt)
    teacher_params_list = []
    loaded_teachers_state_dicts = []
    
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

        # 提取当前教师专属的异构动力学物理参数
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
        
        # 寻找对应教师的权重文件路径（优先读取 best_model.pt，其次按创建时间搜寻 model_*.pt）
        teacher_run_name = f"teacher_{t_id:04d}"
        folder_path = os.path.join(args_cli.teacher_dir, teacher_run_name)
        
        model_path = os.path.join(folder_path, "best_model.pt")
        if not os.path.exists(model_path):
            search_pattern = os.path.join(folder_path, "model_*.pt")
            models = glob.glob(search_pattern)
            if not models:
                raise FileNotFoundError(f"No model found for teacher {t_id} in {folder_path}")
            model_path = max(models, key=os.path.getctime)
            
        print(f"  > [T-{t_id}] Dynamics: Mass={params['mass']:.3f} | Model: {os.path.basename(model_path)}")
        
        # 加载教师模型权重到 CPU 缓存
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        loaded_teachers_state_dicts.append(ckpt)

    # 10. 将多教师动力学参数列表注入环境配置中
    try:
        env_cfg.dynamics.multi_teacher_params = teacher_params_list
    except AttributeError:
        print("[ERROR] env_cfg.dynamics does not have 'multi_teacher_params'.")
        raise

    print(f"{'='*60}\n")

    # 11. 构建结构化的实验日志保存目录
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    
    # 动态拼接日志后缀（包含教师范围和被剔除的 ID）
    log_dir += f"_MultiT_{min(teacher_ids)}-{max(teacher_ids)}"
    if args_cli.exclude_teacher_ids:
        clean_excl = args_cli.exclude_teacher_ids.replace(',', '-')
        log_dir += f"_Excluded{clean_excl}"
        
    log_dir = os.path.join(log_root_path, log_dir)

    # 12. 实例化 Gym 仿真环境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 13. 如果开启了 --video，则包裹视频录制器
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

    # 14. 使用 RSL-RL 向量化环境包装器
    env = RslRlVecEnvWrapper(env)

    # 15. 初始化 RSL-RL 的 OnPolicyRunner 训练运行器
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    print("\n[Distillation] Constructing Multi-Teacher Policy...")
    
    # 获取学生观测空间的维度
    obs, extras = env.get_observations() 
    real_student_obs_dim = obs.shape[1] 
    
    # 从第一个教师权重的输入层自动推断教师观测空间维度 (real_teacher_obs_dim)
    first_ckpt = loaded_teachers_state_dicts[0]
    teacher_input_weight = None
    for key in ['actor.0.weight', 'actor.layers.0.weight', 'actor.actor_mlp.0.weight']:
        if key in first_ckpt['model_state_dict']:
            teacher_input_weight = first_ckpt['model_state_dict'][key]
            break
            
    if teacher_input_weight is None:
        raise ValueError("Could not infer Teacher input dimension from checkpoint.")
    
    real_teacher_obs_dim = teacher_input_weight.shape[1]
    
    # 16. 循环构建每一个神经网络教师模型实例并加载权重
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
            init_noise_std=1.0,  # 仅占位，推理时使用 act_inference()，会被 checkpoint 覆盖
        ).to(agent_cfg.device)
        
        teacher_p.load_state_dict(ckpt['model_state_dict'])
        teacher_p.eval()  # 教师模型设为评估模式（不更新梯度）
        teacher_modules.append(teacher_p)

        # 收集教师专属的观测归一化状态字典（如果存在）
        if 'obs_norm_state_dict' in ckpt:
            teacher_norm_dicts.append(ckpt['obs_norm_state_dict'])
        else:
            teacher_norm_dicts.append(None) 

    # 17. 实例化总的多教师蒸馏策略网络 (MultiTeacherPolicy)
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
    ).to(agent_cfg.device)
    
    # 18. 断点恢复逻辑（如果指定了恢复训练的 checkpoint）
    resume_path = None
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        
    if resume_path:
        print(f"[INFO] Loading Student checkpoint from: {resume_path}")
        loaded_dict = torch.load(resume_path, map_location=agent_cfg.device)
        multi_policy.load_state_dict(loaded_dict['model_state_dict'], strict=False)

    # 19. 将多教师策略赋给运行器，并初始化 Adam 优化器
    runner.alg.policy = multi_policy
    runner.alg.optimizer = torch.optim.Adam(runner.alg.policy.parameters(), lr=runner.alg.learning_rate)
    runner.alg.policy.loaded_teacher = True
    
    # 20. 配置经验归一化（多教师蒸馏中由 MultiTeacherPolicy 内部独立处理归一化）
    if agent_cfg.empirical_normalization:
        print("[INFO] Disabling global privileged_obs_normalizer in Runner.")
        print("       (Normalization is now handled internally by MultiTeacherPolicy per teacher)")
        runner.privileged_obs_normalizer = torch.nn.Identity().to(agent_cfg.device)

    # 21. 持久化备份当前实验的 YAML 和 Pickle 配置文件
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # 22. 正式启动强化学习蒸馏训练循环
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # 23. 训练完成，关闭仿真环境
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()