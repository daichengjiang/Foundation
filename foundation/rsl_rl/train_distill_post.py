# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train Distill Post Agent (Student) with Warm Start capability."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

# --- [CRITICAL IMPORT] ---
# 必须导入定义了 StudentTeacherRecurrentCustom 的模块，
# 这样 RSL-RL 才能通过字符串反射找到该类。
try:
    import student_teacher_recurrent_custom
    print(f"[INFO] Successfully imported custom network module: {student_teacher_recurrent_custom.__file__}")
except ImportError as e:
    print(f"[WARNING] Could not import 'student_teacher_recurrent_custom'. Ensure it is in your Python path. Error: {e}")
# -------------------------

# [新增] 导入 pandas 用于读取动力学配置文件
import pandas as pd
import numpy as np

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train Distill Post Agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--log_timestamp", type=str, default=None, help="Fixed timestamp folder name.")

# [新增] 异构动力学相关参数
parser.add_argument("--teacher_dir", type=str, default=None, help="Path to the directory containing teacher_dynamics.csv (for heterogeneous envs).")
parser.add_argument("--teacher_ids", type=str, default="0", help="Comma-separated list of dynamics IDs to use (e.g., '0,1,2' or '0-4').")

# append RSL-RL cli arguments
# 这会添加 --checkpoint, --resume, --run_name 等参数
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
from isaaclab.utils.io import dump_pickle, dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from foundation import tasks

# 优化设置
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # 1. 配置覆盖
    env_cfg.num_steps_per_env = agent_cfg.num_steps_per_env
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # 2. 设置 Seed 和 Device
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ==================================================================================
    # [新增] 异构动力学配置 (读取 CSV 并注入 env_cfg)
    # ==================================================================================
    if args_cli.teacher_dir:
        print(f"\n{'='*20} [Heterogeneous Dynamics Setup] {'='*20}")
        
        # 1. 解析需要使用的 ID
        teacher_ids = []
        if '-' in args_cli.teacher_ids:
            start, end = map(int, args_cli.teacher_ids.split('-'))
            teacher_ids = list(range(start, end + 1))
        else:
            teacher_ids = [int(x) for x in args_cli.teacher_ids.split(',')]
            
        num_variants = len(teacher_ids)
        print(f"Target Dynamics Variants ({num_variants}): {teacher_ids}")

        # 2. 调整环境数量以保证整除 (均匀分配)
        if env_cfg.scene.num_envs % num_variants != 0:
            old_num = env_cfg.scene.num_envs
            new_num = (old_num // num_variants) * num_variants
            if new_num == 0: new_num = num_variants
            
            print(f"[WARNING] num_envs ({old_num}) is not divisible by num_variants ({num_variants}).")
            print(f"          Adjusting num_envs to {new_num}.")
            env_cfg.scene.num_envs = new_num
        
        # 3. 读取 CSV
        csv_path = os.path.join(args_cli.teacher_dir, "teacher_dynamics.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dynamics CSV not found at: {csv_path}")
            
        df = pd.read_csv(csv_path)
        print(f"Loading dynamics from {csv_path}...")
        
        teacher_params_list = []
        for t_id in teacher_ids:
            row = df[df['id'] == t_id]
            if row.empty:
                raise ValueError(f"ID {t_id} not found in CSV.")
            row = row.iloc[0]

            # 构造参数字典 (字段名需与 distill_post_env.py 兼容)
            params = {
                "id": int(t_id),
                "mass": float(row['mass']),
                "arm_length": float(row['arm_length']),
                "inertia": (float(row['Ixx']), float(row['Iyy']), float(row['Izz'])),
                "twr": float(row['twr']) if 'twr' in row else float(row.get('thrust_to_weight', 2.25)),
                "motor_tau_up": float(row['motor_tau_up']) if 'motor_tau_up' in row else 0.05,
                "motor_tau_down": float(row['motor_tau_down']) if 'motor_tau_down' in row else 0.07,
                "kappa": float(row['kappa']) if 'kappa' in row else 0.016,
            }
            teacher_params_list.append(params)
            print(f"  > [Variant-{t_id}] Mass={params['mass']:.4f} | Arm={params['arm_length']:.4f} | TWR={params['twr']:.2f}")

        # 4. 注入到 env 配置中
        # QuadcopterEnv 会检测到这个列表并进行分配
        try:
            env_cfg.dynamics.multi_teacher_params = teacher_params_list
        except AttributeError:
            print("[ERROR] env_cfg.dynamics does not have 'multi_teacher_params'. Check your environment config definition.")
            raise

        print(f"{'='*60}\n")
    # ==================================================================================

    # 3. 日志目录设置
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    agent_cfg_dict = agent_cfg.to_dict()

    # --- WandB 命名逻辑 ---
    if args_cli.log_timestamp:
        new_run_name = f"{args_cli.log_timestamp}_{agent_cfg.run_name}"
        agent_cfg_dict["run_name"] = new_run_name
        agent_cfg.run_name = new_run_name
        
        if agent_cfg_dict.get("logger") == "wandb":
            agent_cfg_dict["wandb_name"] = new_run_name
            agent_cfg_dict["wandb_id"] = new_run_name
            agent_cfg_dict["wandb_group"] = agent_cfg_dict.get("experiment_name", "distill_post")
        
        run_suffix = new_run_name.split('_')[-1]
        local_run_folder = f"run_{run_suffix}" 
        log_dir = os.path.join(log_root_path, args_cli.log_timestamp, local_run_folder)
        print(f"[INFO] Using fixed timestamp: {args_cli.log_timestamp}")
        print(f"[INFO] Local Log Dir: {log_dir}")
    else:
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)

    # 4. 创建环境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 5. 视频录制
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

    # 6. RSL-RL Runner
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=log_dir, device=agent_cfg.device)

    # 修复 WandB 名字
    if args_cli.log_timestamp:
        runner.run_name = new_run_name
        
    runner.add_git_repo_to_log(__file__)

    # 7. 加载模型逻辑

    if args_cli.checkpoint:
        # [关键部分] 完全按照你的要求实现手动加载、过滤和 Normalizer 加载
        checkpoint_path = retrieve_file_path(args_cli.checkpoint)
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
        # 注意: 这里使用 strict=False 允许加载部分权重（即忽略预训练中缺失的 Critic/Teacher 部分）
        runner.alg.policy.load_state_dict(student_only_state_dict, strict=False)
        print("[INFO] Model weights loaded (Student only, Teachers ignored).")

        # 4. Load Normalizer
        if agent_cfg.empirical_normalization:
            if 'obs_norm_state_dict' in loaded_dict:
                runner.obs_normalizer.load_state_dict(loaded_dict['obs_norm_state_dict'])
                print("[INFO] Observation Normalizer loaded.")
            else:
                print("[WARNING] Empirical normalization is enabled but no 'obs_norm_state_dict' found!")
    
    # 如果需要调整初始噪声 (Distill Post 建议调小噪声)
    if args_cli.init_noise_std:
        runner.load_std(args_cli.init_noise_std)
        print(f"[INFO]: Overwriting init noise std to: {args_cli.init_noise_std}")

    # =================================================================================
    # [CRITICAL FIX] 手动初始化 Hidden States 为正确的维度 [Layers, Num_Envs, Dim]
    # 这防止了 RolloutStorage 因为第一次收到 dummy state [Layers, 1, Dim] 而错误地初始化缓冲区
    # =================================================================================
    policy = runner.alg.policy
    # 检查 policy 是否有 hidden_state 属性 (即是否是我们的自定义 RNN)
    if hasattr(policy, "hidden_state"):
        print(f"[{'='*20}]")
        print(f"[INFO] Manually initializing policy hidden state for {env.num_envs} environments.")
        device = agent_cfg.device
        
        # 根据 RNN 类型构造全0张量
        if hasattr(policy, "rnn_type") and policy.rnn_type == "lstm":
             h = torch.zeros(policy.rnn_num_layers, env.num_envs, policy.rnn_hidden_dim, device=device)
             c = torch.zeros(policy.rnn_num_layers, env.num_envs, policy.rnn_hidden_dim, device=device)
             policy.hidden_state = (h, c)
        else:
             # GRU
             policy.hidden_state = torch.zeros(policy.rnn_num_layers, env.num_envs, policy.rnn_hidden_dim, device=device)
        print(f"[INFO] Hidden state initialized with shape: {policy.hidden_state.shape if hasattr(policy.hidden_state, 'shape') else 'Tuple'}")
        print(f"[{'='*20}]")
    # =================================================================================

    # 8. 保存配置
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # =================================================================================
    # [新增功能] 训练前热身/评估 (Pre-training Evaluation / Play Mode)
    # 目的：使用加载的预训练模型，以 Eval 模式（无噪声、确定性）运行 10 个 Episode
    # =================================================================================
    
    # 配置热身长度
    NUM_EVAL_EPISODES = 10
    # 获取每个 Episode 的步数 (从 cfg 中读取，通常是 256)
    steps_per_episode = env_cfg.num_steps_per_env 
    total_eval_steps = NUM_EVAL_EPISODES * steps_per_episode
    
    print(f"[{'='*30}]")
    print(f"[INFO] Starting Warm-up / Sanity Check (Eval Mode) for {NUM_EVAL_EPISODES} episodes...")
    print(f"[INFO] Total steps: {total_eval_steps}. Watch the viewer to verify behavior.")
    
    # 1. 切换到 Eval 模式
    # 这会通知 PyTorch 层（如 Dropout/BatchNorm）进入推理模式
    runner.alg.policy.eval()
    
    # 2. 获取推理策略
    # RSL-RL 的 get_inference_policy 返回的是确定性的 act_inference 函数
    # 对于你的 RNN 网络，这意味着它会使用内部维护的 hidden_state 进行推理
    inference_policy = runner.get_inference_policy(device=agent_cfg.device)
    
    # 3. 确保环境已重置
    obs, _ = env.get_observations()
    
    # 4. 执行仿真循环
    with torch.inference_mode(): # 关闭梯度计算，节省显存并加速
        for i in range(total_eval_steps):
            # A. 策略推理 (Deterministic)
            # 这里的 actions 是不带噪声的，纯粹由预训练权重决定
            actions = inference_policy(obs)
            
            # B. 环境步进
            obs, rewards, dones, extras = env.step(actions)
            
            # C. [关键] 手动处理 RNN 状态重置
            # 在 runner.learn() 中这是自动的，但在手动循环中，我们需要
            # 告诉策略网络哪些环境已经重置了，以便它清空对应的 hidden_state
            runner.alg.policy.reset(dones)
            
            # 打印进度
            if (i + 1) % steps_per_episode == 0:
                print(f"[Eval] Completed approximately {(i + 1) // steps_per_episode} / {NUM_EVAL_EPISODES} episodes.")

    print(f"[INFO] Warm-up finished. The model seems runnable.")
    print(f"[INFO] Switching back to TRAINING mode (adding noise, enabling grad)...")
    
    # 5. 恢复训练状态
    runner.alg.policy.train() 
    
    # 6. [重要] 彻底重置环境和 RNN 状态
    # 为了防止热身阶段的残留状态（如飞到一半的位置、残留的 hidden_state）污染 PPO 的第一个 Batch
    # 我们强制对所有环境进行一次 Reset
    with torch.inference_mode():
        all_env_ids = torch.arange(env.num_envs, device=agent_cfg.device)
        obs, _ = env.reset() 
        # RNN reset 通常也建议在无梯度模式下做
        runner.alg.policy.reset(torch.ones(env.num_envs, dtype=torch.bool, device=agent_cfg.device))
        print(f"[{'='*30}]")
        # =================================================================================

    # 9. 开始训练
    print(f"[INFO] Starting training for {agent_cfg.max_iterations} iterations...")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # 10. 关闭
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()