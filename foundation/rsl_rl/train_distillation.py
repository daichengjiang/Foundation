# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL (Single Teacher Distillation Mode)."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import pandas as pd # [新增] 用于读取CSV
import numpy as np  # [新增]

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")

# [新增] 教师蒸馏相关参数
parser.add_argument("--teacher_dir", type=str, default=None, help="Path to the teacher experiment directory (containing csv and teacher folders).")
parser.add_argument("--teacher_id", type=int, default=0, help="ID of the teacher to distill/finetune from.")

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


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ==================================================================================
    # [新增/修改] 教师动力学参数覆盖逻辑
    # ==================================================================================
    teacher_model_path = None
    
    if args_cli.teacher_dir is not None:
        print(f"[Distillation] Loading dynamics for Teacher ID {args_cli.teacher_id} from {args_cli.teacher_dir}...")
        
        # 1. 读取 CSV 文件
        csv_path = os.path.join(args_cli.teacher_dir, "teacher_dynamics.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dynamics CSV not found at: {csv_path}")
            
        df = pd.read_csv(csv_path)
        
        # 2. 查找指定 ID 的行
        teacher_row = df[df['id'] == args_cli.teacher_id]
        if teacher_row.empty:
            raise ValueError(f"Teacher ID {args_cli.teacher_id} not found in CSV.")
        
        # 提取参数 (使用 iloc[0] 获取 Series)
        row = teacher_row.iloc[0]
        
        mass = float(row['mass'])
        arm_length = float(row['arm_length'])
        ixx = float(row['Ixx'])
        iyy = float(row['Iyy'])
        izz = float(row['Izz'])
        twr = float(row['twr']) if 'twr' in row else float(row['thrust_to_weight']) # 兼容不同列名
        motor_tau = float(row['motor_tau'])
        
        print(f"[Distillation] Overriding Environment Dynamics:")
        print(f"  > Mass: {mass:.4f}")
        print(f"  > Arm Length: {arm_length:.4f}")
        print(f"  > Inertia: ({ixx:.6f}, {iyy:.6f}, {izz:.6f})")
        print(f"  > TWR: {twr:.2f}")
        print(f"  > Motor Tau: {motor_tau:.4f}")
        
        # 3. 修改 env_cfg (假设 env_cfg.dynamics 存在且为可写对象)
        # 注意：这需要你的 EnvCfg 类中定义了这些字段，通常在 DirectRLEnvCfg 中
        try:
            env_cfg.dynamics.mass = mass
            env_cfg.dynamics.arm_length = arm_length
            env_cfg.dynamics.inertia = (ixx, iyy, izz)
            env_cfg.dynamics.thrust_to_weight = twr
            env_cfg.dynamics.motor_tau = motor_tau
        except AttributeError as e:
            print(f"[WARNING] Could not set dynamics directly on env_cfg: {e}")
            print("Please ensure your Environment Config class has a 'dynamics' structure.")

        # 4. 确定 Teacher Checkpoint 路径
        teacher_run_name = f"teacher_{args_cli.teacher_id:04d}"
        explicit_path = os.path.join(args_cli.teacher_dir, teacher_run_name, "best_model.pt")
        
        if os.path.exists(explicit_path):
            teacher_model_path = explicit_path
        
        if teacher_model_path:
            print(f"[Distillation] Found teacher model: {teacher_model_path}")
        else:
            print(f"[Distillation] WARNING: Could not find model.pt or best_model.pt for {teacher_run_name}")

    # ==================================================================================

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    # 如果是特定教师蒸馏，在日志名中增加标识
    if args_cli.teacher_dir:
        log_dir += f"_distill_T{args_cli.teacher_id}"
        
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    resume_path = None
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    # [新增] 如果没有指定 checkpoint 但指定了 teacher，则使用 teacher 模型
    elif teacher_model_path:
        resume_path = teacher_model_path
        print(f"[Distillation] Setting resume path to teacher model.")

    # wrap for video recording
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

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    
    # load the checkpoint
    if resume_path:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model (Teacher or Checkpoint)
        runner.load(resume_path)
    
    if args_cli.init_noise_std:
        runner.load_std(args_cli.init_noise_std)
        print(f"[INFO]: Loading init noise std from: {args_cli.init_noise_std}")


    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    # close sim app
    simulation_app.close()