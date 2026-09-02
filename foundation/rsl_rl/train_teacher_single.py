# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause



import argparse
import sys
import os
import torch
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
parser.add_argument("--log_timestamp", type=str, default=None, help="Fixed timestamp folder name.")
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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.num_steps_per_env = agent_cfg.num_steps_per_env
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    agent_cfg_dict = agent_cfg.to_dict()

    # [修改] WandB 命名逻辑
    if args_cli.log_timestamp:
        # 0. 记录去掉时间戳之前的原始 run_name
        original_run_name = agent_cfg.run_name 
        
        # 1. 构造用于 WandB 的长运行名称
        new_run_name = f"{args_cli.log_timestamp}_{original_run_name}"
        
        # 2. 更新配置字典中的 run_name
        agent_cfg_dict["run_name"] = new_run_name
        agent_cfg.run_name = new_run_name
        
        # 3. 确保 logger 依然是字符串
        if agent_cfg_dict.get("logger") == "wandb":
            agent_cfg_dict["wandb_name"] = new_run_name
            agent_cfg_dict["wandb_id"] = new_run_name
            agent_cfg_dict["wandb_group"] = agent_cfg_dict.get("experiment_name")
        
        # 4. 计算本地日志路径
        local_run_folder = original_run_name
        
        log_dir = os.path.join(log_root_path, args_cli.log_timestamp, local_run_folder)
        
        print(f"[INFO] Using fixed timestamp: {args_cli.log_timestamp}")
        print(f"[INFO] Local Log Dir: {log_dir}")
        
    else:
        # 默认逻辑 (参数搜索通常走这里，或者你可以根据喜好在 select_params.py 里也不传 timestamp)
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)

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
    runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    
    # 18. 断点恢复逻辑（如果指定了恢复训练的 checkpoint）
    resume_path = None
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        # [NEW] 强制覆盖 Runner 内部的 run_name，确保 WandB 记录正确
    if args_cli.log_timestamp:
        runner.run_name = agent_cfg.run_name
    elif args_cli.run_name_suffix:
         # 如果是参数搜索模式，也确保 Runner 使用带有 Search_ 前缀的名字
        runner.run_name = agent_cfg.run_name

    # write git state to logs
        # load the checkpoint
    if agent_cfg.resume or args_cli.checkpoint:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
    if args_cli.init_noise_std:
        runner.load_std(args_cli.init_noise_std)
        print(f"[INFO]: Loading init noise std from: {args_cli.init_noise_std}")
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