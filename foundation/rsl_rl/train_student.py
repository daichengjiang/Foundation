# foundation/rsl_rl/train_student.py

import argparse
import sys
import os
import torch
from isaaclab.app import AppLauncher

# --- 1. 解析命令行参数 ---
parser = argparse.ArgumentParser(description="Train Raptor Student (Distillation).")
parser.add_argument("--task", type=str, default="point_ctrl_single_dense", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments.")
parser.add_argument("--seed", type=int, default=42, help="Seed.")
parser.add_argument("--max_epochs", type=int, default=1000, help="Distillation epochs.")
# 引入 AppLauncher 参数 (headless, device 等)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 启动 Isaac Sim
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- 2. 导入依赖 (必须在 simulation_app 启动后) ---
import gymnasium as gym
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils.hydra import hydra_task_config
from foundation.rsl_rl.distill_runner import DistillRunner
import foundation.tasks  # 注册你的 task

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    # 覆盖配置
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    
    # 1. 创建环境
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 2. 检查路径
    # 假设你的日志在 Foundation/logs/rsl_rl
    # 你的 teacher_dynamics.csv 在 Foundation/teacher_dynamics.csv
    log_root = os.path.abspath("logs/rsl_rl")
    
    print(f"[Student Train] Log Root: {log_root}")
    print(f"[Student Train] Envs: {args_cli.num_envs}")
    
    # 3. 启动蒸馏 Runner
    runner = DistillRunner(env, log_dir=log_root, device=args_cli.device)
    
    # 4. 开始训练
    runner.learn(max_epochs=args_cli.max_epochs)
    
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()