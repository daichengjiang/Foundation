# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to filter valid dynamics parameters using massive parallel evaluation."""

import argparse
import sys
import csv
import os
import time
import torch
import torchvision
import numpy as np
import gymnasium as gym

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Filter valid dynamics parameters in parallel.")
parser.add_argument("--num_envs", type=int, default=10000, help="Number of environments to simulate (batch size).")
parser.add_argument("--task", type=str, default=None, required=True, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_steps", type=int, default=3000, help="Maximum steps to run for evaluation.")
# [建议] 稍微缩短 warmup，早点开始统计
parser.add_argument("--stats_start_step", type=int, default=500, help="Step to start calculating RMSE.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--output_csv", type=str, default="teacher_dynamics.csv", help="Output CSV filename.")

# append RSL-RL cli arguments (this includes --checkpoint)
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Disable cameras for speed
args_cli.enable_cameras = False
args_cli.video = False

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from foundation import tasks 

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# ==========================================
# DYNAMICS SAMPLER
# ==========================================
def sample_raptor_dynamics():
    """Samples random dynamics parameters for the Raptor."""
    twr = np.random.uniform(1.5, 5.0)
    m_min = 0.02
    m_max = 5.0
    s = np.random.uniform(np.cbrt(m_min), np.cbrt(m_max))
    mass = s ** 3
    
    m_cf = 0.032 
    l_cf = 0.04384 
    base_ratio = l_cf / (m_cf**(1/3)) 
    u = np.random.normal(-0.1, 0.1) 
    if u < 0: s_ms = 1.0 / (1.0 - u)
    else: s_ms = 1.0 + u
    arm_length = base_ratio * (mass**(1/3)) / s_ms
    
    r_t2i = np.random.uniform(40, 1200)
    total_thrust = twr * 9.81 * mass
    tau = total_thrust * np.sqrt(2) * arm_length
    Ixx = tau / r_t2i
    Iyy = Ixx 
    Izz = Ixx * 1.832 
    
    # motor_tau range
    motor_tau_up = np.random.uniform(0.03, 0.1)
    motor_tau_down = np.random.uniform(0.03, 0.3)

    kappa = np.random.uniform(0.005, 0.05)

    return {
        "mass": mass, 
        "arm_length": arm_length, 
        "inertia": (Ixx, Iyy, Izz),
        "thrust_to_weight": twr, 
        "motor_tau_up": motor_tau_up,
        "motor_tau_down": motor_tau_down,
        "kappa": kappa,
        "_ixx": Ixx,
        "_iyy": Iyy,
        "_izz": Izz
    }

# ... (前面的 import 和 args 定义保持不变) ...

# 定义每个参数测试的重复次数
ENV_REPEATS = 10 

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Filter valid dynamics parameters using parallel evaluation with replicates."""
    
    # ---------------------------------------------------------
    # 修改点 1: 检查 num_envs 是否能被 10 整除
    # ---------------------------------------------------------
    if args_cli.num_envs % ENV_REPEATS != 0:
        raise ValueError(f"num_envs ({args_cli.num_envs}) must be divisible by ENV_REPEATS ({ENV_REPEATS}).")

    num_unique_params = args_cli.num_envs // ENV_REPEATS

    print(f"\n{'=' * 80}")
    print(f"Generating {num_unique_params} unique sets of parameters.")
    print(f"Each set will be tested in {ENV_REPEATS} environments (Total Envs: {args_cli.num_envs}).")
    print(f"{'=' * 80}")
    
    # ---------------------------------------------------------
    # 修改点 2: 生成参数并复制扩充
    # ---------------------------------------------------------
    dynamics_params_list = [] # 这个列表长度将是 num_envs
    
    # 只需要生成 num_unique_params 组独立参数
    for _ in range(num_unique_params):
        # 采样一次参数
        unique_param = sample_raptor_dynamics()
        
        # 将这组参数重复添加 10 次
        for _ in range(ENV_REPEATS):
            dynamics_params_list.append(unique_param)
    
    # ... (Step 2 和 Step 3: 环境配置、创建环境、Checkpoint 加载等代码保持原样) ...
    # 这里的代码不需要动，因为 dynamics_params_list 已经被正确填充为 num_envs 长度了
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.prob_null_trajectory = 0.0
    env_cfg.train_or_play = False
    env_cfg.debug_vis = False 
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric if args_cli.disable_fabric is not None else env_cfg.sim.use_fabric
    env_cfg.dynamics.multi_teacher_params = dynamics_params_list # 注入参数
    
    checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {checkpoint_path}")

    try:
        env = gym.make(args_cli.task, cfg=env_cfg)
    except Exception as e:
        print(f"[ERROR] Failed to create env: {e}")
        return

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path, load_optimizer=False)
    runner.eval_mode()
    policy = runner.get_inference_policy(device=agent_cfg.device)
    policy_model = runner.alg.policy
    
    obs, _ = env.get_observations()
    
    # ... (Step 4: 仿真循环 Loop 保持原样，无需修改) ...
    total_squared_error = torch.zeros(env.num_envs, device=env.device)
    valid_steps_count = torch.zeros(env.num_envs, device=env.device)
    has_crashed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    timestep = 0
    print(f"\n[INFO] Starting simulation for {args_cli.max_steps} steps...")
    while timestep < args_cli.max_steps:
        # ... (这里原本的 simulation loop 代码完全不用动) ...
        # 为了节省篇幅，这里省略中间的 loop 代码，请保留你原文件中 Step 4 的内容
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions)
            if hasattr(policy_model, "reset"): policy_model.reset(dones)
            
            if "time_outs" in extras: is_timeout = extras["time_outs"]
            else: is_timeout = torch.zeros_like(dones)
            current_crash = torch.logical_and(dones, torch.logical_not(is_timeout))
            has_crashed = torch.logical_or(has_crashed, current_crash)
            
            if timestep >= args_cli.stats_start_step:
                desired_pos = env.unwrapped.pos_des
                current_pos = env.unwrapped._robot.data.root_pos_w
                pos_error = torch.norm(current_pos - desired_pos, dim=1)
                total_squared_error += pos_error ** 2
                valid_steps_count += 1
        timestep += 1
        if timestep % 100 == 0:
            alive_count = (env.num_envs - has_crashed.sum().item())
            print(f"Step {timestep}/{args_cli.max_steps} | Never Crashed: {alive_count}/{env.num_envs}")


    # ---------------------------------------------------------
    # 修改点 3: 结果筛选与保存 (增量保存版)
    # ---------------------------------------------------------
    print(f"\n{'=' * 80}")
    print(f"Simulation Complete. Filtering results (Group Logic)...")
    
    valid_steps_count = torch.clamp(valid_steps_count, min=1.0)
    rmse_per_env = torch.sqrt(total_squared_error / valid_steps_count)
    
    # 转回 CPU
    rmse_np = rmse_per_env.cpu().numpy()
    has_crashed_np = has_crashed.cpu().numpy()
    
    csv_filename = args_cli.output_csv
    fieldnames = [
        "id", "mass", "arm_length", "Ixx", "Iyy", "Izz", 
        "twr", "motor_tau_up", "motor_tau_down", "kappa", "mean_rmse" 
    ]
    
    # === 新增逻辑：判断是否追加模式 ===
    file_exists = os.path.isfile(csv_filename) and os.path.getsize(csv_filename) > 0
    write_mode = 'a' if file_exists else 'w'
    
    # === 新增逻辑：读取已有 ID 以保持连续 ===
    id_counter = 0
    if file_exists:
        try:
            with open(csv_filename, 'r') as f_read:
                # 使用 DictReader 安全读取最后一行的 ID
                reader = csv.DictReader(f_read)
                for row in reader:
                    # 遍历直到最后一行，获取最后一个 ID
                    if row["id"]:
                        id_counter = int(row["id"])
                # 下一个新的 ID 应该是最后一个 + 1
                id_counter += 1
                print(f"[INFO] Appending to existing CSV. Starting ID: {id_counter}")
        except Exception as e:
            print(f"[WARNING] Could not read existing ID, starting from 0. Error: {e}")

    valid_group_count = 0
    rmse_threshold = 0.12

    # 使用计算好的 mode 打开文件
    with open(csv_filename, mode=write_mode, newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        # 只有在文件是新的时候（写模式）才写入表头
        if write_mode == 'w':
            writer.writeheader()
        
        # 遍历每一组 (Unique Set)
        for i in range(num_unique_params):
            # 计算当前组在数组中的切片范围
            start_idx = i * ENV_REPEATS
            end_idx = start_idx + ENV_REPEATS
            
            # 获取这一组 10 个环境的数据
            group_rmse = rmse_np[start_idx : end_idx]
            group_crashed = has_crashed_np[start_idx : end_idx]
            
            # 核心判断逻辑：
            # 1. 没有任何一个环境发生 Crash (any 为 False)
            # 2. 所有环境的 RMSE 都小于阈值 (all 为 True)
            if not np.any(group_crashed) and np.all(group_rmse < rmse_threshold):
                
                # 因为这一组的 10 个参数是一样的，取第一个即可
                p = dynamics_params_list[start_idx]
                
                # 统计一下这一组的平均误差
                avg_group_rmse = np.mean(group_rmse)
                
                row = {
                    "id": id_counter,
                    "mass": p["mass"],
                    "arm_length": p["arm_length"],
                    "Ixx": p["_ixx"],
                    "Iyy": p["_iyy"],
                    "Izz": p["_izz"],
                    "twr": p["thrust_to_weight"], 
                    "motor_tau_up": p["motor_tau_up"],
                    "motor_tau_down": p["motor_tau_down"],
                    "kappa": p["kappa"],
                    "mean_rmse": float(avg_group_rmse)
                }
                
                writer.writerow(row)
                valid_group_count += 1
                id_counter += 1
    
    print(f"Filtering complete.")
    print(f"Total Unique Params Tested: {num_unique_params}")
    print(f"Valid Groups (All {ENV_REPEATS} reps survived & RMSE < {rmse_threshold}): {valid_group_count}")
    print(f"Results {'appended' if file_exists else 'saved'} to: {os.path.abspath(csv_filename)}")
    print(f"{'=' * 80}\n")
    
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()