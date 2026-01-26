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

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Filter valid dynamics parameters using parallel evaluation."""
    
    # 1. 预先生成所有参数
    print(f"\n{'=' * 80}")
    print(f"Generating {args_cli.num_envs} sets of dynamics parameters...")
    print(f"{'=' * 80}")
    
    dynamics_params_list = []
    for _ in range(args_cli.num_envs):
        dynamics_params_list.append(sample_raptor_dynamics())
    
    # 2. 配置环境
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.prob_null_trajectory = 0.0
    env_cfg.train_or_play = False
    env_cfg.debug_vis = False 
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric if args_cli.disable_fabric is not None else env_cfg.sim.use_fabric

    # 注入异构参数列表
    env_cfg.dynamics.multi_teacher_params = dynamics_params_list

    checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {checkpoint_path}")

    # 3. 创建环境
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
    
    # 4. 初始化统计 Tensor
    total_squared_error = torch.zeros(env.num_envs, device=env.device)
    valid_steps_count = torch.zeros(env.num_envs, device=env.device)
    
    # [新增] 记录是否发生过坠毁 (Crash)
    # 初始化为 False，一旦发生 Crash 就置为 True 并保持
    has_crashed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    timestep = 0
    print(f"\n[INFO] Starting simulation for {args_cli.max_steps} steps...")
    
    while timestep < args_cli.max_steps:
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions)
            
            if hasattr(policy_model, "reset"):
                policy_model.reset(dones)

            # ================= [新增：存活检测逻辑] =================
            # 区分 "TimeOut" (正常重置) 和 "Died" (坠机/违规)
            # DirectRLEnv 通常会将 time_outs 放入 extras 字典
            if "time_outs" in extras:
                is_timeout = extras["time_outs"]
            else:
                # 兜底：如果没有 time_outs 信息，假设所有 done 都是 crash (严格模式)
                # 或者你可以根据 step count 判断
                is_timeout = torch.zeros_like(dones)

            # 如果 Done 了，但不是 Timeout，那就是 Crash
            # logic: is_crash = dones AND (NOT is_timeout)
            current_crash = torch.logical_and(dones, torch.logical_not(is_timeout))
            
            # 累积记录：只要发生过一次 crash，这个环境就被标记为已坠毁
            has_crashed = torch.logical_or(has_crashed, current_crash)
            # =======================================================

            if timestep >= args_cli.stats_start_step:
                desired_pos = env.unwrapped.pos_des
                current_pos = env.unwrapped._robot.data.root_pos_w
                pos_error = torch.norm(current_pos - desired_pos, dim=1)
                total_squared_error += pos_error ** 2
                valid_steps_count += 1
        
        timestep += 1
        if timestep % 100 == 0:
            # 打印当前有多少环境还存活着 (从未坠毁)
            alive_count = (env.num_envs - has_crashed.sum().item())
            print(f"Step {timestep}/{args_cli.max_steps} | Never Crashed: {alive_count}/{env.num_envs}")

    # 5. 计算结果与保存
    print(f"\n{'=' * 80}")
    print(f"Simulation Complete. Filtering results...")
    
    valid_steps_count = torch.clamp(valid_steps_count, min=1.0)
    rmse_per_env = torch.sqrt(total_squared_error / valid_steps_count)
    
    # 转回 CPU
    rmse_np = rmse_per_env.cpu().numpy()
    has_crashed_np = has_crashed.cpu().numpy()
    
    csv_filename = args_cli.output_csv
    fieldnames = [
        "id", "mass", "arm_length", "Ixx", "Iyy", "Izz", 
        "twr", "motor_tau_up", "motor_tau_down", "kappa", "rmse"
    ]
    
    valid_count = 0
    id_counter = 0
    rmse_threshold = 1

    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(args_cli.num_envs):
            rmse_val = rmse_np[i]
            crashed = has_crashed_np[i]
            
            # [修改] 筛选条件：RMSE < 0.1 且 从未坠毁
            if rmse_val < rmse_threshold and not crashed:
                p = dynamics_params_list[i]
                
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
                    "rmse": float(rmse_val)
                }
                
                writer.writerow(row)
                valid_count += 1
                id_counter += 1
    
    print(f"Filtering complete.")
    print(f"Total Envs: {args_cli.num_envs}")
    print(f"Valid Envs (RMSE < {rmse_threshold} & Never Crashed): {valid_count}")
    print(f"Results saved to: {os.path.abspath(csv_filename)}")
    print(f"{'=' * 80}\n")
    
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()