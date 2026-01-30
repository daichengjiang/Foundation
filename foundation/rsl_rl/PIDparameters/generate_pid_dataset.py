import argparse
import sys
import csv
import os
import time
import numpy as np
import gymnasium as gym

from isaaclab.app import AppLauncher

# local imports
from foundation.rsl_rl import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Filter valid dynamics parameters in parallel.")
parser.add_argument("--num_envs", type=int, default=3000, help="Number of environments to simulate (batch size).")
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

import torch
import torchvision
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

import optuna
from foundation.utils.pid_controller import PaperPhysControllerTensor

def run_optimization_study(env, dyn_params, num_trials=400):
    device = env.device
    num_envs = env.num_envs

    def objective(trial):
        # 1. 定义寻优范围
        wn = trial.suggest_float("wn", 0.5, 6.0)
        zeta = trial.suggest_float("zeta", 0.6, 0.9)
        tc_ang_rp = trial.suggest_float("tc_ang_rp", 0.04, 0.12)
        tc_ang_y = trial.suggest_float("tc_ang_y", 0.15, 0.5)
        tc_rate_rp = trial.suggest_float("tc_rate_rp", 0.02, 0.06)
        tc_rate_y = trial.suggest_float("tc_rate_y", 0.1, 0.3)

        # 约束
        if (1.0 / wn) < (tc_ang_rp * 2.0): return 20.0
        if tc_ang_rp < (tc_rate_rp * 2.0): return 20.0
        if tc_ang_y < tc_ang_rp: return 20.0

        # 2. 将增益注入控制器
        wn_t = torch.full((num_envs,), wn, device=device)
        zeta_t = torch.full((num_envs,), zeta, device=device)
        tc_ang_rp_t = torch.full((num_envs,), tc_ang_rp, device=device)
        tc_ang_y_t = torch.full((num_envs,), tc_ang_y, device=device)
        tc_rate_rp_t = torch.full((num_envs,), tc_rate_rp, device=device)
        tc_rate_y_t = torch.full((num_envs,), tc_rate_y, device=device)
        
        env.unwrapped._controller.update_gains(wn_t, zeta_t, tc_ang_rp_t, tc_ang_y_t, tc_rate_rp_t, tc_rate_y_t)

        # 3. 运行短周期评估仿真
        obs, _ = env.reset()
        total_sq_error = torch.zeros(num_envs, device=device)
        steps = 1000 
        warmup = 200 # 建议跳过初始抖动
        crashed = torch.zeros(num_envs, dtype=torch.bool, device=device)

        for t in range(steps):
            # 这里的 actions 虽然被 teacher_env 忽略，但仍需传入正确维度的 Tensor
            actions = torch.zeros((num_envs, env.num_actions), device=device)
            obs, rewards, dones, extras = env.step(actions)
            
            # 只有未超时的 done 才算 crash
            if "time_outs" in extras:
                is_crash = torch.logical_and(dones, ~extras["time_outs"])
            else:
                is_crash = dones
            crashed = torch.logical_or(crashed, is_crash)

            if t >= warmup:
                pos_error = torch.norm(env.unwrapped.pos_des - env.unwrapped._robot.data.root_pos_w, dim=1)
                total_sq_error += pos_error ** 2

        # 4. 计算最终 Cost
        avg_rmse = torch.sqrt(total_sq_error.mean() / (steps - warmup)).item()
        crash_penalty = crashed.float().mean().item() * 10.0
        
        return avg_rmse + crash_penalty

    # 使用 CMA-ES 算法进行高效寻优
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(multivariate=True))
    study.enqueue_trial({
        "wn": 2.0,
        "zeta": 0.7,
        "tc_ang_rp": 0.08,
        "tc_ang_y": 0.4,
        "tc_rate_rp": 0.04,
        "tc_rate_y": 0.20
    })
    study.optimize(objective, n_trials=num_trials)
    
    return study.best_params, study.best_value

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
    # ... 环境初始化代码 (参考 sample_teacher.py) ...
    
    output_file = "hypernetwork_training_data.csv"
    headers = ["mass", "arm_length", "Ixx", "Iyy", "Izz", "thrust_to_weight", "motor_tau_up", "motor_tau_down", "kappa", 
               "wn", "zeta", "tc_ang_rp", "tc_ang_y", "tc_rate_rp", "tc_rate_y", "best_rmse"]

    # 如果文件不存在则创建并写入表头
    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            csv.DictWriter(f, fieldnames=headers).writeheader()

    # 1. 采样动力学
    dynamics_params_list = []
    dyn = sample_raptor_dynamics()
    dynamics_params_list.append(dyn)
    
    # 2. 将动力学参数注入环境（通过覆盖 env_cfg 或直接修改机器人属性）
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.prob_null_trajectory = 0.0
    env_cfg.train_or_play = False
    env_cfg.debug_vis = False 
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric if args_cli.disable_fabric is not None else env_cfg.sim.use_fabric
    env_cfg.dynamics.multi_teacher_params = dynamics_params_list # 注入参数

    try:
        env = gym.make(args_cli.task, cfg=env_cfg)
    except Exception as e:
        print(f"[ERROR] Failed to create env: {e}")
        return

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    env = RslRlVecEnvWrapper(env)

    # 3. 运行自动寻优
    print(f"Optimizing for (Mass: {dyn['mass']:.3f})...")
    best_gains, best_rmse = run_optimization_study(env, dyn)

    # 4. 保存结果
    result_row = {**dyn, **best_gains, "best_rmse": best_rmse}
    # 将 Ixx, Iyy, Izz 拆开写入
    result_row.update({"Ixx": dyn["inertia"][0], "Iyy": dyn["inertia"][1], "Izz": dyn["inertia"][2]})
    
    with open(output_file, 'a') as f:
        # 过滤掉不需要的键，仅保留 headers 中的字段
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
        writer.writerow(result_row)

if __name__ == "__main__":
    main()