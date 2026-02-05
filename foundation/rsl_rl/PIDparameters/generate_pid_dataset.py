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
parser.add_argument("--num_envs", type=int, default=6000, help="Number of environments to simulate (batch size).")
parser.add_argument("--task", type=str, default=None, required=True, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
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
from isaaclab.utils.math import euler_xyz_from_quat
from foundation.utils.pid_controller import PaperPhysControllerTensor

def run_parallel_optimization(env, num_generations=10, n_repeats=3):
    """
    并行进化优化器：利用 GPU 并行同时测试 num_envs 组参数。
    改进版：支持每个参数重复测试 n_repeats 次取平均，减少随机性。
    """
    device = env.device
    num_envs = env.num_envs
    
    # 检查是否整除
    if num_envs % n_repeats != 0:
        raise ValueError(f"[Error] num_envs ({num_envs}) must be divisible by n_repeats ({n_repeats}).")

    # 实际种群大小 = 环境总数 / 重复次数
    population_size = num_envs // n_repeats
    
    print(f"🚀 Starting Parallel Optimization: Total Envs={num_envs}, Population={population_size}, Repeats={n_repeats}, Generations={num_generations}")

    # ================= 1. 定义参数范围 (Min, Max) =================
    # 格式: [wn, zeta, tc_ang_rp, tc_ang_y, tc_rate_rp, tc_rate_y]
    # 针对大质量无人机的宽松范围
    bounds_min = torch.tensor([1.0, 0.65, 0.01, 0.05, 0.01, 0.02], device=device)
    bounds_max = torch.tensor([10.0, 0.95, 0.20, 0.50, 0.1, 0.25], device=device)
    
    # 参数容器 (Population, 6)
    current_params = torch.rand(population_size, 6, device=device)
    # 归一化映射到 [min, max]
    current_params = current_params * (bounds_max - bounds_min) + bounds_min

    # 最优记录
    best_loss = float('inf')
    best_param_dict = {}

    # ================= 2. 进化循环 =================
    for gen in range(num_generations):
        
        # --- A. 提取参数 (Population, ) ---
        wn          = current_params[:, 0]
        zeta        = current_params[:, 1]
        tc_ang_rp   = current_params[:, 2]
        tc_ang_y    = current_params[:, 3]
        tc_rate_rp  = current_params[:, 4]
        tc_rate_y   = current_params[:, 5]

        # --- B. 参数广播 (Expansion) ---
        # 将参数复制 n_repeats 次以填满所有 num_envs 个环境
        # 例如 P1, P2 -> P1, P1, P1, P2, P2, P2
        wn_exp          = wn.repeat_interleave(n_repeats)
        zeta_exp        = zeta.repeat_interleave(n_repeats)
        tc_ang_rp_exp   = tc_ang_rp.repeat_interleave(n_repeats)
        tc_ang_y_exp    = tc_ang_y.repeat_interleave(n_repeats)
        tc_rate_rp_exp  = tc_rate_rp.repeat_interleave(n_repeats)
        tc_rate_y_exp   = tc_rate_y.repeat_interleave(n_repeats)

        # 更新环境中的控制器参数 (传入 num_envs 大小的 Tensor)
        env.unwrapped._controller.update_gains(
            wn_exp, zeta_exp, tc_ang_rp_exp, tc_ang_y_exp, tc_rate_rp_exp, tc_rate_y_exp
        )

        # --- C. 计算惩罚 (Soft Constraints) ---
        # 注意：在 Population 维度计算即可，不需要广播，这样计算量小
        
        # 约束 1: 1/wn < 2 * tc_ang_rp
        violation1 = torch.relu((tc_ang_rp * 3.0) - (1.0 / wn))
        # 约束 2: tc_ang_rp < 2 * tc_rate_rp
        violation2 = torch.relu((tc_rate_rp * 3.0) - tc_ang_rp)
        # 约束 3: tc_ang_y >= tc_ang_rp
        violation3 = torch.relu(tc_ang_rp - tc_ang_y)

        # 总惩罚系数 (Population, )
        penalty_score = (violation1 + violation2 + violation3) * 100.0

        # --- D. 运行仿真评估 ---
        obs, _ = env.reset()
        
        # 累计误差容器 (Total Envs, )
        total_sq_error = torch.zeros(num_envs, device=device)
        crashed_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        
        steps = 5000 
        warmup = 500 

        for t in range(steps):
            actions = torch.zeros((num_envs, env.num_actions), device=device)
            obs, rewards, dones, extras = env.step(actions)
            
            if "time_outs" in extras:
                is_crash = torch.logical_and(dones, ~extras["time_outs"])
            else:
                is_crash = dones
            crashed_mask = torch.logical_or(crashed_mask, is_crash)

            if t >= warmup:
                pos_error = torch.norm(env.unwrapped.pos_des - env.unwrapped._robot.data.root_pos_w, dim=1)
                total_sq_error += pos_error ** 2

                quat_w = env.unwrapped._robot.data.root_quat_w
                _, _, yaw_curr = euler_xyz_from_quat(quat_w)
                
                # 计算误差并归一化到 [-pi, pi]
                yaw_error = env.unwrapped.yaw_des - yaw_curr
                yaw_error = torch.remainder(yaw_error + torch.pi, 2 * torch.pi) - torch.pi
                
                # 累计 Yaw 的平方误差
                # 注意：通常需要给 Yaw 误差一个权重（例如 0.5），因为它的量级和位置(米)不同
                total_sq_error += (yaw_error ** 2) * 0.5

        # --- E. 计算最终得分 (Cost Aggregation) ---
        
        # 1. 计算每个环境的 RMSE (Total Envs, )
        mse_all = total_sq_error / (steps - warmup)
        rmse_all = torch.sqrt(mse_all)
        
        # 2. 计算每个环境的 Crash Score (Total Envs, )
        crash_score_all = crashed_mask.float() * 20.0
        
        # 3. [关键步骤] 取平均值 (Reduce Mean)
        # 将 (P*N, ) 重塑为 (P, N)，然后在维度 1 求平均
        rmse_avg = rmse_all.view(population_size, n_repeats).mean(dim=1)
        crash_avg = crash_score_all.view(population_size, n_repeats).mean(dim=1)
        
        # 4. 总 Cost (Population, )
        total_cost = rmse_avg + penalty_score + crash_avg
        total_cost = torch.nan_to_num(total_cost, nan=100.0, posinf=100.0)

        # --- F. 筛选精英 (Selection) ---
        sorted_indices = torch.argsort(total_cost)
        elites_count = int(population_size * 0.15)
        if elites_count < 1: elites_count = 1
        
        elite_indices = sorted_indices[:elites_count]
        elite_params = current_params[elite_indices]
        elite_costs = total_cost[elite_indices]

        # 记录本代最优
        current_best_loss = elite_costs[0].item()
        print(f"Gen {gen+1}/{num_generations} | Best Cost: {current_best_loss:.4f} | Avg Elite: {elite_costs.mean():.4f}")

        if current_best_loss < best_loss:
            best_loss = current_best_loss
            best_p = elite_params[0]
            best_param_dict = {
                "wn": best_p[0].item(),
                "zeta": best_p[1].item(),
                "tc_ang_rp": best_p[2].item(),
                "tc_ang_y": best_p[3].item(),
                "tc_rate_rp": best_p[4].item(),
                "tc_rate_y": best_p[5].item(),
            }

        # --- G. 繁殖下一代 (Mutation) ---
        num_to_fill = population_size - elites_count
        sample_indices = torch.randint(0, elites_count, (num_to_fill,), device=device)
        offsprings = elite_params[sample_indices].clone()

        mutation_scale = 0.2 * (1.0 - gen / num_generations) 
        if mutation_scale < 0.05: mutation_scale = 0.05

        noise = torch.randn_like(offsprings) * mutation_scale * (bounds_max - bounds_min) * 0.1
        offsprings += noise

        offsprings = torch.max(torch.min(offsprings, bounds_max), bounds_min)
        current_params = torch.cat([elite_params, offsprings], dim=0)

    print(f"✅ Optimization Finished. Best Cost: {best_loss:.4f}")
    return best_param_dict, best_loss


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
    
    output_file = "hypernetwork_training_data.csv"
    # [修改 1] 在最前面添加 id 列
    headers = ["id", "mass", "arm_length", "Ixx", "Iyy", "Izz", "thrust_to_weight", "motor_tau_up", "motor_tau_down", "kappa", 
               "wn", "zeta", "tc_ang_rp", "tc_ang_y", "tc_rate_rp", "tc_rate_y", "best_rmse"]

    # 初始化文件（如果不存在，写入表头）
    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=headers).writeheader()

    # 1. 采样动力学
    dynamics_params_list = []
    dyn = sample_raptor_dynamics()
    dynamics_params_list.append(dyn)
    
    # 2. 将动力学参数注入环境
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.prob_null_trajectory = 0.0
    env_cfg.train_or_play = True
    env_cfg.debug_vis = False 
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric if args_cli.disable_fabric is not None else env_cfg.sim.use_fabric
    env_cfg.dynamics.multi_teacher_params = dynamics_params_list 

    try:
        env = gym.make(args_cli.task, cfg=env_cfg)
    except Exception as e:
        print(f"[ERROR] Failed to create env: {e}")
        return

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    env = RslRlVecEnvWrapper(env)

    # 3. 运行自动寻优
    print(f"Optimizing for (Mass: {dyn['mass']:.3f}) using Parallel Evolution...")
    
    best_gains, best_rmse = run_parallel_optimization(
        env, 
        num_generations=20, 
        n_repeats=3 
    )

    # ================= [关键修改] 筛选与 ID 生成逻辑 =================
    
    # [修改 2] 筛选逻辑：如果 RMSE 太大，直接舍弃
    threshold_rmse = 0.4
    if best_rmse > threshold_rmse:
        print(f"❌ Result Rejected: RMSE {best_rmse:.4f} > {threshold_rmse}. Not saving.")
        return # 直接退出，不保存

    # [修改 3] ID 生成逻辑：读取现有文件获取最后一个 ID
    new_id = 0
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', newline='') as f:
                # 使用 DictReader 读取
                reader = csv.DictReader(f)
                # 这种方式对于超大文件可能稍慢，但对于几千行的数据集完全没问题且稳健
                # 将迭代器转为列表以获取最后一行
                rows = list(reader)
                if rows:
                    last_row = rows[-1]
                    # 确保 id 列存在且有值
                    if "id" in last_row and last_row["id"]:
                        new_id = int(last_row["id"]) + 1
        except Exception as e:
            print(f"[Warning] Failed to read last ID from CSV: {e}. Starting from id=0.")

    # 4. 保存结果
    result_row = {**dyn, **best_gains, "best_rmse": best_rmse}
    result_row.update({"Ixx": dyn["inertia"][0], "Iyy": dyn["inertia"][1], "Izz": dyn["inertia"][2]})
    result_row["id"] = new_id  # 添加 ID
    
    print(f"✅ Result Accepted: ID={new_id}, RMSE={best_rmse:.4f}. Saving to {output_file}...")

    with open(output_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
        writer.writerow(result_row)

if __name__ == "__main__":
    main()