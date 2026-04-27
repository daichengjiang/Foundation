import numpy as np
import subprocess
import os
import time
import argparse
import sys
import shutil 
import json
from datetime import datetime

def sample_raptor_dynamics():
    # 保持原有的动力学采样逻辑不变
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
        "kappa": kappa
    }

def run_training(teacher_id, dynamics, timestamp, experiment_name, gpu_id=0, enable_curriculum=False, headless=False):
    """
    运行单次训练，并返回该次训练的奖励字典。
    """
    # 根据是否开启课程，给 run_name 添加后缀
    suffix = "curr" if enable_curriculum else "nocurr"
    run_name = f"teacher_{teacher_id:04d}_{suffix}"

    inertia_str = f"[{dynamics['inertia'][0]:.10f},{dynamics['inertia'][1]:.10f},{dynamics['inertia'][2]:.10f}]"

    # 将 env.enable_curriculum 传入 Hydra overrides
    overrides = [
        f"env.dynamics.mass={dynamics['mass']:.8f}",
        f"env.dynamics.arm_length={dynamics['arm_length']:.8f}",
        f"env.dynamics.inertia={inertia_str}",
        f"env.dynamics.thrust_to_weight={dynamics['thrust_to_weight']:.5f}",
        f"env.dynamics.motor_tau_up={dynamics['motor_tau_up']:.5f}",
        f"env.dynamics.motor_tau_down={dynamics['motor_tau_down']:.5f}",
        f"env.dynamics.moment_scale={dynamics['kappa']:.5f}",
        
        f"env.enable_curriculum={enable_curriculum}", # <--- 控制课程开关
        
        f"agent.experiment_name={experiment_name}",
        f"agent.run_name={run_name}",
        'env.robot.spawn.usd_path="./USD/cf2x.usd"',
        "env.debug_vis=False"
    ]
    
    train_script = "foundation/rsl_rl/train_teacher_single.py"
    if not os.path.exists(train_script):
        if os.path.exists("train_teacher_single.py"):
            train_script = "train_teacher_single.py"
        else:
            print(f"Error: Could not find {train_script}")
            return None

    max_iters = "1000" if enable_curriculum else "800"

    target_device = f"cuda:{gpu_id}"

    cmd = [
        sys.executable, train_script,
        "--task", "teacher",
        "--num_envs", "8000",
        "--max_iterations", max_iters,
        "--device", target_device,
        "--logger", "wandb",
        "--log_project_name", "Foundation",
        "--log_timestamp", timestamp 
    ] + overrides
    
    if headless:
        cmd.append("--headless")

    teacher_log_dir = os.path.join("logs", "rsl_rl", experiment_name, timestamp, run_name)
    metrics_file = os.path.join(teacher_log_dir, "eval_metrics.json")
    metrics_file_abs = os.path.abspath(metrics_file)

    env_vars = os.environ.copy()
    env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env_vars["TEACHER_REWARD_PATH"] = metrics_file_abs

    print(f"\n[{'CURRICULUM ON' if enable_curriculum else 'CURRICULUM OFF'}] Running {run_name} ...")
    
    # 默认返回 -inf 的字典，防崩
    stats = {
        "position": -float('inf'), 
        "orientation": -float('inf'), 
        "action_smooth": -float('inf'), 
        "base": -float('inf'),      # <--- 新增默认值
        "terminal": -float('inf'),  # <--- 新增默认值
        "total": -float('inf')
    }

    try:
        subprocess.run(cmd, check=True, env=env_vars)
        
        if os.path.exists(metrics_file_abs):
            with open(metrics_file_abs, 'r') as f:
                try:
                    loaded_stats = json.load(f)
                    stats.update(loaded_stats)
                except json.JSONDecodeError:
                    print(f"Error: Could not decode JSON from {metrics_file_abs}")
        else:
            print(f"FAILURE: Metrics file not found at {metrics_file_abs}")

    except subprocess.CalledProcessError as e:
        print(f"!!! Error training {run_name} (Process Crashed) !!!")
        print(e)
        return None # 返回 None 代表进程级别的崩溃（如 CUDA OOM），这种需要重试
                
    return stats

def save_ab_test_to_csv(file_path, teacher_id, dynamics, stats_nocurr, stats_curr):
    """
    将同一套动力学参数，在开/关课程下的得分并排写入 CSV，方便直接对比
    """
    file_exists = os.path.isfile(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, "a") as f:
        if not file_exists:
            # 写入带有完整动力学参数和六项分数的对比表头
            f.write("id,mass,arm_length,Ixx,Iyy,Izz,twr,motor_tau_up,motor_tau_down,kappa,"
                    "pos_nocurr,ori_nocurr,smooth_nocurr,base_nocurr,term_nocurr,total_nocurr,"
                    "pos_curr,ori_curr,smooth_curr,base_curr,term_curr,total_curr\n")
        
        # 写入参数与各项得分
        f.write(f"{teacher_id},{dynamics['mass']},{dynamics['arm_length']},"
                f"{dynamics['inertia'][0]},{dynamics['inertia'][1]},{dynamics['inertia'][2]},"
                f"{dynamics['thrust_to_weight']},{dynamics['motor_tau_up']},"
                f"{dynamics['motor_tau_down']},{dynamics['kappa']},"
                # 无课程各项得分
                f"{stats_nocurr['position']:.2f},{stats_nocurr['orientation']:.2f},"
                f"{stats_nocurr['action_smooth']:.2f},{stats_nocurr['base']:.2f},"
                f"{stats_nocurr['terminal']:.2f},{stats_nocurr['total']:.2f},"
                # 有课程各项得分
                f"{stats_curr['position']:.2f},{stats_curr['orientation']:.2f},"
                f"{stats_curr['action_smooth']:.2f},{stats_curr['base']:.2f},"
                f"{stats_curr['terminal']:.2f},{stats_curr['total']:.2f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--num_pairs", type=int, default=10000, help="测试多少组动力学参数")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--timestamp", type=str, default=None) 
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--experiment_name", type=str, default="ab_test_curriculum")

    args = parser.parse_args()

    if args.timestamp:
        batch_timestamp = args.timestamp
    else:
        batch_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_root_dir = os.path.join("logs", "rsl_rl", args.experiment_name, batch_timestamp)
    os.makedirs(log_root_dir, exist_ok=True)
    csv_path = os.path.join(log_root_dir, "ab_test_results.csv")

    if args.start_id == 0:
        if os.path.exists(csv_path):
            print(f"[Auto-Clean] Removing existing '{csv_path}' to start fresh.")
            try:
                os.remove(csv_path)
            except OSError:
                pass

    print(f"=========================================================")
    print(f" A/B Test Curriculum Learning")
    print(f" Batch Timestamp: {batch_timestamp}")
    print(f" Output CSV: {csv_path}")
    print(f"=========================================================")

    current_id = args.start_id
    end_id = args.start_id + args.num_pairs

    while current_id < end_id:
        # 1. 采样一组动力学参数
        dyn_params = sample_raptor_dynamics()
        
        print(f"\n{'='*50}")
        print(f" [ID: {current_id:04d}] Target Dynamics ")
        print(f" Mass: {dyn_params['mass']:.4f} kg | TWR: {dyn_params['thrust_to_weight']:.2f}")
        print(f" Tau Up: {dyn_params['motor_tau_up']:.3f} | Tau Down: {dyn_params['motor_tau_down']:.3f}")
        print(f"{'='*50}")

        # 2. 第一次训练：不开启课程学习 (enable_curriculum=False)
        stats_nocurr = run_training(
            teacher_id=current_id, 
            dynamics=dyn_params, 
            timestamp=batch_timestamp,
            experiment_name=args.experiment_name, 
            gpu_id=args.gpu_id,
            enable_curriculum=False, # <-- 关闭
            headless=args.headless
        )
        
        # 3. 第二次训练：开启课程学习 (enable_curriculum=True)
        stats_curr = run_training(
            teacher_id=current_id, 
            dynamics=dyn_params, 
            timestamp=batch_timestamp,
            experiment_name=args.experiment_name, 
            gpu_id=args.gpu_id,
            enable_curriculum=True, # <-- 开启
            headless=args.headless
        )
        
        # 4. 如果两次都出现由于爆显存等原因导致 Python 进程崩溃（返回 None），则重试本组
        if stats_nocurr is None or stats_curr is None:
            print(f"!!! Process crashed for ID {current_id}, retrying this parameter set !!!")
            time.sleep(3)
            continue
            
        # 5. 打印对比并保存
        print(f"\n[ID {current_id} Results]")
        print(f"  NO Curriculum Total Score : {stats_nocurr['total']:.2f}")
        print(f"  WITH Curriculum Total Score: {stats_curr['total']:.2f}")
        
        save_ab_test_to_csv(csv_path, current_id, dyn_params, stats_nocurr, stats_curr)
        
        current_id += 1
        time.sleep(1)