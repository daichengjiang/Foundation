import numpy as np
import subprocess
import os
import time
import argparse
import sys
from datetime import datetime

def sample_raptor_dynamics():
    # ... (保持原有的动力学采样逻辑不变) ...
    twr = np.random.uniform(1.5, 5.0)
    m_min = 0.02
    m_max = 5.0
    s = np.random.uniform(np.cbrt(m_min), np.cbrt(m_max))
    mass = s ** 3
    
    m_cf = 0.032 
    l_cf = 0.04384 
    base_ratio = l_cf / (m_cf**(1/3)) 
    u = np.random.normal(0.0, 0.1) 
    u = np.clip(u, -0.3, 0.3) 
    if u < 0: s_ms = 1.0 / (1.0 - u)
    else: s_ms = 1.0 + u
    size_variation = s_ms 
    arm_length = base_ratio * (mass**(1/3)) * size_variation
    
    r_t2i = np.random.uniform(40, 1200)
    total_thrust = twr * 9.81 * mass
    tau = total_thrust * np.sqrt(2) * arm_length
    Ixx = tau / r_t2i
    Iyy = Ixx 
    Izz = Ixx * 1.832 
    
    motor_tau = np.random.uniform(0.02, 0.12)
    return {
        "mass": mass, "arm_length": arm_length, "inertia": (Ixx, Iyy, Izz),
        "thrust_to_weight": twr, "motor_tau": motor_tau
    }

def run_training(teacher_id, dynamics, timestamp, gpu_id=0, csv_path="teacher_dynamics.csv", headless=False):
    """
    调用 train.py 并传入参数
    """
    inertia_str = f"[{dynamics['inertia'][0]:.10f},{dynamics['inertia'][1]:.10f},{dynamics['inertia'][2]:.10f}]"

    overrides = [
        f"env.dynamics.mass={dynamics['mass']:.8f}",
        f"env.dynamics.arm_length={dynamics['arm_length']:.8f}",
        f"env.dynamics.inertia={inertia_str}",
        f"env.dynamics.thrust_to_weight={dynamics['thrust_to_weight']:.5f}",
        f"env.dynamics.motor_tau={dynamics['motor_tau']:.5f}",
        
        f"agent.experiment_name=raptor_teachers",
        f"agent.run_name=teacher_{teacher_id:04d}",
        # 注意：请根据你的实际路径确认 USD 路径
        'env.robot.spawn.usd_path="/home/nv/Foundation/USD/cf2x.usd"'
    ]
    
    # 假设你的 train.py 就在 foundation/rsl_rl 下，或者根据你的项目结构调整
    train_script = "foundation/rsl_rl/train.py"
    if not os.path.exists(train_script):
        # 尝试相对路径回退
        if os.path.exists("train.py"):
            train_script = "train.py"
        else:
            print(f"Error: Could not find {train_script}")
            return

    cmd = [
        sys.executable, train_script,
        "--task", "point_ctrl_single_dense",
        "--num_envs", "6400",
        "--max_iterations", "1000",
        "--device", f"cuda:{gpu_id}",
        "--logger", "wandb",
        "--log_project_name", "Foundation",
        "--log_timestamp", timestamp 
    ] + overrides
    
    # [新增] 如果 headless 为 True，则添加该参数
    if headless:
        cmd.append("--headless")

    # 保存参数到指定的 CSV 路径
    save_params_to_csv(csv_path, teacher_id, dynamics)

    print(f"==================================================")
    print(f"Starting Teacher {teacher_id} | GPU {gpu_id} | Headless: {headless}")
    print(f"Dir: .../{timestamp}/teacher_{teacher_id:04d}")
    print(f"Mass: {dynamics['mass']:.4f} kg | Arm: {dynamics['arm_length']:.4f} m") 
    print(f"TWR : {dynamics['thrust_to_weight']:.2f}    | Tau: {dynamics['motor_tau']:.3f} s")
    print(f"CSV Saved to: {csv_path}")
    print(f"==================================================")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"!!! Error training Teacher {teacher_id} !!!")
        print(e)

def save_params_to_csv(file_path, teacher_id, dynamics):
    """
    将参数追加写入到指定路径的 CSV 文件
    """
    file_exists = os.path.isfile(file_path)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, "a") as f:
        if not file_exists:
            f.write("id,mass,arm_length,Ixx,Iyy,Izz,twr,motor_tau\n")
        
        f.write(f"{teacher_id},{dynamics['mass']},{dynamics['arm_length']},"
                f"{dynamics['inertia'][0]},{dynamics['inertia'][1]},{dynamics['inertia'][2]},"
                f"{dynamics['thrust_to_weight']},{dynamics['motor_tau']}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--num_teachers", type=int, default=1)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--timestamp", type=str, default=None) 
    # [新增] headless 参数
    parser.add_argument("--headless", action="store_true", default=False, help="Run without rendering (Headless mode)")

    args = parser.parse_args()

    # 1. 确定本次运行的统一时间戳
    if args.timestamp:
        batch_timestamp = args.timestamp
    else:
        batch_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 2. 构建目标 CSV 路径
    # 路径格式: logs/rsl_rl/raptor_teachers/{timestamp}/teacher_dynamics.csv
    log_root_dir = os.path.join("logs", "rsl_rl", "raptor_teachers", batch_timestamp)
    
    # 确保日志目录先被创建 (虽然 train.py 也会创建，但我们要先写 CSV)
    os.makedirs(log_root_dir, exist_ok=True)
    
    csv_path = os.path.join(log_root_dir, "teacher_dynamics.csv")

    # 3. 自动清理 (仅当 start_id=0 时，删除该目录下可能已存在的 CSV)
    if args.start_id == 0:
        if os.path.exists(csv_path):
            print(f"[Auto-Clean] Removing existing '{csv_path}' to start fresh.")
            try:
                os.remove(csv_path)
            except OSError as e:
                print(f"Warning: Could not remove file: {e}")

    print(f"Batch Timestamp: {batch_timestamp}")
    print(f"Dynamics CSV will be saved to: {csv_path}")

    # 4. 循环训练
    for i in range(args.start_id, args.start_id + args.num_teachers):
        dyn_params = sample_raptor_dynamics()
        
        run_training(
            teacher_id=i, 
            dynamics=dyn_params, 
            timestamp=batch_timestamp, 
            gpu_id=args.gpu_id,
            csv_path=csv_path, # 传入完整路径
            headless=args.headless # [新增] 传入 headless 参数
        )
        
        time.sleep(2)