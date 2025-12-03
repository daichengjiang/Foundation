import numpy as np
import subprocess
import os
import time
import argparse
import sys
from datetime import datetime # [新增]

# ... sample_raptor_dynamics 函数保持不变 (请保留你原来的代码) ...
def sample_raptor_dynamics():
    # ... (这里不用变) ...
    # 为了完整性，请确保这里是你之前修正过的包含正确惯量计算的版本
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

def run_training(teacher_id, dynamics, timestamp, gpu_id=0):
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
        'env.robot.spawn.usd_path="/home/nv/Foundation/USD/cf2x.usd"'
    ]
    
    train_script = "foundation/rsl_rl/train.py"
    if not os.path.exists(train_script):
        if os.path.exists(os.path.join(os.getcwd(), train_script)):
            pass
        else:
            print(f"Error: Could not find {train_script}")
            return

    cmd = [
        sys.executable, train_script,
        "--task", "point_ctrl_single_dense",
        "--num_envs", "1600",
        "--max_iterations", "1000",
        "--device", f"cuda:{gpu_id}",
        "--logger", "wandb",
        "--log_project_name", "Foundation",
        "--log_timestamp", timestamp 
    ] + overrides
    
    save_params_to_csv(teacher_id, dynamics)

    print(f"==================================================")
    print(f"Starting Teacher {teacher_id} | GPU {gpu_id} | Dir: {timestamp}/teacher_{teacher_id:04d}")
    print(f"Mass: {dynamics['mass']:.4f} kg | Arm: {dynamics['arm_length']:.4f} m")  # [已修复] 加回了 Arm
    print(f"TWR : {dynamics['thrust_to_weight']:.2f}    | Tau: {dynamics['motor_tau']:.3f} s")   # [已修复] 加回了 Tau
    print(f"==================================================")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"!!! Error training Teacher {teacher_id} !!!")
        print(e)

def save_params_to_csv(teacher_id, dynamics):
    file_exists = os.path.isfile("teacher_dynamics.csv")
    with open("teacher_dynamics.csv", "a") as f:
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
    # [可选] 允许手动传入时间戳，方便断点续训时保持目录一致
    parser.add_argument("--timestamp", type=str, default=None) 
    args = parser.parse_args()

    # 1. 确定本次运行的统一时间戳
    if args.timestamp:
        batch_timestamp = args.timestamp
    else:
        batch_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 2. 自动清理 (仅当 start_id=0 且没有手动指定时间戳时，视为全新训练)
    csv_filename = "teacher_dynamics.csv"
    if args.start_id == 0 and args.timestamp is None:
        if os.path.exists(csv_filename):
            print(f"[Auto-Clean] Removing existing '{csv_filename}' to start fresh.")
            try:
                os.remove(csv_filename)
            except OSError as e:
                print(f"Warning: Could not remove file: {e}")

    print(f"Batch Timestamp: {batch_timestamp}")

    # 3. 循环训练
    for i in range(args.start_id, args.start_id + args.num_teachers):
        dyn_params = sample_raptor_dynamics()
        # 将时间戳传进去
        run_training(i, dyn_params, timestamp=batch_timestamp, gpu_id=args.gpu_id)
        
        # 冷却
        time.sleep(2)