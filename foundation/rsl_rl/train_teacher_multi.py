import numpy as np
import subprocess
import os
import time
import argparse
import sys
import shutil # [新增] 用于删除文件夹
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
    
    # motor_tau = np.random.uniform(0.02, 0.12) <-- 删除这行
    motor_tau_up = np.random.uniform(0.03, 0.1)
    motor_tau_down = np.random.uniform(0.03, 0.3)

    kappa = np.random.uniform(0.005, 0.05)

    return {
        "mass": mass, 
        "arm_length": arm_length, 
        "inertia": (Ixx, Iyy, Izz),
        "thrust_to_weight": twr, 
        # 修改返回字典
        "motor_tau_up": motor_tau_up,
        "motor_tau_down": motor_tau_down,
        "kappa": kappa
    }

def run_training(teacher_id, dynamics, timestamp, gpu_id=0, csv_path="teacher_dynamics.csv", headless=False, reward_threshold=11000.0):
    """
    调用 train_teacher_single.py 并传入参数，返回是否训练成功
    """
    inertia_str = f"[{dynamics['inertia'][0]:.10f},{dynamics['inertia'][1]:.10f},{dynamics['inertia'][2]:.10f}]"

    overrides = [
        f"env.dynamics.mass={dynamics['mass']:.8f}",
        f"env.dynamics.arm_length={dynamics['arm_length']:.8f}",
        f"env.dynamics.inertia={inertia_str}",
        f"env.dynamics.thrust_to_weight={dynamics['thrust_to_weight']:.5f}",
        # f"env.dynamics.motor_tau={dynamics['motor_tau']:.5f}",
        f"env.dynamics.motor_tau_up={dynamics['motor_tau_up']:.5f}",
        f"env.dynamics.motor_tau_down={dynamics['motor_tau_down']:.5f}",
        f"env.dynamics.moment_scale={dynamics['kappa']:.5f}",
        
        f"agent.experiment_name=raptor_teachers",
        f"agent.run_name=teacher_{teacher_id:04d}",
        # 注意：请根据你的实际路径确认 USD 路径
        'env.robot.spawn.usd_path="./USD/cf2x.usd"',
        "env.debug_vis=False"
    ]
    
    train_script = "foundation/rsl_rl/train_teacher_single.py"
    if not os.path.exists(train_script):
        if os.path.exists("train_teacher_single.py"):
            train_script = "train_teacher_single.py"
        else:
            print(f"Error: Could not find {train_script}")
            return False

    target_device = "cuda:0"

    cmd = [
        sys.executable, train_script,
        "--task", "teacher",
        "--num_envs", "4000",
        "--max_iterations", "600",
        "--device", target_device,
        "--logger", "wandb",
        "--log_project_name", "Foundation",
        "--log_timestamp", timestamp 
    ] + overrides
    
    if headless:
        cmd.append("--headless")

    # [新增] 构造临时文件路径，用于接收子进程的最大奖励
    result_file = os.path.abspath(f"temp_reward_{timestamp}_{teacher_id}.txt")
    if os.path.exists(result_file):
        os.remove(result_file)

    # [新增] 设置环境变量传给子进程
    env_vars = os.environ.copy()
    env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env_vars["TEACHER_MAX_REWARD_PATH"] = result_file

    print(f"==================================================")
    print(f"Starting Teacher {teacher_id} | GPU {gpu_id} | Headless: {headless}")
    print(f"Dir: .../{timestamp}/teacher_{teacher_id:04d}")
    print(f"Mass: {dynamics['mass']:.4f} kg | Arm: {dynamics['arm_length']:.4f} m") 
    print(f"TWR : {dynamics['thrust_to_weight']:.2f}    | Kappa: {dynamics['kappa']:.4f}")
    print(f"Tau Up: {dynamics['motor_tau_up']:.3f} s | Tau Down: {dynamics['motor_tau_down']:.3f} s")
    print(f"Target Reward: > {reward_threshold}")
    print(f"==================================================")
    
    success = False
    
    try:
        # 传入 env_vars
        subprocess.run(cmd, check=True, env=env_vars)
        
        # [新增] 检查结果
        max_reward = -float('inf')
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                try:
                    content = f.read().strip()
                    if content:
                        max_reward = float(content)
                except ValueError:
                    pass
            # 清理临时文件
            os.remove(result_file)
        
        print(f"Teacher {teacher_id} Finished. Max Reward: {max_reward:.2f}")

        if max_reward > reward_threshold:
            print(f"SUCCESS: Reward {max_reward:.2f} > {reward_threshold}. Saving...")
            # 只有成功了才保存 CSV
            save_params_to_csv(csv_path, teacher_id, dynamics)
            success = True
        else:
            print(f"FAILURE: Reward {max_reward:.2f} < {reward_threshold}. Deleting and Retrying...")
            success = False

    except subprocess.CalledProcessError as e:
        print(f"!!! Error training Teacher {teacher_id} (Process Crashed) !!!")
        print(e)
        success = False
    
    # [新增] 如果失败，清理日志目录
    if not success:
        # 重构日志路径: logs/rsl_rl/raptor_teachers/{timestamp}/teacher_{id}
        log_dir = os.path.join("logs", "rsl_rl", "raptor_teachers", timestamp, f"teacher_{teacher_id:04d}")
        if os.path.exists(log_dir):
            try:
                print(f"[Auto-Clean] Removing failed log dir: {log_dir}")
                shutil.rmtree(log_dir)
            except OSError as e:
                print(f"Warning: Could not remove failed dir: {e}")
                
    return success

def save_params_to_csv(file_path, teacher_id, dynamics):
    """
    将参数追加写入到指定路径的 CSV 文件
    """
    file_exists = os.path.isfile(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, "a") as f:
        if not file_exists:
            # 更新 CSV 表头
            f.write("id,mass,arm_length,Ixx,Iyy,Izz,twr,motor_tau_up,motor_tau_down,kappa\n")
        
        # 写入对应的新 key 值
        f.write(f"{teacher_id},{dynamics['mass']},{dynamics['arm_length']},"
                f"{dynamics['inertia'][0]},{dynamics['inertia'][1]},{dynamics['inertia'][2]},"
                f"{dynamics['thrust_to_weight']},"
                f"{dynamics['motor_tau_up']},{dynamics['motor_tau_down']},{dynamics['kappa']}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--num_teachers", type=int, default=1)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--timestamp", type=str, default=None) 
    parser.add_argument("--headless", action="store_true", default=False, help="Run without rendering")

    args = parser.parse_args()

    if args.timestamp:
        batch_timestamp = args.timestamp
    else:
        batch_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_root_dir = os.path.join("logs", "rsl_rl", "raptor_teachers", batch_timestamp)
    os.makedirs(log_root_dir, exist_ok=True)
    csv_path = os.path.join(log_root_dir, "teacher_dynamics.csv")

    if args.start_id == 0:
        if os.path.exists(csv_path):
            print(f"[Auto-Clean] Removing existing '{csv_path}' to start fresh.")
            try:
                os.remove(csv_path)
            except OSError as e:
                print(f"Warning: Could not remove file: {e}")

    print(f"Batch Timestamp: {batch_timestamp}")
    print(f"Dynamics CSV will be saved to: {csv_path}")

    # [修改] 使用 while 循环来实现重试逻辑
    current_teacher_id = args.start_id
    end_teacher_id = args.start_id + args.num_teachers

    while current_teacher_id < end_teacher_id:
        dyn_params = sample_raptor_dynamics()
        
        is_success = run_training(
            teacher_id=current_teacher_id, 
            dynamics=dyn_params, 
            timestamp=batch_timestamp, 
            gpu_id=args.gpu_id,
            csv_path=csv_path,
            headless=args.headless,
            reward_threshold=11000.0 # 设置阈值
        )
        
        if is_success:
            # 只有成功才移动到下一个 ID
            current_teacher_id += 1
            time.sleep(1)
        else:
            # 失败则不增加 ID，继续循环，重新采样参数进行训练
            print(f"Retrying Teacher {current_teacher_id}...")
            time.sleep(2)