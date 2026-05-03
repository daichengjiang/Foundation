import numpy as np
import subprocess
import os
import time
import argparse
import sys
import shutil 
import json  # [新增] 引入 json 库
from datetime import datetime

def sample_raptor_dynamics():
    mass = np.random.uniform(0.8, 1.2)
    arm_length = np.random.uniform(0.09, 0.13)
    twr = np.random.uniform(2.0, 3.0)
    
    r_t2i = np.random.uniform(250, 750)
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

def run_training(teacher_id, dynamics, timestamp, experiment_name, gpu_id=0, csv_path="teacher_dynamics.csv", headless=False):
    """
    调用 train_teacher_single.py 并传入参数，读取JSON文件判断是否成功
    """
    inertia_str = f"[{dynamics['inertia'][0]:.10f},{dynamics['inertia'][1]:.10f},{dynamics['inertia'][2]:.10f}]"

    overrides = [
        f"env.dynamics.mass={dynamics['mass']:.8f}",
        f"env.dynamics.arm_length={dynamics['arm_length']:.8f}",
        f"env.dynamics.inertia={inertia_str}",
        f"env.dynamics.thrust_to_weight={dynamics['thrust_to_weight']:.5f}",
        f"env.dynamics.motor_tau_up={dynamics['motor_tau_up']:.5f}",
        f"env.dynamics.motor_tau_down={dynamics['motor_tau_down']:.5f}",
        f"env.dynamics.moment_scale={dynamics['kappa']:.5f}",
        
        f"agent.experiment_name={experiment_name}",
        f"agent.run_name=teacher_{teacher_id:04d}",
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
        "--num_envs", "16000",
        "--max_iterations", "1000",
        "--device", target_device,
        "--logger", "wandb",
        "--log_project_name", "Foundation",
        "--log_timestamp", timestamp 
    ] + overrides
    
    if headless:
        cmd.append("--headless")

    # [修改] 预先构建该 Teacher 的日志目录和 Metrics 文件路径
    # 路径结构: logs/rsl_rl/{experiment_name}/{timestamp}/teacher_{id}
    # 注意：这个目录由 RSL-RL 在训练开始时自动创建，但我们可以预知其位置
    teacher_log_dir = os.path.join("logs", "rsl_rl", experiment_name, timestamp, f"teacher_{teacher_id:04d}")
    metrics_file = os.path.join(teacher_log_dir, "eval_metrics.json")
    
    # 确保 metrics_file 是绝对路径，防止子进程工作目录不同导致找不到
    metrics_file_abs = os.path.abspath(metrics_file)

    # [修改] 设置环境变量
    env_vars = os.environ.copy()
    env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # 告诉 teacher_env.py 直接把 JSON 写到这里，永久保存
    env_vars["TEACHER_REWARD_PATH"] = metrics_file_abs

    Pos_threshold = -2000
    Ori_threshold = -800
    Smooth_threshold = -500
    Total_threshold = 8000

    print(f"==================================================")
    print(f"Starting Teacher {teacher_id} | GPU {gpu_id} | Headless: {headless}")
    print(f"Log Dir: {teacher_log_dir}")
    print(f"Mass: {dynamics['mass']:.4f} kg | Arm: {dynamics['arm_length']:.4f} m") 
    print(f"TWR : {dynamics['thrust_to_weight']:.2f}    | Kappa: {dynamics['kappa']:.4f}")
    print(f"Tau Up: {dynamics['motor_tau_up']:.3f} s | Tau Down: {dynamics['motor_tau_down']:.3f} s")
    print(f"Conditions: Pos > {Pos_threshold}, Ori > {Ori_threshold}, Smooth > {Smooth_threshold}")
    print(f"==================================================")
    
    success = False
    
    try:
        # 执行训练
        subprocess.run(cmd, check=True, env=env_vars)
        
        # [修改] 读取 JSON 文件并进行多条件判断
        if os.path.exists(metrics_file_abs):
            with open(metrics_file_abs, 'r') as f:
                try:
                    stats = json.load(f)
                    pos_reward = stats.get("position", -float('inf'))
                    ori_reward = stats.get("orientation", -float('inf'))
                    smooth_reward = stats.get("action_smooth", -float('inf'))
                    total_reward = stats.get("total", -float('inf'))
                    
                    print(f"Teacher {teacher_id} Metrics: Pos={pos_reward:.2f}, Ori={ori_reward:.2f}, Smooth={smooth_reward:.2f}")

                    # [关键修改] 三个条件同时满足
                    if (pos_reward > Pos_threshold and 
                        ori_reward > Ori_threshold and 
                        smooth_reward > Smooth_threshold and
                        total_reward > Total_threshold):
                        
                        print(f"SUCCESS: All conditions met. Saving...")
                        save_params_to_csv(csv_path, teacher_id, dynamics)
                        success = True
                    else:
                        print(f"FAILURE: Conditions not met.")
                        success = False
                        
                except json.JSONDecodeError:
                    print(f"Error: Could not decode JSON from {metrics_file_abs}")
                    success = False
        else:
            print(f"FAILURE: Metrics file not found at {metrics_file_abs}")
            success = False

    except subprocess.CalledProcessError as e:
        print(f"!!! Error training Teacher {teacher_id} (Process Crashed) !!!")
        print(e)
        success = False
    
    # [修改] 如果失败，清理日志目录
    if not success:
        if os.path.exists(teacher_log_dir):
            try:
                print(f"[Auto-Clean] Removing failed log dir: {teacher_log_dir}")
                shutil.rmtree(teacher_log_dir)
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
    parser.add_argument("--experiment_name", type=str, default="c5_teachers", help="Experiment name for saving logs")

    args = parser.parse_args()

    if args.timestamp:
        batch_timestamp = args.timestamp
    else:
        batch_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_root_dir = os.path.join("logs", "rsl_rl", args.experiment_name, batch_timestamp)
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
    print(f"Experiment Name: {args.experiment_name}")
    print(f"Dynamics CSV will be saved to: {csv_path}")

    current_teacher_id = args.start_id
    end_teacher_id = args.start_id + args.num_teachers

    while current_teacher_id < end_teacher_id:
        dyn_params = sample_raptor_dynamics()
        
        is_success = run_training(
            teacher_id=current_teacher_id, 
            dynamics=dyn_params, 
            timestamp=batch_timestamp,
            experiment_name=args.experiment_name, 
            gpu_id=args.gpu_id,
            csv_path=csv_path,
            headless=args.headless
        )
        
        if is_success:
            current_teacher_id += 1
            time.sleep(1)
        else:
            print(f"Retrying Teacher {current_teacher_id}...")
            time.sleep(2)