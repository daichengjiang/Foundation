import numpy as np
import subprocess
import os
import time
import argparse
import sys

# ... sample_raptor_dynamics 函数保持不变 ...
def sample_raptor_dynamics():
    """
    Strict implementation based on RAPTOR Supplementary Materials S5-S26
    """
    # 1. 采样推重比 (S5: Uniform 1.5 - 5)
    twr = np.random.uniform(1.5, 5.0)
    
    # 2. 采样质量 (S6, S7: Cubic distribution 0.02 - 5.0)
    m_min = 0.02
    m_max = 5.0
    s = np.random.uniform(np.cbrt(m_min), np.cbrt(m_max))
    mass = s ** 3
    
    # 3. 采样臂长 (S13 - S18)
    # 论文提到 Crazyflie 的质量-尺寸比约为 7.90 (S13)，但实际分布均值约为 7.24
    # 这里我们直接复用你原本的逻辑，因为你基于 0.032/0.04384 算出的比率正好对应 7.24，符合论文"实际情况"
    m_cf = 0.032
    l_cf = 0.04384
    base_ratio = l_cf / (m_cf**(1/3)) 
    
    # 论文 S15-S18 使用正态分布扰动
    # u ~ N(-0.1, 0.1) -> s_ms
    u = np.random.normal(0.0, 0.1) # 注意这里用 Normal
    # 限制范围防止极端值 (论文隐含)
    u = np.clip(u, -0.3, 0.3) 
    
    if u < 0:
        s_ms = 1.0 / (1.0 - u)
    else:
        s_ms = 1.0 + u
        
    # 计算臂长: l_arm = mass^(1/3) * base_ratio / s_ms (逻辑上类似)
    # 这里简化为你原本的乘法逻辑，但引入 s_ms 的分布特性
    size_variation = s_ms # 使用论文的分布逻辑
    arm_length = base_ratio * (mass**(1/3)) * size_variation
    
    # 4. 采样转矩惯量比 (S19: Uniform 40 - 1200)
    r_t2i = np.random.uniform(40, 1200)
    
    # 5. 计算惯量 (Strictly following S20-S22)
    # S10: Total Thrust T = twr * 9.81 * m
    total_thrust = twr * 9.81 * mass
    
    # S20: Tau = T * sqrt(2) * l_arm  (注: 论文这里的 T 指总推力产生的最大力矩势能)
    # 解释: 单电机最大推力 T/4。力臂 l_arm。对角线电机产生力矩。
    # 论文公式直接给出了 tau 与 T_total 的关系，我们照搬公式
    tau = total_thrust * np.sqrt(2) * arm_length
    
    # S21: Jxx = Jyy = tau / r_t2i
    Ixx = tau / r_t2i
    Iyy = Ixx
    
    # S22: Jzz = (Jxx + Jyy)/2 * 1.832 -> Jxx * 1.832
    Izz = Ixx * 1.832
    
    # 6. 电机时间常数 (Mapping delays to Tau)
    # 论文 S25: 上升延迟 0.03-0.1, S26: 下降延迟 0.03-0.3
    # 我们用一阶低通滤波 Tau 来近似这个延迟。
    # 为了涵盖论文的恶劣情况，我们采样范围设为 0.02 到 0.15
    # (平均值 0.085 对应约 80ms 的响应时间，比原来的 0.05 更"慢")
    motor_tau = np.random.uniform(0.02, 0.12)
    
    return {
        "mass": mass,
        "arm_length": arm_length,
        "inertia": (Ixx, Iyy, Izz),
        "thrust_to_weight": twr,
        "motor_tau": motor_tau
    }

def run_training(teacher_id, dynamics, gpu_id=0):
    """
    调用 train.py 并传入参数 (包含 WandB 和 USD 路径)
    """
    # 1. 准备惯量字符串
    inertia_str = f"[{dynamics['inertia'][0]:.10f},{dynamics['inertia'][1]:.10f},{dynamics['inertia'][2]:.10f}]"

    # 2. 准备 Hydra Overrides (配置覆盖)
    # 这些是 key=value 格式的参数
    overrides = [
        # --- 动力学参数 ---
        f"env.dynamics.mass={dynamics['mass']:.8f}",
        f"env.dynamics.arm_length={dynamics['arm_length']:.8f}",
        f"env.dynamics.inertia={inertia_str}",
        f"env.dynamics.thrust_to_weight={dynamics['thrust_to_weight']:.5f}",
        f"env.dynamics.motor_tau={dynamics['motor_tau']:.5f}",
        
        # --- 训练日志/名称配置 ---
        # 本地日志文件夹名
        f"agent.experiment_name=raptor_teachers",
        # 具体的运行名称 (对应 WandB 的 Run Name)
        f"agent.run_name=teacher_{teacher_id:04d}",
        
        # --- [新增] 指定 USD 路径 ---
        'env.robot.spawn.usd_path="/home/nv/Foundation/USD/cf2x.usd"'
    ]
    
    # 3. 检查脚本路径
    train_script = "foundation/rsl_rl/train.py"
    if not os.path.exists(train_script):
        if os.path.exists(os.path.join(os.getcwd(), train_script)):
            pass
        else:
            print(f"Error: Could not find {train_script}")
            return

    # 4. 构建命令行参数 (CLI Flags)
    # 这些是 --flag value 格式的参数
    cmd = [
        sys.executable, train_script,
        "--task", "point_ctrl_single_dense",
        "--num_envs", "1600",  # 你的原始命令是1600，这里用2048效率通常更高(2的幂次)，也可以改回1600
        "--max_iterations", "1000",
        # "--headless",
        "--device", f"cuda:{gpu_id}",
        
        # --- [新增] WandB 配置 ---
        "--logger", "wandb",
        "--log_project_name", "Foundation"
    ] + overrides  # 将 overrides 追加到列表末尾
    
    # 5. 保存 CSV 记录
    save_params_to_csv(teacher_id, dynamics)

    print(f"==================================================")
    print(f"Starting training for Teacher {teacher_id} on GPU {gpu_id}")
    print(f"Mass: {dynamics['mass']:.4f}, TWR: {dynamics['thrust_to_weight']:.2f}")
    # print(f"CMD: {' '.join(cmd)}") # 调试用，可以取消注释查看完整命令
    print(f"==================================================")
    
    try:
        # 启动子进程
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
    args = parser.parse_args()

    # === [新增逻辑] 自动清理旧文件 ===
    # 只有当你是从第 0 号开始训练时，才视为"新的一轮"，删除旧表
    csv_filename = "teacher_dynamics.csv"
    if args.start_id == 0:
        if os.path.exists(csv_filename):
            print(f"[Auto-Clean] Detected start_id=0. Removing existing '{csv_filename}' to start fresh.")
            try:
                os.remove(csv_filename)
            except OSError as e:
                print(f"Warning: Could not remove file: {e}")
        else:
            print(f"[Auto-Clean] No existing '{csv_filename}' found. Creating new one.")

    for i in range(args.start_id, args.start_id + args.num_teachers):
        dyn_params = sample_raptor_dynamics()
        run_training(i, dyn_params, gpu_id=args.gpu_id)
        
        # 可选：给一点冷却时间，确保显存完全释放
        time.sleep(10)