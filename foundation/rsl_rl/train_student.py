import argparse
from isaaclab.app import AppLauncher

# === 1. 解析参数 (必须在 import omni/torch 之前) ===
parser = argparse.ArgumentParser()
parser.add_argument("--num_teachers", type=int, default=2, help="Number of teachers to distill")
parser.add_argument("--teacher_log_dir", type=str, required=True, help="Path to teacher logs")
parser.add_argument("--dynamics_csv", type=str, default=None, help="Path to dynamics parameters")
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--steps_per_epoch", type=int, default=500, help="Steps per trajectory")
parser.add_argument("--warmup_epochs", type=int, default=10)
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--ckpt_mode", type=str, default="latest", help="Checkpoint to load")
parser.add_argument("--headless", action="store_true", default=False, help="Run without rendering")

args = parser.parse_args()

# === 2. 启动 Isaac Sim (AppLauncher) ===
# 这一步非常关键，必须在 import torch 之前执行
app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

# === 3. 导入其他库 (必须在 simulation_app 启动后) ===
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
from datetime import datetime
import wandb
import gym

from foundation.algo.student_policy import RaptorStudent
from foundation.algo.batched_teacher import BatchedTeacherPolicy
from foundation.utils.distillation_env import DistillationQuadcopterEnv
from foundation.tasks.point_ctrl.quad_point_ctrl_env_single_dense import QuadcopterEnvCfg

device = torch.device(f"cuda:{args.gpu_id}")

def main():

    # === 自动定位 CSV 文件 ===
    if args.dynamics_csv is None:
        csv_path = os.path.join(args.teacher_log_dir, "teacher_dynamics.csv")
    else:
        csv_path = args.dynamics_csv
        
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dynamics CSV not found at: {csv_path}. Please check teacher_log_dir or specify --dynamics_csv.")
    
    print(f"Using Dynamics CSV: {csv_path}")
    # ===============================

    # 1. 初始化 WandB
    run_name = f"student_distill_{args.num_teachers}T_{datetime.now().strftime('%m%d_%H%M')}"
    wandb.init(project="Foundation_Student", name=run_name, config=vars(args))

    # 2. 准备环境配置
    env_cfg = QuadcopterEnvCfg()
    env_cfg.scene.num_envs = args.num_teachers # 强制 1对1
    env_cfg.sim.device = f"cuda:{args.gpu_id}"
    env_cfg.train_or_play = True # 启用训练模式的随机初始化

    # 如果是 headless 模式，虽然 AppLauncher 已经处理了窗口，
    # 但我们最好也关闭相机的渲染逻辑以节省性能 (如果配置里有的话)
    if args.headless:
        # 某些配置下可能需要手动禁用相机渲染，具体取决于你的 EnvCfg 实现
        # 这里是一个示例，通常 AppLauncher 处理完就够了
        pass 

    # 3. 初始化环境 (带动力学覆盖)
    env = DistillationQuadcopterEnv(
        cfg=env_cfg,
        dynamics_csv_path=csv_path, # [修改] 使用处理后的路径
        num_teachers=args.num_teachers
    )
    
    # 4. 加载 Teacher (Batched)
    teachers = BatchedTeacherPolicy(
        args.num_teachers, 
        args.teacher_log_dir, 
        checkpoint_mode=args.ckpt_mode,
        device=device
    ).to(device)
    
    teachers.eval()
    
    # === [关键修复] 动态获取观测维度 ===
    print("[Info] Resetting env to determine observation shape...")
    obs_dict, _ = env.reset()
    sample_obs = obs_dict['policy'] # (Num_Envs, Obs_Dim)
    
    full_obs_dim = sample_obs.shape[1]
    student_obs_dim = full_obs_dim - 4 
    action_dim = 4 # 或者从 env.action_space 获取
    
    print(f"Detected Full Obs Dim : {full_obs_dim}")
    print(f"Student Obs Dim       : {student_obs_dim} (Occluding motor speeds)")
    
    if student_obs_dim <= 0:
        raise ValueError(f"Student obs dim is {student_obs_dim}, something is wrong! Full obs dim was {full_obs_dim}")

    # 5. 初始化 Student
    student = RaptorStudent(input_dim=student_obs_dim, action_dim=action_dim, hidden_dim=16).to(device)
    optimizer = optim.Adam(student.parameters(), lr=1e-3)
    
    print(f"=== Start Distillation ===")
    print(f"Teachers: {args.num_teachers}, Steps/Epoch: {args.steps_per_epoch}")

    try:
        for epoch in range(args.epochs):
            # === 1. Rollout (Data Collection) ===
            obs_dict, _ = env.reset()
            obs = obs_dict['policy'] # Full obs (26 dim)
            
            # Student 隐状态重置
            hidden = student.init_hidden(args.num_teachers, device)
            
            # 存储轨迹用于 BPTT
            # 注意：这里存的必须是 Student 看到的 observation (切片后的)
            traj_student_obs = []     
            traj_teacher_actions = []
            
            student.train()
            
            # 决定谁来控制 (Warmup 阶段 Teacher 飞，之后 Student 飞)
            use_teacher_control = (epoch < args.warmup_epochs)
            
            total_pos_error = 0.0

            for step in range(args.steps_per_epoch):
                # 关键步骤：分离观测
                # Teacher 看到完整的 (包含电机转速)
                teacher_obs = obs 
                # Student 必须切掉最后 4 位 (电机转速)
                # 假设 obs 顺序: ..., last_actions(4), motor_speeds(4)
                student_obs = obs[:, :-4] 

                # A. 获取 Teacher 的“真理”动作
                with torch.no_grad():
                    teacher_action = teachers(teacher_obs) # (N, 4)
                
                # B. 获取 Student 的动作
                # 前向传播 Student (用切片后的 obs)
                student_action, next_hidden = student(student_obs, hidden)
                
                # C. 决定环境执行动作
                if use_teacher_control:
                    exec_action = teacher_action
                    hidden = next_hidden.detach() # Warmup 阶段不传递梯度，类似 Behavior Cloning
                else:
                    exec_action = student_action
                    hidden = next_hidden # On-policy 阶段保留梯度流
                
                # D. 环境步进
                next_obs_dict, rewards, dones, timeouts = env.step(exec_action)
                next_obs = next_obs_dict['policy']
                
                # E. 处理 Reset 的隐状态
                if torch.any(dones):
                    zero_hidden = student.init_hidden(args.num_teachers, device)
                    hidden = torch.where(dones.unsqueeze(0).unsqueeze(-1), zero_hidden, hidden)

                # F. 收集数据 (只收集 Student 能看到的数据)
                traj_student_obs.append(student_obs)
                traj_teacher_actions.append(teacher_action)
                
                obs = next_obs
                
                # 记录 Pos Error (前3维)
                total_pos_error += torch.norm(obs[:, :3], dim=1).mean().item()

            # === 2. Update Student (Backpropagation) ===
            # 堆叠轨迹: (Seq, Batch, Dim) -> (Batch, Seq, Dim)
            batch_student_obs = torch.stack(traj_student_obs, dim=1) # (N, 500, 22)
            batch_targets = torch.stack(traj_teacher_actions, dim=1).detach() # (N, 500, 4)
            
            # 重新进行一次完整序列的前向传播以计算 Loss
            train_hidden = student.init_hidden(args.num_teachers, device)
            
            # Forward 整个序列 (传入的是 22 维的 obs)
            pred_actions, _ = student(batch_student_obs, train_hidden)
            
            # 计算 Loss (MSE)
            loss = F.mse_loss(pred_actions, batch_targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            
            # === 3. Logging ===
            avg_pos_error = total_pos_error / args.steps_per_epoch
            wandb.log({
                "epoch": epoch,
                "loss": loss.item(),
                "pos_error_norm": avg_pos_error,
                "mode": "Teacher_Control" if use_teacher_control else "Student_Control"
            })
            
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f} | Err: {avg_pos_error:.4f} | {'[Warmup]' if use_teacher_control else '[Student]'}")
            
            # Save Checkpoint
            if epoch % 50 == 0:
                save_path = f"student_ckpt_{epoch:04d}.pt"
                torch.save(student.state_dict(), save_path)
    except Exception as e:
        import traceback
        print("\n\n!!! CRITICAL ERROR DURING TRAINING !!!")
        traceback.print_exc()
        print("!!! ------------------------------ !!!\n\n")
        raise e  # 重新抛出，以便看到原始错误码

    finally:
        # 确保正确关闭环境和仿真器
        print("[Info] Closing environment...")
        env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()