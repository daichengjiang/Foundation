import torch
import torch.nn.functional as F
import time
import os
import pandas as pd
from collections import deque

from foundation.algo.teacher_ensemble import TeacherEnsemble
from foundation.algo.student_policy import RaptorStudent

class DistillRunner:
    def __init__(self, env, log_dir, device='cuda:0'):
        self.env = env
        self.device = device
        self.log_dir = log_dir
        
        # 1. 读取 Teacher 参数表
        self.dynamics_df = pd.read_csv("teacher_dynamics.csv")
        self.num_teachers = len(self.dynamics_df)
        print(f"Found {self.num_teachers} teachers in CSV.")
        
        # 2. 初始化 Teacher 集成
        # 指向保存 teacher_xxxx 文件夹的根目录
        teacher_ckpt_root = os.path.join(log_dir, "raptor_teachers") 
        self.teachers = TeacherEnsemble(teacher_ckpt_root, self.num_teachers, device)
        
        # 3. 初始化 Student
        # Obs Dim = 26, Act Dim = 4
        self.student = RaptorStudent(num_inputs=22, num_actions=4).to(device)
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=1e-3)
        
        # 4. 运行时变量
        self.num_envs = env.num_envs
        self.current_teacher_indices = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.hidden_state = self.student.init_hidden(self.num_envs, device)
        
        # 预先将 CSV 数据转为 Tensor 方便索引
        self.mass_list = torch.tensor(self.dynamics_df['mass'].values, device=device, dtype=torch.float32)
        self.arm_list = torch.tensor(self.dynamics_df['arm_length'].values, device=device, dtype=torch.float32)
        self.inertia_list = torch.tensor(self.dynamics_df[['Ixx', 'Iyy', 'Izz']].values, device=device, dtype=torch.float32)
        self.twr_list = torch.tensor(self.dynamics_df['twr'].values, device=device, dtype=torch.float32)
        self.tau_list = torch.tensor(self.dynamics_df['motor_tau'].values, device=device, dtype=torch.float32)

    def randomize_envs(self, env_ids):
        """
        当环境 Reset 时，为它们随机分配新的 Teacher ID，并更新物理参数
        """
        if len(env_ids) == 0: return
        
        # 1. 随机选择 Teacher ID
        new_ids = torch.randint(0, self.num_teachers, (len(env_ids),), device=self.device)
        self.current_teacher_indices[env_ids] = new_ids
        
        # 2. 获取对应的物理参数
        new_mass = self.mass_list[new_ids]
        new_arm = self.arm_list[new_ids]
        new_inertia = self.inertia_list[new_ids]
        new_twr = self.twr_list[new_ids]
        new_tau = self.tau_list[new_ids]
        
        # 3. 更新控制器的参数 (Direct modification of tensors in SimpleQuadrotorController)
        # 注意: 这里的 env.unwrapped 假设了 DirectRLEnv 的结构
        controller = self.env.unwrapped._controller
        
        controller.mass_[env_ids] = new_mass
        controller.arm_l_[env_ids] = new_arm
        controller.inertia_[env_ids] = new_inertia
        # 重新计算推力系数和分配矩阵
        # 这里需要调用我们在 simple_controller.py 里写的逻辑
        # 因为 simple_controller 初始化时才计算，所以我们需要手动更新部分逻辑
        # 或者最好的方法是给 simple_controller 加一个 update_params 方法
        # 这里为了演示，我们假设我们能直接调用 update_params
        controller.update_params(env_ids, new_mass, new_arm, new_inertia, new_twr)
        
        # 更新电机延迟 (tau)
        self.env.unwrapped.motor_tau = 0.0 # reset base
        # 注意：你的 Env 代码里 motor_tau 是 float 还是 tensor? 
        # 如果要支持每个环境不同，Env 里的 self.motor_tau 必须改成 tensor!
        # 下面是一个假设的更新
        # self.env.unwrapped.motor_tau_tensor[env_ids] = new_tau

    def learn(self, max_epochs=1000, steps_per_epoch=500):
        obs, _ = self.env.reset()
        self.randomize_envs(torch.arange(self.num_envs, device=self.device))
        
        for epoch in range(max_epochs):
            total_loss = 0
            
            for step in range(steps_per_epoch):
                # 1. 准备输入数据
                # Teacher (Oracle): 看到全状态 (26维: 含电机转速)
                # Student (Deploy): 只能看到机载传感器数据 (22维: 不含电机转速)
                
                # 假设 obs 的最后 4 维是 motor_speeds
                # obs shape: [Batch, 26]
                student_obs = obs[:, :-4]  # [Batch, 22] <--- 关键切片操作
                
                # 2. Student Forward (用 22 维输入)
                student_actions, self.hidden_state = self.student(student_obs, self.hidden_state)
                self.hidden_state = self.hidden_state.detach()
                
                # 3. Teacher Forward (用 26 维输入)
                with torch.no_grad():
                    teacher_actions = self.teachers(obs, self.current_teacher_indices)
                
                # 4. Loss Calculation
                loss = F.mse_loss(student_actions, teacher_actions)
                
                # 4. Update Student
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # 5. Step Environment (DAgger: Student drives)
                # 使用 Student 的动作推演环境
                obs, rewards, dones, timeouts, infos = self.env.step(student_actions)
                
                # 6. Handle Resets
                reset_env_ids = torch.where(dones)[0]
                if len(reset_env_ids) > 0:
                    self.randomize_envs(reset_env_ids)
                    # Reset hidden states for done envs
                    self.hidden_state[:, reset_env_ids, :] = 0.0
            
            print(f"Epoch {epoch}, Mean Loss: {total_loss / steps_per_epoch:.6f}")
            
            # Save Checkpoint
            if epoch % 10 == 0:
                torch.save(self.student.state_dict(), os.path.join(self.log_dir, f"student_epoch_{epoch}.pt"))