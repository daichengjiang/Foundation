import torch
import pandas as pd
import numpy as np
from foundation.utils.simple_controller import SimpleQuadrotorController
from foundation.tasks.point_ctrl.quad_point_ctrl_env_single_dense import QuadcopterEnv  # 导入你提供的环境类

class DistillationQuadcopterEnv(QuadcopterEnv):
    def __init__(self, cfg, dynamics_csv_path, num_teachers, **kwargs):
        self.dynamics_df = pd.read_csv(dynamics_csv_path)
        self.target_num_teachers = num_teachers
        
        # 确保 CSV 里的行数够
        if len(self.dynamics_df) < num_teachers:
            raise ValueError(f"CSV has {len(self.dynamics_df)} rows, but requested {num_teachers} teachers.")
            
        # 初始化父类
        super().__init__(cfg, **kwargs)
        
        # 覆盖 Controller，因为父类初始化时用的是单一配置
        self._override_dynamics()

    def _override_dynamics(self):
        """根据 CSV 里的参数，为每个环境覆盖特定的动力学参数"""
        device = self.device
        n_envs = self.num_envs
        
        # 准备张量
        masses = torch.zeros(n_envs, device=device)
        arm_lengths = torch.zeros(n_envs, device=device)
        inertias = torch.zeros(n_envs, 3, device=device)
        twrs = torch.zeros(n_envs, device=device)
        motor_taus = torch.zeros(n_envs, device=device)
        
        # 遍历 CSV 填充 (假设环境 i 对应 Teacher i)
        # 如果环境数 > Teacher 数 (例如并行跑多次)，则取模
        for i in range(n_envs):
            teacher_idx = i % self.target_num_teachers
            row = self.dynamics_df.iloc[teacher_idx]
            
            masses[i] = row['mass']
            arm_lengths[i] = row['arm_length']
            inertias[i, 0] = row['Ixx']
            inertias[i, 1] = row['Iyy']
            inertias[i, 2] = row['Izz']
            twrs[i] = row['twr']
            motor_taus[i] = row['motor_tau']
            
        # 1. 更新环境属性
        self.cfg.dynamics.mass = -1 # 标记为已被覆盖
        
        # 2. 重新初始化 Controller (Batched Controller)
        self._controller = SimpleQuadrotorController(
            num_envs=n_envs,
            device=device,
            mass=masses,
            arm_length=arm_lengths,
            inertia=inertias,
            thrust_to_weight=twrs
        )
        # 3. 更新电机时间常数
        self.motor_tau = motor_taus
        self.dt = self.cfg.sim.dt
        self.motor_alpha = self.dt / (self.dt + self.motor_tau).unsqueeze(1)
        
        print(f"[DistillationEnv] Successfully overrode dynamics for {n_envs} envs from CSV.")

    def _reset_idx(self, env_ids: torch.Tensor | None):
        # 调用父类 reset (处理位置重置等)
        super()._reset_idx(env_ids)
        
        # RAPTOR 特定：蒸馏时我们通常不想要太强的随机初始化（除了位置），
        # 因为我们要学的是系统辨识。
        # 这里保留父类的 reset 逻辑即可，因为父类的 train 模式已经包含了位置随机化。
        # 关键是 _override_dynamics 已经保证了 env_i 永远是 物理参数_i