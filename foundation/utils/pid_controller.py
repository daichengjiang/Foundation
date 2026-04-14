import torch
from isaaclab.utils.math import quat_inv, quat_mul, quat_from_angle_axis

class PaperPhysControllerTensor:
    """
    基于用户上传的 positioncontroller.py, attitudecontroller.py, mixer.py, motor.py, vehicle.py
    重构的 PyTorch 向量化控制器。
    """
    def __init__(self,
                 num_envs: int,
                 device: torch.device,
                 mass: torch.Tensor,              # Shape: [num_envs]
                 arm_length: torch.Tensor,        # Shape: [num_envs]
                 inertia: torch.Tensor,           # Shape: [num_envs, 3]
                 thrust_to_weight: torch.Tensor,  # Shape: [num_envs]
                 kappa: torch.Tensor, # Shape: [num_envs] (Optional, aka kappa/c_m)
                 motor_alpha_up: torch.Tensor, # Shape: [num_envs]
                 motor_alpha_down: torch.Tensor, # Shape: [num_envs]
                 gravity: float = 9.81):
        
        self.num_envs = num_envs
        self.device = device
        self.gravity = gravity
        self.mass = mass.to(device)
        self.inertia = inertia.to(device) # [N, 3] 对角阵
        self.arm_length = arm_length.to(device)
        self.kappa = kappa.to(device)
        self.thrust_to_weight = thrust_to_weight.to(device)
        self.motor_alpha_up = motor_alpha_up.to(device)
        self.motor_alpha_down = motor_alpha_down.to(device)
        
        # 质量 < 200g (0.2kg) -> wn = 6.0
        # 质量 >= 200g (0.2kg) -> wn = 2.0
        # threshold_mass = 0.2 # 200g
        # wn_light = 6.0
        # wn_heavy = 2.0

        # # --- 2. 控制器增益 (来自 positioncontroller.py / attitudecontroller.py) ---
        # self.wn = torch.where(
        #         self.mass < threshold_mass,
        #         torch.tensor(wn_light, device=device),
        #         torch.tensor(wn_heavy, device=device)
        # )
        
        # self.zeta = torch.full((num_envs,), 0.7, device=device)
        # self.tc_ang_rp = torch.full((num_envs,), 0.08, device=device)
        # self.tc_ang_y = torch.full((num_envs,), 0.40, device=device)
        # self.tc_rate_rp = torch.full((num_envs,), 0.04, device=device)
        # self.tc_rate_y = torch.full((num_envs,), 0.20, device=device)

        param_list = [2.483903169631958,0.7526758909225464,0.20132757723331451,0.3580484390258789,0.03857744485139847,0.1552553027868271]  # Default gains
        self.wn = torch.full((num_envs,), param_list[0], device=device)
        self.zeta = torch.full((num_envs,), param_list[1], device=device)
        self.tc_ang_rp = torch.full((num_envs,), param_list[2], device=device)
        self.tc_ang_y = torch.full((num_envs,), param_list[3], device=device)
        self.tc_rate_rp = torch.full((num_envs,), param_list[4], device=device)
        self.tc_rate_y = torch.full((num_envs,), param_list[5], device=device)

        # --- 3. 构建混控矩阵 (源自 mixer.py) ---
        # 原 mixer.py 代码:
        # M = [[1, 1, 1, 1], [-l, -l, l, l], [-l, l, l, -l], [-k, k, -k, k]]
        # 这里的 l 直接用于矩阵，意味着 mixer 假设 l 是力臂在轴上的投影，或者坐标系定义如此。
        # 我们严格复刻 mixer.py 的矩阵构造。
        
        d = self.arm_length * 0.70710678 
        k = self.kappa  # Moment coefficient (c_m)
        
        # 原来
        # r0 = torch.ones(self.num_envs, 4, device=self.device)   # Thrust
        # r1 = torch.stack([d, -d, -d, d], dim=1)                 # Roll
        # r2 = torch.stack([-d, -d, d, d], dim=1)                 # Pitch
        # r3 = torch.stack([k, -k, k, -k], dim=1)                 # Yaw (Scaled by Thrust)
        # 实物相同
        # r0 = torch.ones(self.num_envs, 4, device=self.device)   # 推力 (Thrust)
        # r1 = torch.stack([-d,  d,  d, -d], dim=1)               # 滚转 (Roll: 左正右负)
        # r2 = torch.stack([ d, -d,  d, -d], dim=1)               # 俯仰 (Pitch: 前正后负)
        # r3 = torch.stack([-k, -k,  k,  k], dim=1)               # 偏航 (Yaw: CW正, CCW负)
        # 第三版
        r0 = torch.ones(self.num_envs, 4, device=self.device)   # Thrust (恒为正)
        r1 = torch.stack([-d,  d,  d, -d], dim=1)                # Roll: 左正右负 (0右, 1左, 2左, 3右)
        r2 = torch.stack([-d,  d, -d,  d], dim=1)                # Pitch: 后正前负 (0前, 1后, 2前, 3后)
        r3 = torch.stack([-k, -k,  k,  k], dim=1)                # Yaw: 保持对角线同号 (0和1反号, 2和3同号)

        mat = torch.stack([r0, r1, r2, r3], dim=1)
        
        self.mat = mat
        self.mat_inv = torch.linalg.inv(mat)

    # 在 PaperPhysControllerTensor 类中添加
    def update_gains(self, wn, zeta, tc_ang_rp, tc_ang_y, tc_rate_rp, tc_rate_y):
        self.wn = wn.to(self.device)
        self.zeta = zeta.to(self.device)
        self.tc_ang_rp = tc_ang_rp.to(self.device)
        self.tc_ang_y = tc_ang_y.to(self.device)
        self.tc_rate_rp = tc_rate_rp.to(self.device)
        self.tc_rate_y = tc_rate_y.to(self.device)

    def _matrix_to_quat(self, res_R):
        """
        针对 PyTorch 向量化实现的矩阵转四元数 (Shepperd's Algorithm 简化版)
        """
        m00, m11, m22 = res_R[:, 0, 0], res_R[:, 1, 1], res_R[:, 2, 2]
        trace = m00 + m11 + m22

        # 简单的实现，如果 trace > 0 比较安全
        # 在高性能控制器中，通常会根据 trace 分支处理，这里提供一个基础版本
        w = torch.sqrt(torch.clamp(1.0 + trace, min=1e-6)) / 2.0
        x = (res_R[:, 2, 1] - res_R[:, 1, 2]) / (4.0 * w + 1e-6)
        y = (res_R[:, 0, 2] - res_R[:, 2, 0]) / (4.0 * w + 1e-6)
        z = (res_R[:, 1, 0] - res_R[:, 0, 1]) / (4.0 * w + 1e-6)
        
        q = torch.stack([w, x, y, z], dim=-1)
        return q / torch.norm(q, dim=1, keepdim=True)
    
    def compute_target_speeds(self, cur_pos, cur_vel, cur_quat, cur_ang_vel, des_pos, des_vel, des_acc_ff, des_yaw, cur_motor_speed):
        # ============================================
        # 1. Position Controller (位置环计算加速度指令)
        # ============================================
        kp_vec = (self.wn**2).unsqueeze(-1)
        kd_vec = (2 * self.wn * self.zeta).unsqueeze(-1)
        acc_cmd = (des_pos - cur_pos) * kp_vec + (des_vel - cur_vel) * kd_vec + des_acc_ff
        
        # ============================================
        # 2. Attitude Controller (改进方案 A: 矩阵构造法)
        # ============================================
        # 2.1 计算推力矢量和模长 (thrust_norm 这里被提取出来)
        g = torch.zeros_like(acc_cmd); g[:, 2] = self.gravity
        thrust_vec = acc_cmd + g
        
        # 【关键修正】：这里保存 thrust_norm 供后续 Mixer 使用
        thrust_norm = torch.norm(thrust_vec, dim=1, keepdim=True) 
        z_body = thrust_vec / (thrust_norm + 1e-6)
        
        # 2.2 根据 des_yaw 确定参考 X 轴 (世界系水平面)
        x_world_ref = torch.stack([
            torch.cos(des_yaw),
            torch.sin(des_yaw),
            torch.zeros_like(des_yaw)
        ], dim=-1)
        
        # 2.3 叉乘构建机体坐标系 [x_body, y_body, z_body]
        # y 轴垂直于推力方向和期望偏航方向
        y_body = torch.cross(z_body, x_world_ref, dim=1)
        y_body = y_body / (torch.norm(y_body, dim=1, keepdim=True) + 1e-6)
        
        # x 轴重新正交化，确保 Z 轴优先级最高
        x_body = torch.cross(y_body, z_body, dim=1)
        
        # 2.4 构建旋转矩阵并转为四元数
        # res_R shape: [N, 3, 3]
        res_R = torch.stack([x_body, y_body, z_body], dim=-1)
        des_att = self._matrix_to_quat(res_R) # 见下方辅助函数
                
        # 2.5 计算角速度误差 (Rate Loop)
        q_err = quat_mul(des_att, quat_inv(cur_quat))
        q_v = q_err[:, 1:]; q_w = q_err[:, 0:1]
        rot_vec_err = 2.0 * torch.sign(q_w) * q_v 
        
        des_ang_vel = torch.zeros_like(cur_ang_vel)
        des_ang_vel[:, 0] = rot_vec_err[:, 0] / self.tc_ang_rp
        des_ang_vel[:, 1] = rot_vec_err[:, 1] / self.tc_ang_rp
        des_ang_vel[:, 2] = rot_vec_err[:, 2] / self.tc_ang_y
        
        des_ang_acc = (des_ang_vel - cur_ang_vel)
        des_ang_acc[:, 0] /= self.tc_rate_rp
        des_ang_acc[:, 1] /= self.tc_rate_rp
        des_ang_acc[:, 2] /= self.tc_rate_y

        # ============================================
        # 3. Mixer (动力分配)
        # ============================================
        # 使用刚才保存的 thrust_norm 计算总力
        f_total = self.mass * thrust_norm.squeeze() 
        moments = self.inertia * des_ang_acc
        
        u = torch.cat([f_total.unsqueeze(1), moments], dim=1)
        motor_forces = torch.bmm(self.mat_inv, u.unsqueeze(-1)).squeeze(-1)
        motor_forces = torch.clamp(motor_forces, min=0.0)
        
        # ============================================
        # 4. Motor Command (映射到 0.0 - 1.0)
        # ============================================
        max_thrust_motor = self.thrust_to_weight * self.mass * self.gravity / 4.0
        target = torch.sqrt(motor_forces / (max_thrust_motor.unsqueeze(-1)))
        target = torch.clamp(target, 0.0, 1.0)
        
        alpha = torch.where(target > cur_motor_speed, self.motor_alpha_up, self.motor_alpha_down)
        motor_speeds_cmd = cur_motor_speed + (target - cur_motor_speed) / alpha
        return torch.clamp(motor_speeds_cmd, 0.0, 1.0)


    def motor_speeds_to_wrench(self, motor_actions: torch.Tensor) -> tuple:

        # Calculate Thrust per motor (Newtons)
        coeff = (self.thrust_to_weight * self.mass * self.gravity / 4.0).unsqueeze(-1)
        motor_thrusts = coeff * (motor_actions ** 2)
        
        # Mix to Wrench
        # wrench shape: [num_envs, 4]
        wrench = torch.bmm(self.mat, motor_thrusts.unsqueeze(-1)).squeeze(-1)
        
        # Extract Output
        force = torch.zeros(self.num_envs, 3, device=self.device)
        force[:, 2] = wrench[:, 0] # Z-force
        
        torque = wrench[:, 1:4]    # Torques
        
        return force, torque
    
    # def motor_speeds_to_wrench(self, motor_actions: torch.Tensor) -> tuple:
    #     # 【修改点】：模拟低电量，最大推力只有满电的 75%
    #     battery_degradation = 0.75 
        
    #     coeff = (self.thrust_to_weight * self.mass * self.gravity / 4.0).unsqueeze(-1)
    #     # 乘以衰减系数
    #     motor_thrusts = (coeff * battery_degradation) * (motor_actions ** 2)
        
    #     wrench = torch.bmm(self.mat, motor_thrusts.unsqueeze(-1)).squeeze(-1)
    #     force = torch.zeros(self.num_envs, 3, device=self.device)
    #     force[:, 2] = wrench[:, 0]
    #     torque = wrench[:, 1:4]
    #     return force, torque
    
    # def motor_speeds_to_wrench(self, motor_actions: torch.Tensor) -> tuple:
    #     coeff = (self.thrust_to_weight * self.mass * self.gravity / 4.0).unsqueeze(-1)
        
    #     # 【修改点】：不再是完美的平方，加入线性项，或者改变指数
    #     # 混合模型：40% 线性 + 60% 二次方
    #     # 注意：这里保证了 motor_actions=1 时，总系数依然是 1
    #     motor_thrusts = coeff * (0.4 * motor_actions + 0.6 * (motor_actions ** 2))
        
    #     # 或者试试这种：
    #     # motor_thrusts = coeff * (motor_actions ** 1.5)
        
    #     wrench = torch.bmm(self.mat, motor_thrusts.unsqueeze(-1)).squeeze(-1)
    #     force = torch.zeros(self.num_envs, 3, device=self.device)
    #     force[:, 2] = wrench[:, 0]
    #     torque = wrench[:, 1:4]
    #     return force, torque
    
    # def motor_speeds_to_wrench(self, motor_actions: torch.Tensor) -> tuple:
    #     coeff = (self.thrust_to_weight * self.mass * self.gravity / 4.0).unsqueeze(-1)
    #     motor_thrusts = coeff * (motor_actions ** 2)
        
    #     # 【修改点】：模拟 0 号电机（右前）受损，效率只剩 80%
    #     # 这里用张量操作避免报错，针对所有 env 的第 0 个电机打折
    #     damage_mask = torch.tensor([0.8, 1.0, 1.0, 1.0], device=self.device)
    #     motor_thrusts = motor_thrusts * damage_mask
        
    #     wrench = torch.bmm(self.mat, motor_thrusts.unsqueeze(-1)).squeeze(-1)
    #     force = torch.zeros(self.num_envs, 3, device=self.device)
    #     force[:, 2] = wrench[:, 0]
    #     torque = wrench[:, 1:4]
    #     return force, torque
    
    # def motor_speeds_to_wrench(self, motor_actions: torch.Tensor) -> tuple:
    #     coeff = (self.thrust_to_weight * self.mass * self.gravity / 4.0).unsqueeze(-1)
        
    #     # 【修改点】：加上死区，低于 0.1 的指令直接归零
    #     deadband_mask = (motor_actions > 0.1).float()
    #     real_actions = motor_actions * deadband_mask
        
    #     motor_thrusts = coeff * (real_actions ** 2)
        
    #     wrench = torch.bmm(self.mat, motor_thrusts.unsqueeze(-1)).squeeze(-1)
    #     force = torch.zeros(self.num_envs, 3, device=self.device)
    #     force[:, 2] = wrench[:, 0]
    #     torque = wrench[:, 1:4]
    #     return force, torque