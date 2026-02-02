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

        param_list = [3.0309736728668213,0.6789804697036743,0.1660696119070053,0.3423488438129425,0.06988178938627243,0.16802886128425598]  # Default gains
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
        
        r0 = torch.ones(self.num_envs, 4, device=self.device)   # Thrust
        r1 = torch.stack([d, -d, -d, d], dim=1)                 # Roll
        r2 = torch.stack([-d, -d, d, d], dim=1)                 # Pitch
        r3 = torch.stack([k, -k, k, -k], dim=1)                 # Yaw (Scaled by Thrust)

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

    def compute_target_speeds(self, cur_pos, cur_vel, cur_quat, cur_ang_vel, des_pos, des_vel, des_acc_ff, cur_motor_speed):
        """
        流程：PositionController -> AttitudeController -> Mixer -> Speed
        对应文件：positioncontroller.py, attitudecontroller.py, mixer.py
        """
        # ============================================
        # 1. Position Controller (positioncontroller.py)
        # ============================================
        # get_acceleration_command
        kp = self.wn**2
        kd = 2 * self.wn * self.zeta

        kp_vec = kp.unsqueeze(-1)
        kd_vec = kd.unsqueeze(-1)

        # cmd_acc = Kp*e + Kd*e_dot + acc_ff
        acc_cmd = (des_pos - cur_pos) * kp_vec + (des_vel - cur_vel) * kd_vec + des_acc_ff
        
        # ============================================
        # 2. Attitude Controller (attitudecontroller.py)
        # ============================================
        # 2.1 加上重力得到期望推力向量
        g = torch.zeros_like(acc_cmd); g[:, 2] = self.gravity
        thrust_vec = acc_cmd + g
        
        # 2.2 计算期望姿态 (get_angular_acceleration Step 1.1)
        thrust_norm = torch.norm(thrust_vec, dim=1, keepdim=True)
        des_thrust_dir = thrust_vec / (thrust_norm + 1e-6)
        
        e3 = torch.zeros_like(des_thrust_dir); e3[:, 2] = 1.0
        
        # cross product & dot product
        rot_ax = torch.cross(e3, des_thrust_dir, dim=1)
        dot = torch.sum(e3 * des_thrust_dir, dim=1, keepdim=True)
        angle = torch.acos(torch.clamp(dot, -1.0, 1.0))
        
        # 构建四元数
        ax_norm = torch.norm(rot_ax, dim=1, keepdim=True)
        mask = (ax_norm > 1e-6).squeeze()
        des_att = torch.zeros((self.num_envs, 4), device=self.device)
        des_att[:, 0] = 1.0 # Identity
        if mask.any():
            # des_att[mask] = quat_from_angle_axis(angle[mask], rot_ax[mask]/ax_norm[mask])
            
            masked_angle = angle[mask]
            masked_rot_ax = rot_ax[mask]
            masked_ax_norm = ax_norm[mask]

            # 2. 手动应用公式 q = [cos(theta/2), axis * sin(theta/2)]
            theta = masked_angle / 2.0
            w = torch.cos(theta)
            
            axis_normalized = masked_rot_ax / masked_ax_norm
            xyz = axis_normalized * torch.sin(theta)
            
            # 3. 拼接 (此时 w 和 xyz 的维度都是 [M, 1] 和 [M, 3]，拼接不会出错)
            q_val = torch.cat([w, xyz], dim=-1)
            
            # 4. 归一化防止漂移
            q_norm = torch.norm(q_val, dim=-1, keepdim=True)
            des_att[mask] = q_val / (q_norm + 1e-6)
            
        # 2.3 计算角速度误差 (Step 1.2)
        # desRotVec = (desAtt * curAtt.inv()).to_rotation_vector()
        q_err = quat_mul(des_att, quat_inv(cur_quat))
        # Small angle approximation for rotation vector: 2 * v * sign(w)
        q_v = q_err[:, 1:]; q_w = q_err[:, 0:1]
        rot_vec_err = 2.0 * torch.sign(q_w) * q_v 
        
        des_ang_vel = torch.zeros_like(cur_ang_vel)
        des_ang_vel[:, 0] = rot_vec_err[:, 0] / self.tc_ang_rp
        des_ang_vel[:, 1] = rot_vec_err[:, 1] / self.tc_ang_rp
        des_ang_vel[:, 2] = rot_vec_err[:, 2] / self.tc_ang_y
        
        # 2.4 计算角加速度 (Step 2.1)
        des_ang_acc = des_ang_vel - cur_ang_vel
        des_ang_acc[:, 0] /= self.tc_rate_rp
        des_ang_acc[:, 1] /= self.tc_rate_rp
        des_ang_acc[:, 2] /= self.tc_rate_y
        
        # ============================================
        # 3. Mixer (mixer.py)
        # ============================================
        # get_motor_force_cmd
        # F_tot = mass * thrust_norm
        f_total = self.mass * thrust_norm.squeeze()
        # f_total_vec = torch.zeros((self.num_envs, 3), device=self.device)
        # f_total_vec[:, 2] = f_total
        # Moments = Inertia * AngAcc
        moments = self.inertia * des_ang_acc
        
        # [F_tot, Mx, My, Mz]
        u = torch.cat([f_total.unsqueeze(1), moments], dim=1)
        
        # motor_forces = Inv(M) * u
        motor_forces = torch.bmm(self.mat_inv, u.unsqueeze(-1)).squeeze(-1)
        motor_forces = torch.clamp(motor_forces, min=0.0)
        
        # ============================================
        # 4. Command Conversion (motor.py)
        # ============================================
        # line 46: speedCommand = sqrt(cmd / speedSqrToThrust)

        max_thrust_motor = self.thrust_to_weight * self.mass * self.gravity / 4.0
        target = torch.sqrt(motor_forces / (max_thrust_motor.unsqueeze(-1)))
        target = torch.clamp(target, 0.0, 1.0)
        
        alpha = torch.where(target > cur_motor_speed, self.motor_alpha_up, self.motor_alpha_down)
        motor_speeds_cmd = cur_motor_speed + (target - cur_motor_speed) / alpha
        motor_speeds_cmd = torch.clamp(motor_speeds_cmd, 0.0, 1.0)

        # return f_total_vec, moments
        return motor_speeds_cmd


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