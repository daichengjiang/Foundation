import torch
from isaaclab.utils.math import quat_inv, quat_mul, quat_from_angle_axis

class PaperPhysControllerTensor:
    """
    基于用户上传的 positioncontroller.py, attitudecontroller.py, mixer.py, motor.py, vehicle.py
    重构的 PyTorch 向量化控制器。
    """
    def __init__(self, num_envs, device, ctrl_params, phys_params):
        self.num_envs = num_envs
        self.device = device
        
        # --- 1. 物理参数提取 ---
        # 确保全部转为 Tensor 以支持并行和异构
        def as_tensor(val):
            if isinstance(val, torch.Tensor): return val.to(device)
            return torch.full((num_envs,), val, device=device)

        self.mass = as_tensor(phys_params['mass'])
        self.inertia = phys_params['inertia'] # [N, 3] 对角阵
        self.arm_length = as_tensor(phys_params['arm_length'])
        self.k_f = as_tensor(phys_params['mot_k_f']) # SpeedSqrToThrust
        self.k_m = as_tensor(phys_params['mot_k_m']) # SpeedSqrToTorque / DragTorque
        self.kappa = as_tensor(phys_params['kappa'])
        
        # 电机转子惯量 (来自 motor.py 的 inertia 参数，通常很小)
        # 如果你没有这个参数，通常取 1e-6 到 1e-5 之间
        self.rotor_inertia = as_tensor(phys_params.get('rotor_inertia', 0.0))

        # --- 2. 控制器增益 (来自 positioncontroller.py / attitudecontroller.py) ---
        self.wn = ctrl_params['pos_nat_freq']
        self.zeta = ctrl_params['pos_damping']
        self.tc_ang_rp = ctrl_params['tc_angle_rp']
        self.tc_ang_y = ctrl_params['tc_angle_y']
        self.tc_rate_rp = ctrl_params['tc_rate_rp']
        self.tc_rate_y = ctrl_params['tc_rate_y']

        # --- 3. 构建混控矩阵 (源自 mixer.py) ---
        # 原 mixer.py 代码:
        # M = [[1, 1, 1, 1], [-l, -l, l, l], [-l, l, l, -l], [-k, k, -k, k]]
        # 这里的 l 直接用于矩阵，意味着 mixer 假设 l 是力臂在轴上的投影，或者坐标系定义如此。
        # 我们严格复刻 mixer.py 的矩阵构造。
        
        l = self.arm_length * 0.70710678
        # k = self.k_m / self.k_f # ThrustToTorque ratio
        k = self.kappa
        
        # 构造 M [N, 4, 4]
        M = torch.zeros((num_envs, 4, 4), device=device)
        M[:, 0, :] = 1.0
        # Roll: [-l, -l, l, l]
        M[:, 1, 0] = -l; M[:, 1, 1] = -l; M[:, 1, 2] = l;  M[:, 1, 3] = l
        # Pitch: [-l, l, l, -l]
        M[:, 2, 0] = -l; M[:, 2, 1] = l;  M[:, 2, 2] = l;  M[:, 2, 3] = -l
        # Yaw: [-k, k, -k, k]
        M[:, 3, 0] = -k; M[:, 3, 1] = k;  M[:, 3, 2] = -k; M[:, 3, 3] = k
        
        self.mixer_mat_inv = torch.linalg.inv(M)

        # --- 4. 构建电机位置几何 (用于 vehicle.py 的物理计算) ---
        # vehicle.py 通过 totalTorque += pos.cross(thrust) 计算力矩
        # 我们需要定义 4 个电机的位置向量。
        # 假设标准 X 型布局，角度 45 度。
        # Motor 0: FR (+x, -y) ? 参照 mixer 的符号 [-l, -l] -> x负 y负?
        # 让我们根据 Mixer 的符号反推位置：
        # Mixer Roll (My) row is [-l, -l, l, l]. Torque_Roll ~ y_pos * F.
        # Mixer Pitch (Mx) row is [-l, l, l, -l]. Torque_Pitch ~ -x_pos * F.
        # 这取决于坐标系定义，这里我们构建符合 mixer 逻辑的几何位置
        
        # 简化处理：为了 strict compliance with vehicle.py，我们需要 "motorPosition"。
        # 既然 mixer 用了 l 作为系数，我们假设电机在 (l, 0), (0, l) 这种轴上？
        # 或者标准 X: (l/√2, l/√2). 
        # 鉴于 mixer.py 直接用 l，最稳妥的方式是直接使用 mixer 的逆过程，
        # 但既然你要 "process from vehicle.py"，我们需要显式位置。
        # 这里采用标准 X 型布局，并假设 l 是力臂。
        d = self.arm_length # 这里的 d 对应 mixer 中的 l
        
        # 构造 4 个电机的位置向量 [N, 4, 3]
        # 根据 mixer 符号推导：
        # Mot 0: Roll(-), Pitch(-) -> 右后?
        # Mot 1: Roll(-), Pitch(+) -> 右前?
        # Mot 2: Roll(+), Pitch(+) -> 左前?
        # Mot 3: Roll(+), Pitch(-) -> 左后?
        self.motor_pos = torch.zeros((num_envs, 4, 3), device=device)
        # 这是一个近似，确保 cross product 结果与 mixer 一致
        # My = F * x_arm. Mx = -F * y_arm.
        self.motor_pos[:, 0, :] = torch.stack([-d, -d, torch.zeros_like(d)], dim=1) 
        self.motor_pos[:, 1, :] = torch.stack([ d, -d, torch.zeros_like(d)], dim=1)
        self.motor_pos[:, 2, :] = torch.stack([ d,  d, torch.zeros_like(d)], dim=1)
        self.motor_pos[:, 3, :] = torch.stack([-d,  d, torch.zeros_like(d)], dim=1)

        # 电机旋转方向 (来自 mixer Yaw 行 [-k, k, -k, k])
        # -k 表示该电机产生负扭矩 -> 说明它是正向旋转(CCW)还是反向? 
        # 通常 CCW 产生正扭矩。这里 -k 意味着 Mot0 是 CW。
        self.motor_spin_dir = torch.tensor([-1.0, 1.0, -1.0, 1.0], device=device).expand(num_envs, 4)


    def compute_target_speeds(self, cur_pos, cur_vel, cur_quat, cur_ang_vel, des_pos, des_vel, des_acc_ff):
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
        # cmd_acc = Kp*e + Kd*e_dot + acc_ff
        acc_cmd = (des_pos - cur_pos) * kp + (des_vel - cur_vel) * kd + des_acc_ff
        
        # ============================================
        # 2. Attitude Controller (attitudecontroller.py)
        # ============================================
        # 2.1 加上重力得到期望推力向量
        gravity = torch.zeros_like(acc_cmd); gravity[:, 2] = 9.81
        thrust_vec = acc_cmd + gravity
        
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
            des_att[mask] = quat_from_angle_axis(angle[mask], rot_ax[mask]/ax_norm[mask])
            
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
        f_total_vec = torch.zeros((self.num_envs, 3), device=self.device)
        f_total_vec[:, 2] = f_total
        # Moments = Inertia * AngAcc
        moments = self.inertia * des_ang_acc
        
        # # [F_tot, Mx, My, Mz]
        # u = torch.cat([f_total.unsqueeze(1), moments], dim=1)
        
        # # motor_forces = Inv(M) * u
        # motor_forces = torch.bmm(self.mixer_mat_inv, u.unsqueeze(-1)).squeeze(-1)
        # motor_forces = torch.clamp(motor_forces, min=0.0)
        
        # # ============================================
        # # 4. Command Conversion (motor.py)
        # # ============================================
        # # line 46: speedCommand = sqrt(cmd / speedSqrToThrust)
        # motor_speeds_cmd = torch.sqrt(motor_forces / self.k_f.unsqueeze(1))
        
        return f_total_vec, moments
        # return motor_speeds_cmd


    def compute_actual_wrench(self, motor_speeds, dt):
        """
        流程：Motor Physics -> Vehicle Aggregation
        对应文件：motor.py, vehicle.py
        
        Args:
            motor_speeds: 当前真实的电机转速 [N, 4]
            dt: 时间步长 (用于计算电机角加速度产生的惯性力矩)
        Returns:
            force_b: 机体坐标系下的力 [N, 3]
            torque_b: 机体坐标系下的力矩 [N, 3]
        """
        # ============================================
        # 1. Motor Physics (motor.py)
        # ============================================
        # 假设 motor_speeds 已经是经过一阶滤波后的当前转速 (self._speed)
        
        # 1.1 Thrust (line 67: thrust = kf * speed^2 * axis)
        # 注意: vehicle.py 里 axis 是 Z 轴 (0,0,1)
        speeds_sq = motor_speeds ** 2
        thrusts_mag = self.k_f.unsqueeze(1) * speeds_sq # [N, 4]
        
        # 1.2 Aerodynamic Torque (line 70: -km * speed * |speed| * axis)
        # spin_dir: +1 or -1. 
        # Torque dir is opposite to spin dir.
        drag_torques_mag = -self.k_m.unsqueeze(1) * speeds_sq * self.motor_spin_dir # [N, 4]
        
        # 1.3 Inertial Torque (line 73: -angAcc * inertia * axis)
        # 这里需要电机角加速度。如果没有历史数据，通常忽略或假设稳态。
        # 为了完整性，如果 dt > 0 且有上一帧速度，可以计算。
        # 这里暂忽略瞬态项 (Sim 常用做法)，因为我们没有传入 old_speed。
        # 如果必须包含，需要在 env 里维护 old_speed。
        
        # ============================================
        # 2. Vehicle Aggregation (vehicle.py)
        # ============================================
        # run() loop
        
        total_force_b = torch.zeros((self.num_envs, 3), device=self.device)
        total_torque_b = torch.zeros((self.num_envs, 3), device=self.device)
        
        # 这里的计算必须向量化处理 4 个电机
        for i in range(4):
            # Force: 假设推力方向垂直向上 (0,0,1)
            f_i = torch.zeros((self.num_envs, 3), device=self.device)
            f_i[:, 2] = thrusts_mag[:, i]
            
            # Torque 1: Drag Torque (Z axis)
            t_drag_i = torch.zeros((self.num_envs, 3), device=self.device)
            t_drag_i[:, 2] = drag_torques_mag[:, i]
            
            # Torque 2: Position Cross Thrust (line 71: pos.cross(thrust))
            # pos: [N, 3], f_i: [N, 3]
            pos_i = self.motor_pos[:, i, :]
            t_arm_i = torch.cross(pos_i, f_i, dim=1)
            
            # 累加
            total_force_b += f_i
            total_torque_b += (t_drag_i + t_arm_i)
            
        # 3. Body Drag (vehicle.py line 62)
        # totalTorque += -omega.norm * coeff * omega
        # 这部分通常在 Physics 引擎里算，但 vehicle.py 显式加了。
        # 由于我们只输出 Wrench 给 Isaac Sim，Isaac Sim 可能会自己算阻力。
        # 如果你想完全接管动力学，应该加上这个。
        # 但通常作为 Controller，只输出 Actuator Wrench。
        # 这里只返回电机产生的 Wrench。
        
        return total_force_b, total_torque_b