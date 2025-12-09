"""
RAPTOR Dynamics Layer (PyTorch Implementation)

This module replaces the traditional PID controller for RL training.
It acts as the system dynamics layer that converts motor commands (RPMs or normalized actions)
into physical forces and torques applied to the rigid body.

Key Features:
1. Supports Domain Randomization (Mass, Arm Length, Thrust-to-Weight Ratio).
2. Automatically scales thrust curves based on TWR and Mass.
3. Automatically updates allocation matrix based on Arm Length.

Author: Adapted for RAPTOR implementation
"""

import torch
import math

class SimpleQuadrotorController:
    """
    Dynamics Layer for Quadrotors.
    Maps Motor Velocities -> Body Wrench (Force & Torque).
    """

    def __init__(self,
                 num_envs: int,
                 device: torch.device,
                 mass: torch.Tensor,          # Shape: [num_envs]
                 arm_length: torch.Tensor,    # Shape: [num_envs]
                 inertia: torch.Tensor,       # Shape: [num_envs, 3] (Optional for wrench calc, but good for reference)
                 thrust_to_weight: torch.Tensor, # Shape: [num_envs] (Crucial for RAPTOR)
                 gravity: float = 9.81):
        
        # ================= [DEBUG START: 检查控制器接收到的参数] =================
        print(f"\n{'='*20} [DEBUG: SimpleController Init] {'='*20}")
        print(f"Controller Device: {device}")
        print(f"Received Mass[0] (Expected random): {mass[0].item():.6f}")
        print(f"Received TWR[0]  (Expected random): {thrust_to_weight[0].item():.6f}")
        print(f"Received Arm[0]  (Expected random): {arm_length[0].item():.6f}")
        
        # 简单检查是否是 Crazyflie 的默认值 (0.028 / 2.25)
        if abs(mass[0].item() - 0.0282) < 1e-5:
            print("!!! WARNING: Mass looks like Default Crazyflie Mass! Override might have failed!")
        else:
            print(">>> CHECK PASS: Mass is NOT default.")
        print(f"{'='*60}\n")
        # ================= [DEBUG END] =================

        self.num_envs = num_envs
        self.device = device
        self.g = gravity

        # --- 1. Store Physical Parameters ---
        self.mass_ = mass.to(device)
        self.arm_l_ = arm_length.to(device)
        self.inertia_ = inertia.to(device)
        self.thrust_to_weight = thrust_to_weight.to(device)        

        # --- 2. Motor Limits (Baseline: Crazyflie 2.1) ---
        # Used for normalization/denormalization
        rpm_max, rpm_min = 24000.0, 1200.0
        # Convert to rad/s
        self.motor_omega_max_ = torch.full((num_envs,), rpm_max * math.pi / 30, device=device)
        self.motor_omega_min_ = torch.full((num_envs,), rpm_min * math.pi / 30, device=device)

        # --- 3. Thrust Curve Coefficients ---
        # Baseline formula: Thrust = a*w^2 + b*w + c
        # Coefficients from Crazyflie system identification
        base_coeffs = torch.tensor([9.96063125e-08, -2.55003087e-05, 5.84422691e-03], 
                                   device=device).expand(num_envs, 3)
        
        # --- 4. RAPTOR Scaling Logic: Adjust Coefficients based on Mass & TWR ---
        # If we just change the mass but keep the motor coefficients constant,
        # a heavy drone will never take off. We must scale the motors' power.
        
        # Calculate max thrust of the BASELINE drone (single motor)
        w_max = self.motor_omega_max_
        base_thrust_max = (base_coeffs[:, 0] * w_max**2 + 
                           base_coeffs[:, 1] * w_max + 
                           base_coeffs[:, 2])
        
        # Calculate TARGET max thrust per motor based on sampled Mass and TWR
        # F_total_max = Mass * Gravity * TWR
        # F_motor_max = F_total_max / 4
        target_thrust_max = (self.mass_ * self.g * thrust_to_weight) / 4.0
        
        # Calculate scaling factor
        # We assume the shape of the thrust curve remains similar, just scaled in magnitude
        thrust_scale = target_thrust_max / (base_thrust_max + 1e-8)
        
        # Apply scale to coefficients (a, b, c)
        self.thrust_map_ = base_coeffs * thrust_scale.unsqueeze(1)
        
        # --- 5. Torque Coefficients (Kappa) ---
        # Drag torque ratio (Yaw Moment / Thrust)
        # We keep this ratio constant, meaning yaw torque scales linearly with thrust scaling
        self.kappa_ = torch.full((num_envs,), 0.005964552, device=device)

        # --- 6. Compute Allocation Matrix based on Geometry ---
        self.alloc_matrix_ = self._compute_allocation_matrix()

        print(f"[Dynamics] Initialized {num_envs} envs.")
        print(f"[Dynamics] Mass Range: [{self.mass_.min():.3f}, {self.mass_.max():.3f}] kg")
        print(f"[Dynamics] Arm Range:  [{self.arm_l_.min():.3f}, {self.arm_l_.max():.3f}] m")
        print(f"[Dynamics] TWR Range:  [{thrust_to_weight.min():.2f}, {thrust_to_weight.max():.2f}]")

        # [建议] 在最后重新计算一次参数，确保依赖项（如 thrust_map_）被更新
        self.update_dependent_params()

    # [新增] 用于更新依赖参数的方法
    def update_dependent_params(self):
        # 当外部修改了 self.mass_ 或 self.thrust_to_weight_ 后，必须重新计算推力曲线！
        # 否则只改 mass 不改 thrust_map，推力依然是错的
        
        # 1. 重新计算最大推力
        base_coeffs = torch.tensor([9.96063125e-08, -2.55003087e-05, 5.84422691e-03], 
                                   device=self.device).expand(self.num_envs, 3)
        w_max = self.motor_omega_max_
        base_thrust_max = (base_coeffs[:, 0] * w_max**2 + 
                           base_coeffs[:, 1] * w_max + 
                           base_coeffs[:, 2])
        
        # 使用当前的 mass_ 和 thrust_to_weight_
        # 注意：这里需要确保你已经把传入的 thrust_to_weight 存为了 self.thrust_to_weight_
        # 如果你原代码没存，请在 init 里加上 self.thrust_to_weight_ = thrust_to_weight
        if hasattr(self, 'thrust_to_weight_'):
            target_thrust_max = (self.mass_ * self.g * self.thrust_to_weight_) / 4.0
            thrust_scale = target_thrust_max / (base_thrust_max + 1e-8)
            self.thrust_map_ = base_coeffs * thrust_scale.unsqueeze(1)
            
            print(f"[Controller UPDATE] Recalculated Thrust Map for Mass={self.mass_[0]:.4f}, TWR={self.thrust_to_weight_[0]:.2f}")
        else:
            print("[Controller UPDATE] Warning: thrust_to_weight_ not found in controller.")

    def _compute_allocation_matrix(self):
        """
        Computes the mixing matrix that maps [M1, M2, M3, M4] thrusts to [Thrust, Roll, Pitch, Yaw].
        Assumes "X" configuration.
        """
        # Distance from center to motor along X/Y axes
        # d = L * sin(45) = L * 0.707
        d = self.arm_l_ * 0.70710678 
        k = self.kappa_

        # Construction of the allocation matrix (shape: [num_envs, 4, 4])
        # We define the relationship: Wrench = Matrix @ Motor_Thrusts
        # Row 0: Total Thrust (Z)
        # Row 1: Roll Torque  (X)
        # Row 2: Pitch Torque (Y)
        # Row 3: Yaw Torque   (Z)
        
        # Standard X-Config signs (Assuming motors: 0:FR, 1:BR, 2:BL, 3:FL)
        # Note: Signs must match your simulation engine's motor placement definition.
        # Based on your previous code logic:
        # Roll:  d * [1, -1, -1, 1]
        # Pitch: d * [-1, -1, 1, 1]
        # Yaw:   k * [1, -1, 1, -1]
        
        r0 = torch.ones(self.num_envs, 4, device=self.device)   # Thrust
        r1 = torch.stack([d, -d, -d, d], dim=1)                 # Roll
        r2 = torch.stack([-d, -d, d, d], dim=1)                 # Pitch
        r3 = torch.stack([k, -k, k, -k], dim=1)                 # Yaw

        mat = torch.stack([r0, r1, r2, r3], dim=1)
        return mat

    def motor_speeds_to_wrench(self, motor_speeds: torch.Tensor, normalized: bool = False) -> tuple:
        """
        Convert motor angular velocities to body-frame force and torque.
        
        Args:
            motor_speeds: (num_envs, 4) Motor speeds. 
                          If normalized=True, range should be [0, 1] (or RL output).
                          If normalized=False, range is rad/s.
            normalized: Whether inputs are normalized [0,1].
        
        Returns:
            force: (num_envs, 3) Force vector [0, 0, Fz] in body frame (Newtons).
            torque: (num_envs, 3) Torque vector [Tx, Ty, Tz] in body frame (Nm).
        """
        # 1. Denormalize if necessary (RL usually outputs [-1, 1] or [0, 1])
        if normalized:
            # Assuming input is [0, 1]. If your RL outputs [-1, 1], transform it first: (act + 1) / 2
            omega_min = self.motor_omega_min_.unsqueeze(1)
            omega_max = self.motor_omega_max_.unsqueeze(1)
            motor_speeds = omega_min + motor_speeds * (omega_max - omega_min)
        
        # 2. Calculate Thrust per Motor (N)
        # Formula: T = a*w^2 + b*w + c
        # self.thrust_map_ has shape [num_envs, 3] (a, b, c)
        a = self.thrust_map_[:, 0].unsqueeze(1)
        b = self.thrust_map_[:, 1].unsqueeze(1)
        c = self.thrust_map_[:, 2].unsqueeze(1)
        
        motor_thrusts = a * (motor_speeds ** 2) + b * motor_speeds + c
        
        # 3. Allocation (Mixing)
        # Wrench = Matrix @ Thrusts
        # Matrix: [num_envs, 4, 4], Thrusts: [num_envs, 4, 1]
        wrench = torch.bmm(self.alloc_matrix_, motor_thrusts.unsqueeze(-1)).squeeze(-1)
        
        # 4. Extract Force and Torque
        # Wrench format: [Total_Thrust, Roll_Torque, Pitch_Torque, Yaw_Torque]
        
        # Construct Force Vector [0, 0, Z]
        force = torch.zeros(self.num_envs, 3, device=self.device)
        force[:, 2] = wrench[:, 0]
        
        # Construct Torque Vector [X, Y, Z]
        torque = wrench[:, 1:4]
        
        # Optional: Debug info (can be removed for speed)
        # info = {'motor_thrusts': motor_thrusts} 
        
        return force, torque, None

    @property
    def dynamics(self):
        """Property to access internal limits, used by some env wrappers."""
        # Simple shim to make it compatible if env calls controller.dynamics.motor_omega_min_
        return self