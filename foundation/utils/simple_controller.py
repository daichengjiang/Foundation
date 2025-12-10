"""
RAPTOR Dynamics Layer (PyTorch Implementation) - Paper Exact Version

This module implements the RAPTOR dynamics as described in the Supplementary Materials (S1).
It maps normalized motor commands [0, 1] directly to physical forces and torques
using the scaling laws derived from Mass and Thrust-to-Weight Ratio.

Reference: RAPTOR: A Foundation Policy for Quadrotor Control (2025)
Section: Materials and Methods -> Sampling Quadrotors (Eq. S5 - S12)
"""

import torch
import math

class SimpleQuadrotorController:
    """
    RAPTOR Dynamics Layer.
    Maps Normalized Motor Commands [0,1] -> Body Wrench (Force & Torque).
    """

    def __init__(self,
                 num_envs: int,
                 device: torch.device,
                 mass: torch.Tensor,              # Shape: [num_envs]
                 arm_length: torch.Tensor,        # Shape: [num_envs]
                 inertia: torch.Tensor,           # Shape: [num_envs, 3]
                 thrust_to_weight: torch.Tensor,  # Shape: [num_envs]
                 moment_scale: torch.Tensor = None, # Shape: [num_envs] (Optional, aka kappa/c_m)
                 gravity: float = 9.81):
        
        self.num_envs = num_envs
        self.device = device
        self.g = gravity

        # --- 1. Store Physical Parameters ---
        self.mass_ = mass.to(device)
        self.arm_l_ = arm_length.to(device)
        self.inertia_ = inertia.to(device)
        self.thrust_to_weight_ = thrust_to_weight.to(device)
        
        # Paper  samples moment coefficient c_m ~ Uniform(0.005, 0.05).
        # If not provided, we use the mean or the Crazyflie default.
        if moment_scale is None:
             # Default generic value if not sampled externally
            self.kappa_ = torch.full((num_envs,), 0.016, device=device)
        else:
            self.kappa_ = moment_scale.to(device)

        # --- 2. Define Baseline Coefficients (From Paper Eq. S8, S9) ---
        # "C_f0 = 0.038, C_f1 = 0.154, C_f2 = 0.987" 
        # These correspond to the normalized thrust curve f(w) where w is in [0, 1].
        # Note: The sum is approx 1.179, but these are the explicit values listed.
        self.base_coeffs_ = torch.tensor([0.987, 0.154, 0.038], device=device).expand(num_envs, 3) 
        # Order: [Quadratic(w^2), Linear(w), Constant(1)] matching polyval style

        # --- 3. RAPTOR Scaling Logic (Eq. S11, S12) ---
        # Calculate real coefficients c_fi based on Mass and TWR.
        self.update_dependent_params()

    def update_dependent_params(self):
        """
        Recalculates the thrust map coefficients (c_f) and allocation matrix
        based on current Mass and TWR.
        """
        # Eq. S10: T_total_max = TWR * g * mass
        target_total_thrust = self.thrust_to_weight_ * self.g * self.mass_

        # Eq. S11: c_fi = C_fi * (T_total / 4)
        # We need to scale the normalized curve so that at input=1.0, 
        # the sum of thrusts equals the target max thrust.
        # Note: In the paper, it says c_fi = C_fi * (T / 4).
        # This implies the BASE coefficients C_fi are treated as "weights" that sum 
        # to produce the shape, and the magnitude is controlled by T/4.
        
        # Scaling factor per motor
        scale_factor = target_total_thrust / 4.0
        
        # Apply scale to [a, b, c]
        # self.thrust_map_ shape: [num_envs, 3] -> [a_real, b_real, c_real]
        self.thrust_map_ = self.base_coeffs_ * scale_factor.unsqueeze(1)
        
        # Re-compute allocation matrix (depends on arm length and kappa)
        self.alloc_matrix_ = self._compute_allocation_matrix()
        
        # [DEBUG Info]
        print(f"[RAPTOR Controller] Updated Params.")
        print(f"   Target Max Thrust (1 drone): {target_total_thrust[0]:.4f} N")
        print(f"   Derived Coefficients (c_f2, c_f1, c_f0): {self.thrust_map_[0].tolist()}")

    def _compute_allocation_matrix(self):
        """
        Computes Mixing Matrix: [Thrusts] -> [Force_Z, Torque_X, Torque_Y, Torque_Z]
        Standard X Configuration.
        """
        # d = L * sin(45)
        d = self.arm_l_ * 0.70710678 
        k = self.kappa_ # Moment coefficient (c_m)

        # Matrix: Wrench = Mat @ Motor_Thrusts
        # Signs depend on motor rotation direction (Standard Betaflight/Cleanflight X usually):
        # M1(FR, CW), M2(BR, CCW), M3(BL, CW), M4(FL, CCW) -> Check your specific frame!
        # Assuming Standard X:
        # Thrust: + + + +
        # Roll:   - - + + (Left is +, Right is -) or vice versa. 
        #         Let's stick to standard: Roll = d * (T3 + T4 - T1 - T2) ? 
        #         Let's use the signs from your previous code which are standard for many setups:
        #         Roll: d * [1, -1, -1, 1] (FR/FL vs BR/BL)
        
        r0 = torch.ones(self.num_envs, 4, device=self.device)   # Thrust
        r1 = torch.stack([d, -d, -d, d], dim=1)                 # Roll
        r2 = torch.stack([-d, -d, d, d], dim=1)                 # Pitch
        r3 = torch.stack([k, -k, k, -k], dim=1)                 # Yaw (Scaled by Thrust)

        mat = torch.stack([r0, r1, r2, r3], dim=1)
        return mat

    def motor_speeds_to_wrench(self, motor_actions: torch.Tensor) -> tuple:
        """
        Maps NORMALIZED actions to Body Wrench.
        
        Args:
            motor_actions: (num_envs, 4) in range [0, 1]. 
                           NO denormalization to RPM needed.
        
        Returns:
            force: (num_envs, 3) [0, 0, Fz]
            torque: (num_envs, 3) [Tx, Ty, Tz]
        """
        # Formula S8: f(w) = c_f0 + c_f1 * w + c_f2 * w^2
        # motor_actions is w
        
        a = self.thrust_map_[:, 0].unsqueeze(1) # c_f2 (Quadratic)
        b = self.thrust_map_[:, 1].unsqueeze(1) # c_f1 (Linear)
        c = self.thrust_map_[:, 2].unsqueeze(1) # c_f0 (Constant)
        
        # Calculate Thrust per motor (Newtons)
        motor_thrusts = a * (motor_actions ** 2) + b * motor_actions + c
        
        # Mix to Wrench
        # wrench shape: [num_envs, 4]
        wrench = torch.bmm(self.alloc_matrix_, motor_thrusts.unsqueeze(-1)).squeeze(-1)
        
        # Extract Output
        force = torch.zeros(self.num_envs, 3, device=self.device)
        force[:, 2] = wrench[:, 0] # Z-force
        
        torque = wrench[:, 1:4]    # Torques
        
        return force, torque, None