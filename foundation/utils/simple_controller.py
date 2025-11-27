"""
Body-Frame Euler Angle PD Controller (PyTorch Implementation)

Euler Angle Convention (ZYX / Roll-Pitch-Yaw):
- Rotation order: R = Rz(ψ) * Ry(θ) * Rx(φ)
- Euler angles: [φ, θ, ψ] = [roll, pitch, yaw]
  - φ (roll):  rotation around X-axis (body frame forward)
  - θ (pitch): rotation around Y-axis (body frame right)
  - ψ (yaw):   rotation around Z-axis (body frame up)
- All Euler angle vectors follow the order: [roll, pitch, yaw] = [φ, θ, ψ]

Updated control outline:
1. Build a body-frame attitude error quaternion `q_err = q_cur* ⊗ q_des`.
2. Map `q_err` to a rotation vector as the attitude error signal.
3. Convert desired Euler rates to body rates for feed-forward.
4. Apply PD directly in the body frame to obtain angular acceleration.
5. Use Euler’s rigid-body equation to compute torque.

Author: Chengyong Lei
Affiliation: ZJU FAST Lab
"""

import torch
import math


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angles in radians to the range [-pi, pi)."""
    return torch.remainder(angle + torch.pi, 2 * torch.pi) - torch.pi


def quat_from_euler_xyz(euler: torch.Tensor) -> torch.Tensor:
    """Convert batched ZYX Euler angles [roll, pitch, yaw] to quaternion [w, x, y, z]."""
    roll, pitch, yaw = euler[:, 0], euler[:, 1], euler[:, 2]

    half_roll = 0.5 * roll
    half_pitch = 0.5 * pitch
    half_yaw = 0.5 * yaw

    cr = torch.cos(half_roll)
    sr = torch.sin(half_roll)
    cp = torch.cos(half_pitch)
    sp = torch.sin(half_pitch)
    cy = torch.cos(half_yaw)
    sy = torch.sin(half_yaw)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return torch.stack([w, x, y, z], dim=1)


def quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    """Return quaternion conjugate."""
    return torch.stack([quat[:, 0], -quat[:, 1], -quat[:, 2], -quat[:, 3]], dim=1)


def quat_normalize(quat: torch.Tensor) -> torch.Tensor:
    """Normalize quaternion to unit length."""
    norm = torch.norm(quat, dim=1, keepdim=True).clamp(min=1e-12)
    return quat / norm


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Quaternion multiplication."""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=1)


def quat_to_rotvec(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to rotation vector (axis-angle) with shortest representation."""
    quat = torch.where(quat[:, 0:1] < 0.0, -quat, quat)
    w = quat[:, 0:1].clamp(min=-1.0, max=1.0)
    xyz = quat[:, 1:]
    xyz_norm = torch.norm(xyz, dim=1, keepdim=True)

    angle = 2.0 * torch.atan2(xyz_norm, w)
    axis = torch.where(
        xyz_norm > 1e-12,
        xyz / xyz_norm,
        torch.zeros_like(xyz),
    )
    return axis * angle


def quat_to_euler_xyz(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion [w, x, y, z] to ZYX (roll, pitch, yaw) Euler angles."""
    q_w, q_x, q_y, q_z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    sin_roll = 2.0 * (q_w * q_x + q_y * q_z)
    cos_roll = 1.0 - 2.0 * (q_x * q_x + q_y * q_y)
    roll = torch.atan2(sin_roll, cos_roll)

    sin_pitch = 2.0 * (q_w * q_y - q_z * q_x)
    pitch = torch.where(
        torch.abs(sin_pitch) >= 1.0,
        torch.sign(sin_pitch) * (torch.pi / 2.0),
        torch.asin(sin_pitch),
    )

    sin_yaw = 2.0 * (q_w * q_z + q_x * q_y)
    cos_yaw = 1.0 - 2.0 * (q_y * q_y + q_z * q_z)
    yaw = torch.atan2(sin_yaw, cos_yaw)

    roll = wrap_to_pi(roll)
    yaw = wrap_to_pi(yaw)

    return torch.stack([roll, pitch, yaw], dim=1)


class QuadrotorDynamics:
    """
    Quadrotor dynamics model with motor thrust mapping and allocation matrix.
    Based on Crazyflie 2.0 parameters.
    """

    def __init__(self,
                 num_envs: int = 1,
                 mass: torch.Tensor = None,
                 inertia: torch.Tensor = None,
                 arm_l: torch.Tensor = None,
                 device: torch.device = torch.device("cpu")):
        self.device = device
        self.num_envs = num_envs

        # Initialize mass
        if mass is None:
            self.mass_ = torch.full((num_envs,), 0.027, device=self.device)
        else:
            self.mass_ = mass.to(self.device)
            if self.mass_.dim() == 0:
                self.mass_ = self.mass_.expand(num_envs)

        # Initialize arm length
        if arm_l is None:
            self.arm_l_ = torch.full((num_envs,), 0.046, device=self.device)
        else:
            self.arm_l_ = arm_l.to(self.device)
            if self.arm_l_.dim() == 0:
                self.arm_l_ = self.arm_l_.expand(num_envs)

        # Initialize inertia
        default_inertia = torch.tensor([9.19e-6, 9.19e-6, 22.8e-6], device=self.device)
        if inertia is None:
            self.inertia_ = default_inertia.expand(num_envs, 3)
        else:
            if inertia.dim() == 1:
                self.inertia_ = inertia.to(self.device).expand(num_envs, 3)
            elif inertia.dim() == 2:
                self.inertia_ = inertia.to(self.device)
            else:
                raise ValueError(f"Inertia must be tensor of shape (3,) or (num_envs, 3), got {inertia.shape}")

        # Motor limits
        kv_rpm_per_v = 10000.0
        kt = 60.0 / (2 * math.pi * kv_rpm_per_v)
        self.motor_tau_inv_ = torch.full((num_envs,), 1.0 / kt, device=self.device)

        rpm_max, rpm_min = 24000.0, 1200.0
        self.motor_omega_max_ = torch.full((num_envs,), rpm_max * math.pi / 30, device=self.device)
        self.motor_omega_min_ = torch.full((num_envs,), rpm_min * math.pi / 30, device=self.device)

        # Thrust map coefficients: T = a*omega^2 + b*omega + c
        self.thrust_map_ = torch.tensor([9.96063125e-08, -2.55003087e-05, 5.84422691e-03],
                                        device=self.device).expand(num_envs, 3)

        # Drag-torque ratio (yaw moment per unit thrust)
        self.kappa_ = torch.full((num_envs,), 0.005964552, device=self.device)

        # Thrust limits
        self.thrust_min_ = torch.zeros(num_envs, device=self.device)
        a, b, c = self.thrust_map_[:, 0], self.thrust_map_[:, 1], self.thrust_map_[:, 2]
        w = self.motor_omega_max_
        self.thrust_max_ = a * (w ** 2) + b * w + c  # 0.857 N

        # Maximum torque calculation parameter
        self.thrust_ratio = 0.375  # Configurable via set_thrust_ratio()

        self.updateInertiaMatrix()

    def updateInertiaMatrix(self):
        """Build allocation matrix for X-configuration."""
        # t_BM: mapping matrix for X-configuration, shape (num_envs, 3, 4)
        base_mat = torch.tensor([[1, -1, -1, 1],
                                 [-1, -1, 1, 1],
                                 [0, 0, 0, 0]], dtype=torch.float, device=self.device)
        factor = self.arm_l_ * math.sqrt(0.5)
        self.t_BM_ = factor.view(-1, 1, 1) * base_mat

        # Construct the diagonal inertia matrix J_ and its inverse
        self.J_ = torch.diag_embed(self.inertia_)
        self.J_inv_ = torch.diag_embed(1.0 / self.inertia_)

    def set_thrust_ratio(self, ratio: float):
        """
        Set the thrust ratio available for torque generation.

        Args:
            ratio: Thrust ratio for torque [0.0, 1.0]
                   - 0.25: Conservative (assumes hover at 75% throttle)
                   - 0.375: Default (assumes hover at ~62.5% throttle)
                   - 0.5: Balanced (assumes hover at 50% throttle)
                   - 1.0: Aggressive (full motor range available)
        """
        self.thrust_ratio = max(0.0, min(1.0, ratio))

    def clampThrust(self, thrusts: torch.Tensor):
        """Clamp thrust to physical limits."""
        t_min = self.thrust_min_.unsqueeze(1)
        t_max = self.thrust_max_.unsqueeze(1)
        return torch.max(torch.min(thrusts, t_max), t_min)

    def motorOmegaToThrust(self, omega: torch.Tensor):
        """Convert motor angular velocity to thrust."""
        a = self.thrust_map_[:, 0].unsqueeze(1)
        b = self.thrust_map_[:, 1].unsqueeze(1)
        c = self.thrust_map_[:, 2].unsqueeze(1)
        return a * (omega ** 2) + b * omega + c

    def motorThrustToOmega(self, thrusts: torch.Tensor):
        """
        Convert thrust to motor angular velocity using quadratic formula.

        Args:
            thrusts: (num_envs, 4) motor thrust values [N]

        Returns:
            omega: (num_envs, 4) motor angular velocities [rad/s]
        """
        a = self.thrust_map_[:, 0].unsqueeze(1)
        b = self.thrust_map_[:, 1].unsqueeze(1)
        c = self.thrust_map_[:, 2].unsqueeze(1)

        # Solve quadratic: a*omega^2 + b*omega + (c - thrust) = 0
        discriminant = b ** 2 - 4.0 * a * (c - thrusts)
        discriminant_safe = torch.clamp(discriminant, min=0.0)
        omega = (-b + torch.sqrt(discriminant_safe)) / (2.0 * a)

        return omega

    def getAllocationMatrix(self):
        """
        Get control allocation matrix for X-configuration.
        Maps [total_thrust, roll_torque, pitch_torque, yaw_torque] -> [T1, T2, T3, T4]
        """
        num_envs = self.mass_.shape[0]
        ones_row = torch.ones(num_envs, 4, device=self.device)
        alloc = torch.stack([
            ones_row,
            self.t_BM_[:, 0, :],
            self.t_BM_[:, 1, :],
            self.kappa_.unsqueeze(1) * torch.tensor([1, -1, 1, -1], device=self.device).expand(num_envs, 4)
        ], dim=1)
        return alloc


class WorldFramePDControl:
    """
    PD controller in world frame with Euler dynamics.

    Euler Angle Convention:
    - Order: [roll, pitch, yaw] = [φ, θ, ψ] (ZYX convention)
    - φ (roll):  rotation around X-axis
    - θ (pitch): rotation around Y-axis
    - ψ (yaw):   rotation around Z-axis

    Refined body-frame control strategy:
    1. Convert desired/current Euler angles to quaternions.
    2. Compute body-frame attitude error via `q_err = q_cur* ⊗ q_des` and map it to a rotation vector.
    3. Use the rotation-vector error to generate desired body angular velocity.
    4. Run a proportional controller on `(ω_des - ω_cur)` to obtain desired angular acceleration.
    5. Apply Euler’s equation `τ = J α_des + ω × (J ω)` to recover torque.
    """

    def __init__(self,
                 num_envs: int = 1,
                 device: torch.device = torch.device("cpu"),
                 attitude_p_gain: torch.Tensor = None,
                 rate_p_gain: torch.Tensor = None,
                 inertia: torch.Tensor = None,
                 derivative_alpha: float = 0.2,
                 attitude_d_gain: torch.Tensor = None):
        """
        Initialize world-frame PD controller.

        Args:
            num_envs: Number of parallel environments
            device: PyTorch device
            attitude_p_gain: (3,) P gains [roll, pitch, yaw] for attitude loop
            rate_p_gain: (3,) P gains [roll, pitch, yaw] for rate loop
            inertia: (3,) or (num_envs, 3) Inertia tensor [Ixx, Iyy, Izz] in kg*m^2
            derivative_alpha: Low-pass filter coefficient for rotation error derivative (0-1)
            attitude_d_gain: (3,) D gains for the attitude-level PD that produces ω_des
        """
        self.num_envs = num_envs
        self.device = device

        # Default inertia (Crazyflie 2.0)
        if inertia is None:
            inertia = torch.tensor([9.19e-6, 9.19e-6, 22.8e-6], device=device)

        if inertia.dim() == 1:
            self.inertia = inertia.to(device).expand(num_envs, 3)
        else:
            self.inertia = inertia.to(device)

        # Default PD gains
        if attitude_p_gain is None:
            attitude_p_gain = torch.tensor([10.0, 10.0, 5.0], device=device)
        if rate_p_gain is None:
            rate_p_gain = torch.tensor([2.0, 2.0, 1.0], device=device)

        self.attitude_p_gain = attitude_p_gain.to(device)
        self.rate_p_gain = rate_p_gain.to(device)
        if attitude_d_gain is None:
            attitude_d_gain = 0.5 * self.attitude_p_gain
        self.attitude_d_gain = attitude_d_gain.to(device)

        self.prev_rot_error = torch.zeros(num_envs, 3, device=device)
        self.rot_error_rate = torch.zeros(num_envs, 3, device=device)
        self.derivative_alpha = derivative_alpha

    def update(self,
               euler_des: torch.Tensor,
               euler_cur: torch.Tensor,
               omega_body: torch.Tensor,
               dt: float) -> torch.Tensor:
        """
        Compute torque using body-frame PD control.

        Control law (body-frame formulation):
        1. Form quaternion attitude error in the body frame: `q_err = q_cur* ⊗ q_des`.
        2. Map the quaternion error to a rotation vector `e_R` (axis-angle, body frame).
        3. Map the attitude error to a desired body angular velocity via PD on the rotation-vector error.
        4. Body-frame P control on angular-rate error: `α_des = Kd ⊙ (ω_des - ω_cur)`.
        5. Euler equation: `τ = J α_des + ω × (J ω)`.

        Args:
            euler_des: (num_envs, 3) desired Euler angles [φ, θ, ψ] = [roll, pitch, yaw] in radians
            euler_cur: (num_envs, 3) current Euler angles [φ, θ, ψ] = [roll, pitch, yaw] in radians
            omega_body: (num_envs, 3) current body angular velocity [ωx, ωy, ωz] = [p, q, r] in rad/s
            dt: Integration time-step in seconds for derivative estimation

        Returns:
            torque: (num_envs, 3) torque command [τx, τy, τz] = [τ_roll, τ_pitch, τ_yaw] in Nm
        """
        # Step 1: build target/current quaternions and recover body-frame attitude error
        q_des = quat_normalize(quat_from_euler_xyz(euler_des))
        q_cur = quat_normalize(quat_from_euler_xyz(euler_cur))

        q_err = quat_normalize(quat_mul(quat_conjugate(q_cur), q_des))
        rot_error = quat_to_rotvec(q_err)

        # Step 2: convert rotation-vector error into a desired body angular velocity via PD
        dt = max(dt, 1e-5)
        raw_rot_error_rate = (rot_error - self.prev_rot_error) / dt
        alpha = self.derivative_alpha
        self.rot_error_rate = alpha * raw_rot_error_rate + (1.0 - alpha) * self.rot_error_rate
        self.prev_rot_error = rot_error.detach()

        omega_des = (
            self.attitude_p_gain.unsqueeze(0) * rot_error
            + self.attitude_d_gain.unsqueeze(0) * self.rot_error_rate
        )

        # Step 3: body-frame angular-rate error
        omega_error = omega_des - omega_body

        # Step 4: body-frame proportional controller on angular-rate error -> desired angular acceleration
        alpha_body = self.rate_p_gain.unsqueeze(0) * omega_error

        # Step 5: Compute torque using Euler's equation: τ = J*α + ω × (J*ω)
        # Gyroscopic term: ω × (J*ω)
        J_omega = self.inertia * omega_body  # Element-wise multiplication
        gyroscopic_torque = torch.cross(omega_body, J_omega, dim=1)

        # Total torque
        torque = self.inertia * alpha_body + gyroscopic_torque

        return torque

    def _euler_rates_to_body_rates(self, euler_rates: torch.Tensor, euler: torch.Tensor) -> torch.Tensor:
        """
        Convert Euler angle rates to body angular rates using Jacobian matrix W.

        Jacobian W (ZYX Euler rates → body rates):
        [ωx]   [1    0        -sin(θ)    ] [φ̇ ]
        [ωy] = [0   cos(φ)   sin(φ)cos(θ)] [θ̇ ]
        [ωz]   [0  -sin(φ)   cos(φ)cos(θ)] [ψ̇ ]

        Where: [φ, θ, ψ] = [roll, pitch, yaw], [ωx, ωy, ωz] = [p, q, r]

        Args:
            euler_rates: (num_envs, 3) Euler angle rates [φ̇, θ̇, ψ̇ ] = [roll_rate, pitch_rate, yaw_rate]
            euler: (num_envs, 3) current Euler angles [φ, θ, ψ] = [roll, pitch, yaw]

        Returns:
            omega_body: (num_envs, 3) body angular rates [ωx, ωy, ωz] = [p, q, r]
        """
        phi, theta, psi = euler[:, 0], euler[:, 1], euler[:, 2]
        phi_dot, theta_dot, psi_dot = euler_rates[:, 0], euler_rates[:, 1], euler_rates[:, 2]

        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        # Avoid division by zero at gimbal lock (θ = ±90°)
        cos_theta = torch.clamp(cos_theta, min=1e-6)

        # W matrix multiplication
        omega_x = phi_dot - psi_dot * sin_theta
        omega_y = theta_dot * cos_phi + psi_dot * sin_phi * cos_theta
        omega_z = -theta_dot * sin_phi + psi_dot * cos_phi * cos_theta

        return torch.stack([omega_x, omega_y, omega_z], dim=1)

    def _body_rates_to_euler_rates(self, omega_body: torch.Tensor, euler: torch.Tensor) -> torch.Tensor:
        """
        Convert body angular rates to Euler angle rates using inverse Jacobian W_inv.

        Inverse Jacobian W_inv (body rates → ZYX Euler rates):
        [φ̇ ]   [1  sin(φ)tan(θ)  cos(φ)tan(θ)] [ωx]
        [θ̇ ] = [0     cos(φ)        -sin(φ)  ] [ωy]
        [ψ̇ ]   [0  sin(φ)/cos(θ)  cos(φ)/cos(θ)] [ωz]

        Where: [φ, θ, ψ] = [roll, pitch, yaw], [ωx, ωy, ωz] = [p, q, r]

        Args:
            omega_body: (num_envs, 3) body angular rates [ωx, ωy, ωz] = [p, q, r]
            euler: (num_envs, 3) current Euler angles [φ, θ, ψ] = [roll, pitch, yaw]

        Returns:
            euler_rates: (num_envs, 3) Euler angle rates [φ̇, θ̇, ψ̇] = [roll_rate, pitch_rate, yaw_rate]
        """
        phi, theta, psi = euler[:, 0], euler[:, 1], euler[:, 2]
        omega_x, omega_y, omega_z = omega_body[:, 0], omega_body[:, 1], omega_body[:, 2]

        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        tan_theta = torch.tan(theta)

        # Avoid division by zero at gimbal lock (θ = ±90°)
        cos_theta = torch.clamp(cos_theta, min=1e-6)

        # W_inv matrix multiplication
        phi_dot = omega_x + omega_y * sin_phi * tan_theta + omega_z * cos_phi * tan_theta
        theta_dot = omega_y * cos_phi - omega_z * sin_phi
        psi_dot = (omega_y * sin_phi + omega_z * cos_phi) / cos_theta

        return torch.stack([phi_dot, theta_dot, psi_dot], dim=1)

    def _normalize_angle(self, angle: torch.Tensor) -> torch.Tensor:
        """Normalize angle to [-pi, pi]."""
        return torch.atan2(torch.sin(angle), torch.cos(angle))


class SimpleQuadrotorController:
    """
    Simple quadrotor controller using world-frame PD control.

    Features:
    - World-frame Euler angle control
    - PD control with gyroscopic compensation
    - Motor allocation and thrust mapping
    """

    def __init__(self,
                 num_envs: int = 1,
                 device: torch.device = torch.device("cpu"),
                 # PD control params
                 attitude_p_gain: torch.Tensor = None,
                 attitude_d_gain: torch.Tensor = None,
                 rate_p_gain: torch.Tensor = None,
                 # Dynamics params
                 mass: torch.Tensor = None,
                 inertia: torch.Tensor = None,
                 arm_length: torch.Tensor = None):
        """
        Initialize simple quadrotor controller.

        Args:
            num_envs: Number of parallel environments
            device: PyTorch device
            attitude_p_gain: (3,) attitude-loop P gains [roll, pitch, yaw]
            attitude_d_gain: (3,) attitude-loop D gains [roll, pitch, yaw]
            rate_p_gain: (3,) rate-loop P gains [roll, pitch, yaw]
            mass: Quadrotor mass (kg)
            inertia: Inertia tensor [Ixx, Iyy, Izz] (kg*m^2)
            arm_length: Motor arm length (m)
        """
        self.num_envs = num_envs
        self.device = device

        # Initialize dynamics model
        self.dynamics = QuadrotorDynamics(
            num_envs=num_envs,
            mass=mass,
            inertia=inertia,
            arm_l=arm_length,
            device=device
        )

        # Initialize PD controller
        self.pd_control = WorldFramePDControl(
            num_envs=num_envs,
            device=device,
            attitude_p_gain=attitude_p_gain,
            rate_p_gain=rate_p_gain,
            attitude_d_gain=attitude_d_gain,
            inertia=self.dynamics.inertia_
        )

        # Precompute allocation matrices
        self.alloc_matrix_ = self.dynamics.getAllocationMatrix()
        self.alloc_matrix_pinv_ = torch.linalg.pinv(self.alloc_matrix_)

        print(f"[Simple Controller] World-frame PD controller initialized")
        print(f"[Simple Controller] Attitude Kp: {self.pd_control.attitude_p_gain}")
        print(f"[Simple Controller] Attitude Kd: {self.pd_control.attitude_d_gain}")
        print(f"[Simple Controller] Rate Kp: {self.pd_control.rate_p_gain}")
        print(f"[Simple Controller] Inertia: {self.dynamics.inertia_[0]}")
        print(f"[Simple Controller] {num_envs} environments")

    def compute_control(self,
                       state: torch.Tensor,
                       cmd: torch.Tensor,
                       dt: float = 0.01) -> tuple:
        """
        Main control loop.

        Args:
            state: (num_envs, 19+) tensor containing:
                   [0:3] position, [3:7] quaternion [w,x,y,z],
                   [7:10] velocity, [10:13] angular velocity (world frame),
                   [13:16] body torque, [16:19] linear acceleration
            cmd: (num_envs, 4) normalized command [roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd]
                 - roll_cmd: normalized roll command, range [-1, 1] → ±45°
                 - pitch_cmd: normalized pitch command, range [-1, 1] → ±45°
                 - yaw_cmd: normalized yaw command, range [-1, 1] → ±90°
                 - thrust_cmd: normalized thrust [0, 1]
            dt: integration timestep (s), used for derivative filtering in the attitude loop

        Returns:
            force: (num_envs, 3) body frame force [Fx, Fy, Fz]
            torque: (num_envs, 3) body frame torque [τ_roll, τ_pitch, τ_yaw]
            motor_speeds: (num_envs, 4) motor angular velocities [rad/s]
            info: dict with control information
        """
        batch = state.shape[0]

        # Parse state
        q_cur = state[:, 3:7]  # Current quaternion (body-to-world)
        omega_world = state[:, 10:13]  # Angular velocity in world frame

        # Convert quaternion to Euler angles (world frame)
        euler_cur = self._quat_to_euler(q_cur)

        # Convert angular velocity to body frame
        omega_body = self._quat_rotate_inverse(q_cur, omega_world)

        # Parse command and convert normalized values to Euler angles
        roll_des = cmd[:, 0] * (math.pi / 4)   # ±45deg
        pitch_des = cmd[:, 1] * (math.pi / 4)  # ±45deg
        yaw_des = cmd[:, 2] * (math.pi / 2)    # ±90deg
        euler_des = torch.stack([roll_des, pitch_des, yaw_des], dim=1)

        thrust_cmd = cmd[:, 3]   # Normalized thrust [0, 1]

        # Compute desired thrust force
        F_des = thrust_cmd * self.dynamics.thrust_max_ * self.dynamics.thrust_ratio * 4.0

        # Compute torque using PD control
        tau_des = self.pd_control.update(
            euler_des=euler_des,
            euler_cur=euler_cur,
            omega_body=omega_body,
            dt=dt
        )

        # === Control allocation ===
        # Step 1: Compute baseline thrust per motor
        T0 = F_des / 4.0

        # Step 2: Calculate differential thrust bounds
        T_motor_max = self.dynamics.thrust_max_ * self.dynamics.thrust_ratio
        T_motor_min = self.dynamics.thrust_min_
        delta_T_max_pos = (T_motor_max - T0).clamp(min=0.0)
        delta_T_max_neg = (T0 - T_motor_min).clamp(min=0.0)

        # Step 3: Build control wrench [thrust, roll_torque, pitch_torque, yaw_torque]
        torque_only_wrench = torch.cat([
            torch.zeros_like(F_des).unsqueeze(1),  # No additional thrust
            tau_des  # Physical torque [Nm]
        ], dim=1)

        # Use pseudo-inverse for control allocation
        delta_T = torch.bmm(self.alloc_matrix_pinv_, torque_only_wrench.unsqueeze(-1)).squeeze(-1)

        # Step 4: Scale differential thrust to respect bounds
        eps = 1e-6
        pos_mask = delta_T > 0
        neg_mask = delta_T < 0

        delta_T_max_pos_safe = torch.clamp(delta_T_max_pos.unsqueeze(1), min=eps)
        delta_T_max_neg_safe = torch.clamp(delta_T_max_neg.unsqueeze(1), min=eps)

        scale_factors_pos = torch.where(pos_mask, delta_T / delta_T_max_pos_safe,
                                        torch.zeros_like(delta_T))
        scale_factors_neg = torch.where(neg_mask, -delta_T / delta_T_max_neg_safe,
                                        torch.zeros_like(delta_T))

        max_scale_per_batch, _ = torch.max(torch.maximum(scale_factors_pos, scale_factors_neg),
                                           dim=1, keepdim=True)

        delta_T_scaling = torch.where(
            max_scale_per_batch > 1.0,
            1.0 / max_scale_per_batch,
            torch.ones_like(max_scale_per_batch)
        )
        delta_T = delta_T * delta_T_scaling

        # Step 5: Combine baseline and differential thrust
        motor_thrusts = T0.unsqueeze(1) + delta_T

        # Step 6: Clamp to non-negative
        motor_thrusts = torch.clamp(motor_thrusts, min=0.0)

        # Zero throttle → zero thrust
        zero_throttle = (cmd[:, 3] == 0.0).unsqueeze(1)
        motor_thrusts = torch.where(zero_throttle, torch.zeros_like(motor_thrusts), motor_thrusts)

        # Step 7: Limit to maximum thrust
        thrust_max_expanded = T_motor_max.unsqueeze(1).expand_as(motor_thrusts)
        scale_factors = motor_thrusts / thrust_max_expanded
        max_scale_per_batch, _ = torch.max(scale_factors, dim=1, keepdim=True)

        scaling = torch.where(
            max_scale_per_batch > 1.0,
            1.0 / max_scale_per_batch,
            torch.ones_like(max_scale_per_batch)
        )
        motor_thrusts = motor_thrusts * scaling

        # Step 8: Convert to motor speeds
        motor_speeds = self.dynamics.motorThrustToOmega(motor_thrusts)

        # Step 9: Reconstruct actual force & torque
        alloc = torch.bmm(self.alloc_matrix_, motor_thrusts.unsqueeze(-1)).squeeze(-1)

        force = torch.zeros((batch, 3), device=self.device)
        force[:, 2] = alloc[:, 0]  # Thrust in z-direction
        torque = alloc[:, 1:4]     # Actual torque [roll, pitch, yaw]

        # Build info dictionary
        euler_error = euler_des - euler_cur
        euler_error[:, 2] = self._normalize_angle(euler_error[:, 2])

        info = {
            'euler_cur': euler_cur,
            'euler_des': euler_des,
            'euler_error': euler_error,
            'omega_body': omega_body,
            'tau_des': tau_des,
            'motor_thrusts': motor_thrusts,
            'thrust_cmd': thrust_cmd,
            'force_z': force[:, 2],
            'torque': torque
        }

        return force, torque, motor_speeds, info

    def _quat_to_euler(self, quat: torch.Tensor) -> torch.Tensor:
        """Convert quaternion batch to ZYX Euler angles [roll, pitch, yaw]."""
        return quat_to_euler_xyz(quat)

    def _quat_rotate_inverse(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Rotate vector from world frame to body frame."""
        q_inv = torch.stack([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]], dim=1)
        v_quat = torch.cat([torch.zeros(v.shape[0], 1, device=v.device), v], dim=1)
        temp = self._quat_mul(q_inv, v_quat)
        result = self._quat_mul(temp, q)
        return result[:, 1:]

    def _quat_mul(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Quaternion multiplication."""
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return torch.stack([w, x, y, z], dim=1)

    def _normalize_angle(self, angle: torch.Tensor) -> torch.Tensor:
        """Normalize angle to [-pi, pi]."""
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def motor_speeds_to_wrench(self, motor_speeds: torch.Tensor, normalized: bool = False) -> tuple:
        """
        Convert motor angular velocities to body-frame force and torque.
        
        This function performs the forward calculation from motor speeds to wrench:
        1. Denormalize motor speeds if they are normalized (optional)
        2. Convert motor speeds to individual motor thrusts using thrust map
        3. Apply allocation matrix to get total force and torque
        
        Args:
            motor_speeds: (num_envs, 4) motor angular velocities [rad/s] or normalized [0, 1]
                         Motors in X-configuration order: [M1, M2, M3, M4]
            normalized: If True, input motor_speeds are normalized in [0, 1] range
                       and will be denormalized to [motor_omega_min, motor_omega_max]
        
        Returns:
            force: (num_envs, 3) body frame force [Fx, Fy, Fz] in Newtons
                   For quadrotors, typically Fx=Fy=0, Fz=total_thrust
            torque: (num_envs, 3) body frame torque [τ_roll, τ_pitch, τ_yaw] in Nm
        
        Example:
            >>> controller = SimpleQuadrotorController(num_envs=2)
            >>> # Using absolute motor speeds (rad/s)
            >>> motor_speeds = torch.tensor([[2000, 2000, 2000, 2000],
            ...                               [2500, 2400, 2500, 2400]], device=controller.device)
            >>> force, torque = controller.motor_speeds_to_wrench(motor_speeds)
            >>> 
            >>> # Using normalized motor speeds [0, 1]
            >>> motor_speeds_norm = torch.tensor([[0.5, 0.5, 0.5, 0.5],
            ...                                    [0.7, 0.6, 0.7, 0.6]], device=controller.device)
            >>> force, torque = controller.motor_speeds_to_wrench(motor_speeds_norm, normalized=True)
        """
        batch = motor_speeds.shape[0]
        
        # Step 0: Denormalize motor speeds if necessary
        if normalized:
            # Denormalize from [0, 1] to [motor_omega_min, motor_omega_max]
            omega_min = self.dynamics.motor_omega_min_.unsqueeze(1)  # (num_envs, 1)
            omega_max = self.dynamics.motor_omega_max_.unsqueeze(1)  # (num_envs, 1)
            motor_speeds = omega_min + motor_speeds * (omega_max - omega_min)
        
        # Step 1: Convert motor speeds to thrusts using quadratic thrust map
        # T = a*omega^2 + b*omega + c
        motor_thrusts = self.dynamics.motorOmegaToThrust(motor_speeds)
        
        # Step 2: Apply allocation matrix to get wrench
        # wrench = [total_thrust, roll_torque, pitch_torque, yaw_torque]
        # alloc_matrix shape: (num_envs, 4, 4)
        # motor_thrusts shape: (num_envs, 4)
        alloc = torch.bmm(self.alloc_matrix_, motor_thrusts.unsqueeze(-1)).squeeze(-1)
        
        # Step 3: Extract force and torque
        # Force: only vertical thrust in body z-direction
        force = torch.zeros((batch, 3), device=self.device)
        force[:, 2] = alloc[:, 0]  # Total thrust in z-direction
        
        # Torque: [roll, pitch, yaw] moments
        torque = alloc[:, 1:4]  # [τ_roll, τ_pitch, τ_yaw]
        
        return force, torque

    def reset(self, env_ids: torch.Tensor = None):
        """
        Reset controller states for specified environments.

        Args:
            env_ids: (N,) tensor of environment indices to reset. If None, resets all.

        Note:
            This PD controller is stateless, so reset does nothing.
            Kept for API compatibility.
        """
        pass
