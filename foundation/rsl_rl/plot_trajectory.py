import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

class LangevinTrajectory:
    def __init__(self, device='cpu', dt=0.02, num_envs=1):
        self.device = device
        self.dt = dt
        self.num_envs = num_envs

        # 初始状态 (XYZ)
        self.pos_des = torch.zeros((num_envs, 3), device=device)
        self.vel_des = torch.zeros((num_envs, 3), device=device)
        self.acc_des = torch.zeros((num_envs, 3), device=device)
        self._spawn_pos_w = torch.zeros((num_envs, 3), device=device)
        
        # 初始状态 (Yaw)
        self.yaw_des = torch.zeros(num_envs, device=device)
        self.yaw_rate_des = torch.zeros(num_envs, device=device)

    def step(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        n_envs = len(env_ids)
        dt = self.dt

        # ==================== 1. XYZ 动力学 ====================
        k_pos = torch.tensor([1.0, 1.0, 2.0], device=self.device) 
        k_vel = 1.5   
        acc_inertia = 0.1 
        noise_scale = 10.0 
        max_vel = 3.0 
        max_acc = 15.0

        pos_current = self.pos_des[env_ids]
        vel_current = self.vel_des[env_ids]
        acc_current = self.acc_des[env_ids]
        spawn_pos = self._spawn_pos_w[env_ids]

        vel_next = vel_current + acc_current * dt

        vel_norm = torch.norm(vel_next, dim=1, keepdim=True)
        scale = torch.clamp(max_vel / (vel_norm + 1e-6), max=1.0)
        vel_next = vel_next * scale

        pos_next = pos_current + vel_current * dt
        pos_err = pos_current - spawn_pos

        noise = torch.randn(n_envs, 3, device=self.device) * noise_scale
        z_noise_attenuation = 0.3  
        noise[:, 2] *= z_noise_attenuation

        force_total = noise - k_pos * pos_err - k_vel * vel_current

        acc_next = (1.0 - acc_inertia) * acc_current + acc_inertia * force_total
        acc_next = torch.clamp(acc_next, -max_acc, max_acc)

        self.acc_des[env_ids] = acc_next
        self.vel_des[env_ids] = vel_next
        self.pos_des[env_ids] = pos_next

        # ==================== 2. Yaw 动力学 ====================
        yaw_limit = math.pi / 2  # 限制在 +/- 90 度 (1.57 rad)
        yaw_k_pos = 0.2          # 弹簧刚度 (回复力)
        yaw_k_vel = 0.5          # 阻尼
        yaw_noise_scale = 2.0    # 噪声强度
        max_yaw_rate = 2.5       # 最大角速度
            
        current_yaw = self.yaw_des[env_ids]
        current_yaw_rate = self.yaw_rate_des[env_ids]
        
        yaw_noise = torch.randn(len(env_ids), device=self.device) * yaw_noise_scale
        
        # 核心 OU 过程
        yaw_acc = yaw_noise - yaw_k_vel * current_yaw_rate - yaw_k_pos * current_yaw
        next_yaw_rate = current_yaw_rate + yaw_acc * dt
        
        # 限制角速度
        next_yaw_rate = torch.clamp(next_yaw_rate, -max_yaw_rate, max_yaw_rate)
        next_yaw = current_yaw + next_yaw_rate * dt
        
        # 硬边界与非弹性碰撞处理
        over_max = next_yaw > yaw_limit
        under_min = next_yaw < -yaw_limit
        if over_max.any() or under_min.any():
            next_yaw = torch.clamp(next_yaw, -yaw_limit, yaw_limit)
            hit_limit_mask = over_max | under_min
            next_yaw_rate[hit_limit_mask] = 0.0 # 撞墙后速度清零
            
        self.yaw_des[env_ids] = next_yaw
        self.yaw_rate_des[env_ids] = next_yaw_rate

        return (self.pos_des.clone(), self.vel_des.clone(), self.acc_des.clone(),
                self.yaw_des.clone(), self.yaw_rate_des.clone())


def run_and_plot(steps=1000, dt=0.02):
    sim = LangevinTrajectory(dt=dt, num_envs=1)

    # 记录轨迹 (增加 yaw 和 yaw_rate)
    history = {'pos': [], 'vel': [], 'acc': [], 'yaw': [], 'yaw_rate': []}

    # 运行模拟 (1000步 = 20秒)
    for _ in range(steps):
        p, v, a, y, yr = sim.step()
        history['pos'].append(p[0].numpy())
        history['vel'].append(v[0].numpy())
        history['acc'].append(a[0].numpy())
        history['yaw'].append(y[0].numpy())
        history['yaw_rate'].append(yr[0].numpy())

    # 转化为 numpy 数组
    for key in history.keys():
        history[key] = np.array(history[key])

    time_axis = np.arange(steps) * dt

    # ================= 使用 Plotly 开始绘制交互式图表 =================
    
    # 创建 3x2 的子图网格
    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=("3D Target Trajectory", "Position Components over Time",
                        "Velocity Magnitude over Time", "Acceleration Magnitude over Time",
                        "Yaw Angle over Time (rad)", "Yaw Rate over Time (rad/s)")
    )

    # 1. 3D 轨迹图 (Row 1, Col 1)
    fig.add_trace(go.Scatter3d(
        x=history['pos'][:, 0], y=history['pos'][:, 1], z=history['pos'][:, 2],
        mode='lines', line=dict(color='purple', width=4), name='Trajectory'
    ), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode='markers', marker=dict(color='black', symbol='x', size=5), name='Spawn Point'
    ), row=1, col=1)

    # 2. X, Y, Z (Row 1, Col 2)
    fig.add_trace(go.Scatter(x=time_axis, y=history['pos'][:, 0], mode='lines', name='X Pos', line=dict(color='red', width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=time_axis, y=history['pos'][:, 1], mode='lines', name='Y Pos', line=dict(color='green', width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=time_axis, y=history['pos'][:, 2], mode='lines', name='Z Pos', line=dict(color='blue', width=2)), row=1, col=2)

    # 3. 速度 (Row 2, Col 1)
    vel_norm = np.linalg.norm(history['vel'], axis=1)
    fig.add_trace(go.Scatter(x=time_axis, y=vel_norm, mode='lines', name='Velocity', line=dict(color='orange', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=[time_axis[0], time_axis[-1]], y=[3.0, 3.0], mode='lines', name='Max Vel', line=dict(color='red', width=2, dash='dash')), row=2, col=1)

    # 4. 加速度 (Row 2, Col 2)
    acc_norm = np.linalg.norm(history['acc'], axis=1)
    fig.add_trace(go.Scatter(x=time_axis, y=acc_norm, mode='lines', name='Acceleration', line=dict(color='magenta', width=2)), row=2, col=2)
    fig.add_trace(go.Scatter(x=[time_axis[0], time_axis[-1]], y=[15.0, 15.0], mode='lines', name='Max Acc', line=dict(color='red', width=2, dash='dash')), row=2, col=2)

    # 5. Yaw 角度 (Row 3, Col 1)
    fig.add_trace(go.Scatter(x=time_axis, y=history['yaw'], mode='lines', name='Yaw Angle', line=dict(color='cyan', width=2)), row=3, col=1)
    # 画出正负 90 度的红虚线界限
    yaw_limit = math.pi / 2
    fig.add_trace(go.Scatter(x=[time_axis[0], time_axis[-1]], y=[yaw_limit, yaw_limit], mode='lines', name='+90° Limit', line=dict(color='red', width=1, dash='dash')), row=3, col=1)
    fig.add_trace(go.Scatter(x=[time_axis[0], time_axis[-1]], y=[-yaw_limit, -yaw_limit], mode='lines', name='-90° Limit', line=dict(color='red', width=1, dash='dash'), showlegend=False), row=3, col=1)

    # 6. Yaw 角速度 (Row 3, Col 2)
    fig.add_trace(go.Scatter(x=time_axis, y=history['yaw_rate'], mode='lines', name='Yaw Rate', line=dict(color='teal', width=2)), row=3, col=2)
    # 画出最大角速度限制
    fig.add_trace(go.Scatter(x=[time_axis[0], time_axis[-1]], y=[2.5, 2.5], mode='lines', name='Max Yaw Rate', line=dict(color='red', width=1, dash='dash')), row=3, col=2)
    fig.add_trace(go.Scatter(x=[time_axis[0], time_axis[-1]], y=[-2.5, -2.5], mode='lines', name='-Max Yaw Rate', line=dict(color='red', width=1, dash='dash'), showlegend=False), row=3, col=2)

    # 布局设置
    fig.update_layout(
        height=1200, width=1400, 
        title_text="Langevin Trajectory Simulation with Yaw Dynamics (Interactive)",
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)'),
        showlegend=True
    )
    
    # 轴标签
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Position (m)", row=1, col=2)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Velocity (m/s)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Acceleration (m/s²)", row=2, col=2)
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="Yaw (rad)", row=3, col=1)
    fig.update_xaxes(title_text="Time (s)", row=3, col=2)
    fig.update_yaxes(title_text="Yaw Rate (rad/s)", row=3, col=2)

    # 显示与保存
    fig.show()
    html_filename = "trajectory_with_yaw.html"
    fig.write_html(html_filename)
    print(f"✅ 包含偏航角动态的交互式图表已生成：{html_filename}")

if __name__ == "__main__":
    run_and_plot(steps=1000, dt=0.02)