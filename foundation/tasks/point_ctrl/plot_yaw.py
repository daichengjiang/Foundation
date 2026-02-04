import numpy as np
import matplotlib.pyplot as plt

def simulate_constrained_yaw_langevin(duration=100.0, dt=0.01):
    """
    模拟带回复力和硬限位的 Yaw 轨迹生成
    """
    steps = int(duration / dt)
    
    # --- 参数设置 (与你的代码完全一致) ---
    yaw_limit = np.pi / 2  # 限制在 +/- 90 度 (1.57 rad)
    yaw_k_pos = 0.2        # 弹簧刚度 (回复力)
    yaw_k_vel = 0.5        # 阻尼
    yaw_noise_scale = 2.0  # 噪声强度
    max_yaw_rate = 2.5     # 最大角速度
    
    # 状态初始化
    current_yaw = 0.0
    current_yaw_rate = 0.0
    
    # 记录数据
    times = []
    yaws = []
    yaw_rates = []
    
    for i in range(steps):
        times.append(i * dt)
        yaws.append(current_yaw)
        yaw_rates.append(current_yaw_rate)
        
        # 1. 生成随机扰动
        noise = np.random.randn() * yaw_noise_scale
        
        # 2. 动力学更新 (加速度 = 噪声 - 阻尼 - 弹簧回复)
        yaw_acc = noise - yaw_k_vel * current_yaw_rate - yaw_k_pos * current_yaw
        
        # 3. 积分更新角速度
        next_yaw_rate = current_yaw_rate + yaw_acc * dt
        
        # 4. 限制角速度 (物理能力限制)
        next_yaw_rate = np.clip(next_yaw_rate, -max_yaw_rate, max_yaw_rate)
        
        # 5. 积分更新角度
        next_yaw = current_yaw + next_yaw_rate * dt
        
        # 6. 硬截断与边界处理
        over_max = next_yaw > yaw_limit
        under_min = next_yaw < -yaw_limit
        
        if over_max or under_min:
            # 截断角度
            next_yaw = np.clip(next_yaw, -yaw_limit, yaw_limit)
            
            # 撞墙处理：速度归零
            next_yaw_rate = 0.0
            
        # 更新状态
        current_yaw = next_yaw
        current_yaw_rate = next_yaw_rate

    return np.array(times), np.array(yaws), np.array(yaw_rates)

# --- 运行模拟 ---
t, yaw, yaw_rate = simulate_constrained_yaw_langevin(duration=100.0)

# --- 绘图 ---
plt.figure(figsize=(12, 8))

# 1. Yaw 角度
plt.subplot(2, 1, 1)
plt.plot(t, np.degrees(yaw), label='Yaw Angle (deg)', color='blue', linewidth=1.5)
plt.axhline(90, color='red', linestyle='--', linewidth=2, label='Limit (+90°)')
plt.axhline(-90, color='red', linestyle='--', linewidth=2, label='Limit (-90°)')
plt.axhline(0, color='black', linestyle=':', alpha=0.3)
plt.title('Constrained Yaw Trajectory (Target: 0°, Limit: ±90°)', fontsize=14)
plt.ylabel('Yaw Angle (degrees)', fontsize=12)
plt.ylim(-110, 110) #稍微留点余量以便观察
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')

# 2. Yaw 角速度
plt.subplot(2, 1, 2)
plt.plot(t, yaw_rate, label='Yaw Rate (rad/s)', color='orange', linewidth=1.5)
plt.axhline(1.5, color='red', linestyle='--', alpha=0.5, label='Max Rate (±1.5)')
plt.axhline(-1.5, color='red', linestyle='--', alpha=0.5)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Yaw Rate (rad/s)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()