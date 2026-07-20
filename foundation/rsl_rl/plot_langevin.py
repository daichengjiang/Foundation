import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ================= 参数设置 (与之前完全相同) =================
dt = 0.02
steps = 2000
k_pos = np.array([1.0, 1.0, 2.0])  
k_vel = 1.5                        
acc_inertia = 0.1                  
noise_scale = 10.0                 
max_vel = 3.0                      
max_acc = 15.0                     
z_noise_attenuation = 0.3          

pos = np.array([0.0, 0.0, 0.0])
vel = np.array([0.0, 0.0, 0.0])
acc = np.array([0.0, 0.0, 0.0])
spawn_pos = np.array([0.0, 0.0, 0.0])

trajectory = []

np.random.seed(42)  
for _ in range(steps):
    vel_next = vel + acc * dt
    vel_norm = np.linalg.norm(vel_next)
    scale = min(max_vel / (vel_norm + 1e-6), 1.0)
    vel_next = vel_next * scale
    
    pos_next = pos + vel * dt
    pos_err = pos - spawn_pos
    
    noise = np.random.randn(3) * noise_scale
    noise[2] *= z_noise_attenuation
    
    force_total = noise - k_pos * pos_err - k_vel * vel
    
    acc_next = (1.0 - acc_inertia) * acc + acc_inertia * force_total
    acc_norm = np.linalg.norm(acc_next)
    if acc_norm > max_acc:
        acc_next = (acc_next / acc_norm) * max_acc
        
    acc = acc_next
    vel = vel_next
    pos = pos_next
    trajectory.append(pos.copy())

trajectory = np.array(trajectory)

# ================= 修复版的 3D 绘图与保存 =================
fig = plt.figure(figsize=(10, 8), dpi=300) 
ax = fig.add_subplot(111, projection='3d')

x = trajectory[:, 0]
y = trajectory[:, 1]
z = trajectory[:, 2]

# 绘制中心原点 (出生点)
ax.scatter([0], [0], [0], color='red', s=100, label='Spawn Point', edgecolors='black', zorder=5)

# 颜色渐变
points = np.array([x, y, z]).T.reshape(-1, 1, 3)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
cmap = plt.get_cmap('cool')
colors = cmap(np.linspace(0, 1, len(segments)))

for i in range(len(segments)):
    ax.plot(segments[i, :, 0], segments[i, :, 1], segments[i, :, 2], color=colors[i], linewidth=1.5, alpha=0.8)

# --- [修复核心 1：动态扩展轴边界，留出 15% 的呼吸空间] ---
margin = 0.15 
x_range = x.max() - x.min()
y_range = y.max() - y.min()
z_range = z.max() - z.min()

ax.set_xlim([x.min() - x_range * margin, x.max() + x_range * margin])
ax.set_ylim([y.min() - y_range * margin, y.max() + y_range * margin])
ax.set_zlim([z.min() - z_range * margin, z.max() + z_range * margin])

# 设置视角与标签
ax.view_init(elev=30, azim=45) 
ax.set_xlabel('X Position (m)', fontsize=12, labelpad=12)
ax.set_ylabel('Y Position (m)', fontsize=12, labelpad=12)
ax.set_zlabel('Z Position (m)', fontsize=12, labelpad=12)

# 网格背景调整
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(True, linestyle='--', alpha=0.6)

# --- [修复核心 2：移除紧凑布局，改用带内边距的保存方式] ---
# 注意这里不要写 plt.tight_layout() 了！
# pad_inches=0.3 会强制在图片最外围加一圈 0.3 英寸的白边，保证坐标轴绝对不会被裁掉
plt.savefig('Figure_3_2_Langevin_Trajectory_Fixed.pdf', format='pdf', bbox_inches='tight', pad_inches=0.4)
plt.savefig('Figure_3_2_Langevin_Trajectory_Fixed.png', format='png', bbox_inches='tight', pad_inches=0.4)

plt.show()