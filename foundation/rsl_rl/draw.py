# import pandas as pd
# import matplotlib.pyplot as plt

# # 读取CSV文件，跳过第一行数据，手动指定列名
# column_names = ['roll', 'pitch', 'yaw', 'roll_es', 'pitch_es', 'yaw_es']
# data = pd.read_csv('/home/zjr/CrazyE2E/logs/rsl_rl/point_ctrl_direct/v=3/policy_actions.csv', header=1, names=column_names)  # header=1跳过第一行数据，names指定列名

# # 获取数据的行数（即X轴的范围）
# num_rows = len(data)

# # 创建图形
# plt.figure(figsize=(10, 6))

# # 设置变量的列名
# columns = ['roll', 'pitch', 'yaw', 'roll_es', 'pitch_es', 'yaw_es']

# # 逐列绘制每个变量
# for column in columns:
#     plt.plot(range(num_rows), data[column], label=column)  # 横坐标使用 0 到 num_rows-1

# # 添加标题和标签
# plt.title('Variables over Time')
# plt.xlabel('Index')
# plt.ylabel('Values')

# # 显示图例
# plt.legend()

# # 显示图形
# plt.grid(True)
# plt.tight_layout()
# plt.show()



import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
column_names = ['action_0', 'action_1', 'action_2', 'thrust', 'mean_action_0', 'mean_action_1', 'mean_action_2', 'mean_thrust', 'roll', 'pitch', 'yaw']
data = pd.read_csv('/home/chenyog/ImageE2E/logs/rsl_rl/point_ctrl_direct/policy_actions.csv', header=0, names=column_names)

num_rows = len(data)
time = [i * 0.01 for i in range(num_rows)]  # 时间轴，每步0.01秒

# 创建四个子图
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('All Four Dimensions over Time', fontsize=16)

# 绘制 action_0 (roll) vs mean_action_0 vs 实际 roll
axes[0, 0].plot(time, data['action_0'], label='action_0 (roll)', color='blue', linewidth=2)
axes[0, 0].plot(time, data['mean_action_0'], label='mean_action_0 (roll)', color='red', linestyle='--', linewidth=2)
axes[0, 0].plot(time, data['roll'], label='roll (actual)', color='green', linestyle=':', linewidth=2, marker='o', markersize=1)
axes[0, 0].set_title('Action 0 (Roll) vs Mean Action vs Actual Roll')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Roll Value (degrees)')
axes[0, 0].legend()
axes[0, 0].grid(True)

# 绘制 action_1 (pitch) vs mean_action_1 vs 实际 pitch
axes[0, 1].plot(time, data['action_1'], label='action_1 (pitch)', color='blue', linewidth=2)
axes[0, 1].plot(time, data['mean_action_1'], label='mean_action_1 (pitch)', color='red', linestyle='--', linewidth=2)
axes[0, 1].plot(time, data['pitch'], label='pitch (actual)', color='green', linestyle=':', linewidth=2, marker='s', markersize=1)
axes[0, 1].set_title('Action 1 (Pitch) vs Mean Action vs Actual Pitch')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Pitch Value (degrees)')
axes[0, 1].legend()
axes[0, 1].grid(True)

# 绘制 action_2 (yaw) vs mean_action_2 vs 实际 yaw
axes[1, 0].plot(time, data['action_2'], label='action_2 (yaw)', color='blue', linewidth=2)
axes[1, 0].plot(time, data['mean_action_2'], label='mean_action_2 (yaw)', color='red', linestyle='--', linewidth=2)
axes[1, 0].plot(time, data['yaw'], label='yaw (actual)', color='green', linestyle=':', linewidth=2, marker='^', markersize=1)
axes[1, 0].set_title('Action 2 (Yaw) vs Mean Action vs Actual Yaw')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Yaw Value (degrees)')
axes[1, 0].legend()
axes[1, 0].grid(True)

# 绘制 thrust vs mean_thrust (第四维推力)
axes[1, 1].plot(time, data['thrust'], label='thrust (normalized)', color='blue', linewidth=2)
axes[1, 1].plot(time, data['mean_thrust'], label='mean_thrust (normalized)', color='red', linestyle='--', linewidth=2)
axes[1, 1].set_title('Thrust vs Mean Thrust (4th Dimension)')
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Thrust Value (0-1)')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()