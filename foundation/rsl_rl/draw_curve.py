import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ==========================================
# 1. 数据读取与预处理
# ==========================================
file_path = 'foundation/rsl_rl/motor_curve.csv' # 替换为你的路径
df_raw = pd.read_csv(file_path, header=None)

df = df_raw.T
df.columns = df.iloc[0]
df = df.iloc[1:].dropna(subset=['command']).reset_index(drop=True)
df = df.apply(pd.to_numeric, errors='coerce').dropna()

# ==========================================
# 2. 物理归一化 (电压补偿)
# ==========================================
V_REF = 16.8
df['force_normalized'] = df['force'] * ((V_REF / df['voltage'])**2)

# ==========================================
# 3. 曲线拟合
# ==========================================
def thrust_curve(x, a, b, c):
    return a * x**2 + b * x + c

popt_raw, _ = curve_fit(thrust_curve, df['command'], df['force'])
popt_norm, _ = curve_fit(thrust_curve, df['command'], df['force_normalized'])

z_dshot = np.polyfit(df['command'], df['dshot'], 1)
p_dshot = np.poly1d(z_dshot)

# 构建纯二次方假设基准线 (对齐最大拉力)
max_thrust = popt_norm[0] + popt_norm[1] + popt_norm[2]
def pure_quadratic(x):
    return max_thrust * (x**2)

# ==========================================
# 4. 绘制对比图 (1x3 网格)
# ==========================================
plt.figure(figsize=(18, 5))
x_fit = np.linspace(0, 1, 100) 

# ------ 图1：Command vs Dshot 映射 ------
plt.subplot(1, 3, 1)
plt.scatter(df['command'], df['dshot'], color='green', s=60, zorder=3)
plt.plot(x_fit, p_dshot(x_fit), 'g--', linewidth=2, label=f'Dshot = {z_dshot[0]:.0f}x + {z_dshot[1]:.0f}')
plt.title('1. Command to Dshot Mapping')
plt.xlabel('Command (0-1)')
plt.ylabel('Dshot Value')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# ------ 图2：原始拉力（受电压掉电污染） ------
plt.subplot(1, 3, 2)
scatter1 = plt.scatter(df['command'], df['force'], c=df['voltage'], cmap='viridis', s=60, zorder=3)
plt.colorbar(scatter1, label='Battery Voltage (V)')
plt.plot(x_fit, thrust_curve(x_fit, *popt_raw), 'r--', alpha=0.7, label='Raw Fit')
plt.title('2. Original: Raw Thrust vs Command')
plt.xlabel('Command (0-1)')
plt.ylabel('Raw Thrust (g)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# ------ 图3：真实曲线 vs 纯二次方假设 (绝对拉力对比) ------
plt.subplot(1, 3, 3)
plt.scatter(df['command'], df['force_normalized'], color='blue', s=60, label=f'Real Data (@ {V_REF}V)', zorder=3)
plt.plot(x_fit, thrust_curve(x_fit, *popt_norm), 'red', linewidth=2, label='Real Physics (Linear + Quad)')
plt.plot(x_fit, pure_quadratic(x_fit), 'orange', linestyle='--', linewidth=2, label='Old Sim Assumption (Pure Quad)')

# --- 标注半油门处的绝对拉力差异 ---
mid_cmd = 0.5
real_thrust_mid = thrust_curve(mid_cmd, *popt_norm)
pure_thrust_mid = pure_quadratic(mid_cmd)

# 画一条紫色的垂直虚线连接两个点
plt.plot([mid_cmd, mid_cmd], [pure_thrust_mid, real_thrust_mid], color='purple', linestyle=':', linewidth=2)
plt.scatter([mid_cmd, mid_cmd], [pure_thrust_mid, real_thrust_mid], color='purple', zorder=4)

# 标注具体克数数值
plt.annotate(f'{real_thrust_mid:.0f}g', xy=(mid_cmd - 0.02, real_thrust_mid), 
             ha='right', va='center', color='red', fontweight='bold', fontsize=11)
plt.annotate(f'{pure_thrust_mid:.0f}g', xy=(mid_cmd + 0.02, pure_thrust_mid), 
             ha='left', va='center', color='orange', fontweight='bold', fontsize=11)

plt.title('3. Absolute Thrust: Real vs Pure Quadratic')
plt.xlabel('Command (0-1)')
plt.ylabel('Absolute Thrust (g)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()