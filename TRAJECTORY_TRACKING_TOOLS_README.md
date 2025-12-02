# 轨迹跟踪评估工具 - 文件说明

本目录包含用于评估轨迹跟踪控制策略的完整工具链。

## 📦 文件清单

### 主要脚本

1. **`foundation/rsl_rl/play_best_model.py`** ⭐
   - 轨迹跟踪评估的主脚本
   - 功能：
     - 加载best_model.pt
     - 生成Langevin轨迹
     - 实时计算跟踪误差
     - 可视化期望vs实际轨迹
     - 保存统计数据和轨迹数据
   - 用法：见下文"快速开始"

2. **`foundation/rsl_rl/visualize_trajectory.py`** 📊
   - 轨迹数据可视化脚本
   - 功能：
     - 生成3D轨迹对比图
     - 生成2D投影视图
     - 位置/速度误差分析
     - 控制动作可视化
   - 需要：matplotlib, numpy
   - 输入：trajectory_data.npz
   - 输出：5张高质量PNG图表


## 🚀 快速开始

### 第一步：运行评估

**直接使用Python：**
```bash
python foundation/rsl_rl/play_best_model.py --task Isaac-Quadcopter-Point-Ctrl-v0 --checkpoint logs/rsl_rl/your_experiment/best_model.pt \
    --num_envs 4 \
    --max_steps 10000
```

### 第二步：查看结果

评估完成后，检查输出目录：
```bash
ls -l logs/rsl_rl/your_experiment/YYYY-MM-DD_HH-MM-SS_trajectory_tracking/
```

你会看到：
- `tracking_statistics.txt` - 打开查看统计报告
- `tracking_errors.npz` - 所有误差数据
- `trajectory_data.npz` - 完整轨迹数据

### 第三步：生成可视化图表

```bash
python foundation/rsl_rl/visualize_trajectory.py --data_dir logs/rsl_rl/your_experiment/YYYY-MM-DD_HH-MM-SS_trajectory_tracking
```

图表保存在 `plots/` 子目录中。

## 📊 评估指标说明

### Position Error (位置误差)
- 无人机实际位置与期望轨迹点之间的欧几里得距离
- 单位：米 (m)
- 关键统计量：Mean, Std, Median, Max, 95th percentile

### Velocity Error (速度误差)
- 实际速度与期望速度之间的差异
- 单位：米/秒 (m/s)
- 反映动态跟踪性能

### 评估维度
- **总体统计**：所有环境、所有时间步的综合表现
- **各环境统计**：每个环境的独立性能（检测一致性）
- **时间序列**：误差随时间的变化（检测稳定性）

## 🎯 使用场景

### 场景1：模型验证
**目标**：快速检查模型是否正常工作
```bash
./scripts/eval_trajectory_tracking.sh your_model.pt --num_envs 1 --max_steps 2000
```
**检查**：观察Isaac Sim中的可视化，误差是否在合理范围

### 场景2：性能评估
**目标**：获得可靠的性能统计
```bash
./scripts/eval_trajectory_tracking.sh your_model.pt --num_envs 16 --max_steps 20000
```
**检查**：tracking_statistics.txt中的平均误差和标准差

### 场景3：论文图表
**目标**：生成高质量的可视化图表
```bash
# 1. 运行评估并保存轨迹
python foundation/rsl_rl/play_best_model.py \
    --task Isaac-Quadcopter-Point-Ctrl-v0 \
    --checkpoint your_model.pt \
    --num_envs 1 \
    --max_steps 10000 \
    --save_trajectory

# 2. 生成图表
python foundation/rsl_rl/visualize_trajectory.py \
    --data_dir logs/.../trajectory_tracking \
    --save_plots
```
**输出**：5张300 DPI的PNG图表

### 场景4：对比多个模型
**目标**：选择最佳checkpoint
```bash
for model in model_1000.pt model_2000.pt best_model.pt; do
    ./scripts/eval_trajectory_tracking.sh logs/rsl_rl/exp/$model --num_envs 8
done
```
**对比**：各模型的tracking_statistics.txt

### 场景5：泛化性测试
**目标**：测试不同随机种子下的稳定性
```bash
for seed in 42 123 456 789; do
    python foundation/rsl_rl/play_best_model.py \
        --task Isaac-Quadcopter-Point-Ctrl-v0 \
        --checkpoint your_model.pt \
        --seed $seed \
        --num_envs 16 \
        --max_steps 10000
done
```
**分析**：对比不同seed下的误差分布

## 🔧 配置说明

### Langevin轨迹参数
轨迹由环境中的Langevin动力学生成，参数在 `quad_point_ctrl_env_single_dense.py` 中：

```python
self._langevin_dt = 0.01          # 积分步长
self._langevin_friction = 1.0     # 阻尼系数（gamma）
self._langevin_omega = 2.0        # 振荡频率（omega）
self._langevin_sigma = 1.0        # 噪声强度（sigma）
self._langevin_alpha = 1.0        # 平滑系数（alpha）
```

**影响**：
- `friction` 越大，轨迹越趋向于静止
- `omega` 越大，振荡越快（轨迹更复杂）
- `sigma` 越大，随机性越强
- `alpha` 控制平滑程度（0=无平滑，1=完全平滑）

### 评估脚本参数
关键参数在 `play_best_model.py` 中：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_envs` | 4 | 并行环境数 |
| `--max_steps` | 10000 | 最大运行步数 |
| `--save_trajectory` | True | 保存轨迹数据 |
| `--video` | False | 录制视频 |
| `--realtime` | False | 实时运行 |

## 📈 数据格式说明

### tracking_statistics.txt
文本格式的统计报告，包含：
- 基本信息（checkpoint路径、任务名、参数）
- 总体统计（所有环境的综合性能）
- 各环境统计（每个环境的独立性能）

### tracking_errors.npz
NumPy压缩数组，包含：
```python
data = np.load('tracking_errors.npz')
position_errors = data['position_errors']  # shape: (total_steps,)
velocity_errors = data['velocity_errors']  # shape: (total_steps,)
```

### trajectory_data.npz
完整的轨迹数据（仅环境0），包含：
```python
data = np.load('trajectory_data.npz')
desired_pos = data['desired_pos']      # shape: (N, 3)
actual_pos = data['actual_pos']        # shape: (N, 3)
desired_vel = data['desired_vel']      # shape: (N, 3)
actual_vel = data['actual_vel']        # shape: (N, 3)
position_error = data['position_error'] # shape: (N,)
velocity_error = data['velocity_error'] # shape: (N,)
actions = data['actions']              # shape: (N, 4)
timestamps = data['timestamps']        # shape: (N,)
```

## 🎨 可视化说明

### 1. 3D Trajectory
- 绿色线：期望轨迹
- 蓝色线：实际轨迹
- 绿色球：起点（期望）
- 蓝色球：起点（实际）
- 红色方：终点

### 2. 2D Projections
- 三个视图：XY（俯视）、XZ（侧视）、YZ（正视）
- 颜色同3D图

### 3. Position Errors
- 上图：误差随时间变化
- 下图：误差分布直方图

### 4. Velocity Comparison
- 三个子图：Vx, Vy, Vz
- 绿色：期望
- 蓝色：实际

### 5. Actions
- 四个子图：四个电机的控制指令
- 范围：[0, 1]

## 💡 最佳实践

### 1. 评估前检查
- ✅ 确认best_model.pt路径正确
- ✅ 确认任务名称与训练时一致
- ✅ 确认环境配置未修改

### 2. 评估中观察
- 👀 在Isaac Sim中实时观察轨迹
- 👀 注意绿色箭头（期望）和蓝色箭头（实际）的距离
- 👀 检查终端输出的实时误差

### 3. 评估后分析
- 📊 查看tracking_statistics.txt
- 📊 生成可视化图表
- 📊 对比不同模型/参数的结果

### 4. 报告撰写
- 📝 使用平均误差±标准差表示性能
- 📝 提供95th percentile作为最坏情况参考
- 📝 使用生成的图表展示轨迹对比
- 📝 说明评估环境数和总步数

## 🐛 常见问题

### Q1: 为什么看不到绿色箭头？
A: 确保：
1. 没有使用 `--headless`
2. `debug_vis = True`（脚本自动设置）
3. 在Isaac Sim窗口中，可能需要调整相机视角

### Q2: 误差突然很大是什么原因？
A: 可能原因：
1. Langevin轨迹超出阈值（查看 `position_exceeded_langevin`）
2. 数值不稳定（查看 `numerical_is_unstable`）
3. 模型未正确加载

### Q3: 如何加快评估速度？
A: 
1. 使用 `--headless` 禁用渲染
2. 增加 `--num_envs` 并行评估
3. 减少 `--max_steps`

### Q4: 可视化图表不清晰？
A: 图表默认300 DPI，如需更高分辨率，修改 `visualize_trajectory.py` 中的 `dpi` 参数。

## 📞 支持

遇到问题？检查：
1. **QUICK_REFERENCE.md** - 快速参考
2. **PLAY_BEST_MODEL_README.md** - 详细文档
3. 终端输出的错误信息
4. Isaac Sim日志

## 🔄 更新日志

- **2025-12-01**: 初始版本
  - 轨迹跟踪评估脚本
  - 可视化工具
  - 完整文档

---

**祝评估顺利！** 🚁✨
