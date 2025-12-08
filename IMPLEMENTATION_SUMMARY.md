# 两阶段训练实现总结

## 修改内容

已成功实现两阶段distillation训练模式：

### 阶段1：教师引导阶段
- 使用教师（teacher）action更新环境
- 收集 state、obs、teacher action、student action 到数据集
- 学生网络使用数据集训练

### 阶段2：学生自主阶段  
- 使用学生（student）action更新环境
- 收集 state、obs、teacher action、student action 到数据集
- 学生网络继续使用数据集训练

## 修改的文件

### 1. `/home/frd/Foundation/rsl_rl/rsl_rl/algorithms/distillation.py`

**添加的功能：**
- 构造函数新增参数：
  - `use_two_stage_training`: 是否启用两阶段训练
  - `phase1_iterations`: 第一阶段的迭代次数
  - `training_phase`: 当前训练阶段（1或2）
  - `current_iteration`: 当前迭代计数

- 修改 `act()` 方法：
  - 计算并保存学生和教师的action
  - 根据 `training_phase` 返回相应的action用于环境更新
  
- 修改 `update()` 方法：
  - 增加迭代计数
  - 自动检测并切换到第二阶段

- 新增辅助方法：
  - `switch_to_phase2()`: 切换到第二阶段并打印提示信息
  - `get_training_phase_info()`: 获取当前训练阶段信息

### 2. `/home/frd/IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/distillation_cfg.py`

**添加的配置项：**
```python
use_two_stage_training: bool = False
"""是否启用两阶段训练模式"""

phase1_iterations: int = 500  
"""第一阶段的迭代次数"""
```

### 3. `/home/frd/Foundation/rsl_rl/rsl_rl/runners/on_policy_runner.py`

**添加的日志功能：**
- 在 WandB/TensorBoard 中记录：
  - `Distillation/training_phase`: 当前训练阶段
  - `Distillation/current_iteration`: 当前迭代次数

- 在终端输出中显示：
  - 当前训练阶段（Phase 1/2）
  - 使用的action源（teacher/student actions）
  - 阶段进度

### 4. `/home/frd/Foundation/foundation/tasks/point_ctrl/agents/rsl_rl_ppo_cfg.py`

**更新的配置：**
```python
algorithm = RslRlDistillationAlgorithmCfg(
    num_learning_epochs=4,  
    learning_rate=1e-3,
    max_grad_norm=1.0,
    gradient_length=15,
    class_name="Distillation",
    use_two_stage_training=True,      # 启用两阶段训练
    phase1_iterations=500,             # 前500次迭代使用teacher actions
)
```

## 使用方法

### 启动训练

使用您现有的训练命令即可：

```bash
python foundation/rsl_rl/train.py \
    --num_envs 100 \
    --task distillation \
    --checkpoint logs/rsl_rl/point_ctrl_direct/2025-12-01_18-51-49/best_model.pt \
    --logger wandb \
    --log_project_name Foundation \
    env.robot.spawn.usd_path="/home/frd/Foundation/USD/cf2x.usd"
```

### 预期行为

1. **训练开始（迭代 0-499）**
   - 终端显示："Training Phase: Phase 1 (teacher actions)"
   - 环境使用教师action更新
   - 数据集同时收集教师和学生action
   - 学生网络学习模仿教师

2. **阶段切换（迭代 500）**
   - 自动切换到Phase 2
   - 终端显示醒目的切换提示信息

3. **第二阶段（迭代 500+）**
   - 终端显示："Training Phase: Phase 2 (student actions)"
   - 环境使用学生action更新
   - 数据集继续收集教师和学生action
   - 学生网络继续训练

## 关键特性

1. **无缝切换**: 阶段切换完全自动，无需人工干预
2. **持续学习**: 两个阶段都进行数据收集和网络训练
3. **灵活配置**: 可以通过配置文件调整阶段划分
4. **向后兼容**: 默认禁用，不影响现有训练流程

## 设计优势

1. **稳定的初始训练**: 
   - Phase 1使用教师action，确保环境交互质量
   - 学生可以从高质量轨迹中学习

2. **渐进式过渡**:
   - Phase 2切换到学生action
   - 检验学生的实际表现

3. **完整数据收集**:
   - 两个阶段都收集完整数据（state, obs, teacher action, student action）
   - 保证训练数据的连续性

## 调试信息

如果需要查看更详细的阶段信息，可以在代码中调用：

```python
phase_info = runner.alg.get_training_phase_info()
print(phase_info)
# 输出示例:
# {
#     'use_two_stage_training': True,
#     'training_phase': 1,
#     'current_iteration': 450,
#     'phase1_iterations': 500,
#     'action_source': 'teacher'
# }
```

## 测试建议

1. 运行训练观察第一阶段的表现
2. 在迭代500时观察切换提示信息
3. 比较两个阶段的reward变化
4. 检查WandB日志中的phase指标

## 注意事项

- 确保教师模型已正确加载（通过--checkpoint参数）
- phase1_iterations应该设置为合理的值（建议总迭代数的1/3到1/2）
- 阶段切换时可能会观察到性能波动，这是正常现象
