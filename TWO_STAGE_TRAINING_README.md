# 两阶段训练模式说明

## 概述

现在的distillation训练支持两阶段训练模式：

- **第一阶段（Phase 1）**：使用教师（teacher）action更新环境，同时收集教师和学生的action存入数据集用于训练学生网络
- **第二阶段（Phase 2）**：使用学生（student）action更新环境，继续收集数据并训练学生网络

## 配置方法

### 1. 在配置文件中启用两阶段训练

在 `foundation/tasks/point_ctrl/agents/rsl_rl_ppo_cfg.py` 中：

```python
class QuadcopterDistillationRunnerCfg(RslRlOnPolicyRunnerCfg):
    # ... 其他配置 ...
    
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=4,  
        learning_rate=1e-3,
        max_grad_norm=1.0,
        gradient_length=15,
        class_name="Distillation",
        # 启用两阶段训练
        use_two_stage_training=True,
        phase1_iterations=500,  # 第一阶段持续500个迭代
    )
```

### 2. 配置参数说明

- `use_two_stage_training`: 布尔值，是否启用两阶段训练模式
  - `True`: 启用两阶段训练
  - `False`: 默认模式（只使用学生action更新环境）

- `phase1_iterations`: 整数，第一阶段的迭代次数
  - 示例：500 表示前500个迭代使用教师action
  - 超过这个次数后，自动切换到第二阶段

## 训练命令

使用相同的训练命令即可：

```bash
python foundation/rsl_rl/train.py \
    --num_envs 100 \
    --task distillation \
    --checkpoint logs/rsl_rl/point_ctrl_direct/2025-12-01_18-51-49/best_model.pt \
    --logger wandb \
    --log_project_name Foundation \
    env.robot.spawn.usd_path="/home/frd/Foundation/USD/cf2x.usd"
```

## 训练过程

### 阶段切换

当训练达到 `phase1_iterations` 次迭代时，系统会自动切换到第二阶段，并在终端显示：

```
================================================================================
================================================================================
  SWITCHING TO PHASE 2: Now using STUDENT actions to update environment
  Iteration: 500
================================================================================
================================================================================
```

### 日志信息

训练过程中，日志会显示当前的训练阶段信息：

```
################################################################################
            Learning iteration 450/1500

Computation: 1234 steps/s (collection: 0.081s, learning 0.000s)
Mean action noise std: 0.50
Mean behavior loss: 0.1234
Mean reward: 12.34
Mean episode length: 100.00
--------------------------------------------------------------------------------
Training Phase: Phase 1 (teacher actions)
Phase Iteration: 450/500 (Phase 1)
--------------------------------------------------------------------------------
Total timesteps: 450000
Iteration time: 0.08s
Time elapsed: 00:36:00
ETA: 01:24:00
```

### WandB日志

在WandB中，会记录以下指标：

- `Distillation/training_phase`: 当前训练阶段（1或2）
- `Distillation/current_iteration`: 当前迭代次数
- `Loss/behavior`: 行为克隆损失

## 工作原理

### 数据收集

无论在哪个阶段，系统都会：
1. 根据学生观测计算学生action
2. 根据教师观测计算教师action
3. 将两者都存入数据集

### 环境更新

- **Phase 1**: 使用教师action执行 `env.step(teacher_action)`
- **Phase 2**: 使用学生action执行 `env.step(student_action)`

### 网络训练

无论在哪个阶段，学生网络都使用收集到的数据进行训练，学习模仿教师的动作。

## 修改的文件

1. **算法核心逻辑**
   - `rsl_rl/rsl_rl/algorithms/distillation.py`
     - 添加了两阶段训练参数
     - 修改了 `act()` 方法以根据阶段返回不同action
     - 添加了 `switch_to_phase2()` 和 `get_training_phase_info()` 方法

2. **配置类**
   - `IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/distillation_cfg.py`
     - 添加了 `use_two_stage_training` 和 `phase1_iterations` 配置项

3. **训练器**
   - `rsl_rl/rsl_rl/runners/on_policy_runner.py`
     - 添加了阶段信息的日志记录

4. **任务配置**
   - `foundation/tasks/point_ctrl/agents/rsl_rl_ppo_cfg.py`
     - 启用了两阶段训练配置

## 使用建议

1. **第一阶段迭代次数**：建议设置为总迭代次数的1/3到1/2
   - 例如：总共1500次迭代，可以设置phase1_iterations=500或750

2. **教师策略质量**：确保教师策略性能良好，因为第一阶段会使用教师action
   - 可以通过 `--checkpoint` 参数加载训练好的教师模型

3. **监控切换点**：注意观察阶段切换后的性能变化
   - 如果切换后性能急剧下降，可能需要延长第一阶段

## 禁用两阶段训练

如果想恢复到原来的单阶段训练模式，只需在配置中设置：

```python
algorithm = RslRlDistillationAlgorithmCfg(
    # ... 其他参数 ...
    use_two_stage_training=False,  # 禁用两阶段训练
)
```

或者直接删除这两个参数（默认就是禁用的）。
