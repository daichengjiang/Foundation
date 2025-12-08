# 两阶段Distillation训练 - 快速开始

## 🎯 核心变化

原训练模式：
- 全程使用 **student action** 更新环境

新训练模式（两阶段）：
- **Phase 1**: 使用 **teacher action** 更新环境 (前N次迭代)
- **Phase 2**: 使用 **student action** 更新环境 (剩余迭代)
- 两个阶段都收集完整数据：state, obs, teacher action, student action

## ⚡ 快速启用

### 1. 修改配置文件

`foundation/tasks/point_ctrl/agents/rsl_rl_ppo_cfg.py`:

```python
algorithm = RslRlDistillationAlgorithmCfg(
    num_learning_epochs=4,
    learning_rate=1e-3,
    max_grad_norm=1.0,
    gradient_length=15,
    class_name="Distillation",
    use_two_stage_training=True,    # ← 添加这行
    phase1_iterations=500,           # ← 添加这行
)
```

### 2. 运行训练（命令不变）

```bash
python foundation/rsl_rl/train.py \
    --num_envs 100 \
    --task distillation \
    --checkpoint logs/rsl_rl/point_ctrl_direct/2025-12-01_18-51-49/best_model.pt \
    --logger wandb \
    --log_project_name Foundation \
    env.robot.spawn.usd_path="/home/frd/Foundation/USD/cf2x.usd"
```

## 📊 预期输出

### Phase 1 (迭代 0-499)
```
################################################################################
            Learning iteration 450/1500

Training Phase: Phase 1 (teacher actions)  ← 关键信息
Phase Iteration: 450/500 (Phase 1)
```

### 切换点 (迭代 500)
```
================================================================================
================================================================================
  SWITCHING TO PHASE 2: Now using STUDENT actions to update environment
  Iteration: 500
================================================================================
================================================================================
```

### Phase 2 (迭代 500+)
```
################################################################################
            Learning iteration 550/1500

Training Phase: Phase 2 (student actions)  ← 关键信息
Phase Iteration: 550/500 (Phase 1)
```

## 🔧 配置参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `use_two_stage_training` | 是否启用两阶段训练 | `True` / `False` |
| `phase1_iterations` | Phase 1 的迭代次数 | 总迭代数的 1/3 到 1/2 |

**示例**：
- 总迭代 1500 → `phase1_iterations=500` (33%)
- 总迭代 1500 → `phase1_iterations=750` (50%)
- 总迭代 2000 → `phase1_iterations=800` (40%)

## 📖 详细文档

- `TWO_STAGE_TRAINING_README.md` - 完整功能说明
- `IMPLEMENTATION_SUMMARY.md` - 实现细节
- `TWO_STAGE_TRAINING_CONFIG_EXAMPLES.md` - 配置示例
- `VERIFICATION_CHECKLIST.md` - 验证清单

## ❓ 常见问题

**Q: 为什么需要两阶段训练？**
A: Phase 1使用teacher action提供稳定的高质量交互，帮助学生网络快速学习。Phase 2切换到student action检验实际性能。

**Q: 必须使用两阶段吗？**
A: 不是。设置 `use_two_stage_training=False` 或删除该参数即可恢复原始训练模式。

**Q: 如何选择 phase1_iterations？**
A: 建议从总迭代数的40%开始，根据训练效果调整。如果切换后性能下降明显，可以增加这个值。

**Q: 两个阶段的数据都用来训练吗？**
A: 是的。两个阶段都收集数据并训练学生网络，区别只在于用哪个action更新环境。

## 🎓 工作原理

```
每个训练步骤:
1. student_action = student_policy(obs)
2. teacher_action = teacher_policy(privileged_obs)
3. 存储到数据集: (obs, student_action, teacher_action)
4. 选择action更新环境:
   - Phase 1: env.step(teacher_action)  ← 使用教师
   - Phase 2: env.step(student_action)  ← 使用学生
5. 训练学生网络学习模仿教师
```

## ✅ 修改的文件

1. `rsl_rl/rsl_rl/algorithms/distillation.py` - 核心逻辑
2. `IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/distillation_cfg.py` - 配置类
3. `rsl_rl/rsl_rl/runners/on_policy_runner.py` - 日志记录
4. `foundation/tasks/point_ctrl/agents/rsl_rl_ppo_cfg.py` - 任务配置

无需修改其他代码，所有改动向后兼容。
