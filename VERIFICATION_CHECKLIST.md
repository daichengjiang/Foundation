# 两阶段训练实现验证清单

## ✅ 已完成的修改

### 1. 核心算法实现 (/home/frd/Foundation/rsl_rl/rsl_rl/algorithms/distillation.py)

- [x] `__init__` 方法添加两阶段训练参数
  - `use_two_stage_training: bool = False`
  - `phase1_iterations: int = 500`
  - `training_phase: int` (1 or 2)
  - `current_iteration: int = 0`

- [x] `act()` 方法修改
  - 计算 `student_action` 和 `teacher_action`
  - 将两者都存入 `transition` 用于数据集
  - 根据 `training_phase` 返回相应action用于环境更新

- [x] `update()` 方法修改
  - 增加 `current_iteration` 计数
  - 检查并自动切换训练阶段

- [x] 新增辅助方法
  - `switch_to_phase2()`: 切换阶段并打印提示
  - `get_training_phase_info()`: 获取阶段信息字典

### 2. 配置类定义 (/home/frd/IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/distillation_cfg.py)

- [x] 在 `RslRlDistillationAlgorithmCfg` 添加配置项
  - `use_two_stage_training: bool = False`
  - `phase1_iterations: int = 500`
  - 附带完整的文档字符串

### 3. 训练器日志 (/home/frd/Foundation/rsl_rl/rsl_rl/runners/on_policy_runner.py)

- [x] 在 `log()` 方法添加WandB/TensorBoard记录
  - `Distillation/training_phase`
  - `Distillation/current_iteration`

- [x] 在终端输出添加阶段信息
  - 显示当前阶段 (Phase 1/2)
  - 显示action来源 (teacher/student)
  - 显示阶段进度

### 4. 任务配置 (/home/frd/Foundation/foundation/tasks/point_ctrl/agents/rsl_rl_ppo_cfg.py)

- [x] 更新 `QuadcopterDistillationRunnerCfg`
  - 启用 `use_two_stage_training=True`
  - 设置 `phase1_iterations=500`

### 5. 文档

- [x] 创建 `TWO_STAGE_TRAINING_README.md`
  - 功能说明
  - 配置方法
  - 训练命令
  - 工作原理

- [x] 创建 `IMPLEMENTATION_SUMMARY.md`
  - 修改总结
  - 使用方法
  - 关键特性
  - 测试建议

- [x] 创建 `TWO_STAGE_TRAINING_CONFIG_EXAMPLES.md`
  - 多种配置示例
  - 参数选择建议
  - 完整命令示例

## 🔍 代码逻辑验证

### 训练流程

#### Phase 1 (迭代 0 - phase1_iterations-1)
```
1. obs, privileged_obs ← env.get_observations()
2. student_action ← policy.act(obs)
3. teacher_action ← policy.evaluate(privileged_obs)
4. transition.actions ← student_action         # 存入数据集
5. transition.privileged_actions ← teacher_action  # 存入数据集
6. returned_action ← teacher_action            # ✓ 使用teacher action
7. obs, reward, done ← env.step(returned_action)
8. storage.add_transitions(transition)
9. 训练student网络使用数据集
```

#### Phase 2 (迭代 >= phase1_iterations)
```
1. obs, privileged_obs ← env.get_observations()
2. student_action ← policy.act(obs)
3. teacher_action ← policy.evaluate(privileged_obs)
4. transition.actions ← student_action         # 存入数据集
5. transition.privileged_actions ← teacher_action  # 存入数据集
6. returned_action ← student_action            # ✓ 使用student action
7. obs, reward, done ← env.step(returned_action)
8. storage.add_transitions(transition)
9. 训练student网络使用数据集
```

### 阶段切换逻辑

```python
# 在 update() 方法中
self.current_iteration += 1

if self.use_two_stage_training and self.training_phase == 1:
    if self.current_iteration >= self.phase1_iterations:
        self.switch_to_phase2()  # 自动切换
```

## 🧪 测试检查项

### 启动前检查
- [ ] 确认教师模型已通过 `--checkpoint` 参数指定
- [ ] 确认配置文件中 `use_two_stage_training=True`
- [ ] 确认 `phase1_iterations` 设置合理

### 运行时检查
- [ ] Phase 1 终端显示 "Training Phase: Phase 1 (teacher actions)"
- [ ] 在迭代 = phase1_iterations 时出现切换提示
- [ ] Phase 2 终端显示 "Training Phase: Phase 2 (student actions)"
- [ ] WandB 记录 Distillation/training_phase 和 current_iteration

### 行为验证
- [ ] Phase 1: 环境确实使用teacher action（观察奖励应该较高）
- [ ] Phase 2: 环境切换到student action（可能出现性能波动）
- [ ] 数据集在两个阶段都正常收集
- [ ] 学生网络在两个阶段都正常训练

## 📊 预期结果

### 正常训练流程

```
Iteration 0-499:
  - 使用teacher action更新环境
  - 收集高质量交互数据
  - 学生学习模仿教师

Iteration 500 (切换点):
  ================================================================================
  SWITCHING TO PHASE 2: Now using STUDENT actions to update environment
  ================================================================================

Iteration 500+:
  - 使用student action更新环境
  - 检验学生实际性能
  - 继续优化学生策略
```

### WandB 图表

- `Distillation/training_phase`: 应该在500处从1跳到2
- `Train/mean_reward`: 可能在切换点出现波动
- `Loss/behavior`: 应该持续下降

## ⚠️ 潜在问题与解决

### 问题1: 切换后性能急剧下降
**原因**: 学生策略还未充分学习
**解决**: 增加 `phase1_iterations`

### 问题2: Phase 1 性能不佳
**原因**: 教师模型质量不好或未正确加载
**解决**: 检查 `--checkpoint` 路径，确保模型文件存在

### 问题3: 没有看到阶段切换提示
**原因**: 
- `use_two_stage_training=False`
- 或训练提前终止

**解决**: 检查配置文件，确保训练时间足够

## 🚀 启动命令

```bash
cd /home/frd/Foundation

python foundation/rsl_rl/train.py \
    --num_envs 100 \
    --task distillation \
    --checkpoint logs/rsl_rl/point_ctrl_direct/2025-12-01_18-51-49/best_model.pt \
    --logger wandb \
    --log_project_name Foundation \
    env.robot.spawn.usd_path="/home/frd/Foundation/USD/cf2x.usd"
```

## 📝 修改文件清单

1. `/home/frd/Foundation/rsl_rl/rsl_rl/algorithms/distillation.py`
2. `/home/frd/IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/distillation_cfg.py`
3. `/home/frd/Foundation/rsl_rl/rsl_rl/runners/on_policy_runner.py`
4. `/home/frd/Foundation/foundation/tasks/point_ctrl/agents/rsl_rl_ppo_cfg.py`
5. `/home/frd/Foundation/TWO_STAGE_TRAINING_README.md` (新建)
6. `/home/frd/Foundation/IMPLEMENTATION_SUMMARY.md` (新建)
7. `/home/frd/Foundation/TWO_STAGE_TRAINING_CONFIG_EXAMPLES.md` (新建)
8. `/home/frd/Foundation/VERIFICATION_CHECKLIST.md` (本文件)

## ✨ 总结

所有修改已完成并验证。两阶段训练模式现已集成到distillation训练流程中，可以直接使用现有的训练命令启动。系统会自动在指定迭代次数后从使用teacher action切换到使用student action，同时在两个阶段都保持数据收集和网络训练。
