# 两阶段训练配置示例

## 示例1：启用两阶段训练（推荐）

```python
# foundation/tasks/point_ctrl/agents/rsl_rl_ppo_cfg.py

class QuadcopterDistillationRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 100
    max_iterations = 1500
    save_interval = 100
    experiment_name = "distillation"
    empirical_normalization = True
    
    policy = QuadcopterDistillationPolicyCfg()
    
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=4,  
        learning_rate=1e-3,
        max_grad_norm=1.0,
        gradient_length=15,
        class_name="Distillation",
        
        # 启用两阶段训练
        use_two_stage_training=True,
        phase1_iterations=500,  # 前500次迭代使用teacher action
    )
```

**效果：**
- 迭代 0-499: 使用teacher action更新环境
- 迭代 500+: 使用student action更新环境

---

## 示例2：禁用两阶段训练（原始模式）

```python
class QuadcopterDistillationRunnerCfg(RslRlOnPolicyRunnerCfg):
    # ... 其他配置 ...
    
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=4,  
        learning_rate=1e-3,
        max_grad_norm=1.0,
        gradient_length=15,
        class_name="Distillation",
        
        # 禁用两阶段训练（默认行为）
        use_two_stage_training=False,
    )
```

**效果：**
- 全程使用student action更新环境（原始训练模式）

---

## 示例3：更长的第一阶段

```python
class QuadcopterDistillationRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 100
    max_iterations = 2000  # 增加总迭代次数
    save_interval = 100
    experiment_name = "distillation"
    empirical_normalization = True
    
    policy = QuadcopterDistillationPolicyCfg()
    
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=4,  
        learning_rate=1e-3,
        max_grad_norm=1.0,
        gradient_length=15,
        class_name="Distillation",
        
        # 使用更长的第一阶段
        use_two_stage_training=True,
        phase1_iterations=1000,  # 前1000次迭代使用teacher action
    )
```

**效果：**
- 迭代 0-999: 使用teacher action更新环境（50%的训练时间）
- 迭代 1000+: 使用student action更新环境

---

## 示例4：短暂的教师引导

```python
class QuadcopterDistillationRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 100
    max_iterations = 1500
    save_interval = 100
    experiment_name = "distillation"
    empirical_normalization = True
    
    policy = QuadcopterDistillationPolicyCfg()
    
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=4,  
        learning_rate=1e-3,
        max_grad_norm=1.0,
        gradient_length=15,
        class_name="Distillation",
        
        # 短暂的教师引导阶段
        use_two_stage_training=True,
        phase1_iterations=200,  # 仅前200次迭代使用teacher action
    )
```

**效果：**
- 迭代 0-199: 使用teacher action更新环境（初始化阶段）
- 迭代 200+: 使用student action更新环境（主要训练阶段）

---

## 配置参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_two_stage_training` | bool | False | 是否启用两阶段训练 |
| `phase1_iterations` | int | 500 | 第一阶段持续的迭代次数 |

## 参数选择建议

### `phase1_iterations` 的选择：

1. **保守策略**（推荐初次使用）
   - 设置为总迭代数的 40-50%
   - 例如：1500次迭代 → phase1_iterations=600-750

2. **激进策略**（教师策略很强时）
   - 设置为总迭代数的 20-30%
   - 例如：1500次迭代 → phase1_iterations=300-450

3. **稳健策略**（学生网络复杂时）
   - 设置为总迭代数的 50-60%
   - 例如：1500次迭代 → phase1_iterations=750-900

### 调整原则：

- **增加 phase1_iterations** 如果：
  - 学生action质量不足，导致环境交互失败
  - 训练初期reward下降明显
  - 需要更多高质量数据进行学习

- **减少 phase1_iterations** 如果：
  - 学生快速学会教师行为
  - 想要更多实际环境反馈
  - 教师策略可能存在次优性

---

## 完整训练命令示例

```bash
# 使用两阶段训练
python foundation/rsl_rl/train.py \
    --num_envs 100 \
    --task distillation \
    --checkpoint logs/rsl_rl/point_ctrl_direct/2025-12-01_18-51-49/best_model.pt \
    --logger wandb \
    --log_project_name Foundation \
    env.robot.spawn.usd_path="/home/frd/Foundation/USD/cf2x.usd"
```

命令本身无需修改，所有配置在配置文件中完成。
