# DAgger (Dataset Aggregation) 实现说明

## 概述

现在的训练已实现**DAgger (Dataset Aggregation)**算法，模型会学习**整个历史数据集**而不是只在最新的rollout数据上训练。

## 核心变化

### 原训练模式
```
每次迭代:
1. 收集当前rollout数据 (100步 × 100环境 = 10,000个transition)
2. 只在这10,000个transition上训练
3. 清空数据
4. 重复
```

**问题**: 每次只学习最新数据，无法利用历史经验。

### DAgger模式 (新实现)
```
每次迭代:
1. 收集当前rollout数据 (10,000个transition)
2. 将新数据聚合到历史buffer中
3. 在整个历史buffer上训练 (可能有几十万到百万个transition)
4. 保留历史数据，继续下一轮
```

**优势**: 
- ✅ 充分利用所有收集的数据
- ✅ 避免灾难性遗忘
- ✅ 数据效率更高
- ✅ 训练更稳定

## 技术实现

### 1. 数据聚合Buffer

```python
self.dagger_buffer = {
    'observations': Tensor[N, obs_dim],      # 学生观测
    'teacher_actions': Tensor[N, action_dim], # 教师action (标签)
    'masks': Tensor[N],                       # 有效数据标记
    'size': int,                              # 当前数据量
    'capacity': int,                          # 最大容量
}
```

- **动态扩展**: Buffer容量不足时自动扩展（加倍）
- **容量限制**: 达到`max_buffer_size`后保留最新数据，丢弃旧数据

### 2. 数据聚合流程

每次rollout后:
```python
def aggregate_current_rollout_to_buffer():
    # 1. 从current storage获取数据
    current_obs = storage.observations[:step]
    current_teacher_actions = storage.privileged_actions[:step]
    
    # 2. 展平: [num_steps, num_envs, dim] -> [num_steps*num_envs, dim]
    current_obs = current_obs.reshape(-1, obs_dim)
    current_teacher_actions = current_teacher_actions.reshape(-1, action_dim)
    
    # 3. 添加到历史buffer
    dagger_buffer['observations'][start:end] = current_obs
    dagger_buffer['teacher_actions'][start:end] = current_teacher_actions
    dagger_buffer['size'] += num_new_transitions
```

### 3. 训练流程

```python
def update():
    # 1. 聚合当前rollout到历史buffer
    aggregate_current_rollout_to_buffer()
    
    # 2. 在整个历史buffer上训练
    for epoch in range(num_learning_epochs):
        # 随机打乱索引
        indices = torch.randperm(buffer_size)
        
        for batch in mini_batches:
            # 从历史buffer采样
            obs_batch = dagger_buffer['observations'][indices[batch]]
            target_batch = dagger_buffer['teacher_actions'][indices[batch]]
            
            # 前向传播
            pred_actions = policy.act(obs_batch)
            
            # 计算损失并更新
            loss = loss_fn(pred_actions, target_batch)
            loss.backward()
            optimizer.step()
    
    # 3. 清空当前rollout storage (数据已聚合)
    storage.clear()
```

## 配置参数

### Algorithm配置

```python
algorithm = RslRlDistillationAlgorithmCfg(
    # ... 其他参数 ...
    
    # DAgger配置
    use_dagger=True,           # 启用DAgger
    max_buffer_size=1000000,   # 最多存储100万个transition
)
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_dagger` | bool | `True` | 是否启用DAgger数据聚合 |
| `max_buffer_size` | int | `1000000` | 历史buffer最大容量 |

## 内存管理

### 内存占用估算

假设:
- 观测维度: 100
- 动作维度: 4
- 数据类型: float32 (4 bytes)

每个transition占用:
```
(100 + 4) × 4 bytes = 416 bytes
```

100万个transition:
```
1,000,000 × 416 bytes ≈ 397 MB
```

### 动态扩展策略

1. **初始容量**: `num_transitions_per_env × num_envs × 10`
   - 例如: 100步 × 100环境 × 10 = 100,000个transition

2. **扩展策略**: 容量不足时加倍
   - 100k → 200k → 400k → 800k → 1M (达到上限)

3. **容量上限**: 达到`max_buffer_size`后
   - 新数据覆盖最旧的数据（FIFO）
   - 保持buffer大小恒定

## 与两阶段训练结合

DAgger与两阶段训练可以完美结合：

### Phase 1 (使用teacher action)
```
迭代 0-499:
  - 环境用teacher action更新
  - 收集高质量数据
  - 数据聚合到buffer: [0, 10k, 20k, ..., 5M]
  - 训练时使用所有历史数据
```

### Phase 2 (使用student action)  
```
迭代 500+:
  - 环境用student action更新
  - 收集student exploration数据
  - 继续聚合: [5M, 5.01M, 5.02M, ...]
  - 训练时使用所有历史数据（包括Phase 1的高质量数据）
```

**优势**: Phase 2的训练仍然能从Phase 1的高质量数据中学习！

## 训练输出

### 日志信息

```
[DAgger] Initialized aggregated buffer with capacity: 100000 transitions
[DAgger] Maximum buffer size: 1000000 transitions

Iteration 0:
  [DAgger] Aggregated 10000 transitions. Total buffer size: 10000/1000000

Iteration 1:
  [DAgger] Aggregated 10000 transitions. Total buffer size: 20000/1000000

Iteration 2:
  [DAgger] Aggregated 10000 transitions. Total buffer size: 30000/1000000

...

Iteration 100:
  [DAgger] Expanding buffer from 100000 to 200000 transitions
  [DAgger] Aggregated 10000 transitions. Total buffer size: 1010000/1000000
```

## 性能考虑

### 优点
1. **数据效率高**: 每个数据点被重复使用多次
2. **训练稳定**: 大数据集减少方差
3. **避免遗忘**: 保留早期高质量数据

### 缺点
1. **内存占用**: 需要存储大量历史数据
2. **训练时间**: 在大数据集上训练较慢

### 优化建议

1. **调整batch size**: 数据量大时可以增加batch size
   ```python
   batch_size = min(512, buffer_size // num_mini_batches)
   ```

2. **调整buffer大小**: 根据GPU内存调整
   ```python
   max_buffer_size=500000   # 约200MB (如果内存有限)
   max_buffer_size=2000000  # 约800MB (如果内存充足)
   ```

3. **采样策略**: 可以优先采样最新数据
   ```python
   # 可选: 指数衰减权重，偏向新数据
   weights = torch.exp(-0.01 * torch.arange(buffer_size))
   ```

## 禁用DAgger

如果想恢复到原始训练模式（只在最新rollout上训练）：

```python
algorithm = RslRlDistillationAlgorithmCfg(
    # ... 其他参数 ...
    use_dagger=False,  # 禁用DAgger
)
```

## 完整训练命令

```bash
python foundation/rsl_rl/train.py \
    --num_envs 100 \
    --task distillation \
    --checkpoint logs/rsl_rl/point_ctrl_direct/2025-12-01_18-51-49/best_model.pt \
    --logger wandb \
    --log_project_name Foundation \
    env.robot.spawn.usd_path="/home/frd/Foundation/USD/cf2x.usd"
```

命令无需修改，DAgger会自动启用！

## 实现文件

- `/home/frd/Foundation/rsl_rl/rsl_rl/algorithms/distillation.py`
  - `init_storage()`: 初始化DAgger buffer
  - `aggregate_current_rollout_to_buffer()`: 数据聚合
  - `_train_on_aggregated_buffer()`: 在历史数据上训练
  - `_expand_dagger_buffer()`: 动态扩展buffer

- `/home/frd/IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/distillation_cfg.py`
  - 新增`use_dagger`和`max_buffer_size`配置

- `/home/frd/Foundation/foundation/tasks/point_ctrl/agents/rsl_rl_ppo_cfg.py`
  - 启用DAgger配置

## 总结

DAgger实现让模型能够：
- 📚 学习整个历史数据集
- 🔄 不断积累经验
- 📈 提高数据利用率
- 🎯 获得更好的泛化能力

结合两阶段训练，训练流程更加稳定和高效！
