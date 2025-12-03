# Teacher Policy Distillation

这个文档介绍如何使用 `foundation/rsl_rl/distillation.py` 脚本进行教师策略蒸馏。

## 概述

该脚本实现了从教师策略到学生策略的知识蒸馏，基于 C++ 实现 (`raptor/rl-tools/src/foundation_policy/post_training/main.cpp`)。主要功能包括：

- **行为克隆 (Behavior Cloning)**: 学生策略学习模仿教师策略的动作
- **数据收集**: 使用教师策略收集示范数据
- **监督学习**: 训练学生策略最小化与教师动作的差异
- **评估**: 定期评估学生策略的性能

## 主要特性

### 与 train.py 一致的环境管理

脚本现在完全遵循 `train.py` 的环境创建和管理方式：

1. **Hydra 配置系统**: 使用 `@hydra_task_config` 装饰器自动加载任务配置
2. **多环境类型支持**: 
   - `ManagerBasedRLEnvCfg`: 基于管理器的环境
   - `DirectRLEnvCfg`: 直接RL环境
   - `DirectMARLEnvCfg`: 多智能体环境（自动转换为单智能体）
3. **视频录制**: 使用 `--video` 标志启用训练过程录制
4. **配置持久化**: 自动保存环境和智能体配置到日志目录

### 与 C++ 实现的对应关系

| C++ 功能 | Python 实现 | 说明 |
|---------|------------|------|
| `ACTOR_TEACHER` | `TeacherPolicyWrapper` | 教师策略加载和推理 |
| `ACTOR` | `StudentPolicy` | 学生策略网络 |
| `dataset_*` tensors | `DistillationDataset` | 数据集管理 |
| `gather_epoch` | `collect_episodes` | 数据收集 |
| `rlt::nn::loss_functions::mse` | `nn.functional.mse_loss` | MSE 损失函数 |
| `EPOCH_TEACHER_FORCING` | `--epoch_teacher_forcing` | 教师强制学习的轮数 |

### 新增功能（相比初始版本）

1. ✅ **Hydra 集成**: 完整支持 Isaac Lab 的配置系统
2. ✅ **视频录制**: 可视化训练过程
3. ✅ **配置管理**: 自动保存和加载配置
4. ✅ **多环境支持**: 支持所有 Isaac Lab 环境类型
5. ✅ **RSL-RL 参数**: 继承所有 RSL-RL 命令行参数

### 简化的设计

为简化实现，这个版本：
- **单教师**: 只支持一个教师策略（C++ 版本支持多个）
- **MLP架构**: 使用简单的多层感知机（未来可扩展到 RNN）
- **批量训练**: 使用标准的批量梯度下降

## 使用方法

### 基本命令

```bash
cd /home/frd/Foundation
python foundation/rsl_rl/distillation.py \
    --task Isaac-Quadcopter-Point-Ctrl-v0 \
    --teacher_checkpoint /path/to/teacher/checkpoint.pt \
    --num_envs 512 \
    --seed 42
```

### 命令行参数

#### 必需参数

- `--task`: 环境任务名称（例如：`Isaac-Quadcopter-Point-Ctrl-v0`）
- `--teacher_checkpoint`: 教师策略检查点路径

#### 环境参数

- `--num_envs`: 并行环境数量（默认：从环境配置读取）
- `--seed`: 随机种子（默认：从 agent 配置读取）
- `--video`: 是否录制视频（默认：False）
- `--video_length`: 录制视频长度（步数）（默认：200）
- `--video_interval`: 视频录制间隔（步数）（默认：2000）

#### RSL-RL 通用参数

通过 `cli_args` 继承所有 RSL-RL 参数：
- `--device`: 计算设备（例如：`cuda:0`）
- `--experiment_name`: 实验名称
- `--run_name`: 运行名称

#### 蒸馏超参数

- `--n_epochs`: 训练轮数（默认：100）
- `--num_episodes`: 每轮收集的episodes数量（默认：100）
- `--batch_size`: 训练批量大小（默认：256）
- `--sequence_length`: 序列长度，用于RNN（默认：1，MLP不需要）
- `--learning_rate`: 学习率（默认：1e-4）
- `--epoch_teacher_forcing`: 仅使用教师收集数据的轮数（默认：50）
- `--shuffle`: 是否打乱episodes顺序（默认：True）
- `--on_policy`: 是否使用on-policy数据收集（默认：False）
- `--teacher_deterministic`: 教师策略是否使用确定性动作（默认：True）

### 完整示例

```bash
# 基础蒸馏（使用 Hydra 配置）
python foundation/rsl_rl/distillation.py \
    --task Isaac-Quadcopter-Point-Ctrl-v0 \
    --teacher_checkpoint logs/rsl_rl/experiment_name/model_5000.pt \
    --num_envs 1024 \
    --seed 123

# 带视频录制的蒸馏
python foundation/rsl_rl/distillation.py \
    --task Isaac-Quadcopter-Point-Ctrl-v0 \
    --teacher_checkpoint teacher.pt \
    --video \
    --video_interval 1000 \
    --num_envs 512

# 高级配置 - On-policy 蒸馏
python foundation/rsl_rl/distillation.py \
    --task Isaac-Quadcopter-Point-Ctrl-v0 \
    --teacher_checkpoint teacher.pt \
    --num_envs 1024 \
    --n_epochs 200 \
    --num_episodes 200 \
    --batch_size 512 \
    --learning_rate 5e-5 \
    --epoch_teacher_forcing 100 \
    --on_policy \
    --device cuda:0
```

### 使用 Hydra 覆盖环境配置

由于使用了 Hydra 配置系统，可以直接覆盖环境参数：

```bash
python foundation/rsl_rl/distillation.py \
    --task Isaac-Quadcopter-Point-Ctrl-v0 \
    --teacher_checkpoint teacher.pt \
    scene.num_envs=2048 \
    sim.device="cuda:1"
```

## 训练流程

### 1. 数据收集阶段

在前 `epoch_teacher_forcing` 轮中：
- 使用**教师策略**收集示范数据
- 在环境中运行教师策略
- 记录 (observation, action) 对

### 2. 蒸馏训练阶段

在剩余轮中：
- 使用**学生策略**收集数据（自我改进）
- 对收集的数据进行监督学习
- 最小化学生和教师动作之间的 MSE 损失

### 3. 评估阶段

每轮训练后：
- 评估学生策略性能
- 记录平均回报和标准差
- 保存最佳模型

## 输出文件

训练过程中会生成以下文件结构（与 `train.py` 一致）：

```
logs/distillation/experiment_name/YYYY-MM-DD_HH-MM-SS/
├── events.out.tfevents.*          # TensorBoard日志
├── best_student.pt                # 最佳学生模型
├── final_student.pt               # 最终学生模型
├── checkpoint_epoch_*.pt          # 定期检查点
├── params/                        # 配置文件目录
│   ├── env.yaml                   # 环境配置（YAML）
│   ├── env.pkl                    # 环境配置（Pickle）
│   ├── agent.yaml                 # 智能体配置（YAML）
│   └── agent.pkl                  # 智能体配置（Pickle）
└── videos/                        # 视频录制（如果启用）
    └── distillation/
        ├── rl-video-episode-*.mp4
        └── rl-video-episode-*.meta.json
```

### 检查点内容

每个检查点包含：
- `epoch`: 训练轮数
- `model_state_dict`: 模型参数
- `optimizer_state_dict`: 优化器状态
- `best_return`: 最佳平均回报

## 监控训练

使用 TensorBoard 查看训练进度：

```bash
tensorboard --logdir logs/distillation
```

可视化指标：
- `data_collection/mean_return`: 数据收集期间的平均回报
- `data_collection/std_return`: 回报标准差
- `training/loss`: 训练损失（MSE）
- `evaluation/mean_return`: 评估平均回报
- `evaluation/std_return`: 评估回报标准差
- `evaluation/mean_length`: 平均episode长度

## 加载训练好的学生策略

训练完成后，可以这样加载学生策略：

```python
import torch
from distillation import StudentPolicy

# 加载检查点
checkpoint = torch.load("logs/distillation/.../best_student.pt")

# 创建学生策略
student = StudentPolicy(num_obs, num_actions)
student.load_state_dict(checkpoint['model_state_dict'])
student.eval()

# 使用策略
with torch.no_grad():
    action = student(observation)
```

## 高级用法

### On-Policy 蒸馏

使用 `--on_policy` 标志，每轮重新收集数据：

```bash
python foundation/rsl_rl/distillation.py \
    --task Isaac-Quadcopter-Point-Ctrl-v0 \
    --teacher_checkpoint teacher.pt \
    --on_policy
```

这种方式：
- ✓ 避免分布漂移
- ✓ 适合动态环境
- ✗ 需要更多计算资源

### Off-Policy 蒸馏（默认）

不使用 `--on_policy`，重复使用历史数据：

```bash
python foundation/rsl_rl/distillation.py \
    --task Isaac-Quadcopter-Point-Ctrl-v0 \
    --teacher_checkpoint teacher.pt
```

这种方式：
- ✓ 更高效，数据复用
- ✓ 训练更快
- ✗ 可能存在分布漂移

## 调试技巧

### 检查教师策略加载

脚本会打印教师网络架构：
```
Teacher architecture: obs=26, actions=4, hidden=[256, 256, 256]
```

### 监控数据收集

查看每轮的数据收集统计：
```
Data collection - Mean return: 45.23 ± 12.45
```

### 验证训练损失

损失应该逐渐下降：
```
Training - Mean loss: 0.012345
```

## 故障排除

### 问题：损失不下降

可能原因：
- 学习率太高或太低 → 调整 `--learning_rate`
- 批量大小不合适 → 调整 `--batch_size`
- 数据不足 → 增加 `--num_episodes`

### 问题：评估性能差

可能原因：
- 教师策略质量不佳 → 使用更好的教师检查点
- 训练不充分 → 增加 `--n_epochs`
- 过拟合 → 使用 `--on_policy` 或增加数据多样性

### 问题：内存不足

解决方案：
- 减少 `--num_envs`
- 减少 `--batch_size`
- 减少 `--num_episodes`

## 与 train.py 的集成

### 架构一致性

蒸馏脚本现在与 `train.py` 完全一致：

```python
# 两者都使用相同的模式
@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    # 环境创建
    env = gym.make(args_cli.task, cfg=env_cfg, ...)
    
    # 配置覆盖
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    
    # 日志管理
    log_dir = os.path.join("logs", "rsl_rl/distillation", ...)
    
    # 配置保存
    dump_yaml/dump_pickle(...)
```

### 优势

1. **无缝切换**: 可以直接使用训练好的策略作为教师
2. **配置兼容**: 所有环境配置都能直接复用
3. **统一日志**: 日志结构与 PPO 训练一致
4. **易于调试**: 使用相同的环境包装器和工具

### 工作流程示例

```bash
# 1. 使用 train.py 训练教师策略
python foundation/rsl_rl/train.py \
    --task Isaac-Quadcopter-Point-Ctrl-v0 \
    --num_envs 4096 \
    --seed 42

# 2. 使用 distillation.py 蒸馏学生策略
python foundation/rsl_rl/distillation.py \
    --task Isaac-Quadcopter-Point-Ctrl-v0 \
    --teacher_checkpoint logs/rsl_rl/quadcopter/2025-12-03_10-00-00/model_5000.pt \
    --num_envs 1024 \
    --seed 123

# 3. 评估学生策略（使用同样的环境）
python foundation/rsl_rl/play.py \
    --task Isaac-Quadcopter-Point-Ctrl-v0 \
    --checkpoint logs/distillation/quadcopter/2025-12-03_12-00-00/best_student.pt
```

## 与 C++ 版本的主要区别

| 特性 | C++ 版本 | Python 版本 |
|------|---------|------------|
| 多教师支持 | ✓ | ✗（单教师）|
| RNN 支持 | ✓ | ✗（MLP only）|
| 位置偏移校正 | ✓ | ✗ |
| 活跃教师选择 | ✓ | ✗ |
| 动态参数加载 | ✓ | ✗ |

未来可以根据需要添加这些功能。

## 参考资料

- C++ 实现: `/home/frd/raptor/rl-tools/src/foundation_policy/post_training/main.cpp`
- RSL-RL Distillation: `/home/frd/Foundation/rsl_rl/rsl_rl/algorithms/distillation.py`
- 环境配置: `/home/frd/Foundation/foundation/tasks/point_ctrl/quad_point_ctrl_env_single_dense.py`

## 贡献

如需添加新功能（如多教师支持、RNN支持等），请参考 C++ 实现并保持 API 一致性。
