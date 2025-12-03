# Distillation Script Updates - 模仿 train.py 的改进

## 更新日期
2025年12月3日

## 主要变更

### 1. 环境创建方式改进

**之前 (旧版本)**:
```python
env = gym.make(args.task, num_envs=args.num_envs)
env = RslRlVecEnvWrapper(env)
```

**现在 (新版本)**:
```python
# 使用 Hydra 配置
@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    # 完整的环境配置支持
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=...)
    
    # 支持多智能体环境
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    # 视频录制
    if args_cli.video:
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # RSL-RL 包装
    env = RslRlVecEnvWrapper(env)
```

### 2. 参数系统改进

**之前**:
- 直接使用 `args` 对象
- 手动指定所有参数

**现在**:
- 使用 `args_cli` (命令行参数)
- 集成 `agent_cfg` (Hydra 配置)
- 继承所有 RSL-RL 参数 (`cli_args.add_rsl_rl_args`)
- 支持 Hydra 覆盖

### 3. 日志和配置管理

**新增功能**:
```python
# 自动保存配置
dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

# 结构化日志目录
logs/distillation/experiment_name/YYYY-MM-DD_HH-MM-SS/
```

### 4. 新增命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--video` | 录制视频 | False |
| `--video_length` | 视频长度 | 200 |
| `--video_interval` | 录制间隔 | 2000 |
| 所有 RSL-RL 参数 | 继承自 `cli_args` | - |
| 所有 AppLauncher 参数 | 继承自 Isaac Lab | - |

### 5. 设备和种子管理

**现在**:
```python
# 从 agent_cfg 读取
env_cfg.seed = agent_cfg.seed
env_cfg.sim.device = args_cli.device or env_cfg.sim.device

# 使用一致的设备
teacher = TeacherPolicyWrapper(checkpoint, agent_cfg.device)
student = StudentPolicy(num_obs, num_actions).to(agent_cfg.device)
```

## 使用变更

### 命令行调用

**之前**:
```bash
python foundation/rsl_rl/distillation.py \
    --task Isaac-Quadcopter-Point-Ctrl-v0 \
    --teacher_checkpoint teacher.pt \
    --num_envs 512 \
    --seed 42 \
    --device cuda:0
```

**现在**:
```bash
# 基本用法（设备和种子从配置读取）
python foundation/rsl_rl/distillation.py \
    --task Isaac-Quadcopter-Point-Ctrl-v0 \
    --teacher_checkpoint teacher.pt \
    --num_envs 512

# 或显式指定
python foundation/rsl_rl/distillation.py \
    --task Isaac-Quadcopter-Point-Ctrl-v0 \
    --teacher_checkpoint teacher.pt \
    --num_envs 512 \
    --seed 42 \
    --device cuda:0

# 使用 Hydra 覆盖环境参数
python foundation/rsl_rl/distillation.py \
    --task Isaac-Quadcopter-Point-Ctrl-v0 \
    --teacher_checkpoint teacher.pt \
    scene.num_envs=2048 \
    sim.device="cuda:1"
```

### 新增功能调用

```bash
# 带视频录制
python foundation/rsl_rl/distillation.py \
    --task Isaac-Quadcopter-Point-Ctrl-v0 \
    --teacher_checkpoint teacher.pt \
    --video \
    --video_interval 1000

# 使用实验名称和运行名称
python foundation/rsl_rl/distillation.py \
    --task Isaac-Quadcopter-Point-Ctrl-v0 \
    --teacher_checkpoint teacher.pt \
    --experiment_name my_distillation \
    --run_name trial_01
```

## 兼容性

### 向后兼容

✅ **完全兼容**: 所有原有的蒸馏参数仍然可用
- `--n_epochs`
- `--num_episodes`
- `--batch_size`
- `--learning_rate`
- `--epoch_teacher_forcing`
- `--shuffle`
- `--on_policy`
- `--teacher_deterministic`

### 与 train.py 的一致性

✅ **完全一致**: 
- 环境创建流程
- 配置管理系统
- 日志目录结构
- 参数处理方式

## 优势总结

1. **更好的集成**: 与 Isaac Lab 生态系统完全集成
2. **灵活配置**: 支持 Hydra 的所有配置覆盖功能
3. **可视化**: 内置视频录制支持
4. **可重现性**: 自动保存所有配置
5. **统一接口**: 与 PPO 训练使用相同的接口
6. **类型安全**: 完整的类型提示支持

## 迁移指南

如果你有现有的蒸馏脚本调用，只需：

1. 移除 `--device` 参数（现在从配置或 `--device` 读取）
2. （可选）添加 `--video` 来录制视频
3. （可选）使用 Hydra 语法覆盖特定配置

就这样！其他参数保持不变。

## 测试

建议测试场景：

```bash
# 1. 基本测试
python foundation/rsl_rl/distillation.py \
    --task Isaac-Quadcopter-Point-Ctrl-v0 \
    --teacher_checkpoint /path/to/teacher.pt \
    --num_envs 64 \
    --n_epochs 2

# 2. 视频测试
python foundation/rsl_rl/distillation.py \
    --task Isaac-Quadcopter-Point-Ctrl-v0 \
    --teacher_checkpoint /path/to/teacher.pt \
    --num_envs 64 \
    --video \
    --n_epochs 2

# 3. Hydra 覆盖测试
python foundation/rsl_rl/distillation.py \
    --task Isaac-Quadcopter-Point-Ctrl-v0 \
    --teacher_checkpoint /path/to/teacher.pt \
    scene.num_envs=128 \
    --n_epochs 2
```

## 问题排查

### 如果遇到 Hydra 错误

确保任务配置文件存在：
```bash
ls foundation/tasks/point_ctrl/agents/rsl_rl_ppo_cfg.py
```

### 如果设备不匹配

显式指定设备：
```bash
--device cuda:0
```

或在 Hydra 中覆盖：
```bash
sim.device="cuda:0"
```

## 贡献

这些改进使得蒸馏脚本与整个 Isaac Lab 框架更加一致和易用。欢迎反馈和改进建议！
