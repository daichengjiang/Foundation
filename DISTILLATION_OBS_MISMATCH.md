# Distillation Observation Mismatch Update

## Overview
Modified the distillation system to support different observation spaces for teacher and student policies. The teacher uses full 26D observations while the student uses reduced 22D observations (excluding motor speeds).

## Changes Made

### 1. Observation Dimension Split

**Teacher Observation (26D):**
- `pos_error_b` (3D): Position error in body frame
- `rotation_matrix_flat` (9D): Flattened rotation matrix
- `vel_error_b` (3D): Velocity error in body frame
- `ang_vel_b` (3D): Angular velocity in body frame
- `last_actions` (4D): Previous actions
- `motor_speeds_obs` (4D): Current motor speeds ✓

**Student Observation (22D):**
- `pos_error_b` (3D): Position error in body frame
- `rotation_matrix_flat` (9D): Flattened rotation matrix
- `vel_error_b` (3D): Velocity error in body frame
- `ang_vel_b` (3D): Angular velocity in body frame
- `last_actions` (4D): Previous actions
- ~~`motor_speeds_obs` (4D): Current motor speeds~~ ✗ (removed)

### 2. StudentPolicy Class Updates

**Added Parameters:**
```python
def __init__(self, num_obs: int, num_actions: int, hidden_dim: int = 16, 
             activation: str = "relu", obs_slice: tuple = None):
```

- `obs_slice`: Optional `(start_idx, end_idx)` tuple to automatically slice observations
- This allows the student to accept full 26D observations from the environment but only use the first 22D

**Forward Pass Modification:**
```python
def forward(self, obs, hidden_states=None):
    # Apply observation slicing if configured
    if self.obs_slice is not None:
        start_idx, end_idx = self.obs_slice
        obs = obs[..., start_idx:end_idx]
    # ... rest of forward pass
```

### 3. Data Collection Updates

**Modified `collect_episodes()` function:**
```python
def collect_episodes(env, policy, num_episodes: int, deterministic: bool = True, 
                    teacher_policy=None, student_obs_slice=None):
```

- Added `teacher_policy` parameter: Use teacher for action generation during behavioral cloning
- Added `student_obs_slice` parameter: For future extensibility (currently handled by StudentPolicy)
- Always stores full 26D observations from environment
- Uses teacher policy for action generation during teacher forcing phase

### 4. Training Loop Updates

**Main changes in `train_distillation()`:**

```python
# Calculate observation dimensions
num_obs_teacher = obs.shape[1]  # 26D
num_obs_student = num_obs_teacher - 4  # 22D (remove motor_speeds)
student_obs_slice = (0, num_obs_student)

# Create student with slicing capability
student = StudentPolicy(num_obs_student, num_actions, 
                       obs_slice=student_obs_slice).to(device)

# Dataset stores full teacher observations
dataset = DistillationDataset(dataset_size, num_obs_teacher, 
                              num_actions, device)
```

**Data Collection Phase:**
```python
if epoch < args_cli.epoch_teacher_forcing:
    # Teacher forcing: use teacher for actions
    teacher_for_actions = teacher
else:
    # Student exploration: use student for actions
    teacher_for_actions = None

episode_obs_list, episode_actions_list, episode_returns = collect_episodes(
    env, student, num_episodes, deterministic,
    teacher_policy=teacher_for_actions,
    student_obs_slice=student_obs_slice
)
```

**Training Phase:**
```python
# Student automatically slices observations in forward()
predicted_actions = student(batch_obs)  # batch_obs is 26D, student uses 22D
```

## Benefits

### 1. **Reduced Student Complexity**
- Student network operates on 22D instead of 26D observations
- Motor speeds are internal state, not needed for high-level control decisions
- Matches embodiment-agnostic policy design

### 2. **Maintained Teacher Performance**
- Teacher continues using full 26D observations (trained that way)
- No degradation in teacher demonstration quality

### 3. **Clean Data Pipeline**
- Environment always outputs consistent 26D observations
- Observation slicing handled internally by StudentPolicy
- No manual slicing needed in data collection/training loops

### 4. **Extensibility**
- `obs_slice` parameter allows arbitrary observation filtering
- Easy to experiment with different observation subsets
- Future-proof for multi-teacher with different observation spaces

## Usage

### Training Command (unchanged)
```bash
python distillation.py \
    --teacher_checkpoint logs/teacher/model.pt \
    --task Isaac-Quadcopter-Direct-v0 \
    --num_envs 512 \
    --n_epochs 500 \
    --batch_size 64 \
    --sequence_length 500
```

### Verification
Check logs for correct dimensions:
```
[INFO] Teacher observation dim: 26, Student observation dim: 22, Action dim: 4
```

## Implementation Details

### Observation Slicing Logic

The slicing happens in `StudentPolicy.forward()`:
```python
# Before: obs.shape = (seq_len, batch, 26)
if self.obs_slice is not None:
    start_idx, end_idx = self.obs_slice  # (0, 22)
    obs = obs[..., start_idx:end_idx]
# After: obs.shape = (seq_len, batch, 22)
```

The `...` ellipsis ensures slicing works for both:
- 2D inputs: `(batch, 26)` → `(batch, 22)`
- 3D inputs: `(seq_len, batch, 26)` → `(seq_len, batch, 22)`

### Why Remove Motor Speeds?

1. **Teacher-Student Mismatch**: Teacher was trained with motor speed feedback, but this creates a dependency on specific hardware
2. **Embodiment Agnostic**: Student should learn control policies that work across different actuator dynamics
3. **Delayed Observation**: Motor speeds are delayed (first-order low-pass filtered), making them less informative than desired actions
4. **State vs Action**: Motor speeds are internal actuation state, not environmental state

## Testing

### Unit Test Checklist
- [ ] Student forward pass with 26D input → correctly slices to 22D
- [ ] Student forward pass with 22D input (no slicing) → works correctly
- [ ] Data collection stores full 26D observations
- [ ] Training loop handles dimension mismatch correctly
- [ ] Evaluation uses student's slicing automatically

### Integration Test
```python
# Verify observation dimensions
obs, _ = env.reset()
assert obs.shape[1] == 26, "Environment should output 26D observations"

# Verify student slicing
student_obs = student.forward(obs)
assert student.input_layer.in_features == 22, "Student should use 22D observations"
```

## Future Enhancements

1. **Multi-Teacher Support**: Each teacher could have different observation spaces
2. **Observation Masking**: Random masking for robustness training
3. **Adaptive Slicing**: Learn which observations to use during training
4. **Cross-Embodiment**: Train on multiple robot morphologies with different sensors

## Troubleshooting

### Common Errors

**Error: "RuntimeError: mat1 and mat2 shapes cannot be multiplied"**
- Cause: Observation dimension mismatch
- Solution: Verify `student_obs_slice = (0, 22)` is set correctly

**Error: "Teacher checkpoint expects 22D but environment provides 26D"**
- Cause: Using wrong teacher checkpoint
- Solution: Use teacher trained on full 26D observations

**Error: "Index out of range in slicing"**
- Cause: `student_obs_slice` end index > observation dimension
- Solution: Verify `num_obs_student = num_obs_teacher - 4`

## References

- Original C++ implementation: `raptor/rl-tools/src/foundation_policy/post_training/main.cpp`
- Environment observation: `foundation/tasks/point_ctrl/quad_point_ctrl_env_single_dense.py:_get_observations()`
- Student policy: `foundation/rsl_rl/distillation.py:StudentPolicy`
