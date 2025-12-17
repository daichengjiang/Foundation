# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# rsl-rl
from rsl_rl.modules import StudentTeacher, StudentTeacherRecurrent, StudentTeacherRecurrentCustom
from rsl_rl.storage import RolloutStorage


class Distillation:
    """Distillation algorithm for training a student model to mimic a teacher model."""

    policy: StudentTeacher | StudentTeacherRecurrent | StudentTeacherRecurrentCustom
    """The student teacher model."""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        gradient_length=15,
        learning_rate=1e-3,
        max_grad_norm=None,
        loss_type="mse",
        # device="cpu",
        device="cuda",
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,

        # additional parameters
        num_mini_batches=8,
        # Two-stage training parameters
        use_two_stage_training=True,
        phase1_iterations=10,
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.rnd = None  # TODO: remove when runner has a proper base class

        # distillation components
        self.policy = policy
        self.policy.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()
        self.last_hidden_states = None

        # distillation parameters
        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.num_mini_batches = num_mini_batches

        # Two-stage training parameters
        self.use_two_stage_training = use_two_stage_training
        self.phase1_iterations = phase1_iterations
        self.training_phase = 1 if use_two_stage_training else 2  # Phase 1: use teacher actions, Phase 2: use student actions
        self.current_iteration = 0

        # initialize the loss function
        if loss_type == "mse":
            self.loss_fn = nn.functional.mse_loss
        elif loss_type == "huber":
            self.loss_fn = nn.functional.huber_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported types are: mse, huber")

        self.num_updates = 0

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, student_obs_shape, teacher_obs_shape, actions_shape
    ):
        # create rollout storage for current rollout
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            student_obs_shape,
            teacher_obs_shape,
            actions_shape,
            None,
            self.device,
        )

    def act(self, obs, teacher_obs):
        # compute the actions
        student_action = self.policy.act(obs).detach()
        teacher_action = self.policy.evaluate(teacher_obs).detach()
        
        # store both actions in transition for dataset
        self.transition.actions = student_action
        self.transition.privileged_actions = teacher_action
        
        # record the observations
        self.transition.observations = obs
        self.transition.privileged_observations = teacher_obs
        
        # Return appropriate action based on training phase
        if self.use_two_stage_training:
            if self.training_phase == 1:
                # Phase 1: Use teacher action to update environment
                return teacher_action
            else:
                # Phase 2: Use student action to update environment
                return student_action
        else:
            # Default behavior: use student action
            return student_action

    def process_env_step(self, rewards, dones, infos):
        # record the rewards and dones
        self.transition.rewards = rewards
        self.transition.dones = dones
        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)


    def update(self):
        self.num_updates += 1
        self.current_iteration += 1
        
        # Check if we should switch from phase 1 to phase 2
        if self.use_two_stage_training and self.training_phase == 1:
            if self.current_iteration >= self.phase1_iterations:
                self.switch_to_phase2()
        
        mean_behavior_loss = 0
        loss_accum = 0
        cnt = 0

        # Check if policy supports act_batch (Our new custom class)
        is_recurrent_custom = hasattr(self.policy, 'act_batch')

        if not is_recurrent_custom:
            # Fallback for standard Non-Recurrent policies
            # ... (Old logic for non-recurrent if needed) ...
            # For brevity, assuming you are using the new Recurrent class
            pass

        for epoch in range(self.num_learning_epochs):
            generator = self.storage.recurrent_distillation_batch_generator(self.num_mini_batches)

            for obs_batch, target_actions_batch, masks_batch in generator:
                # obs_batch: [Seq_Len, Batch_Size, Dim]
                
                # ================= [FIX 1: 保留原始 Mask 用于隐状态管理] =================
                # masks_batch 即将用于 Loss 计算（会被修改以进行 Burn-in）
                # 我们需要一份"纯净"的 Mask，仅包含环境的 Done 信息，用于重置 RNN
                # 假设 generator 返回的 masks_batch 中，1.0 代表继续，0.0 代表 Done/Reset
                reset_masks_batch = masks_batch.clone() 
                
                # ================= [Burn-in 策略 (原代码)] =================
                burn_in_steps = 20 
                if masks_batch.shape[0] > burn_in_steps:
                    # 这只影响 Loss 的计算，不应影响 RNN 记忆的传递
                    masks_batch[:burn_in_steps, :] = 0.0
                
                T, B, _ = obs_batch.shape
                
                # 2. Initialize Hidden State (Zeros)
                if hasattr(self.policy, 'rnn_type') and self.policy.rnn_type == 'lstm':
                     hidden_state = (
                         torch.zeros(self.policy.rnn_num_layers, B, self.policy.rnn_hidden_dim, device=self.device),
                         torch.zeros(self.policy.rnn_num_layers, B, self.policy.rnn_hidden_dim, device=self.device)
                     )
                else:
                     hidden_state = torch.zeros(
                         self.policy.rnn_num_layers, B, self.policy.rnn_hidden_dim, 
                         device=self.device
                     )

                # 3. Iterate Time
                for t in range(0, T, self.gradient_length):
                    end_t = min(t + self.gradient_length, T)
                    
                    obs_window = obs_batch[t:end_t]
                    target_window = target_actions_batch[t:end_t]
                    mask_window = masks_batch[t:end_t]         # 用于 Loss (含 Burn-in)
                    
                    # [FIX 1 Continued] 获取当前窗口用于重置隐状态的 Mask
                    # 我们需要当前窗口“最后一步”的 Mask
                    # 如果最后一步是 Done (0.0)，则传给下一段的隐状态应为 0
                    current_reset_mask_window = reset_masks_batch[t:end_t]
                    
                    # Forward Batch
                    pred_actions_window, next_hidden_state = self.policy.act_batch(obs_window, hidden_state)
                    
                    # Calculate Loss
                    loss = self.loss_fn(pred_actions_window, target_window, reduction='none')
                    if len(loss.shape) > 2:
                        loss = loss.mean(dim=-1)
                    
                    loss = loss * mask_window # 使用含 Burn-in 的 mask 屏蔽 Loss
                    
                    valid_tokens = mask_window.sum()
                    if valid_tokens > 0:
                        loss_val = loss.sum() / valid_tokens
                    else:
                        loss_val = torch.tensor(0.0, device=self.device, requires_grad=True)
                    
                    self.optimizer.zero_grad()
                    loss_val.backward()
                    
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                        
                    self.optimizer.step()

                    mean_behavior_loss += loss_val.item()
                    cnt += 1
                    
                    # ================= [FIX 2: 根据 Mask 重置隐状态] =================
                    # 取出当前窗口最后一步的 mask: [Batch_Size]
                    # reset_masks通常是: 1.0 (Alive), 0.0 (Done)
                    # 形状调整为: [1, Batch, 1] 以便与 hidden_state [Layers, Batch, Dim] 广播相乘
                    
                    # 注意：我们使用 reset_masks_batch 而不是 mask_window
                    # 这样可以避免 Burn-in 阶段把隐状态清零了
                    last_step_mask = current_reset_mask_window[-1].view(1, -1, 1)

                    if isinstance(next_hidden_state, tuple):
                        # LSTM: (h, c)
                        h_next, c_next = next_hidden_state
                        # Detach AND Mask
                        h_next = h_next.detach() * last_step_mask
                        c_next = c_next.detach() * last_step_mask
                        hidden_state = (h_next, c_next)
                    else:
                        # GRU: h
                        # Detach AND Mask
                        hidden_state = next_hidden_state.detach() * last_step_mask
                    # ===============================================================

        self.storage.clear()
        self.policy.reset() 
        return {"behavior": mean_behavior_loss / max(cnt, 1)}

        

    """
    Helper functions
    """

    def switch_to_phase2(self):
        """Switch from phase 1 (teacher actions) to phase 2 (student actions)."""
        if self.training_phase == 1:
            self.training_phase = 2
            print(f"\n{'='*80}")
            print(f"{'='*80}")
            print(f"  SWITCHING TO PHASE 2: Now using STUDENT actions to update environment")
            print(f"  Iteration: {self.current_iteration}")
            print(f"{'='*80}")
            print(f"{'='*80}\n")

    def get_training_phase_info(self):
        """Get information about current training phase."""
        return {
            "use_two_stage_training": self.use_two_stage_training,
            "training_phase": self.training_phase,
            "current_iteration": self.current_iteration,
            "phase1_iterations": self.phase1_iterations,
            "action_source": "teacher" if self.training_phase == 1 else "student"
        }

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel
