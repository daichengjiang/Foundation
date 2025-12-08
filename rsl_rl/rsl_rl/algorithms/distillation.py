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
        device="cpu",
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,

        # additional parameters
        num_mini_batches=8,
        # Two-stage training parameters
        use_two_stage_training=True,
        phase1_iterations=10,
        # DAgger parameters
        use_dagger=True,
        max_buffer_size=1000000,
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

        # DAgger parameters: aggregated dataset for historical data
        self.use_dagger = use_dagger
        self.dagger_buffer = None  # Will store all historical data
        self.max_buffer_size = max_buffer_size

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
        
        # Initialize DAgger aggregated buffer for historical data
        if self.use_dagger:
            # Start with capacity for multiple rollouts
            initial_capacity = num_transitions_per_env * num_envs * 10  # Start with 10x rollout size
            self.dagger_buffer = {
                'observations': torch.zeros(initial_capacity, *student_obs_shape, device=self.device),
                'teacher_actions': torch.zeros(initial_capacity, actions_shape[0], device=self.device),
                'masks': torch.zeros(initial_capacity, device=self.device),  # For valid data marking
                'size': 0,  # Current number of valid transitions
                'capacity': initial_capacity,
            }
            print(f"[DAgger] Initialized aggregated buffer with capacity: {initial_capacity} transitions")
            print(f"[DAgger] Maximum buffer size: {self.max_buffer_size} transitions")

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

    def aggregate_current_rollout_to_buffer(self):
        """Aggregate current rollout data into the DAgger historical buffer."""
        if not self.use_dagger or self.dagger_buffer is None:
            return
        
        # Get data from current rollout storage
        # Shape: [num_steps, num_envs, obs_dim] -> flatten to [num_steps * num_envs, obs_dim]
        current_obs = self.storage.observations[:self.storage.step].reshape(-1, self.storage.observations.shape[-1])
        current_teacher_actions = self.storage.privileged_actions[:self.storage.step].reshape(-1, self.storage.actions.shape[-1])
        
        num_new_transitions = current_obs.shape[0]
        
        # Check if we need to expand the buffer
        if self.dagger_buffer['size'] + num_new_transitions > self.dagger_buffer['capacity']:
            self._expand_dagger_buffer(num_new_transitions)
        
        # Add new data to buffer
        start_idx = self.dagger_buffer['size']
        end_idx = start_idx + num_new_transitions
        
        # Ensure we don't exceed max buffer size
        if end_idx > self.max_buffer_size:
            # Use reservoir sampling or simply keep most recent data
            # Here we keep the most recent data
            overflow = end_idx - self.max_buffer_size
            # Shift old data
            self.dagger_buffer['observations'][:-overflow] = self.dagger_buffer['observations'][overflow:self.max_buffer_size]
            self.dagger_buffer['teacher_actions'][:-overflow] = self.dagger_buffer['teacher_actions'][overflow:self.max_buffer_size]
            self.dagger_buffer['masks'][:-overflow] = self.dagger_buffer['masks'][overflow:self.max_buffer_size]
            # Update indices
            start_idx = self.max_buffer_size - num_new_transitions
            end_idx = self.max_buffer_size
            self.dagger_buffer['size'] = self.max_buffer_size
        else:
            self.dagger_buffer['size'] = end_idx
        
        # Copy data to buffer
        self.dagger_buffer['observations'][start_idx:end_idx] = current_obs.to(self.device)
        self.dagger_buffer['teacher_actions'][start_idx:end_idx] = current_teacher_actions.to(self.device)
        self.dagger_buffer['masks'][start_idx:end_idx] = 1.0  # Mark as valid
        
        print(f"[DAgger] Aggregated {num_new_transitions} transitions. Total buffer size: {self.dagger_buffer['size']}/{self.max_buffer_size}")
    
    def _expand_dagger_buffer(self, min_additional_space):
        """Expand the DAgger buffer capacity."""
        old_capacity = self.dagger_buffer['capacity']
        # Double the capacity or add enough space for new data, whichever is larger
        new_capacity = min(max(old_capacity * 2, old_capacity + min_additional_space), self.max_buffer_size)
        
        print(f"[DAgger] Expanding buffer from {old_capacity} to {new_capacity} transitions")
        
        # Create new larger buffers
        new_obs = torch.zeros(new_capacity, *self.dagger_buffer['observations'].shape[1:], device=self.device)
        new_actions = torch.zeros(new_capacity, *self.dagger_buffer['teacher_actions'].shape[1:], device=self.device)
        new_masks = torch.zeros(new_capacity, device=self.device)
        
        # Copy old data
        new_obs[:old_capacity] = self.dagger_buffer['observations']
        new_actions[:old_capacity] = self.dagger_buffer['teacher_actions']
        new_masks[:old_capacity] = self.dagger_buffer['masks']
        
        # Update buffer
        self.dagger_buffer['observations'] = new_obs
        self.dagger_buffer['teacher_actions'] = new_actions
        self.dagger_buffer['masks'] = new_masks
        self.dagger_buffer['capacity'] = new_capacity

    def update(self):
        self.num_updates += 1
        self.current_iteration += 1
        
        # Check if we should switch from phase 1 to phase 2
        if self.use_two_stage_training and self.training_phase == 1:
            if self.current_iteration >= self.phase1_iterations:
                self.switch_to_phase2()
        
        # DAgger: Aggregate current rollout data to historical buffer
        if self.use_dagger:
            self.aggregate_current_rollout_to_buffer()
        
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

        # DAgger: Train on entire historical dataset instead of just current rollout
        if self.use_dagger and self.dagger_buffer is not None and self.dagger_buffer['size'] > 0:
            mean_behavior_loss = self._train_on_aggregated_buffer()
        else:
            # Fallback to original training on current rollout only
            mean_behavior_loss = self._train_on_current_rollout()
        
        # Clear current rollout storage after aggregation and training
        self.storage.clear()
        
        # Reset policy internal state (just in case)
        self.policy.reset() 

        return {"behavior": mean_behavior_loss}
    
    def _train_on_current_rollout(self):
        """Original training method: train only on current rollout."""
        mean_behavior_loss = 0
        cnt = 0

        # Check if policy supports act_batch (Our new custom class)
        is_recurrent_custom = hasattr(self.policy, 'act_batch')

        if not is_recurrent_custom:
            # Fallback for standard Non-Recurrent policies
            # ... (Old logic for non-recurrent if needed) ...
            # For brevity, assuming you are using the new Recurrent class
            pass

        for epoch in range(self.num_learning_epochs):
            # 1. Get Batch Generator
            generator = self.storage.recurrent_distillation_batch_generator(self.num_mini_batches)

            for obs_batch, target_actions_batch, masks_batch in generator:
                # obs_batch: [Seq_Len, Batch_Size, Dim]
                # masks_batch: [Seq_Len, Batch_Size]
                
                T, B, _ = obs_batch.shape
                
                # 2. Initialize Hidden State for this batch (Zeros)
                # Note: RNN hidden state shape is [Num_Layers, Batch, Hidden_Dim]
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

                # 3. Iterate Time (TBPTT: Truncated Backpropagation Through Time)
                for t in range(0, T, self.gradient_length):
                    end_t = min(t + self.gradient_length, T)
                    
                    # Slice window
                    obs_window = obs_batch[t:end_t]         # [Grad_Len, B, D]
                    target_window = target_actions_batch[t:end_t]
                    mask_window = masks_batch[t:end_t]
                    
                    # Forward Batch
                    # hidden_state passes information from previous window to this one
                    pred_actions_window, next_hidden_state = self.policy.act_batch(obs_window, hidden_state)
                    
                    # Calculate Loss
                    # reduction='none' allows us to mask invalid timesteps (padded data)
                    loss = self.loss_fn(pred_actions_window, target_window, reduction='none')
                    
                    # Average over action dim: [Grad_Len, B, A] -> [Grad_Len, B]
                    if len(loss.shape) > 2:
                        loss = loss.mean(dim=-1)
                    
                    # Apply Mask
                    loss = loss * mask_window
                    
                    # Compute scalar loss (Average over valid tokens only)
                    valid_tokens = mask_window.sum()
                    if valid_tokens > 0:
                        loss_val = loss.sum() / valid_tokens
                    else:
                        loss_val = torch.tensor(0.0, device=self.device, requires_grad=True)
                    
                    # Backward & Step
                    self.optimizer.zero_grad()
                    loss_val.backward()
                    
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                        
                    self.optimizer.step()
                    
                    # Pass hidden state to next window (Detach to stop gradient flow)
                    if isinstance(next_hidden_state, tuple):
                        hidden_state = (next_hidden_state[0].detach(), next_hidden_state[1].detach())
                    else:
                        hidden_state = next_hidden_state.detach()
                    
                    mean_behavior_loss += loss_val.item()
                    cnt += 1
        
        return mean_behavior_loss / max(cnt, 1)
    
    def _train_on_aggregated_buffer(self):
        """DAgger training: train on entire historical aggregated dataset."""
        mean_behavior_loss = 0
        cnt = 0
        
        buffer_size = self.dagger_buffer['size']
        batch_size = min(512, buffer_size // self.num_mini_batches)  # Adjust batch size based on buffer
        
        # Train for multiple epochs on the aggregated dataset
        for epoch in range(self.num_learning_epochs):
            # Shuffle indices for random sampling
            indices = torch.randperm(buffer_size, device=self.device)
            
            # Mini-batch training
            for start_idx in range(0, buffer_size, batch_size):
                end_idx = min(start_idx + batch_size, buffer_size)
                batch_indices = indices[start_idx:end_idx]
                
                # Sample batch from aggregated buffer
                obs_batch = self.dagger_buffer['observations'][batch_indices]  # [Batch, Obs_Dim]
                target_actions_batch = self.dagger_buffer['teacher_actions'][batch_indices]  # [Batch, Action_Dim]
                
                # For recurrent policy: need to handle sequences
                # For simplicity, treat each sample independently (no temporal relationship)
                if hasattr(self.policy, 'act_batch'):
                    # Reshape to [Seq_Len=1, Batch, Dim] for recurrent interface
                    obs_batch = obs_batch.unsqueeze(0)  # [1, Batch, Obs_Dim]
                    target_actions_batch = target_actions_batch.unsqueeze(0)  # [1, Action_Dim]
                    
                    # Initialize hidden state
                    B = obs_batch.shape[1]
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
                    
                    # Forward pass
                    pred_actions, _ = self.policy.act_batch(obs_batch, hidden_state)
                    pred_actions = pred_actions.squeeze(0)  # [Batch, Action_Dim]
                    target_actions_batch = target_actions_batch.squeeze(0)
                else:
                    # Non-recurrent policy
                    pred_actions = self.policy.act(obs_batch)
                
                # Compute loss
                loss = self.loss_fn(pred_actions, target_actions_batch, reduction='mean')
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                if self.is_multi_gpu:
                    self.reduce_parameters()
                if self.max_grad_norm:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                mean_behavior_loss += loss.item()
                cnt += 1
        
        return mean_behavior_loss / max(cnt, 1)

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
