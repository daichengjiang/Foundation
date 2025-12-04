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
        # create rollout storage
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
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.privileged_actions = self.policy.evaluate(teacher_obs).detach()
        # record the observations
        self.transition.observations = obs
        self.transition.privileged_observations = teacher_obs
        return self.transition.actions

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
        mean_behavior_loss = 0
        loss = 0
        cnt = 0

        # Check if the policy is recurrent
        is_recurrent = hasattr(self.policy, 'is_recurrent') and self.policy.is_recurrent

        for epoch in range(self.num_learning_epochs):
            if is_recurrent:
                # For recurrent networks, use trajectory-based generator
                generator = self.storage.recurrent_distillation_generator()
            else:
                # For non-recurrent networks, use simple timestep generator
                generator = self.storage.generator()

            # Reset policy at the start of each epoch
            if is_recurrent:
                # For recurrent policies, always reset (don't use last_hidden_states from update)
                self.policy.reset()
            else:
                # For non-recurrent policies, can restore hidden states from last update
                self.policy.reset(hidden_states=self.last_hidden_states)
                self.policy.detach_hidden_states()

            if is_recurrent:
                # Process complete trajectories for RNN
                for obs_traj, privileged_obs_traj, _, privileged_action_traj, dones_traj, traj_length in generator:
                    # Reset hidden states at the start of each trajectory
                    self.policy.reset()
                    
                    # Process trajectory step by step
                    for t in range(traj_length):
                        obs = obs_traj[t:t+1]  # [1, obs_dim] - keep batch dimension
                        privileged_actions = privileged_action_traj[t:t+1]  # [1, action_dim]
                        
                        # Student inference with RNN
                        actions = self.policy.act_inference(obs)
                        
                        # Behavior cloning loss
                        behavior_loss = self.loss_fn(actions, privileged_actions)
                        
                        # Accumulate loss
                        loss = loss + behavior_loss
                        mean_behavior_loss += behavior_loss.item()
                        cnt += 1
                        
                        # Gradient step every gradient_length steps
                        if cnt % self.gradient_length == 0:
                            self.optimizer.zero_grad()
                            loss.backward()
                            if self.is_multi_gpu:
                                self.reduce_parameters()
                            if self.max_grad_norm:
                                nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
                            self.optimizer.step()
                            self.policy.detach_hidden_states()
                            loss = 0
                        
                        # Check if this timestep ends an episode
                        if dones_traj[t].item():
                            # Detach hidden states at episode boundary
                            self.policy.detach_hidden_states()
                            # Note: Don't reset here, we'll reset at the start of next trajectory
            else:
                # Process individual timesteps for non-recurrent networks
                for obs, privileged_obs, _, privileged_actions, dones in generator:
                    # inference the student for gradient computation
                    actions = self.policy.act_inference(obs)

                    # behavior cloning loss
                    behavior_loss = self.loss_fn(actions, privileged_actions)

                    # total loss
                    loss = loss + behavior_loss
                    mean_behavior_loss += behavior_loss.item()
                    cnt += 1

                    # gradient step
                    if cnt % self.gradient_length == 0:
                        self.optimizer.zero_grad()
                        loss.backward()
                        if self.is_multi_gpu:
                            self.reduce_parameters()
                        if self.max_grad_norm:
                            nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                        self.policy.detach_hidden_states()
                        loss = 0

                    # reset dones
                    self.policy.reset(dones.view(-1))
                    self.policy.detach_hidden_states(dones.view(-1))

        mean_behavior_loss /= cnt
        self.storage.clear()
        
        # Only save hidden states for non-recurrent policies
        # For recurrent policies, hidden states from update (batch_size=1) don't match act() (batch_size=num_envs)
        if not is_recurrent:
            self.last_hidden_states = self.policy.get_hidden_states()
            self.policy.detach_hidden_states()
        else:
            # For recurrent policies, reset hidden states after update
            self.policy.reset()
            self.last_hidden_states = None

        # construct the loss dictionary
        loss_dict = {"behavior": mean_behavior_loss}

        return loss_dict

    """
    Helper functions
    """

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
