# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain

from rsl_rl.modules import ActorCritic
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import string_to_callable


class PPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347).
    
    Modified to support: Bootstrapping Reinforcement Learning with Imitation for Vision-Based Agile Flight (CoRL 2024)
    Algorithm 1: Adaptive Fine-tuning
    """

    policy: ActorCritic
    """The actor critic module."""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cuda",
        normalize_advantage_per_mini_batch=False,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        
        # [新增] 论文 Algorithm 1 专用参数
        critic_learning_rate=1e-3,
        adaptive_c_v=1e-5,       # c_V: 用于增加 Actor LR
        adaptive_c_pi=1e-5,      # c_pi: 用于减小 Critic LR
        adaptive_c_epsilon=0.01, # c_epsilon: 用于增加 Clip Range
        lr_max=5e-4,
        lr_min=1e-6,
        epsilon_max=0.5,
        **kwargs,  # [修复] 必须添加这个，才能在函数体内使用 kwargs.pop
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm # Clip gradient norm

        # [新增] 保存自适应算法参数
        self.adaptive_c_v = adaptive_c_v
        self.adaptive_c_pi = adaptive_c_pi
        self.adaptive_c_epsilon = adaptive_c_epsilon
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.epsilon_max = epsilon_max
        
        # 记录当前状态
        self.current_lr_actor = learning_rate
        self.current_lr_critic = critic_learning_rate
        self.clip_param = clip_param # 这是动态变化的 epsilon

        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches

        self.gamma = gamma
        self.lam = lam
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.use_clipped_value_loss = use_clipped_value_loss
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        # Distributed training parameters
        self.multi_gpu_cfg = multi_gpu_cfg
        self.is_multi_gpu = multi_gpu_cfg is not None
        if self.is_multi_gpu:
            self.gpu_local_rank = multi_gpu_cfg.get("local_rank", 0)
            self.gpu_global_rank = multi_gpu_cfg.get("global_rank", 0)
            self.gpu_world_size = multi_gpu_cfg.get("world_size", 1)
        else:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.policy = policy.to(self.device)

        # [修改] 分离 Actor 和 Critic 的参数
        # 假设 model 中 Critic 的网络包含在名为 'critic' 的模块中
        actor_params = []
        critic_params = []
        for name, param in self.policy.named_parameters():
            if "critic" in name:
                critic_params.append(param)
            else:
                actor_params.append(param)

        # 初始化优化器，使用两个参数组
        # group[0]: Actor, group[1]: Critic
        self.optimizer = optim.Adam([
            {'params': actor_params, 'lr': self.current_lr_actor},
            {'params': critic_params, 'lr': self.current_lr_critic}
        ])
        
        # RND Logic (保持原样)
        self.rnd = None
        if rnd_cfg is not None:
            self.rnd = RandomNetworkDistillation(**rnd_cfg, device=self.device)
            self.optimizer.add_param_group({"params": self.rnd.parameters(), "lr": rnd_cfg["learning_rate"]})

        # Symmetry Logic (保持原样)
        self.symmetry = None
        if symmetry_cfg is not None:
            symmetry_class = string_to_callable(symmetry_cfg["class_name"])
            self.symmetry = symmetry_class(**symmetry_cfg, device=self.device)

        self.storage = RolloutStorage(
            self.policy.is_recurrent,
            self.gpu_world_size,
            kwargs.pop("num_transitions_per_env"),
            kwargs.pop("obs_shape"),
            kwargs.pop("privileged_obs_shape"),
            kwargs.pop("actions_shape"),
            rnd_state_shape=self.rnd.output_shape if self.rnd else None,
            device=self.device,
        )

        self.transition = RolloutStorage.Transition()

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape, actions_shape):
        self.storage = RolloutStorage(
            self.policy.is_recurrent,
            self.gpu_world_size,
            num_transitions_per_env,
            actor_obs_shape,
            privileged_obs_shape,
            actions_shape,
            rnd_state_shape=self.rnd.output_shape if self.rnd else None,
            device=self.device,
        )

    def test_mode(self):
        self.policy.eval()
        if self.rnd:
            self.rnd.eval()

    def train_mode(self):
        self.policy.train()
        if self.rnd:
            self.rnd.train()

    def act(self, obs, privileged_obs):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        
        # Compute the actions and values
        # Note: we clone the observations to avoid modifying the original tensor
        # since it is used for the rnd forward pass later
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(privileged_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        
        # RND update (保持原样)
        if self.rnd:
            self.transition.rnd_state = self.rnd.act(obs.clone()).detach()

        # Need to save observations and privileged observations to the buffer
        self.transition.observations = obs
        self.transition.privileged_observations = privileged_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self, performance_ratio=None, update_actor=True):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_rnd_loss = 0
        mean_symmetry_loss = 0
        
        # =================================================================================
        # [Algorithm 1] 自适应调整逻辑
        # =================================================================================
        if self.schedule == "performance_adaptive" and performance_ratio is not None and update_actor:
            alpha = performance_ratio
            # Line 10: Compute improvement, only if alpha > 1
            improvement = max(alpha - 1.0, 0.0)
            
            if improvement > 0:
                # Line 11: Update Actor LR (Increase)
                self.current_lr_actor = min(
                    self.current_lr_actor + improvement * self.adaptive_c_v, 
                    self.lr_max
                )
                
                # Line 13: Update Critic LR (Decrease)
                self.current_lr_critic = max(
                    self.current_lr_critic - improvement * self.adaptive_c_pi, 
                    self.lr_min
                )
                
                # Line 14: Update Clip Range (Increase)
                self.clip_param = min(
                    self.clip_param + improvement * self.adaptive_c_epsilon, 
                    self.epsilon_max
                )

            # 应用到优化器
            # Group 0 is Actor, Group 1 is Critic
            self.optimizer.param_groups[0]["lr"] = self.current_lr_actor
            self.optimizer.param_groups[1]["lr"] = self.current_lr_critic
        
        # 原有的 Adaptive KL 逻辑 (如果不使用 performance_adaptive，仍可回退到此逻辑)
        elif self.desired_kl is not None and self.schedule == "adaptive":
            # 注意：如果启用了 adaptive schedule 但没有传入 performance_ratio，
            # 这里的逻辑会基于 KL 调整 Group 0 (Actor) 的学习率
            # 这是一个兼容性保留，但实际上 Algorithm 1 不使用 KL 调度
            pass 

        # =================================================================================
        # Training Loop
        # =================================================================================
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
            old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, rnd_state_batch in generator:

            # 1. Forward Pass
            # --------------------------------------------------------------------------
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch)
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch)
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            # KL calculation
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    # 如果还在使用传统 adaptive 模式，这里调整 Actor LR
                    if self.schedule == "adaptive":
                        if kl_mean > self.desired_kl * 2.0:
                            self.current_lr_actor = max(1e-5, self.current_lr_actor / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.current_lr_actor = min(1e-2, self.current_lr_actor * 1.5)
                        self.optimizer.param_groups[0]["lr"] = self.current_lr_actor

            # 2. Surrogate Loss (Actor Loss)
            # --------------------------------------------------------------------------
            # ratio = P_new / P_old
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            
            # PPO Objective
            surrogate = -torch.squeeze(advantages_batch) * ratio
            
            # [关键] 使用动态的 self.clip_param 进行裁剪
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # 3. Value Function Loss (Critic Loss)
            # --------------------------------------------------------------------------
            if self.use_clipped_value_loss:
                # 这里 Value 的 clip 也通常跟随 self.clip_param
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # 4. RND Loss (Optional)
            # --------------------------------------------------------------------------
            rnd_loss = 0
            if self.rnd:
                # The RND target (predictor_target) is fixed, we only train the predictor
                # We need to compute the predictor output for the current observations
                # The target output is computed in the storage (fixed for the batch)
                current_rnd_output = self.rnd.predictor(obs_batch)
                # The target is the rnd_state_batch (computed during rollout by target network)
                rnd_loss = (current_rnd_output - rnd_state_batch).pow(2).mean()

            # 5. Symmetry Loss (Optional)
            # --------------------------------------------------------------------------
            symmetry_loss = 0
            if self.symmetry:
                symmetry_loss = self.symmetry.get_loss(obs_batch, actions_batch, self.policy)

            # 6. Total Loss & Backward
            # --------------------------------------------------------------------------
            if update_actor:
                # 正常训练：同时优化 Actor, Critic, Entropy
                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() \
                     + rnd_loss + symmetry_loss
            else:
                # Critic Warm-up：只反向传播 Value Loss
                # 注意：为了避免梯度图问题，我们只取 value_loss 的梯度
                # 即使计算了 surrogate_loss，只要不加到 loss 里，它是不会产生梯度的（或者梯度为0）
                loss = self.value_loss_coef * value_loss + rnd_loss

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.multi_gpu_cfg is not None:
                self.reduce_parameters()
                
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Logging
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy_loss += entropy_batch.mean().item()
            if self.rnd:
                mean_rnd_loss += rnd_loss.item()
            if self.symmetry:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_rnd_loss /= num_updates
        mean_symmetry_loss /= num_updates
        
        self.storage.clear()
        
        # 返回训练统计信息，包含自适应参数状态
        stats = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy_loss,
            "param/lr_actor": self.current_lr_actor,
            "param/lr_critic": self.current_lr_critic,
            "param/clip_range": self.clip_param,
        }
        if self.rnd:
            stats["rnd_loss"] = mean_rnd_loss
        if self.symmetry:
            stats["symmetry_loss"] = mean_symmetry_loss
            
        return stats

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them."""
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        
        if len(grads) == 0:
            return

        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel