# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules import StudentTeacher
from rsl_rl.networks import Memory
from rsl_rl.utils import resolve_nn_activation


class StudentTeacherRecurrentCustom(StudentTeacher):
    """
    Custom Student-Teacher architecture adapted for PPO:
    - Actor (Student): Input(22) -> Dense -> GRU -> Dense -> Output(4)
    - Critic: Input(22) -> MLP -> Output(1)
    """
    is_recurrent = True

    def __init__(
        self,
        num_student_obs,
        num_teacher_obs, # PPO Runner 会传入 22
        num_actions,
        student_hidden_dims=[256, 256, 128],
        teacher_hidden_dims=[64, 64, 64],
        activation="elu",
        rnn_type="gru",
        rnn_hidden_dim=16,
        rnn_num_layers=1,
        pre_rnn_dim=16,
        post_rnn_dim=16,
        init_noise_std=0.1,
        teacher_recurrent=False,
        noise_std_type="scalar", 
        **kwargs,
    ):
        if "rnn_hidden_size" in kwargs:
             if rnn_hidden_dim == 16: rnn_hidden_dim = kwargs.pop("rnn_hidden_size")

        self.teacher_recurrent = teacher_recurrent
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num_layers = rnn_num_layers
        self.rnn_type = rnn_type
        self.noise_std_type = noise_std_type

        base_student_dims = student_hidden_dims if len(student_hidden_dims) > 0 else [64]

        super().__init__(
            num_student_obs=post_rnn_dim, 
            num_teacher_obs=rnn_hidden_dim if teacher_recurrent else num_teacher_obs,
            num_actions=num_actions,
            student_hidden_dims=base_student_dims,
            teacher_hidden_dims=teacher_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        activation_fn = resolve_nn_activation(activation)

        # --------------------------------------------------------
        # 1. Student / Actor Architecture (RNN)
        # --------------------------------------------------------
        self.pre_rnn_mlp = nn.Sequential(
            nn.Linear(num_student_obs, pre_rnn_dim),
            activation_fn
        )

        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(input_size=pre_rnn_dim, hidden_size=rnn_hidden_dim, num_layers=rnn_num_layers)
        else:
            self.rnn = nn.GRU(input_size=pre_rnn_dim, hidden_size=rnn_hidden_dim, num_layers=rnn_num_layers)
        
        if post_rnn_dim == rnn_hidden_dim:
            self.post_rnn_mlp = nn.Identity()
        else:
            self.post_rnn_mlp = nn.Sequential(
                nn.Linear(rnn_hidden_dim, post_rnn_dim),
                activation_fn
            )

        student_layers = []
        input_dim = post_rnn_dim
        if len(student_hidden_dims) > 0:
            student_layers.append(nn.Linear(input_dim, student_hidden_dims[0]))
            student_layers.append(activation_fn)
            for i in range(len(student_hidden_dims) - 1):
                student_layers.append(nn.Linear(student_hidden_dims[i], student_hidden_dims[i + 1]))
                student_layers.append(activation_fn)
            student_layers.append(nn.Linear(student_hidden_dims[-1], num_actions))
            student_layers.append(nn.Tanh())
        else:
            student_layers.append(nn.Linear(input_dim, num_actions))
            student_layers.append(nn.Identity())
        
        self.student = nn.Sequential(*student_layers)
        
        # --------------------------------------------------------
        # 2. Critic Architecture (Value Function)
        # --------------------------------------------------------
        critic_layers = []
        critic_input_dim = num_teacher_obs 
        
        critic_layers.append(nn.Linear(critic_input_dim, teacher_hidden_dims[0]))
        critic_layers.append(activation_fn)
        for i in range(len(teacher_hidden_dims) - 1):
            critic_layers.append(nn.Linear(teacher_hidden_dims[i], teacher_hidden_dims[i + 1]))
            critic_layers.append(activation_fn)
            
        # Critic 输出维度强制为 1 (Value)
        critic_layers.append(nn.Linear(teacher_hidden_dims[-1], 1))
        
        self.critic = nn.Sequential(*critic_layers)
        # --------------------------------------------------------
        
        self.hidden_state = None 
        self._last_mean = None

    def load_state_dict(self, state_dict, strict=True):
        new_state_dict = state_dict.copy()
        keys_to_rename = [k for k in new_state_dict.keys() if "memory_s.rnn" in k]
        
        if len(keys_to_rename) > 0:
            print(f"[INFO] StudentTeacherRecurrentCustom: Renaming {len(keys_to_rename)} legacy keys...")
            for key in keys_to_rename:
                val = new_state_dict.pop(key)
                new_key = key.replace("memory_s.rnn", "rnn")
                new_state_dict[new_key] = val
        
        return super().load_state_dict(new_state_dict, strict=strict)

    def reset(self, dones=None, hidden_states=None):
        if self.hidden_state is not None and dones is not None:
             env_ids = dones.nonzero(as_tuple=False).flatten()
             if len(env_ids) > 0:
                 if self.rnn_type.lower() == 'lstm':
                     self.hidden_state[0][:, env_ids, :] = 0.0
                     self.hidden_state[1][:, env_ids, :] = 0.0
                 else:
                     self.hidden_state[:, env_ids, :] = 0.0


    def _forward_head(self, observations, hidden_states=None):
        """
        处理前向传播，自动适配 Rollout (2D输入) 和 PPO Training (3D输入)。
        """
        # 判断是训练模式(Sequence)还是推理模式(Step)
        # 训练时 PPO 传入的 observations 是 [Seq_Len, Batch, Dim] (3D)
        # 推理时 Runner 传入的 observations 是 [Batch, Dim] (2D)
        is_training_seq = observations.dim() == 3
        
        if is_training_seq:
            x = observations # [Seq, Batch, Dim]
        else:
            x = observations.unsqueeze(0) # [1, Batch, Dim]

        device = x.device
        batch_size = x.shape[1]

        # 1. MLP Pre-processing
        # nn.Linear 支持多维输入，只要最后一维对齐即可
        x = self.pre_rnn_mlp(x)

        # 2. RNN Processing
        # 如果是 PPO 训练，必须使用传入的 hidden_states (历史状态)
        if hidden_states is not None:
            h = hidden_states
        else:
            # 如果是推理，使用内部维护的 self.hidden_state
            if self.hidden_state is None or self.hidden_state.shape[1] != batch_size:
                 if self.rnn_type.lower() == 'lstm':
                     self.hidden_state = (torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_dim, device=device),
                                          torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_dim, device=device))
                 else:
                     self.hidden_state = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_dim, device=device)
            h = self.hidden_state

        # RNN Forward
        # x: [Seq, Batch, Dim], h: [Layers, Batch, Hidden]
        x, h_out = self.rnn(x, h)

        # 3. Post-processing & Output
        if not isinstance(self.post_rnn_mlp, nn.Identity):
            x = self.post_rnn_mlp(x)
            
        mean = self.student(x)

        # 4. State Update Logic
        if not is_training_seq:
            # 如果是推理模式，我们要更新内部状态，并且移除 Sequence 维度
            self.hidden_state = h_out
            mean = mean.squeeze(0) # [1, Batch, Act] -> [Batch, Act]
        
        # 如果是训练模式，mean 保持 [Seq, Batch, Act]，不需要更新 self.hidden_state
        
        return mean
    
    def act_inference(self, observations):
        return self._forward_head(observations)

    # def act(self, observations, **kwargs):
    #     mean = self._forward_head(observations)
    #     self._last_mean = mean
        
    #     if self.std.mean() < 1e-6:
    #         self.distribution = None
    #         return mean
            
    #     self.distribution = Normal(mean, self.std.expand_as(mean))
    #     return self.distribution.sample()
    def act(self, observations, **kwargs):
        # [关键修复] 获取 PPO 传入的 hidden_states (训练时存在)
        hidden_states = kwargs.get('hidden_states', None)
        
        mean = self._forward_head(observations, hidden_states)
        self._last_mean = mean
        
        if self.std.mean() < 1e-6:
            self.distribution = None
            return mean
            
        self.distribution = Normal(mean, self.std.expand_as(mean))
        return self.distribution.sample()
    # === PPO Compatibility Interfaces ===
    
    def get_actions_log_prob(self, actions):
        if self.distribution is None:
            return torch.zeros(actions.shape[0], device=actions.device)
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self):
        if self.distribution is not None:
            return self.distribution.mean
        if self._last_mean is not None:
            return self._last_mean
        return None

    @property
    def action_std(self):
        return self.std

    @property
    def entropy(self):
        if self.distribution is None:
             return torch.zeros(1, device=self.std.device)
        return self.distribution.entropy().sum(dim=-1)

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
    
    def act_batch(self, observations, hidden_states):
        T, B, D = observations.shape
        x = observations.view(T * B, -1)
        x = self.pre_rnn_mlp(x)
        x = x.view(T, B, -1)
        x, new_hidden_states = self.rnn(x, hidden_states)
        x = x.contiguous().view(T * B, -1)
        if not isinstance(self.post_rnn_mlp, nn.Identity):
            x = self.post_rnn_mlp(x)
        actions = self.student(x)
        actions = actions.view(T, B, -1)
        return actions, new_hidden_states

    def get_hidden_states(self):
        # RolloutStorage 要求 Actor 和 Critic 的隐状态数量必须一致（循环次数基于 hid_a）
        # 所以我们必须给 Critic 一个假的隐状态（直接复用 Actor 的），以骗过 assert/index error。
        # 这样会多占一点点显存，但是能跑通。
        
        actor_state = self.hidden_state
        
        # 1. 如果尚未初始化 (None)，构造全 0 的 dummy state
        if actor_state is None:
            device = self.std.device
            if self.rnn_type.lower() == 'lstm':
                actor_state = (torch.zeros(self.rnn_num_layers, 1, self.rnn_hidden_dim, device=device),
                               torch.zeros(self.rnn_num_layers, 1, self.rnn_hidden_dim, device=device))
            else:
                actor_state = torch.zeros(self.rnn_num_layers, 1, self.rnn_hidden_dim, device=device)
        
        # 2. 将 Actor State 赋值给 Critic State (Dummy)
        critic_state = actor_state
        
        # 3. 返回 (Actor, Critic)
        # 此时 len(hid_a) == len(hid_c)，RolloutStorage 就会开心地工作了
        return actor_state, critic_state
        
    def detach_hidden_states(self, dones=None):
        if self.hidden_state is not None:
             if self.rnn_type.lower() == 'lstm':
                 self.hidden_state = (self.hidden_state[0].detach(), self.hidden_state[1].detach())
             else:
                 self.hidden_state = self.hidden_state.detach()