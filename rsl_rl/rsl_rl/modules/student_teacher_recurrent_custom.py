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
    Custom Student-Teacher architecture with:
    - Student: Input -> Dense(pre_rnn_dim) -> GRU -> Dense(post_rnn_dim) -> Output
    - Teacher: Input -> MLP -> Output (non-recurrent)
    - Optimized for Batch Training
    """
    is_recurrent = True

    def __init__(
        self,
        num_student_obs,
        num_teacher_obs,
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
        **kwargs,
    ):
        # ... (Parameter handling same as before) ...
        if "rnn_hidden_size" in kwargs:
             if rnn_hidden_dim == 16: rnn_hidden_dim = kwargs.pop("rnn_hidden_size")

        self.teacher_recurrent = teacher_recurrent
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num_layers = rnn_num_layers
        self.rnn_type = rnn_type

        super().__init__(
            num_student_obs=post_rnn_dim, 
            num_teacher_obs=rnn_hidden_dim if teacher_recurrent else num_teacher_obs,
            num_actions=num_actions,
            student_hidden_dims=student_hidden_dims,
            teacher_hidden_dims=teacher_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        activation_fn = resolve_nn_activation(activation)

        # 1. Pre-RNN Dense layer
        self.pre_rnn_mlp = nn.Sequential(
            nn.Linear(num_student_obs, pre_rnn_dim),
            activation_fn
        )

        # 2. RNN Layer (Using nn.GRU directly for control)
        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(input_size=pre_rnn_dim, hidden_size=rnn_hidden_dim, num_layers=rnn_num_layers)
        else:
            self.rnn = nn.GRU(input_size=pre_rnn_dim, hidden_size=rnn_hidden_dim, num_layers=rnn_num_layers)
        
        # 3. Post-RNN Dense layer
        self.post_rnn_mlp = nn.Sequential(
            nn.Linear(rnn_hidden_dim, post_rnn_dim),
            activation_fn
        )

        # 4. Student MLP (Rebuild input layer)
        student_layers = []
        input_dim = post_rnn_dim
        if len(student_hidden_dims) > 0:
            student_layers.append(nn.Linear(input_dim, student_hidden_dims[0]))
            student_layers.append(activation_fn)
            for i in range(len(student_hidden_dims) - 1):
                student_layers.append(nn.Linear(student_hidden_dims[i], student_hidden_dims[i + 1]))
                student_layers.append(activation_fn)
            student_layers.append(nn.Linear(student_hidden_dims[-1], num_actions))
        else:
            student_layers.append(nn.Linear(input_dim, num_actions))
        
        self.student = nn.Sequential(*student_layers)
        
        # Internal state for Inference/Rollout
        self.hidden_state = None 

    def load_state_dict(self, state_dict, strict=True):
        """
        Overridden to handle backward compatibility with checkpoints trained 
        using the 'Memory' wrapper (where keys start with 'memory_s.rnn').
        """
        new_state_dict = state_dict.copy()
        
        # 查找所有旧格式的键 (memory_s.rnn...)
        keys_to_rename = [k for k in new_state_dict.keys() if "memory_s.rnn" in k]
        
        if len(keys_to_rename) > 0:
            print(f"[INFO] StudentTeacherRecurrentCustom: Detected legacy checkpoint keys. Renaming {len(keys_to_rename)} keys from 'memory_s.rnn' to 'rnn'...")
            for key in keys_to_rename:
                # 取出旧参数值
                val = new_state_dict.pop(key)
                # 构造新键名: memory_s.rnn.weight... -> rnn.weight...
                new_key = key.replace("memory_s.rnn", "rnn")
                new_state_dict[new_key] = val
                
        # [CRITICAL FIX] 必须加 return，否则 Runner 以为加载失败了
        return super().load_state_dict(new_state_dict, strict=strict)

    def reset(self, dones=None, hidden_states=None):
        # Reset internal hidden state for inference
        if self.hidden_state is not None and dones is not None:
             env_ids = dones.nonzero(as_tuple=False).flatten()
             if len(env_ids) > 0:
                 if self.rnn_type.lower() == 'lstm':
                     self.hidden_state[0][:, env_ids, :] = 0.0
                     self.hidden_state[1][:, env_ids, :] = 0.0
                 else:
                     self.hidden_state[:, env_ids, :] = 0.0

    def act(self, observations):
        # Inference: [Batch, Dim]
        batch_size = observations.shape[0]
        
        # Init hidden state if needed
        if self.hidden_state is None or self.hidden_state.shape[1] != batch_size:
             device = observations.device
             if self.rnn_type.lower() == 'lstm':
                 self.hidden_state = (torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_dim, device=device),
                                      torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_dim, device=device))
             else:
                 self.hidden_state = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_dim, device=device)

        x = self.pre_rnn_mlp(observations)
        x = x.unsqueeze(0) # [Seq=1, Batch, Dim]
        
        # Forward RNN
        x, self.hidden_state = self.rnn(x, self.hidden_state)
        
        x = x.squeeze(0)
        x = self.post_rnn_mlp(x)
        mean = self.student(x)
        
        self.distribution = Normal(mean, self.std.expand_as(mean))
        return self.distribution.sample()
    
    def act_inference(self, observations):
        # Similar to act but returns mean
        return self.act(observations) # act() updates self.distribution, we can just return mean from there if needed, but here let's reuse logic. 
        # Actually standard rsl_rl act_inference returns just mean actions.
        # Let's just call act and return mean to be safe and consistent
        _ = self.act(observations)
        return self.distribution.mean

    def act_batch(self, observations, hidden_states):
        """
        [NEW] Efficient forward pass for a whole batch of sequences.
        Args:
            observations: [Seq_Len, Batch, Dim]
            hidden_states: [Num_Layers, Batch, Hidden_Dim] (Initial state)
        Returns:
            actions: [Seq_Len, Batch, Action_Dim]
            next_hidden_states: [Num_Layers, Batch, Hidden_Dim] (Final state)
        """
        T, B, D = observations.shape
        
        # 1. Pre-RNN (Merge T and B)
        x = observations.view(T * B, -1)
        x = self.pre_rnn_mlp(x)
        
        # 2. RNN (Reshape to T, B, D)
        x = x.view(T, B, -1)
        x, new_hidden_states = self.rnn(x, hidden_states)
        
        # 3. Post-RNN (Merge T and B)
        x = x.contiguous().view(T * B, -1)
        x = self.post_rnn_mlp(x)
        actions = self.student(x)
        
        # Reshape back to [T, B, A]
        actions = actions.view(T, B, -1)
        
        return actions, new_hidden_states

    # Keep compatibility with existing code
    def evaluate(self, teacher_observations):
        return super().evaluate(teacher_observations)
    
    def get_hidden_states(self):
        return self.hidden_state
        
    def detach_hidden_states(self, dones=None):
        if self.hidden_state is not None:
             if self.rnn_type.lower() == 'lstm':
                 self.hidden_state = (self.hidden_state[0].detach(), self.hidden_state[1].detach())
             else:
                 self.hidden_state = self.hidden_state.detach()