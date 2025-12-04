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
        pre_rnn_dim=16,  # Dense layer before GRU
        post_rnn_dim=16,  # Dense layer after GRU
        init_noise_std=0.1,
        teacher_recurrent=False,
        **kwargs,
    ):
        if "rnn_hidden_size" in kwargs:
            warnings.warn(
                "The argument `rnn_hidden_size` is deprecated and will be removed in a future version. "
                "Please use `rnn_hidden_dim` instead.",
                DeprecationWarning,
            )
            if rnn_hidden_dim == 16:  # Only override if the new argument is at its default
                rnn_hidden_dim = kwargs.pop("rnn_hidden_size")
        if kwargs:
            print(
                "StudentTeacherRecurrentCustom.__init__ got unexpected arguments, which will be ignored: "
                + str(kwargs.keys()),
            )

        self.teacher_recurrent = teacher_recurrent
        self.pre_rnn_dim = pre_rnn_dim
        self.post_rnn_dim = post_rnn_dim

        # Initialize parent class with modified dimensions
        # Student MLP will process: post_rnn_dim -> output
        # We'll override the student network to add pre-RNN processing
        super().__init__(
            num_student_obs=post_rnn_dim,  # Input to the MLP part after RNN
            num_teacher_obs=rnn_hidden_dim if teacher_recurrent else num_teacher_obs,
            num_actions=num_actions,
            student_hidden_dims=student_hidden_dims,
            teacher_hidden_dims=teacher_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        activation_fn = resolve_nn_activation(activation)

        # Override student network to add pre-RNN processing
        # Architecture: Input(num_student_obs) -> Dense(pre_rnn_dim) -> GRU(rnn_hidden_dim) -> Dense(post_rnn_dim) -> MLP -> Output
        
        # Pre-RNN Dense layer
        self.pre_rnn_mlp = nn.Sequential(
            nn.Linear(num_student_obs, pre_rnn_dim),
            activation_fn
        )

        # RNN layer (receives pre_rnn_dim, outputs rnn_hidden_dim)
        self.memory_s = Memory(pre_rnn_dim, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
        
        # Post-RNN Dense layer
        self.post_rnn_mlp = nn.Sequential(
            nn.Linear(rnn_hidden_dim, post_rnn_dim),
            activation_fn
        )

        # The self.student MLP (from parent class) will process post_rnn_dim -> output
        # We need to rebuild it to ensure correct input dimension
        student_layers = []
        if len(student_hidden_dims) > 0:
            student_layers.append(nn.Linear(post_rnn_dim, student_hidden_dims[0]))
            student_layers.append(activation_fn)
            for layer_index in range(len(student_hidden_dims)):
                if layer_index == len(student_hidden_dims) - 1:
                    student_layers.append(nn.Linear(student_hidden_dims[layer_index], num_actions))
                else:
                    student_layers.append(nn.Linear(student_hidden_dims[layer_index], student_hidden_dims[layer_index + 1]))
                    student_layers.append(activation_fn)
        else:
            # No hidden layers, direct connection
            student_layers.append(nn.Linear(post_rnn_dim, num_actions))
        
        self.student = nn.Sequential(*student_layers)

        if self.teacher_recurrent:
            self.memory_t = Memory(
                num_teacher_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim
            )

        print(f"Student Architecture:")
        print(f"  Pre-RNN MLP: {self.pre_rnn_mlp}")
        print(f"  Student RNN: {self.memory_s}")
        print(f"  Post-RNN MLP: {self.post_rnn_mlp}")
        print(f"  Student Output MLP: {self.student}")
        if self.teacher_recurrent:
            print(f"Teacher RNN: {self.memory_t}")
        print(f"Teacher MLP: {self.teacher}")

    def reset(self, dones=None, hidden_states=None):
        if hidden_states is None:
            hidden_states = (None, None)
        self.memory_s.reset(dones, hidden_states[0])
        if self.teacher_recurrent:
            self.memory_t.reset(dones, hidden_states[1])

    def act(self, observations):
        # Student forward: obs -> pre_rnn_mlp -> GRU -> post_rnn_mlp -> student_mlp -> action
        x = self.pre_rnn_mlp(observations)
        x = self.memory_s(x)
        x = self.post_rnn_mlp(x.squeeze(0))
        # Update distribution and sample action
        mean = self.student(x)
        std = self.std.expand_as(mean)
        self.distribution = Normal(mean, std)
        return self.distribution.sample()

    def act_inference(self, observations):
        # Student inference: obs -> pre_rnn_mlp -> GRU -> post_rnn_mlp -> student_mlp -> action
        x = self.pre_rnn_mlp(observations)
        x = self.memory_s(x)
        x = self.post_rnn_mlp(x.squeeze(0))
        actions_mean = self.student(x)
        return actions_mean

    def evaluate(self, teacher_observations):
        if self.teacher_recurrent:
            teacher_observations = self.memory_t(teacher_observations)
            teacher_observations = teacher_observations.squeeze(0)
        with torch.no_grad():
            actions = self.teacher(teacher_observations)
        return actions

    def get_hidden_states(self):
        if self.teacher_recurrent:
            return self.memory_s.hidden_states, self.memory_t.hidden_states
        else:
            return self.memory_s.hidden_states, None

    def detach_hidden_states(self, dones=None):
        self.memory_s.detach_hidden_states(dones)
        if self.teacher_recurrent:
            self.memory_t.detach_hidden_states(dones)
