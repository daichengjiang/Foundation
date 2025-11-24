


from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.networks import Memory
from math import floor

from rsl_rl.utils import resolve_nn_activation

class ActorCriticAtten(nn.Module):
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],  # For fusion MLP
        critic_hidden_dims=[256, 256, 256],
        state_hidden_dims=[128, 64],  # For state MLP
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        depth_cnn_out_dim=256,
        depth_height=60,
        depth_width=80,
        obs_size=20,
        depth_history_length = 1, #2
        obs_history_length =0, #10
        rnn_type="gru",
        rnn_hidden_dim=512,
        rnn_num_layers=1,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        # Observation dimensions
        self.depth_height = depth_height
        self.depth_width = depth_width
        self.depth_history_length = depth_history_length
        self.depth_cnn_out_dim = depth_cnn_out_dim
        self.obs_size = obs_size
        self.obs_history_length = obs_history_length

        # Validate observation dimension
        expected_obs_dim = self.obs_size + self.obs_history_length * self.obs_size + self.depth_history_length * self.depth_width * self.depth_height
        if num_actor_obs != expected_obs_dim or num_critic_obs != expected_obs_dim:
            raise ValueError(f"Observation dimensions mismatch: actor ({num_actor_obs}), critic ({num_critic_obs}), expected ({expected_obs_dim})")

        # Calculate CNN output size
        def calc_conv_output_size(h, w, kernel_size=4, stride=2, padding=1):
            h_out = floor((h + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
            w_out = floor((w + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
            return h_out, w_out

        h, w = depth_height, depth_width
        for _ in range(3):  # 3 conv layers
            h, w = calc_conv_output_size(h, w)
        cnn_out_channels = 64
        cnn_out_dim = cnn_out_channels * h * w  # e.g., 64 * 7 * 10 = 4480 for 60x80

        # Actor CNN for depth image
        self.actor_cnn = nn.Sequential(
            nn.Conv2d(self.depth_history_length, 16, kernel_size=4, stride=2, padding=1),  # [batch, 2, h, w] -> [batch, 16, h/2, w/2]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(cnn_out_dim, self.depth_cnn_out_dim),  # [batch, 256]
            nn.ReLU(),
        )

        # Critic CNN for depth image
        self.critic_cnn = nn.Sequential(
            nn.Conv2d(self.depth_history_length, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(cnn_out_dim, self.depth_cnn_out_dim),
            nn.ReLU(),
        )

        if self.obs_history_length != 0:
            # Actor observation history mlp
            self.actor_obs_his_mlp = nn.Sequential(
                nn.Linear(self.obs_history_length * self.obs_size, state_hidden_dims[0]),
                nn.Linear(state_hidden_dims[0], state_hidden_dims[1]), 
                activation,
            )

            # Critic observation history mlp
            self.critic_obs_his_mlp = nn.Sequential(
                nn.Linear(self.obs_history_length * self.obs_size, state_hidden_dims[0]), 
                nn.Linear(state_hidden_dims[0], state_hidden_dims[1]), 
                activation,
            )
        else:
            self.actor_obs_his_mlp = nn.Identity()
            self.critic_obs_his_mlp = nn.Identity()

        # self.memory_a = Memory(depth_cnn_out_dim+state_hidden_dims[-1], type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
        # self.memory_c = Memory(depth_cnn_out_dim+state_hidden_dims[-1], type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
        encoder_layer_a = nn.TransformerEncoderLayer(depth_cnn_out_dim+self.obs_size, nhead=12, dropout=0.0)
        encoder_layer_c = nn.TransformerEncoderLayer(depth_cnn_out_dim+self.obs_size, nhead=12, dropout=0.0)
        self.atten_a = nn.TransformerEncoder(encoder_layer_a, num_layers=1)
        self.atten_c = nn.TransformerEncoder(encoder_layer_c, num_layers=1)

        self.memory_a = Memory(depth_cnn_out_dim+self.obs_size, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
        self.memory_c = Memory(depth_cnn_out_dim+self.obs_size, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)

        # MLP for actor
        actor_layers = []
        actor_layers.append(nn.Linear(rnn_hidden_dim, actor_hidden_dims[0]))  
        # actor_layers.append(nn.Linear(rnn_hidden_dim, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))  
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # MLP for critic
        critic_layers = []
        critic_layers.append(nn.Linear(rnn_hidden_dim, critic_hidden_dims[0]))  
        # critic_layers.append(nn.Linear(rnn_hidden_dim, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))  
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)
        
        
        print(f"Actor CNN: {self.actor_cnn}")
        print(f"Actor Observation History MLP: {self.actor_obs_his_mlp}")
        print(f"Actor Fusion GRU: {self.memory_a}")
        print(f"Actor: {self.actor}")
        print(f"Critic CNN: {self.critic_cnn}")
        print(f"Critic Observation History MLP: {self.critic_obs_his_mlp}")
        print(f"Critic Fusion GRU: {self.memory_c}")
        print(f"Critic: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution
        self.distribution = None
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, masks=None, hidden_states=None):
        if masks is not None:
            unpadded_obs = observations[masks].reshape(-1, self.obs_size + self.obs_history_length * self.obs_size + self.depth_history_length * self.depth_height * self.depth_width)
            max_seq_len = observations.shape[0]
            num_traj = observations.shape[1]
        else:
            unpadded_obs = observations

        obs = unpadded_obs[:, :self.obs_size]

        depth_obs = unpadded_obs[:, self.obs_size + self.obs_history_length * self.obs_size:]

        # Reshape depth for CNN
        depth_obs = depth_obs.reshape(-1, self.depth_history_length, self.depth_height, self.depth_width)

        # Process depth with Actor CNN
        depth_features = self.actor_cnn(depth_obs)  

        if self.obs_history_length != 0:
            obs_history = unpadded_obs[:, self.obs_size:self.obs_history_length * self.obs_size]
            obs_history = unpadded_obs[:, self.obs_size:self.obs_size + self.obs_history_length * self.obs_size]
            obs_history_features = self.actor_obs_his_mlp(obs_history)  
            # Concatenate features
            fused_features = torch.cat([obs_history_features, depth_features], dim=-1) 

        else:
            fused_features = torch.cat([obs, depth_features], dim=-1)

        if masks is not None:
            # Generate the memory of surroundings
            features = torch.zeros(max_seq_len, num_traj, fused_features.shape[-1], device=unpadded_obs.device)
            features[masks] = fused_features
            features = self.atten_a(features)  # Apply attention
            features = self.memory_a(features, masks, hidden_states).squeeze(0)
        else:
            features = fused_features
            features = self.atten_a(features)  # Apply attention
            features = self.memory_a(features, masks, hidden_states).squeeze(0)

        # Compute mean with fusion MLP
        mean = self.actor(features)  

        # Compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations, masks=None, hidden_states=None, **kwargs):
        self.update_distribution(observations, masks, hidden_states)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        obs = observations[:, :self.obs_size]

        depth_obs = observations[:, self.obs_size + self.obs_history_length * self.obs_size:]
        # Reshape depth for CNN
        depth_obs = depth_obs.reshape(-1, self.depth_history_length, self.depth_height, self.depth_width)

        # Process depth with Actor CNN
        depth_features = self.actor_cnn(depth_obs)

        if self.obs_history_length != 0:
            obs_history = observations[:, self.obs_size:self.obs_history_length*self.obs_size]
            obs_history = observations[:, self.obs_size:self.obs_size + self.obs_history_length * self.obs_size]
            obs_history_features = self.actor_obs_his_mlp(obs_history)
            fused_features = torch.cat([obs_history_features, depth_features], dim=-1)
        else:
            fused_features = torch.cat([obs, depth_features], dim=-1)
        
        fused_features = self.atten_a(fused_features)  # Apply attention
        # Generate the memory of surroundings
        fused_features = self.memory_a(fused_features).squeeze(0)  

        # Compute mean
        actions_mean = self.actor(fused_features)
        return actions_mean

    def evaluate(self, critic_observations, masks=None, hidden_states=None, **kwargs):
        if masks is not None:
            unpadded_obs = critic_observations[masks].reshape(-1, self.obs_size + self.obs_history_length * self.obs_size + self.depth_history_length * self.depth_height * self.depth_width)
            max_seq_len = critic_observations.shape[0]
            num_traj = critic_observations.shape[1]
        else:
            unpadded_obs = critic_observations

        obs = unpadded_obs[:, :self.obs_size]

        depth_obs = unpadded_obs[:, self.obs_size + self.obs_history_length * self.obs_size:]
        # Reshape depth for CNN
        depth_obs = depth_obs.reshape(-1, self.depth_history_length, self.depth_height, self.depth_width)

        # Process depth with Actor CNN
        depth_features = self.critic_cnn(depth_obs) 

        if self.obs_history_length != 0:
            obs_history = unpadded_obs[:, self.obs_size:self.obs_history_length*self.obs_size]
            obs_history = unpadded_obs[:, self.obs_size:self.obs_size + self.obs_history_length * self.obs_size]
            obs_history_features = self.critic_obs_his_mlp(obs_history)
            fused_features = torch.cat([obs_history_features, depth_features], dim=-1)
        else:
            fused_features = torch.cat([obs, depth_features], dim=-1)

        if masks is not None:
            # Generate the memory of surroundings
            features = torch.zeros(max_seq_len, num_traj, fused_features.shape[-1], device=unpadded_obs.device)
            features[masks] = fused_features
            self.atten_c(features)  # Apply attention
            features = self.memory_c(features, masks, hidden_states).squeeze(0)
        else:
            features = fused_features
            self.atten_c(features)
            features = self.memory_c(features, masks, hidden_states).squeeze(0)

        # Compute value
        value = self.critic(features)  # [batch, 1]
        return value

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True