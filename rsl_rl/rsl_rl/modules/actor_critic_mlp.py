


from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from math import floor

from rsl_rl.utils import resolve_nn_activation


class ActorCriticMLP(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256,256],  # For fusion MLP
        critic_hidden_dims=[256,256],
        state_hidden_dims=[256,256],  # For state MLP
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        depth_height=60,
        depth_width=80,
        history_length=1,
        frame_state_dim=20,  # Per-frame state (24 + 4 TOF)
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
        self.history_length = history_length
        self.frame_depth_dim = depth_height * depth_width  # e.g., 60 * 80 = 4800
        self.depth_dim = self.frame_depth_dim * history_length  # e.g., 4800 * 2 = 9600
        self.frame_state_dim = frame_state_dim  # 27 (24 state + 3 TOF)
        self.state_dim = frame_state_dim * history_length  # 54
        self.frame_pre_depth_dim = 20  # State before depth_image
        self.frame_post_depth_dim = 0  # TOF after depth_image
        self.frame_obs_dim = self.frame_pre_depth_dim + self.frame_depth_dim + self.frame_post_depth_dim  # e.g., 24 + 4800 + 3 = 4827

        # Validate observation dimension
        expected_obs_dim = self.frame_obs_dim * history_length  # e.g., 4827 * 2 = 9654
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
            nn.Conv2d(history_length, 16, kernel_size=4, stride=2, padding=1),  # [batch, 2, h, w] -> [batch, 16, h/2, w/2]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(cnn_out_dim, 256),  # [batch, 256]
            nn.ReLU(),
        )

        # Critic CNN for depth image
        self.critic_cnn = nn.Sequential(
            nn.Conv2d(history_length, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(cnn_out_dim, 256),
            nn.ReLU(),
        )

        # Actor state MLP
        self.actor_state_mlp = nn.Sequential(
            nn.Linear(self.state_dim, state_hidden_dims[0]),  # [batch, 54] -> [batch, 128]
            activation,
            nn.Linear(state_hidden_dims[0], state_hidden_dims[1]),  # [batch, 128] -> [batch, 128]
            activation,
        )

        # Critic state MLP
        self.critic_state_mlp = nn.Sequential(
            nn.Linear(self.state_dim, state_hidden_dims[0]),  # [batch, 54] -> [batch, 128]
            activation,
            nn.Linear(state_hidden_dims[0], state_hidden_dims[1]),  # [batch, 128] -> [batch, 128]
            activation,
        )
        self.num_actions = int(num_actions)
        # Fusion MLP for actor
        actor_layers = []
        actor_layers.append(nn.Linear(256 + state_hidden_dims[-1], actor_hidden_dims[0]))  # [batch, 256+128=384] -> [batch, 256]
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))  # [batch, 256] -> [batch, 4]
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Fusion MLP for critic
        critic_layers = []
        critic_layers.append(nn.Linear(256 + state_hidden_dims[-1], critic_hidden_dims[0]))  # [batch, 256+128=384] -> [batch, 256]
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))  # [batch, 256] -> [batch, 1]
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor CNN: {self.actor_cnn}")
        print(f"Actor State MLP: {self.actor_state_mlp}")
        print(f"Actor Fusion MLP: {self.actor}")
        print(f"Critic CNN: {self.critic_cnn}")
        print(f"Critic State MLP: {self.critic_state_mlp}")
        print(f"Critic Fusion MLP: {self.critic}")

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
        pass

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

    def update_distribution(self, observations):
        # Extract depth and state
        depth_t1 = observations[:, self.frame_pre_depth_dim:self.frame_pre_depth_dim + self.frame_depth_dim]  # t-1 depth [batch, frame_depth_dim]
        depth_t = observations[:, self.frame_obs_dim + self.frame_pre_depth_dim:self.frame_obs_dim + self.frame_pre_depth_dim + self.frame_depth_dim]  # t depth [batch, frame_depth_dim]
        state_t1 = torch.cat([
            observations[:, :self.frame_pre_depth_dim],  # t-1 state [batch, 24]
            # observations[:, self.frame_pre_depth_dim + self.frame_depth_dim:self.frame_pre_depth_dim + self.frame_depth_dim + self.frame_post_depth_dim],  # t-1 TOF [batch, 3]
        ], dim=-1)  # [batch, 27]
        state_t2 = torch.cat([
            observations[:, self.frame_obs_dim:self.frame_obs_dim + self.frame_pre_depth_dim],  # t state [batch, 24]
            # observations[:, self.frame_obs_dim + self.frame_pre_depth_dim + self.frame_depth_dim:],  # t TOF [batch, 3]
        ], dim=-1)  # [batch, 27]

        # Combine depth and state
        depth_obs = torch.cat([depth_t1, depth_t], dim=-1)  # [batch, depth_dim]
        state_obs = torch.cat([state_t1, state_t2], dim=-1)  # [batch, 54]

        # Reshape depth for CNN
        depth_obs = depth_obs.view(-1, self.history_length, self.depth_height, self.depth_width)

        # Process depth with Actor CNN
        depth_features = self.actor_cnn(depth_obs)  # [batch, 256]

        # Process state with Actor MLP
        state_features = self.actor_state_mlp(state_obs)  # [batch, 128]

        # Concatenate features
        fused_features = torch.cat([depth_features, state_features], dim=-1)  # [batch, 256+128=384]

        # Compute mean with fusion MLP
        mean = self.actor(fused_features)  # [batch, 4]

        # Compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        # Extract depth and state
        depth_t1 = observations[:, self.frame_pre_depth_dim:self.frame_pre_depth_dim + self.frame_depth_dim]
        depth_t = observations[:, self.frame_obs_dim + self.frame_pre_depth_dim:self.frame_obs_dim + self.frame_pre_depth_dim + self.frame_depth_dim]
        state_t1 = torch.cat([
            observations[:, :self.frame_pre_depth_dim],
        #     observations[:, self.frame_pre_depth_dim + self.frame_depth_dim:self.frame_pre_depth_dim + self.frame_depth_dim + self.frame_post_depth_dim],
        ], dim=-1)
        state_t2 = torch.cat([
            observations[:, self.frame_obs_dim:self.frame_obs_dim + self.frame_pre_depth_dim],
            # observations[:, self.frame_obs_dim + self.frame_pre_depth_dim + self.frame_depth_dim:],
        ], dim=-1)

        depth_obs = torch.cat([depth_t1, depth_t], dim=-1)
        state_obs = torch.cat([state_t1, state_t2], dim=-1)

        # Reshape depth
        depth_obs = depth_obs.view(-1, self.history_length, self.depth_height, self.depth_width)

        # Process features
        depth_features = self.actor_cnn(depth_obs)
        state_features = self.actor_state_mlp(state_obs)
        fused_features = torch.cat([depth_features, state_features], dim=-1)

        # Compute mean
        actions_mean = self.actor(fused_features)
        return actions_mean

    def evaluate(self, critic_observations, masks=None, hidden_states=None, **kwargs):
        # Extract depth and state
        depth_t1 = critic_observations[:, self.frame_pre_depth_dim:self.frame_pre_depth_dim + self.frame_depth_dim]
        depth_t = critic_observations[:, self.frame_obs_dim + self.frame_pre_depth_dim:self.frame_obs_dim + self.frame_pre_depth_dim + self.frame_depth_dim]
        state_t1 = torch.cat([
            critic_observations[:, :self.frame_pre_depth_dim],
            # critic_observations[:, self.frame_pre_depth_dim + self.frame_depth_dim:self.frame_pre_depth_dim + self.frame_depth_dim + self.frame_post_depth_dim],
        ], dim=-1)
        state_t2 = torch.cat([
            critic_observations[:, self.frame_obs_dim:self.frame_obs_dim + self.frame_pre_depth_dim],
            # critic_observations[:, self.frame_obs_dim + self.frame_pre_depth_dim + self.frame_depth_dim:],
        ], dim=-1)

        depth_obs = torch.cat([depth_t1, depth_t], dim=-1)
        state_obs = torch.cat([state_t1, state_t2], dim=-1)

        # Reshape depth
        depth_obs = depth_obs.view(-1, self.history_length, self.depth_height, self.depth_width)

        # Process depth with Critic CNN
        depth_features = self.critic_cnn(depth_obs)  # [batch, 256]

        # Process state with Critic MLP
        state_features = self.critic_state_mlp(state_obs)  # [batch, 128]

        # Concatenate features
        fused_features = torch.cat([depth_features, state_features], dim=-1)  # [batch, 256+128=384]

        # Compute value
        value = self.critic(fused_features)  # [batch, 1]
        return value

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True