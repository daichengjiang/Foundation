#!/usr/bin/env python3
# Copyright (c) 2025, Foundation Project
# Teacher Policy Distillation Script
# Based on the C++ implementation in raptor/rl-tools

"""
Teacher policy distillation for training a student policy to mimic a single teacher policy.
This script implements behavior cloning through supervised learning.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# Add command-line arguments
parser = argparse.ArgumentParser(description="Distill a teacher policy into a student policy")
parser.add_argument("--teacher_checkpoint", type=str, required=True, help="Path to teacher checkpoint")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

# Distillation hyperparameters
parser.add_argument("--n_epochs", type=int, default=500, help="Number of training epochs")
parser.add_argument("--num_episodes", type=int, default=10, help="Episodes per epoch for data collection")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--sequence_length", type=int, default=1, help="Sequence length for RNN (1 for MLP)")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--epoch_teacher_forcing", type=int, default=50, help="Epochs to use teacher only")
parser.add_argument("--shuffle", action="store_true", default=True, help="Shuffle episodes")
parser.add_argument("--on_policy", action="store_true", default=False, help="Use on-policy data collection")
parser.add_argument("--teacher_deterministic", action="store_true", default=True, help="Use deterministic teacher")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# Import after launching Isaac Sim
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from foundation import tasks

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


class TeacherPolicyWrapper:
    """Wrapper for teacher policy to handle inference."""
    
    def __init__(self, checkpoint_path: str, device: str):
        self.device = torch.device(device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract policy
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "policy" in checkpoint:
            state_dict = checkpoint["policy"]
        else:
            state_dict = checkpoint
        
        # Extract actor parameters
        actor_state_dict = {}
        for key, value in state_dict.items():
            if "actor" in key:
                actor_state_dict[key.replace("actor.", "")] = value
        
        # Infer network architecture from state dict
        self.hidden_dims = []
        layer_idx = 0
        while f"{layer_idx}.weight" in actor_state_dict:
            weight = actor_state_dict[f"{layer_idx}.weight"]
            self.hidden_dims.append(weight.shape[0])
            layer_idx += 2  # Skip activation layers
        
        if len(self.hidden_dims) > 0:
            self.num_obs = actor_state_dict["0.weight"].shape[1]
            self.num_actions = self.hidden_dims[-1]
            self.hidden_dims = self.hidden_dims[:-1]
        
        print(f"Teacher architecture: obs={self.num_obs}, actions={self.num_actions}, hidden={self.hidden_dims}")
        
        # Reconstruct network (assuming ELU activation)
        layers = []
        in_dim = self.num_obs
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ELU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, self.num_actions))
        
        self.network = nn.Sequential(*layers).to(self.device)
        self.network.load_state_dict(actor_state_dict)
        self.network.eval()
    
    @torch.no_grad()
    def act(self, obs, deterministic=True):
        """Get actions from teacher policy."""
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        
        actions = self.network(obs)
        
        if not deterministic:
            # Add small noise for exploration
            actions = actions + torch.randn_like(actions) * 0.01
        
        return actions


class StudentPolicy(nn.Module):
    """Student policy network with GRU (matching C++ implementation).
    
    Architecture (from C++ config.h):
    - INPUT_LAYER: Dense(obs_dim -> hidden_dim=16) + ReLU
    - GRU: GRU(hidden_dim=16)
    - OUTPUT_LAYER: Dense(hidden_dim=16 -> action_dim) + Identity
    
    Note: Student uses reduced observation (22D) without motor_speeds (last 4D),
          while teacher uses full observation (26D).
    """
    
    def __init__(self, num_obs: int, num_actions: int, hidden_dim: int = 16, 
                 activation: str = "relu", obs_slice: tuple = None):
        super().__init__()
        
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.is_recurrent = True
        self.obs_slice = obs_slice  # (start_idx, end_idx) to slice observations
        
        # INPUT_LAYER: Dense + Activation
        self.input_layer = nn.Linear(num_obs, hidden_dim)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=False  # (seq_len, batch, features)
        )
        
        # OUTPUT_LAYER: Dense (no activation, identity)
        self.output_layer = nn.Linear(hidden_dim, num_actions)
        
        # Hidden states for recurrent processing
        self.hidden_states = None
        
        # Initialize weights (matching rl-tools default initialization)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights similar to rl-tools DefaultInitializer."""
        if isinstance(module, nn.Linear):
            # Use orthogonal initialization for linear layers
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.GRU):
            # Initialize GRU weights
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)
    
    def reset(self, batch_size: int = None, device: str = None):
        """Reset hidden states."""
        if batch_size is not None:
            device = device or next(self.parameters()).device
            self.hidden_states = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        else:
            self.hidden_states = None
    
    def forward(self, obs, hidden_states=None):
        """Forward pass.
        
        Args:
            obs: Observations, shape (seq_len, batch, obs_dim) or (batch, obs_dim)
            hidden_states: GRU hidden states, shape (1, batch, hidden_dim)
        
        Returns:
            actions: Output actions, shape (seq_len, batch, action_dim) or (batch, action_dim)
            hidden_states: Updated hidden states
        """
        # Apply observation slicing if configured
        if self.obs_slice is not None:
            start_idx, end_idx = self.obs_slice
            obs = obs[..., start_idx:end_idx]
        
        # Handle both (batch, obs_dim) and (seq_len, batch, obs_dim) inputs
        input_is_2d = obs.dim() == 2
        if input_is_2d:
            # Add sequence dimension: (batch, obs_dim) -> (1, batch, obs_dim)
            obs = obs.unsqueeze(0)
        
        # INPUT_LAYER: obs -> hidden with activation
        # Shape: (seq_len, batch, obs_dim) -> (seq_len, batch, hidden_dim)
        x = self.input_layer(obs)
        x = self.activation(x)
        
        # GRU: process sequence
        # x: (seq_len, batch, hidden_dim)
        # hidden_states: (1, batch, hidden_dim) or None
        if hidden_states is None:
            hidden_states = self.hidden_states
        
        x, new_hidden_states = self.gru(x, hidden_states)
        
        # OUTPUT_LAYER: hidden -> actions (no activation)
        # Shape: (seq_len, batch, hidden_dim) -> (seq_len, batch, action_dim)
        actions = self.output_layer(x)
        
        # Remove sequence dimension if input was 2D
        if input_is_2d:
            actions = actions.squeeze(0)
        
        # Update internal hidden states
        self.hidden_states = new_hidden_states.detach()
        
        return actions
    
    def act(self, obs, hidden_states=None):
        """Get actions from observations (inference mode)."""
        with torch.no_grad():
            return self.forward(obs, hidden_states)
    
    def detach_hidden_states(self):
        """Detach hidden states from computation graph."""
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.detach()


class DistillationDataset:
    """Dataset for storing teacher demonstrations."""
    
    def __init__(self, max_size: int, obs_dim: int, action_dim: int, device: str):
        self.max_size = max_size
        self.device = torch.device(device)
        
        # Preallocate buffers
        self.observations = torch.zeros((max_size, obs_dim), device=self.device)
        self.actions = torch.zeros((max_size, action_dim), device=self.device)
        self.episode_starts = torch.zeros(max_size, dtype=torch.long, device=self.device)
        self.truncated = torch.zeros(max_size, dtype=torch.bool, device=self.device)
        
        self.size = 0
        self.num_episodes = 0
    
    def add_episode(self, observations, actions, truncated):
        """Add a complete episode to the dataset."""
        episode_len = len(observations)
        
        if self.size + episode_len > self.max_size:
            print(f"Warning: Dataset full, skipping episode")
            return
        
        start_idx = self.size
        end_idx = self.size + episode_len
        
        self.observations[start_idx:end_idx] = observations
        self.actions[start_idx:end_idx] = actions
        self.episode_starts[self.num_episodes] = start_idx
        self.truncated[end_idx - 1] = truncated
        
        self.size = end_idx
        self.num_episodes += 1
    
    def clear(self):
        """Clear the dataset."""
        self.size = 0
        self.num_episodes = 0
    
    def get_batches(self, batch_size: int, sequence_length: int, shuffle: bool = True):
        """Generate batches for training with sequence support.
        
        For RNN training, generates batches of shape (sequence_length, batch_size, feature_dim).
        For MLP training (sequence_length=1), generates batches of shape (batch_size, feature_dim).
        """
        if self.num_episodes == 0:
            return
        
        # Get episode indices
        episode_indices = torch.arange(self.num_episodes, device=self.device)
        
        if shuffle:
            episode_indices = episode_indices[torch.randperm(self.num_episodes)]
        
        if sequence_length > 1:
            # RNN mode: generate sequences
            for ep_idx in episode_indices:
                start_idx = self.episode_starts[ep_idx].item()
                
                # Find episode end
                if ep_idx + 1 < self.num_episodes:
                    end_idx = self.episode_starts[ep_idx + 1].item()
                else:
                    end_idx = self.size
                
                episode_len = end_idx - start_idx
                
                # Split episode into sequences
                num_sequences = max(1, episode_len // sequence_length)
                
                for seq_i in range(num_sequences):
                    seq_start = start_idx + seq_i * sequence_length
                    seq_end = min(seq_start + sequence_length, end_idx)
                    actual_seq_len = seq_end - seq_start
                    
                    if actual_seq_len > 0:
                        # Collect sequences until batch is full
                        seq_obs = self.observations[seq_start:seq_end]
                        seq_actions = self.actions[seq_start:seq_end]
                        
                        # Pad if needed
                        if actual_seq_len < sequence_length:
                            pad_len = sequence_length - actual_seq_len
                            seq_obs = torch.cat([seq_obs, torch.zeros(pad_len, seq_obs.shape[1], device=self.device)])
                            seq_actions = torch.cat([seq_actions, torch.zeros(pad_len, seq_actions.shape[1], device=self.device)])
                        
                        # Yield as (seq_len, 1, feature_dim) for single batch element
                        # In practice, we'll accumulate these to form proper batches
                        yield (
                            seq_obs.unsqueeze(1),  # (seq_len, 1, obs_dim)
                            seq_actions.unsqueeze(1)  # (seq_len, 1, action_dim)
                        )
        else:
            # MLP mode: generate flat batches (original behavior)
            batch_obs = []
            batch_actions = []
            
            for ep_idx in episode_indices:
                start_idx = self.episode_starts[ep_idx].item()
                
                # Find episode end
                if ep_idx + 1 < self.num_episodes:
                    end_idx = self.episode_starts[ep_idx + 1].item()
                else:
                    end_idx = self.size
                
                # Add samples from this episode
                for i in range(start_idx, end_idx):
                    batch_obs.append(self.observations[i])
                    batch_actions.append(self.actions[i])
                    
                    if len(batch_obs) == batch_size:
                        yield (
                            torch.stack(batch_obs),
                            torch.stack(batch_actions)
                        )
                        batch_obs = []
                        batch_actions = []
            
            # Yield remaining samples
            if len(batch_obs) > 0:
                yield (
                    torch.stack(batch_obs),
                    torch.stack(batch_actions)
                )


def collect_episodes(env, policy, num_episodes: int, deterministic: bool = True, 
                    teacher_policy=None, student_obs_slice=None):
    """Collect episodes using a policy.
    
    Supports both MLP and RNN policies. For RNN policies, resets hidden states at episode start.
    
    Args:
        env: The environment
        policy: The policy to use for action selection
        num_episodes: Number of episodes to collect
        deterministic: Whether to use deterministic actions
        teacher_policy: Optional teacher policy to generate actions (for behavioral cloning)
        student_obs_slice: Optional (start_idx, end_idx) to slice observations for student
    
    Returns:
        all_observations: List of observation tensors (one per episode)
        all_actions: List of action tensors (one per episode)
        all_returns: List of episode returns
    """
    all_observations = []
    all_actions = []
    all_returns = []
    
    episodes_collected = 0
    
    # Reset environment
    obs, _ = env.reset()
    batch_size = obs.shape[0]
    
    # Reset hidden states for RNN policies
    if isinstance(policy, StudentPolicy) and policy.is_recurrent:
        policy.reset(batch_size=batch_size, device=obs.device)
    
    episode_obs = []
    episode_actions = []
    episode_reward = 0.0
    
    while episodes_collected < num_episodes:
        # Get action from policy (or teacher if provided)
        if teacher_policy is not None:
            # Use teacher to generate actions, but still collect full observations
            action = teacher_policy.act(obs, deterministic=deterministic)
        elif isinstance(policy, StudentPolicy):
            action = policy.act(obs)
        else:
            action = policy.act(obs, deterministic=deterministic)
        
        # Store transition (always store full observation from environment)
        episode_obs.append(obs.clone())
        episode_actions.append(action.clone())
        
        # Step environment
        obs, rewards, dones, infos = env.step(action)
        episode_reward += rewards.mean().item()


        if dones.any():
            # Store complete episodes
            for env_idx in range(len(dones)):
                if dones[env_idx]:
                    all_observations.append(torch.stack([o[env_idx] for o in episode_obs]))
                    all_actions.append(torch.stack([a[env_idx] for a in episode_actions]))
                    all_returns.append(episode_reward / len(episode_obs))
                    
                    episodes_collected += 1
                    
                    if episodes_collected >= num_episodes:
                        break
            
            # Reset episode buffers and hidden states
            episode_obs = []
            episode_actions = []
            episode_reward = 0.0
            
            # Reset hidden states for completed episodes (RNN)
            if isinstance(policy, StudentPolicy) and policy.is_recurrent:
                # For simplicity, reset all hidden states when any episode ends
                policy.reset(batch_size=batch_size, device=obs.device)
    
    return all_observations, all_actions, all_returns


def evaluate_policy(env, policy, num_episodes: int = 10):
    """Evaluate a policy and return mean return.
    
    Supports both MLP and RNN policies.
    """
    returns = []
    episode_lengths = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        batch_size = obs.shape[0]
        
        # Reset hidden states for RNN policies
        if isinstance(policy, StudentPolicy) and policy.is_recurrent:
            policy.reset(batch_size=batch_size, device=obs.device)
        
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done:
            with torch.no_grad():
                if isinstance(policy, StudentPolicy):
                    action = policy.act(obs)
                else:
                    action = policy.act(obs, deterministic=True)

            obs, rewards, dones, infos = env.step(action)
            
            # rewards is a 1D tensor, use mean() for overall reward tracking
            episode_reward += rewards.mean().item()
            episode_length += 1
            # dones is the combined signal (terminated | truncated)
            done = dones.all()
        
        returns.append(episode_reward / episode_length)
        episode_lengths.append(episode_length)
    
    return np.mean(returns), np.std(returns), np.mean(episode_lengths)


def train_distillation(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, 
    agent_cfg: RslRlOnPolicyRunnerCfg
):
    """Main distillation training loop."""
    
    # Override configurations with command-line arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    
    # Set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    
    # Set random seeds
    torch.manual_seed(agent_cfg.seed)
    np.random.seed(agent_cfg.seed)
    
    # Specify directory for logging experiments
    log_root_path = os.path.join("logs", "distillation", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging distillation experiment in directory: {log_root_path}")
    
    # Specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    
    # Create isaac environment
    print(f"[INFO] Creating environment: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # Convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    # Wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "distillation"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during distillation.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # Wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    
    # Get observation and action dimensions
    obs, _ = env.get_observations()
    num_obs_teacher = obs.shape[1]  # Full observation for teacher (26D)
    num_actions = env.num_actions
    
    # Student uses reduced observation (without last 4D motor_speeds)
    num_obs_student = num_obs_teacher - 4  # 22D = 26D - 4D
    student_obs_slice = (0, num_obs_student)  # Slice first 22 dimensions
    
    print(f"[INFO] Teacher observation dim: {num_obs_teacher}, Student observation dim: {num_obs_student}, Action dim: {num_actions}")
    
    # Load teacher policy
    print(f"[INFO] Loading teacher policy from: {args_cli.teacher_checkpoint}")
    teacher = TeacherPolicyWrapper(args_cli.teacher_checkpoint, agent_cfg.device)
    
    # Verify teacher observation dimension
    if teacher.num_obs != num_obs_teacher:
        print(f"[WARNING] Teacher checkpoint expects {teacher.num_obs}D observations, "
              f"but environment provides {num_obs_teacher}D. This may cause issues.")
    
    # Create student policy with reduced observation dimension
    print("[INFO] Creating student policy with reduced observation (no motor speeds)")
    student = StudentPolicy(num_obs_student, num_actions, obs_slice=student_obs_slice).to(agent_cfg.device)
    
    # Setup optimizer
    optimizer = optim.Adam(student.parameters(), lr=args_cli.learning_rate)
    
    # Setup logging
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    print(f"[INFO] Logging to: {log_dir}")
    
    # Dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
    
    # Calculate dataset size
    episode_length = env.unwrapped.max_episode_length
    dataset_size = (1 if args_cli.on_policy else args_cli.n_epochs) * args_cli.num_episodes * episode_length
    
    print(f"[INFO] Dataset size: {dataset_size} samples")
    print(f"[INFO] Episode length: {episode_length}")
    
    # Create dataset (stores teacher's full observations)
    dataset = DistillationDataset(dataset_size, num_obs_teacher, num_actions, agent_cfg.device)
    
    # Training loop
    best_return = float('-inf')
    best_student_path = log_dir_path / "best_student.pt"
    
    for epoch in range(args_cli.n_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args_cli.n_epochs}")
        print(f"{'='*60}")
        
        # Clear dataset if on-policy
        if args_cli.on_policy:
            dataset.clear()
        
        # Collect data
        if epoch < args_cli.epoch_teacher_forcing:
            # Use teacher to collect data (always use teacher for action generation)
            print(f"[INFO] Collecting {args_cli.num_episodes} episodes using teacher...")
            policy_for_collection = student  # Use student's forward for collection loop
            teacher_for_actions = teacher    # But use teacher for generating actions
        else:
            # Use student to collect data
            print(f"[INFO] Collecting {args_cli.num_episodes} episodes using student...")
            policy_for_collection = student
            teacher_for_actions = None
        
        episode_obs_list, episode_actions_list, episode_returns = collect_episodes(
            env, policy_for_collection, args_cli.num_episodes, args_cli.teacher_deterministic,
            teacher_policy=teacher_for_actions, student_obs_slice=student_obs_slice
        )
        
        # Add to dataset
        for obs, actions in zip(episode_obs_list, episode_actions_list):
            dataset.add_episode(obs, actions, truncated=False)
        
        mean_return = np.mean(episode_returns)
        std_return = np.std(episode_returns)
        
        print(f"Data collection - Mean return: {mean_return:.2f} ± {std_return:.2f}")
        
        writer.add_scalar("data_collection/mean_return", mean_return, epoch)
        writer.add_scalar("data_collection/std_return", std_return, epoch)
        
        # Training
        print(f"[INFO] Training on {dataset.size} samples...")
        student.train()
        
        # Reset hidden states before training
        if student.is_recurrent:
            student.reset()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_obs, batch_actions in dataset.get_batches(
            args_cli.batch_size, args_cli.sequence_length, args_cli.shuffle
        ):
            # Reset hidden states at the start of each batch (for RNN)
            if student.is_recurrent:
                batch_size_actual = batch_obs.shape[1] if batch_obs.dim() == 3 else batch_obs.shape[0]
                student.reset(batch_size=batch_size_actual, device=batch_obs.device)
            
            # Forward pass (student will slice observations internally using obs_slice)
            predicted_actions = student(batch_obs)
            
            # Compute loss (MSE)
            loss = nn.functional.mse_loss(predicted_actions, batch_actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (optional)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Detach hidden states for next iteration (RNN)
            if student.is_recurrent:
                student.detach_hidden_states()

            total_loss += loss.item()
            num_batches += 1
        
        mean_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"Training - Mean loss: {mean_loss:.6f}")
        
        writer.add_scalar("training/loss", mean_loss, epoch)
        
        # Evaluation
        if epoch >= args_cli.epoch_teacher_forcing:
            print("[INFO] Evaluating student policy...")
            student.eval()
            
            eval_return, eval_std, eval_length = evaluate_policy(env, student, num_episodes=10)
            
            print(f"Evaluation - Mean return: {eval_return:.2f} ± {eval_std:.2f}, "
                  f"Mean length: {eval_length:.1f}")
            
            writer.add_scalar("evaluation/mean_return", eval_return, epoch)
            writer.add_scalar("evaluation/std_return", eval_std, epoch)
            writer.add_scalar("evaluation/mean_length", eval_length, epoch)
            
            # Save best model
            if eval_return > best_return:
                best_return = eval_return
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': student.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_return': best_return,
                }, best_student_path)
                print(f"[INFO] New best model saved! Return: {best_return:.2f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = log_dir_path / f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_return': best_return,
            }, checkpoint_path)
            print(f"[INFO] Checkpoint saved to {checkpoint_path}")
    
    # Final save
    final_path = log_dir_path / "final_student.pt"
    torch.save({
        'epoch': args_cli.n_epochs,
        'model_state_dict': student.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_return': best_return,
    }, final_path)
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best return: {best_return:.2f}")
    print(f"Final model saved to: {final_path}")
    print(f"Best model saved to: {best_student_path}")
    print(f"{'='*60}")
    
    writer.close()
    env.close()


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Main entry point for distillation training."""
    train_distillation(env_cfg, agent_cfg)


if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()
