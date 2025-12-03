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
parser.add_argument("--n_epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--num_episodes", type=int, default=100, help="Episodes per epoch for data collection")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
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
    """Student policy network (MLP architecture)."""
    
    def __init__(self, num_obs: int, num_actions: int, hidden_dims: list = [256, 256, 256], 
                 activation: str = "elu"):
        super().__init__()
        
        self.num_obs = num_obs
        self.num_actions = num_actions
        
        # Build network
        layers = []
        in_dim = num_obs
        
        activation_fn = nn.ELU() if activation == "elu" else nn.ReLU()
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(activation_fn)
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, num_actions))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs):
        return self.network(obs)


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
        """Generate batches for training."""
        if self.num_episodes == 0:
            return
        
        # Get episode indices
        episode_indices = torch.arange(self.num_episodes, device=self.device)
        
        if shuffle:
            episode_indices = episode_indices[torch.randperm(self.num_episodes)]
        
        # Generate batches
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


def collect_episodes(env, policy, num_episodes: int, deterministic: bool = True):
    """Collect episodes using a policy."""
    all_observations = []
    all_actions = []
    all_returns = []
    
    episodes_collected = 0
    
    # Reset environment
    obs, _ = env.reset()
    episode_obs = []
    episode_actions = []
    episode_reward = 0.0
    
    while episodes_collected < num_episodes:
        # Get action from policy
        action = policy.act(obs, deterministic=deterministic)
        
        # Store transition
        episode_obs.append(obs.clone())
        episode_actions.append(action.clone())
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward.mean().item()
        
        # Check for episode end
        done = terminated | truncated
        
        if done.any():
            # Store complete episodes
            for env_idx in range(len(done)):
                if done[env_idx]:
                    all_observations.append(torch.stack([o[env_idx] for o in episode_obs]))
                    all_actions.append(torch.stack([a[env_idx] for a in episode_actions]))
                    all_returns.append(episode_reward / len(episode_obs))
                    
                    episodes_collected += 1
                    
                    if episodes_collected >= num_episodes:
                        break
            
            # Reset episode buffers
            episode_obs = []
            episode_actions = []
            episode_reward = 0.0
    
    return all_observations, all_actions, all_returns


def evaluate_policy(env, policy, num_episodes: int = 10):
    """Evaluate a policy and return mean return."""
    returns = []
    episode_lengths = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done:
            with torch.no_grad():
                if isinstance(policy, StudentPolicy):
                    action = policy(obs)
                else:
                    action = policy.act(obs, deterministic=True)
            
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward.mean().item()
            episode_length += 1
            done = (terminated | truncated).all()
        
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
    num_obs = obs.shape[1]
    num_actions = env.num_actions
    
    print(f"[INFO] Observation dim: {num_obs}, Action dim: {num_actions}")
    
    # Load teacher policy
    print(f"[INFO] Loading teacher policy from: {args_cli.teacher_checkpoint}")
    teacher = TeacherPolicyWrapper(args_cli.teacher_checkpoint, agent_cfg.device)
    
    # Create student policy
    print("[INFO] Creating student policy")
    student = StudentPolicy(num_obs, num_actions).to(agent_cfg.device)
    
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
    
    # Create dataset
    dataset = DistillationDataset(dataset_size, num_obs, num_actions, agent_cfg.device)
    
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
            # Use teacher to collect data
            print(f"[INFO] Collecting {args_cli.num_episodes} episodes using teacher...")
            policy_for_collection = teacher
        else:
            # Use student to collect data
            print(f"[INFO] Collecting {args_cli.num_episodes} episodes using student...")
            policy_for_collection = student
        
        episode_obs_list, episode_actions_list, episode_returns = collect_episodes(
            env, policy_for_collection, args_cli.num_episodes, args_cli.teacher_deterministic
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
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_obs, batch_actions in dataset.get_batches(
            args_cli.batch_size, args_cli.sequence_length, args_cli.shuffle
        ):
            # Forward pass
            predicted_actions = student(batch_obs)
            
            # Compute loss (MSE)
            loss = nn.functional.mse_loss(predicted_actions, batch_actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (optional)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            
            optimizer.step()
            
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
