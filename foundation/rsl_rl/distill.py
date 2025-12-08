# Copyright (c) 2025
# RAPTOR Implementation - Stage 2: Distillation (Meta-Imitation Learning)

"""
Usage:
    python distill.py \
      --teacher_dir logs/rsl_rl/raptor_teachers/2025-XX-XX_XX-XX-XX \
      --num_teachers 2 \
      --epochs 50 \
      --headless  # (Optional) Run without GUI
"""

import argparse
import sys
import os

# ---------------------------------------------------------------------------
# 0. App Launcher Setup (Must be first!)
# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

# Define arguments first
parser = argparse.ArgumentParser(description="RAPTOR Distillation")

# Add Isaac Lab standard args (headless, device, etc.)
AppLauncher.add_app_launcher_args(parser)

# Add RAPTOR specific args
parser.add_argument("--teacher_dir", type=str, required=True, help="Path to timestamp folder containing teacher_XXXX folders and csv")
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--steps_per_epoch", type=int, default=500, help="Sequence length (5s @ 100Hz)")
parser.add_argument("--warmup_epochs", type=int, default=10, help="Epochs where teacher drives")
parser.add_argument("--student_hidden", type=int, default=16)
# Note: --save_path is now optional/ignored as we auto-generate the path inside teacher_dir
parser.add_argument("--num_teachers", type=int, default=None, help="Number of teachers to use (default: use all in CSV)")

# Parse args
args = parser.parse_args()

# Launch App (This initializes the simulator and omni bindings)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# 1. Imports (Must happen AFTER app launch)
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import gymnasium as gym
import torch.nn.functional as F

# Isaac Lab imports
from isaaclab.envs import DirectRLEnvCfg
# Ensure your env file is accessible in python path
from foundation.tasks.point_ctrl.quad_point_ctrl_env_single_dense import QuadcopterEnv, QuadcopterEnvCfg

# Helper for vmap
from torch.func import functional_call, vmap

# ---------------------------------------------------------------------------
# 2. Student Policy Architecture (RAPTOR Paper Fig 7B)
# ---------------------------------------------------------------------------

class StudentPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=16):
        super().__init__()
        # RAPTOR student is very small: ~2084 parameters
        self.input_layer = nn.Linear(obs_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, action_dim)
        
        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, x, hidden_state=None):
        # x shape: (Batch, Seq_Len, Obs_Dim) or (Batch, Obs_Dim)
        # If input is 2D (Batch, Obs), we treat it as seq_len=1 for GRU
        if x.dim() == 2:
            x = x.unsqueeze(1) 
        
        x = F.tanh(self.input_layer(x))
        x, new_hidden = self.gru(x, hidden_state)
        x = self.output_layer(x)
        # Output shape is always (Batch, Seq_Len, Action_Dim)
        return x, new_hidden

# ---------------------------------------------------------------------------
# 3. Vectorized Teacher Policy (Ensemble)
# ---------------------------------------------------------------------------

class EnsembleTeacherPolicy:
    def __init__(self, num_teachers, teacher_log_dir, device, obs_dim, action_dim):
        self.device = device
        self.num_teachers = num_teachers
        print(f"[Distill] Loading first {num_teachers} teachers from {teacher_log_dir}...")

        # Define the teacher architecture (Must match what RSL-RL trained)
        self.base_model = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ELU(), 
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, action_dim)
        ).to(device)

        models_state_dicts = []
        
        # Load weights for exactly num_teachers
        for i in tqdm(range(num_teachers), desc="Loading Checkpoints"):
            run_name = f"teacher_{i:04d}"
            
            teacher_path = os.path.join(teacher_log_dir, run_name)
            if not os.path.exists(teacher_path):
                candidates = [d for d in os.listdir(teacher_log_dir) if d.endswith(f"_{i:04d}")]
                if candidates:
                    teacher_path = os.path.join(teacher_log_dir, candidates[0])
                else:
                    raise FileNotFoundError(f"Could not find folder for teacher {i}")

            ckpt_dir = teacher_path
            if not any(f.endswith('.pt') for f in os.listdir(ckpt_dir)):
                 possible_subs = [os.path.join(ckpt_dir, "checkpoints"), ckpt_dir]
            else:
                possible_subs = [ckpt_dir]

            model_path = None
            
            for sub in possible_subs:
                if not os.path.exists(sub): continue
                
                # Priority 1: User requested 'best_model.pt'
                if os.path.exists(os.path.join(sub, "best_model.pt")):
                    model_path = os.path.join(sub, "best_model.pt")
                    break
                
                # Priority 2: Standard RSL-RL 'model.pt'
                if os.path.exists(os.path.join(sub, "model.pt")):
                    model_path = os.path.join(sub, "model.pt")
                    break
                
                # Priority 3: Parse numbered files
                files = [f for f in os.listdir(sub) if f.endswith('.pt')]
                numbered_files = []
                for f in files:
                    parts = f.replace(".pt", "").split("_")
                    if len(parts) > 1 and parts[-1].isdigit():
                        numbered_files.append((int(parts[-1]), f))
                
                if numbered_files:
                    numbered_files.sort(key=lambda x: x[0])
                    model_path = os.path.join(sub, numbered_files[-1][1])
                    break
            
            if model_path is None:
                 raise FileNotFoundError(f"No checkpoint found for teacher {i} in {teacher_path}")

            # Load Checkpoint
            ckpt = torch.load(model_path, map_location=device)
            
            # Extract Actor Weights
            full_dict = ckpt['model_state_dict']
            actor_dict = {}
            
            target_keys = set(self.base_model.state_dict().keys())
            
            for key, val in full_dict.items():
                if 'actor' in key and 'critic' not in key and 'std' not in key:
                    new_key = key
                    prefixes = [
                        "actor_architecture.actor.layers.", 
                        "actor.layers.", 
                        "actor."
                    ]
                    for p in prefixes:
                        if new_key.startswith(p):
                            new_key = new_key[len(p):]
                            break
                    
                    if new_key in target_keys:
                        actor_dict[new_key] = val
            
            if len(actor_dict) == 0:
                print(f"\n[Error] Failed to map weights for Teacher {i} from {model_path}")
                print(f"Expected keys: {list(target_keys)[:3]}...")
                print(f"Found keys in ckpt (first 5): {list(full_dict.keys())[:5]}...")
                raise ValueError(f"Could not map actor weights for teacher {i}.")
            
            models_state_dicts.append(actor_dict)

        # Manually stack weights
        print("[Distill] Stacking teacher weights for vmap...")
        self.stacked_params = {}
        if len(models_state_dicts) > 0:
            keys = models_state_dicts[0].keys()
            for key in keys:
                stacked_tensor = torch.stack([d[key] for d in models_state_dicts]).to(device)
                self.stacked_params[key] = stacked_tensor
        
        # Define vmap function
        def call_single_model(params, data):
            return functional_call(self.base_model, params, data)

        self.vmap_forward = vmap(call_single_model, (0, 0))

    def get_actions(self, obs):
        actions = self.vmap_forward(self.stacked_params, obs)
        return torch.tanh(actions) 

# ---------------------------------------------------------------------------
# 4. Environment Wrapper for Distillation
# ---------------------------------------------------------------------------

class DistillationEnv(QuadcopterEnv):
    def __init__(self, cfg, dynamics_csv_path, device, target_num_teachers=None):
        # Read CSV
        self.dynamics_df = pd.read_csv(dynamics_csv_path)
        
        if target_num_teachers is not None:
            if target_num_teachers > len(self.dynamics_df):
                print(f"[Warning] Requested {target_num_teachers} teachers but CSV only has {len(self.dynamics_df)}.")
            else:
                print(f"[Distill] Slicing dynamics to first {target_num_teachers} teachers.")
                self.dynamics_df = self.dynamics_df.iloc[:target_num_teachers]
        
        self.num_teachers = len(self.dynamics_df)
        
        # Ensure cfg matches the number of teachers
        cfg.scene.num_envs = self.num_teachers
        
        super().__init__(cfg, render_mode=None)
        
        # Pre-load dynamics tensors
        self.mass_tensor = torch.tensor(self.dynamics_df['mass'].values, device=device, dtype=torch.float32)
        self.arm_tensor = torch.tensor(self.dynamics_df['arm_length'].values, device=device, dtype=torch.float32)
        self.twr_tensor = torch.tensor(self.dynamics_df['twr'].values, device=device, dtype=torch.float32)
        self.tau_tensor = torch.tensor(self.dynamics_df['motor_tau'].values, device=device, dtype=torch.float32)
        
        ixx = torch.tensor(self.dynamics_df['Ixx'].values, device=device, dtype=torch.float32)
        iyy = torch.tensor(self.dynamics_df['Iyy'].values, device=device, dtype=torch.float32)
        izz = torch.tensor(self.dynamics_df['Izz'].values, device=device, dtype=torch.float32)
        self.inertia_tensor = torch.stack([ixx, iyy, izz], dim=1) # (N, 3)

        # Force Update Controller parameters
        self._controller.mass = self.mass_tensor
        self._controller.arm_length = self.arm_tensor
        self._controller.inertia = self.inertia_tensor
        self._controller.thrust_to_weight = self.twr_tensor
        
        # Update motor tau
        self.motor_tau = self.tau_tensor.view(self.num_envs, 1) # Ensure correct shape
        self.dt = self.cfg.sim.dt
        self.motor_alpha = self.dt / (self.dt + self.motor_tau)


    def _reset_idx(self, env_ids):
        # Call parent reset to handle physics/state reset
        super()._reset_idx(env_ids)
        pass

# ---------------------------------------------------------------------------
# 5. Main Training Logic
# ---------------------------------------------------------------------------

def main():
    # Use args parsed globally
    device = torch.device(f"cuda:{args.device}" if ":" not in str(args.device) else args.device)
    if str(args.device) == "cuda":
        device = torch.device("cuda:0")

    csv_path = os.path.join(args.teacher_dir, "teacher_dynamics.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV not found at {csv_path}")
        return

    # [新增] 自动设置保存路径: {teacher_dir}/student/
    student_save_dir = os.path.join(args.teacher_dir, "student")
    os.makedirs(student_save_dir, exist_ok=True)
    final_save_path = os.path.join(student_save_dir, "student_policy.pt")
    
    print(f"==================================================")
    print(f"[Distill] Student models will be saved to:")
    print(f"          {student_save_dir}")
    print(f"==================================================")

    # 1. Setup Environment
    env_cfg = QuadcopterEnvCfg()
    env_cfg.train_or_play = True 
    env_cfg.sim.device = str(device) 
    env_cfg.seed = 42
    
    # Initialize Distillation Env
    env = DistillationEnv(env_cfg, csv_path, device, target_num_teachers=args.num_teachers)
    print(f"[Distill] Environment Initialized with {env.num_envs} unique dynamics.")

    # 2. Setup Dimensions
    teacher_obs_dim = env.observation_space.shape[-1]
    
    # Student Observation (Masked): 26 - 4 (motor speeds) = 22
    student_obs_dim = teacher_obs_dim - 4
    
    action_dim = env.action_space.shape[-1]
    
    print(f"[Distill] Teacher Obs Dim: {teacher_obs_dim} (Full)")
    print(f"[Distill] Student Obs Dim: {student_obs_dim} (No Motor Speeds)")
    print(f"[Distill] Action Dim: {action_dim}")
    
    # 3. Load Teachers (With FULL observation dimension)
    teachers = EnsembleTeacherPolicy(
        num_teachers=env.num_envs, 
        teacher_log_dir=args.teacher_dir,
        device=device,
        obs_dim=teacher_obs_dim, # Teacher sees motor speeds
        action_dim=action_dim
    )

    # 4. Initialize Student (With MASKED observation dimension)
    student = StudentPolicy(obs_dim=student_obs_dim, action_dim=action_dim, hidden_dim=args.student_hidden).to(device)
    optimizer = optim.Adam(student.parameters(), lr=1e-3) 
    mse_loss = nn.MSELoss()

    print("[Distill] Starting Meta-Imitation Learning Loop...")
    
    # -----------------------------------------------------------------------
    # Training Loop
    # -----------------------------------------------------------------------
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # A. Rollout Phase
        # ----------------
        obs_dict, _ = env.reset()
        full_obs = obs_dict['policy'] # (N, 26)
        
        # Reset Student Hidden State
        hidden_state = torch.zeros(env.num_teachers, 1, args.student_hidden, device=device).transpose(0, 1) # (1, Batch, Hidden)
        
        traj_student_obs = [] # Store masked observations
        traj_teacher_actions = []
        
        student.train() 
        
        with torch.no_grad():
            for step in range(args.steps_per_epoch):
                # 1. Get Teacher Actions (Uses Full Obs)
                teacher_actions = teachers.get_actions(full_obs)
                
                # 2. Prepare Student Obs (Mask Motor Speeds)
                student_obs = full_obs[:, :-4] 
                
                # 3. Get Student Actions
                # Output shape: (N, 1, Action)
                student_actions_seq, hidden_state = student(student_obs, hidden_state)
                # [Fix] Squeeze to (N, Action) for environment step
                student_actions = student_actions_seq.squeeze(1)
                
                # 4. Select Action to Drive Environment
                if epoch < args.warmup_epochs:
                    env_actions = teacher_actions
                else:
                    env_actions = student_actions
                
                # Store data for training (Use Student's view)
                traj_student_obs.append(student_obs.clone())
                traj_teacher_actions.append(teacher_actions.clone())
                
                # 5. Step Environment
                obs_dict, rewards, dones, timeouts, extras = env.step(env_actions)
                full_obs = obs_dict['policy']
                
                if dones.any():
                     hidden_state[:, dones, :] = 0.0

        # B. Training Phase (BPTT)
        # ------------------------
        # Stack trajectories: (Seq_Len, Batch, Dim) -> (Batch, Seq_Len, Dim)
        batch_obs = torch.stack(traj_student_obs).permute(1, 0, 2) 
        batch_targets = torch.stack(traj_teacher_actions).permute(1, 0, 2)
        
        optimizer.zero_grad()
        
        # Forward pass Student on WHOLE trajectory using Masked Obs
        pred_actions, _ = student(batch_obs)
        
        loss = mse_loss(pred_actions, batch_targets)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        
        # C. Logging
        # ----------
        epoch_time = time.time() - epoch_start
        mode = "Teacher (Warmup)" if epoch < args.warmup_epochs else "Student (On-Policy)"
        
        # [新增] 只有在 On-Policy 阶段（Student 自己飞的时候）才开始评选 Best Model
        # 因为 Warmup 阶段数据是 Teacher 产生的，Loss 低不代表 Student 能力强
        is_best = ""
        if epoch >= args.warmup_epochs:
            if current_loss < min_loss:
                min_loss = current_loss
                torch.save(student.state_dict(), best_save_path)
                is_best = " [New Best!]"

        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {current_loss:.6f}{is_best} | Mode: {mode} | Time: {epoch_time:.2f}s")
        
        # Save Checkpoint periodically (每100轮存一个备份)
        if (epoch + 1) % 100 == 0:
            ckpt_filename = f"student_policy_ep{epoch+1}.pt"
            ckpt_path = os.path.join(student_save_dir, ckpt_filename)
            torch.save(student.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Final Save (这是最后一轮的)
    torch.save(student.state_dict(), final_save_path)
    print("Distillation Complete!")
    print(f"Last policy saved to: {final_save_path}")
    print(f"Best policy saved to: {best_save_path} (Loss: {min_loss:.6f})")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()