import torch
import torch.nn as nn
import os
import re

class TeacherEnsemble(nn.Module):
    def __init__(self, log_dir, num_teachers, device):
        super().__init__()
        self.device = device
        self.num_teachers = num_teachers
        self.loaded = False
        
        # 假设 Teacher 网络结构是固定的 [26 -> 64 -> 64 -> 4] (根据你之前的日志)
        # 我们需要存储所有 Teacher 的权重堆叠
        # Layer 0
        self.l0_weight = None # shape: [num_teachers, 64, 26]
        self.l0_bias = None   # shape: [num_teachers, 64]
        # Layer 2
        self.l2_weight = None # shape: [num_teachers, 64, 64]
        self.l2_bias = None   # shape: [num_teachers, 64]
        # Layer 4
        self.l4_weight = None # shape: [num_teachers, 64, 64]
        self.l4_bias = None   # shape: [num_teachers, 64]
        # Layer 6 (Output)
        self.l6_weight = None # shape: [num_teachers, 4, 64]
        self.l6_bias = None   # shape: [num_teachers, 4]

        self.load_teachers(log_dir)

    def load_teachers(self, log_dir):
        print(f"Loading {self.num_teachers} teachers from {log_dir}...")
        
        weights = {k: [] for k in ['0.weight', '0.bias', '2.weight', '2.bias', '4.weight', '4.bias', '6.weight', '6.bias']}
        
        for i in range(self.num_teachers):
            # 路径构建：需要根据你的实际保存路径调整
            # 假设路径是 logs/rsl_rl/raptor_teachers/teacher_0000/model_1000.pt
            model_path = os.path.join(log_dir, f"teacher_{i:04d}", "model_1000.pt")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Teacher model not found: {model_path}")
                
            # 加载 checkpoint
            ckpt = torch.load(model_path, map_location=self.device)
            # 提取 actor 的 state_dict (RSL-RL PPO 保存的是整个 runner state)
            # 通常在 'model' -> 'actor_architecture_state_dict' 或者直接是 actor 的权重
            # 根据 RSL-RL 的习惯，通常是 ckpt['model_state_dict']['actor_state_dict']
            # 这里我们需要根据实际 saved dict 结构调整。
            # 假设 RSL-RL OnPolicyRunner 保存结构:
            actor_dict = ckpt['model_state_dict']['actor_state_dict']
            
            for name, val in actor_dict.items():
                if name in weights:
                    weights[name].append(val)
        
        # Stack tensors
        self.l0_weight = torch.stack(weights['0.weight'])
        self.l0_bias = torch.stack(weights['0.bias'])
        self.l2_weight = torch.stack(weights['2.weight'])
        self.l2_bias = torch.stack(weights['2.bias'])
        self.l4_weight = torch.stack(weights['4.weight'])
        self.l4_bias = torch.stack(weights['4.bias'])
        self.l6_weight = torch.stack(weights['6.weight'])
        self.l6_bias = torch.stack(weights['6.bias'])
        
        self.loaded = True
        print("All teachers loaded successfully into GPU memory.")

    def forward(self, obs, teacher_indices):
        """
        obs: [batch_size, obs_dim]
        teacher_indices: [batch_size] (每个环境对应哪个 teacher ID)
        """
        # 1. 根据索引提取对应的权重 [batch_size, out_dim, in_dim]
        w0 = self.l0_weight[teacher_indices]
        b0 = self.l0_bias[teacher_indices]
        
        w2 = self.l2_weight[teacher_indices]
        b2 = self.l2_bias[teacher_indices]
        
        w4 = self.l4_weight[teacher_indices]
        b4 = self.l4_bias[teacher_indices]
        
        w6 = self.l6_weight[teacher_indices]
        b6 = self.l6_bias[teacher_indices]
        
        # 2. 手动执行 MLP 前向传播 (使用 bmm: batch matrix multiply)
        # Layer 0
        # obs: [B, 26] -> [B, 26, 1]
        x = obs.unsqueeze(2)
        # w0: [B, 64, 26], x: [B, 26, 1] -> [B, 64, 1]
        x = torch.bmm(w0, x).squeeze(2) + b0
        x = torch.nn.functional.elu(x, alpha=1.0)
        
        # Layer 2
        x = x.unsqueeze(2)
        x = torch.bmm(w2, x).squeeze(2) + b2
        x = torch.nn.functional.elu(x, alpha=1.0)
        
        # Layer 4
        x = x.unsqueeze(2)
        x = torch.bmm(w4, x).squeeze(2) + b4
        x = torch.nn.functional.elu(x, alpha=1.0)
        
        # Output Layer
        x = x.unsqueeze(2)
        x = torch.bmm(w6, x).squeeze(2) + b6
        x = torch.tanh(x) # Actor 输出通常有 Tanh
        
        return x