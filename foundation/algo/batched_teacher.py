import torch
import torch.nn as nn
import os
import re

class BatchedTeacherPolicy(nn.Module):
    """
    能够并行运行 N 个不同权重的 MLP。
    假设所有 Teacher 的网络结构完全相同（层数、维度）。
    """
    def __init__(self, num_teachers, teacher_log_dir, checkpoint_mode="latest", device='cuda:0'):
        """
        Args:
            num_teachers: Teacher 的数量
            teacher_log_dir: 包含 teacher_0000, teacher_0001... 文件夹的根目录
            checkpoint_mode: 
                - "latest": 自动寻找数字最大的模型 (例如 model_2000.pt)
                - 具体文件名: 例如 "model_1000.pt" 或 "model_best.pt"，将强制读取该名称
            device: 运行设备
        """
        super().__init__()
        self.num_teachers = num_teachers
        self.device = device
        self.checkpoint_mode = checkpoint_mode
        
        print(f"[BatchedTeacher] Loading {num_teachers} teachers from {teacher_log_dir}...")
        print(f"[BatchedTeacher] Checkpoint Mode: {checkpoint_mode}")

        self.weights = []
        self.biases = []
        self.obs_mean = None
        self.obs_std = None
        
        # 临时存储权重列表
        layers_w = {} # {layer_idx: [w_teacher_0, w_teacher_1, ...]}
        layers_b = {}
        
        # 1. 加载所有 Teacher 的权重
        # 我们先加载第0个来确定结构
        path_0 = self._get_checkpoint_path(teacher_log_dir, 0)
        print(f"[BatchedTeacher] Template path (Teacher 0): {path_0}")
        
        ckpt_0 = torch.load(path_0, map_location='cpu')
        # 处理 RSL-RL 结构差异，尝试找到 model state dict
        if 'model_state_dict' in ckpt_0:
            state_dict_0 = ckpt_0['model_state_dict']
        else:
            state_dict_0 = ckpt_0 # 可能是纯 dict

        # 筛选出 actor 的 keys
        # RSL-RL 通常是 actor_architecture.layers.0.weight, actor_architecture.layers.2.weight ...
        actor_keys = [k for k in state_dict_0.keys() if 'actor' in k]
        # 按层排序
        actor_keys.sort()
        
        # print(f"[BatchedTeacher] Detected Actor Keys: {actor_keys}")

        # 初始化列表
        for k in actor_keys:
            if 'weight' in k: layers_w[k] = []
            if 'bias' in k: layers_b[k] = []
            
        # 批量加载
        obs_means = []
        obs_stds = []
        
        for i in range(num_teachers):
            path = self._get_checkpoint_path(teacher_log_dir, i)
            ckpt = torch.load(path, map_location='cpu')
            sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            
            # 加载 Normalization 统计量
            if 'running_mean_std.running_mean' in sd:
                obs_means.append(sd['running_mean_std.running_mean'])
                obs_stds.append(sd['running_mean_std.running_var'].sqrt()) # RSL-RL 存的是 var
            else:
                # [修复] 如果没有 Norm 层，必须手动创建 Mean=0, Std=1 的占位符
                
                # 寻找第一个真正的权重层 (weight)
                # 我们假设第一个出现的 'weight' 键就是第一层，且它是二维矩阵
                first_weight_key = None
                for k in actor_keys:
                    # 确保是 weight 且是 2D 张量 (Out, In)
                    if 'weight' in k and sd[k].dim() >= 2: 
                        first_weight_key = k
                        break
                
                if first_weight_key is None:
                    raise RuntimeError(f"Could not find any 2D weight tensor in checkpoint for teacher {i}. Keys: {actor_keys}")

                first_weight = sd[first_weight_key]
                input_dim = first_weight.shape[1]
                
                obs_means.append(torch.zeros(input_dim)) 
                obs_stds.append(torch.ones(input_dim))

            for k in actor_keys:
                if 'weight' in k: layers_w[k].append(sd[k])
                if 'bias' in k: layers_b[k].append(sd[k])
                
        # 2. 堆叠权重 -> (Num_Teachers, Out_Dim, In_Dim)
        self.layer_names = sorted(list(layers_w.keys())) 
        
        self.batched_w = nn.ParameterList()
        self.batched_b = nn.ParameterList()
        
        for name in self.layer_names:
            # stack: (N, Out, In)
            w = torch.stack(layers_w[name]).to(device)
            # bias: (N, Out)
            bias_name = name.replace('weight', 'bias')
            b = torch.stack(layers_b[bias_name]).to(device)
            
            self.batched_w.append(nn.Parameter(w, requires_grad=False))
            self.batched_b.append(nn.Parameter(b, requires_grad=False))
            
        # 堆叠 Normalization
        # (N, Obs_Dim)
        if len(obs_means) > 0:
            self.batch_mean = torch.stack(obs_means).to(device)
            self.batch_std = torch.stack(obs_stds).to(device) + 1e-6 # 避免除零
        else:
            # 防御性：如果不应该发生，但发生了
            raise RuntimeError("No observation statistics found (obs_means is empty).")

        print("[BatchedTeacher] Teachers loaded and stacked successfully.")

    def _get_checkpoint_path(self, log_dir, teacher_id):
        # 构造文件夹路径: log_dir/teacher_0000
        teacher_folder = os.path.join(log_dir, f"teacher_{teacher_id:04d}")
        
        if not os.path.exists(teacher_folder):
            raise FileNotFoundError(f"Teacher folder not found: {teacher_folder}")

        # === 模式 A: 指定文件名 (例如 'model_1000.pt') ===
        if self.checkpoint_mode != "latest":
            target_path = os.path.join(teacher_folder, self.checkpoint_mode)
            if os.path.exists(target_path):
                return target_path
            else:
                # 尝试容错：有些时候 best_model 没有 .pt 后缀? 通常都有
                raise FileNotFoundError(f"Specified checkpoint '{self.checkpoint_mode}' not found in {teacher_folder}")

        # === 模式 B: 自动寻找最新 (latest) ===
        # 扫描所有 .pt 文件
        files = [f for f in os.listdir(teacher_folder) if f.endswith('.pt')]
        if not files:
            raise FileNotFoundError(f"No .pt files found in {teacher_folder}")
        
        # 过滤出包含数字的文件，以便排序
        numbered_files = [f for f in files if re.search(r'\d+', f)]
        
        if not numbered_files:
            # 如果没有带数字的文件，且没指定文件名，就随便返回一个
            print(f"[Warning] No numbered checkpoints found in {teacher_folder}, taking {files[0]}")
            return os.path.join(teacher_folder, files[0])

        # 按文件名中的最后一个数字排序
        numbered_files.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
        
        # 返回数字最大的那个
        return os.path.join(teacher_folder, numbered_files[-1])

    def forward(self, obs):
        """
        obs: (Batch_Size, Obs_Dim)
        假设 Batch_Size == Num_Teachers (每个环境对应一个 Teacher)
        """
        if obs.shape[0] != self.num_teachers:
            raise ValueError(f"Batch size {obs.shape[0]} must match num_teachers {self.num_teachers} for 1-to-1 mapping.")
            
        # 1. Normalize
        x = (obs - self.batch_mean) / self.batch_std
        x = torch.clamp(x, -5.0, 5.0) # RSL-RL 默认 clip
        
        # 2. Parallel MLP Inference
        # Input x: (N, In) -> (N, In, 1) for matmul
        x = x.unsqueeze(-1)
        
        for i, (w, b) in enumerate(zip(self.batched_w, self.batched_b)):
            # w: (N, Out, In)
            # x: (N, In, 1)
            # b: (N, Out)
            
            # Matmul: (N, Out, In) @ (N, In, 1) -> (N, Out, 1)
            x = torch.bmm(w, x)
            x = x.squeeze(-1) + b # Add bias -> (N, Out)
            
            # Activation
            if i < len(self.batched_w) - 1: 
                x = torch.nn.functional.elu(x)
                x = x.unsqueeze(-1) # Prepare for next matmul

        # Final Layer (假设 Teacher 输出 Mean，加 Tanh)
        return torch.tanh(x)