# multi_teacher_policy.py

import torch
import torch.nn as nn
from rsl_rl.modules import StudentTeacherRecurrentCustom, EmpiricalNormalization

class MultiTeacherPolicy(StudentTeacherRecurrentCustom):
    def __init__(self, *args, 
                 teacher_models: list[nn.Module] = None, 
                 teacher_norm_state_dicts: list[dict] = None,
                 teacher_offsets: list[tuple] = None,
                 **kwargs):
        
        super().__init__(*args, **kwargs)
        
        if teacher_models is None or len(teacher_models) == 0:
            raise ValueError("MultiTeacherPolicy requires a list of teacher_models")

        self.num_teachers = len(teacher_models)
        self.teachers_list = nn.ModuleList(teacher_models)
        
        # [新增] 1. 初始化 Student Normalizer
        # 注意：这里我们使用 num_student_obs (kwargs中已经由父类处理，但在父类初始化前我们拿不到，所以用 self.num_student_obs)
        # 父类 StudentTeacherRecurrentCustom 会把 num_student_obs 存为属性吗？如果不存，我们需要从 args 或 kwargs 获取
        # 根据 RSL-RL 源码，StudentTeacher 基类通常不保存 num_student_obs 为 public 属性，但我们可以从 kwargs 的 'num_student_obs' 或者输入维度推断。
        # 稳妥起见，我们从 args[0] (num_student_obs) 获取，或者从 kwargs 获取。
        # StudentTeacherRecurrentCustom.__init__ 签名是 (num_student_obs, ...)
        
        # 获取 obs 维度用于 Normalizer
        if args:
            student_obs_dim = args[0]
        else:
            student_obs_dim = kwargs.get('num_student_obs')
            
        self.student_normalizer = EmpiricalNormalization(shape=[student_obs_dim], until=1.0e8)
        
        # [新增] 存储每个教师的稳态误差
        if teacher_offsets is None:
            teacher_offsets = [(0.0, 0.0, 0.0)] * self.num_teachers
        
        offsets_tensor = torch.tensor(teacher_offsets, dtype=torch.float32)
        self.register_buffer('teacher_offsets', offsets_tensor)
        
        print(f"  > [MultiPolicy] Registered {self.num_teachers} teacher offsets.")
        
        # ... (Teacher Normalizer 初始化逻辑保持不变) ...
        self.teacher_normalizers = nn.ModuleList()
        norm_dim = kwargs.get('num_teacher_obs')
        for i in range(self.num_teachers):
            normalizer = EmpiricalNormalization(shape=[norm_dim], until=1.0e8)
            if teacher_norm_state_dicts and i < len(teacher_norm_state_dicts) and teacher_norm_state_dicts[i] is not None:
                normalizer.load_state_dict(teacher_norm_state_dicts[i])
            normalizer.eval()
            self.teacher_normalizers.append(normalizer)

        # 冻结教师网络参数
        for teacher in self.teachers_list:
            for param in teacher.parameters():
                param.requires_grad = False
            teacher.eval()

    def train(self, mode=True):
        # 1. 调用父类 train，这会把 Student 网络 (RNN, MLP) 和 self.student_normalizer 设为 mode
        super().train(mode)
        
        # 2. [关键] 确保 Student Normalizer 跟随 mode (训练时更新均值方差，评估时停止)
        self.student_normalizer.train(mode)
        
        # 3. 强制 Teacher 和 Teacher Normalizer 永远处于 Eval
        for teacher in self.teachers_list:
            teacher.eval()
        for norm in self.teacher_normalizers:
            norm.eval()
            
        return self

    def evaluate(self, teacher_observations):
        # Teacher 的评估逻辑保持不变 (Raw Obs -> Slice -> Norm -> Net)
        # 注意：Teacher 的 Normalizer 是独立的，不需要加 offset (或者 offset 已经在环境/wrapper层处理了? 
        # 根据你的描述，Teacher 是已经训练好的，所以这里我们只负责复现 Teacher 的行为)
        # 如果 Teacher 训练时输入就是 Raw Obs，那么这里也是 Raw Obs -> Teacher Norm -> Teacher Net
        
        total_envs = teacher_observations.shape[0]
        envs_per_teacher = total_envs // self.num_teachers
        outputs = []
        
        for i in range(self.num_teachers):
            start_idx = i * envs_per_teacher
            end_idx = start_idx + envs_per_teacher if i < self.num_teachers - 1 else total_envs
            
            obs_slice = teacher_observations[start_idx:end_idx] # .clone() not strictly needed for read-only
            
            with torch.no_grad():
                # 确保 Teacher 组件处于 eval
                self.teacher_normalizers[i].eval()
                self.teachers_list[i].eval()
                
                # Teacher 直接归一化原始观测 (假设 Teacher 训练时没有 offset trick，或者 offset 隐含在 dynamics 中)
                normalized_obs = self.teacher_normalizers[i](obs_slice)
                action_slice = self.teachers_list[i].act_inference(normalized_obs)
                
            outputs.append(action_slice)
            
        return torch.cat(outputs, dim=0)

    def _forward_head(self, observations):
        """
        重写父类方法:
        1. 接收 Raw Observations (因为 Runner 的 Normalizer 被我们架空了)
        2. 加上稳态误差 Offset
        3. 通过内部的 Student Normalizer 进行归一化
        4. 传给父类处理 (RNN -> MLP)
        """
        
        # 1. 克隆观测，避免修改原始数据 (Runner 可能还需要用原始数据做其他 log)
        obs_compensated = observations.clone()
        
        # 2. 根据环境索引应用对应教师的 Offset
        batch_size = observations.shape[0]
        envs_per_teacher = batch_size // self.num_teachers
        
        for i in range(self.num_teachers):
            start_idx = i * envs_per_teacher
            end_idx = start_idx + envs_per_teacher if i < self.num_teachers - 1 else batch_size
            
            # [修改] 按照你的要求：在 obs 加上 offset
            # 假设 offset 补偿的是前 3 维 (Pos Error)
            obs_compensated[start_idx:end_idx, 0:3] -= self.teacher_offsets[i]
        
        # 3. [新增] 内部归一化
        # 这个 normalizer 会根据 self.training 状态决定是更新均值方差还是仅使用
        obs_normalized = self.student_normalizer(obs_compensated)
        
        # 4. 调用父类的真正推理逻辑 (传入归一化后的数据)
        return super()._forward_head(obs_normalized)
    
    def act_batch(self, observations, hidden_states):
        """
        重写父类 act_batch，确保在进入 RNN 训练前：
        1. 加上 Offset (对 Raw Obs)
        2. 进行归一化 (使用 student_normalizer)
        """
        # observations shape: [Seq_Len, Batch, Dim]
        T, B, D = observations.shape
        
        # 1. Clone，避免修改 Storage 中的原始数据
        obs_processed = observations.clone()
        
        # 2. Apply Offsets (需要处理 T 维度)
        envs_per_teacher = B // self.num_teachers
        
        # 对前3维 (pos_error) 加 Offset
        # 这里的切片需要同时覆盖 T 和 B 维度
        for i in range(self.num_teachers):
            start_idx = i * envs_per_teacher
            end_idx = start_idx + envs_per_teacher if i < self.num_teachers - 1 else B
            
            # [T, Envs_Slice, 3] += [3] (Broadcast)
            obs_processed[:, start_idx:end_idx, 0:3] -= self.teacher_offsets[i]

        # 3. Apply Normalization
        # EmpiricalNormalization 通常处理 (N, D) 的输入
        # 我们先 flatten 成 (T*B, D)
        obs_reshaped = obs_processed.view(-1, D)
        
        # 使用内部的 student_normalizer 进行归一化
        # 注意：在 update 阶段，self.student_normalizer 处于 train() 模式 (由 self.train() 设定)
        # 这意味着它会继续更新均值方差。这是符合 RSL-RL 原设计的 (On-policy数据更新Norm)。
        obs_norm = self.student_normalizer(obs_reshaped)
        
        # 还原形状 [T, B, D] 以传给 RNN
        obs_input = obs_norm.view(T, B, D)
        
        # 4. 调用父类的 act_batch 继续后续网络计算 (MLP -> RNN -> MLP)
        # 注意：父类 act_batch 会再次 view 数据，但这没关系
        return super().act_batch(obs_input, hidden_states)