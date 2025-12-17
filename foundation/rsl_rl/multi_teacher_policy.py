# multi_teacher_policy.py

import torch
import torch.nn as nn
from rsl_rl.modules import StudentTeacherRecurrentCustom, EmpiricalNormalization

class MultiTeacherPolicy(StudentTeacherRecurrentCustom):
    def __init__(self, *args, 
                 teacher_models: list[nn.Module] = None, 
                 teacher_norm_state_dicts: list[dict] = None,
                 teacher_offsets: list[tuple] = None,  # [新增] 每个教师的稳态误差 (x_offset, y_offset, z_offset)
                 **kwargs):
        
        super().__init__(*args, **kwargs)
        
        if teacher_models is None or len(teacher_models) == 0:
            raise ValueError("MultiTeacherPolicy requires a list of teacher_models")

        self.num_teachers = len(teacher_models)
        self.teachers_list = nn.ModuleList(teacher_models)
        
        # [新增] 存储每个教师的稳态误差，转为 Tensor
        if teacher_offsets is None:
            teacher_offsets = [(0.0, 0.0, 0.0)] * self.num_teachers
        
        # 将稳态误差转为 tensor 并注册为 buffer (不参与梯度计算)
        offsets_tensor = torch.tensor(teacher_offsets, dtype=torch.float32)  # shape: (num_teachers, 3)
        self.register_buffer('teacher_offsets', offsets_tensor)
        print(f"  > [MultiPolicy] Registered {self.num_teachers} teacher offsets:")
        for i, offset in enumerate(teacher_offsets):
            print(f"    Teacher {i}: offset=({offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f})")
        
        # 初始化每个 Teacher 对应的 Normalizer
        self.teacher_normalizers = nn.ModuleList()
        
        # 获取 Teacher Obs 维度
        norm_dim = kwargs.get('num_teacher_obs')
        
        for i in range(self.num_teachers):
            # 创建 Normalizer
            normalizer = EmpiricalNormalization(shape=[norm_dim], until=1.0e8)
            
            # 如果提供了参数，加载参数
            if teacher_norm_state_dicts and i < len(teacher_norm_state_dicts):
                if teacher_norm_state_dicts[i] is not None:
                    normalizer.load_state_dict(teacher_norm_state_dicts[i])
                    print(f"  > [MultiPolicy] Teacher {i} normalizer loaded. Mean[0]={normalizer.mean[0]:.4f}, Std[0]={normalizer.std[0]:.4f}")
                else:
                    print(f"  > [MultiPolicy] Teacher {i} has NO normalizer stats. Using Identity behavior.")
            
            # 初始冻结
            normalizer.eval()
            self.teacher_normalizers.append(normalizer)

        # 冻结教师网络参数
        for teacher in self.teachers_list:
            for param in teacher.parameters():
                param.requires_grad = False
            teacher.eval()

    # ========================================================================
    # [CRITICAL FIX] 重写 train 方法
    # PyTorch 的 train() 会递归调用所有子模块的 train()。
    # 我们必须拦截这个调用，确保 Teacher 和 Normalizer 永远处于 Eval 模式。
    # ========================================================================
    def train(self, mode=True):
        # 1. 让父类处理 Student 部分的模式切换 (包括 RNN, MLP 等)
        # 注意：这里调用 super().train(mode) 会把所有子模块(包括 teachers)先设为 mode
        super().train(mode)
        
        # 2. 强制将 Teacher 部分改回 Eval 模式
        for teacher in self.teachers_list:
            teacher.eval()
        
        # 3. 强制将 Teacher Normalizer 改回 Eval 模式 (严禁更新均值方差)
        for norm in self.teacher_normalizers:
            norm.eval()
            
        return self

    def evaluate(self, teacher_observations):
        """
        Args:
            teacher_observations: (Total_Envs, Obs_Dim) -> 此时应该是原始的、未归一化的数据
        
        在对教师进行推理前，需要对观测的前3维（pos_error）进行稳态误差补偿：
        pos_error_compensated = pos_error + offset
        """
        total_envs = teacher_observations.shape[0]
        envs_per_teacher = total_envs // self.num_teachers
        
        outputs = []
        
        for i in range(self.num_teachers):
            # 1. 切片
            start_idx = i * envs_per_teacher
            # 处理最后一个 Teacher 可能承担剩余所有环境的情况
            end_idx = start_idx + envs_per_teacher if i < self.num_teachers - 1 else total_envs
            
            # 2. 取出原始观测
            obs_slice = teacher_observations[start_idx:end_idx].clone()  # 复制一份避免修改原数据
            
            # 3. [新增] 对前3维（pos_error）进行稳态误差补偿
            # obs_slice[:, 0:3] 是 pos_error，加上该教师的稳态误差
            obs_slice[:, 0:3] += self.teacher_offsets[i]
            
            # 4. 使用该 Teacher 对应的 Normalizer 进行归一化
            with torch.no_grad():
                # 双重保险：确保使用时是 eval 模式
                self.teacher_normalizers[i].eval()
                self.teachers_list[i].eval()
                
                normalized_obs = self.teacher_normalizers[i](obs_slice)
                
                # 5. 推理 
                action_slice = self.teachers_list[i].act_inference(normalized_obs)
                
            outputs.append(action_slice)
            
        final_actions = torch.cat(outputs, dim=0)
        
        # Teacher 的 act_inference 输出已经通过 Tanh，范围在 [-1, 1]
        return final_actions

    # 兼容性方法，不再需要手动调用，留空即可
    def train_mode(self):
        self.train(True)
            
    def eval_mode(self):
        self.train(False)

    # # 如果需要对学生观测应用offset补偿，可以重写此方法
    # # 另外还需要：❌ 删除这行：obs_slice[:, 0:3] += self.teacher_offsets[i]
    # def _forward_head(self, observations):
    #     """重写父类方法，在学生推理前对观测应用 offset 补偿"""
        
    #     # 1. 克隆观测，避免修改原始数据
    #     obs_compensated = observations.clone()
        
    #     # 2. 根据环境索引应用对应教师的 offset
    #     batch_size = observations.shape[0]
    #     envs_per_teacher = batch_size // self.num_teachers
        
    #     for i in range(self.num_teachers):
    #         start_idx = i * envs_per_teacher
    #         end_idx = start_idx + envs_per_teacher if i < self.num_teachers - 1 else batch_size
            
    #         # 对 pos_error (前3维) 加上该教师的稳态误差
    #         obs_compensated[start_idx:end_idx, 0:3] += self.teacher_offsets[i]
        
    #     # 3. 调用父类的真正推理逻辑
    #     return super()._forward_head(obs_compensated)