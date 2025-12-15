# multi_teacher_policy.py

import torch
import torch.nn as nn
from rsl_rl.modules import StudentTeacherRecurrentCustom, EmpiricalNormalization

class MultiTeacherPolicy(StudentTeacherRecurrentCustom):
    def __init__(self, *args, 
                 teacher_models: list[nn.Module] = None, 
                 teacher_norm_state_dicts: list[dict] = None, 
                 **kwargs):
        
        super().__init__(*args, **kwargs)
        
        if teacher_models is None or len(teacher_models) == 0:
            raise ValueError("MultiTeacherPolicy requires a list of teacher_models")

        self.num_teachers = len(teacher_models)
        self.teachers_list = nn.ModuleList(teacher_models)
        
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
                    print(f"  > [MultiPolicy] Teacher {i} normalizer loaded. Mean[0]={normalizer.mean[0]:.4f}")
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
            obs_slice = teacher_observations[start_idx:end_idx]
            
            # 3. 使用该 Teacher 对应的 Normalizer 进行归一化
            with torch.no_grad():
                # 双重保险：确保使用时是 eval 模式
                self.teacher_normalizers[i].eval()
                self.teachers_list[i].eval()
                
                normalized_obs = self.teacher_normalizers[i](obs_slice)
                
                # 4. 推理 (act_inference 已经包含 Tanh，输出范围 [-1, 1])
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