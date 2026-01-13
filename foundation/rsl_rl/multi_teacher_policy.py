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
        # 调用父类 train，这会把 Student 网络 (RNN, MLP) 和 self.student_normalizer 设为 mode
        super().train(mode)
        
        # 强制 Teacher 和 Teacher Normalizer 永远处于 Eval
        for teacher in self.teachers_list:
            teacher.eval()
        for norm in self.teacher_normalizers:
            norm.eval()
            
        return self

    def evaluate(self, teacher_observations):

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
                
                normalized_obs = self.teacher_normalizers[i](obs_slice)
                action_slice = self.teachers_list[i].act_inference(normalized_obs)
                
            outputs.append(action_slice)
            
        return torch.cat(outputs, dim=0)
