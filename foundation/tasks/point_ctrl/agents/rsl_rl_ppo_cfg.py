# Copyright (c) 2025 Xu Yang
# HKUST UAV Group
#
# Author: Xu Yang
# Affiliation: HKUST UAV Group
# Date: April 2025
# License: MIT License

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlDistillationAlgorithmCfg

from isaaclab.utils import configclass


@configclass
class QuadcopterTeacherRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 256
    max_iterations = 5000
    save_interval = 200
    experiment_name = "single_teacher"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[64, 64, 64],
        critic_hidden_dims=[64, 64, 64],
        activation="elu",
        class_name="ActorCritic",  # "ActorCriticRNN" or "ActorCriticAtten" or "ActorCriticMLP"
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0002, #0.0002
        num_learning_epochs=1,  #4
        num_mini_batches=64,   #4
        learning_rate=1.0e-4,
        schedule="fixed",  #"adaptive"
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class QuadcopterUpperRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 128
    max_iterations = 5000
    save_interval = 500
    experiment_name = "upper"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
        class_name="ActorCriticRNN",  # "ActorCriticRNN" or "ActorCriticAtten" or "ActorCriticMLP"
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0002, #0.0002
        num_learning_epochs=1,  #4
        num_mini_batches=32,   #4
        learning_rate=1.0e-4,
        schedule="adaptive",  #"adaptive"
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class QuadcopterDistillationPolicyCfg(RslRlPpoActorCriticCfg):
    """
    Quadcopter蒸馏任务的策略配置。
    该类继承自 RslRlPpoActorCriticCfg，并添加了教师网络所需的参数。
    """
    
    # *** 教师网络参数 (用于蒸馏) ***
    teacher_hidden_dims: list[int] = [64, 64, 64]  # 教师网络的隐藏层维度
    teacher_recurrent: bool = False               # 教师为MLP (非循环网络)
    
    # *** 学生网络/策略通用参数 ***
    init_noise_std = 0.0
    activation = "elu"
    class_name = "StudentTeacherRecurrentCustom"  # 核心：指定自定义的策略实现类
    
    # *** 学生网络MLP/RNN架构参数 ***
    student_hidden_dims = []
    rnn_type = "gru"         # 使用GRU
    rnn_hidden_dim = 16      # GRU隐藏层维度
    rnn_num_layers = 1       # GRU层数
    pre_rnn_dim = 16         # GRU前的Dense层输出维度
    post_rnn_dim = 16        # GRU后的Dense层输出维度

@configclass
class QuadcopterDistillationRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 400
    max_iterations = 1500
    save_interval = 100
    experiment_name = "distillation"
    empirical_normalization = True
    
    # *** 策略配置：直接使用我们新定义的配置类 ***
    policy = QuadcopterDistillationPolicyCfg()
    
    # *** 算法配置保持不变 ***
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=4, 
        learning_rate=1e-4,
        max_grad_norm=1.0,
        gradient_length=15,
        class_name="Distillation",
    )

@configclass
class QuadcopterDistillPostPolicyCfg(RslRlPpoActorCriticCfg):
    """
    专为 Distill Post 阶段 (PPO From Scratch) 设计的策略配置。
    使用 StudentTeacherRecurrentCustom 网络结构。
    """
    # 指定自定义网络类名 (必须在 python path 下可被 import)
    class_name = "StudentTeacherRecurrentCustom"

    # *** 自定义网络的特定参数 ***
    # 1. 网络架构参数 (对应 StudentTeacherRecurrentCustom.__init__)
    rnn_type = "gru"
    rnn_hidden_dim = 16      # GRU 隐层维度
    rnn_num_layers = 1
    pre_rnn_dim = 16         # 输入 -> GRU 之前的 MLP 维度
    post_rnn_dim = 16        # GRU -> 输出 之前的 MLP 维度
    
    # 2. Student Head (GRU 之后的 MLP)
    # 如果留空，则直接输出动作；如果不为空，则添加额外的 MLP 层
    student_hidden_dims = [] 
    
    # 3. Teacher/Critic 参数
    # 在 PPO 训练中，Teacher 分支通常充当 Critic (Value Function)
    teacher_hidden_dims = [64, 64, 64]
    teacher_recurrent = False # Critic 使用 MLP
    
    # 4. 初始化噪声
    init_noise_std = 1.0
    activation = "elu"

# @configclass
# class QuadcopterDistillPostRunnerCfg(RslRlOnPolicyRunnerCfg):
#     """
#     对应 Distill Post Env 的 Runner 配置
#     """
#     num_steps_per_env = 256  # 与 env 配置保持一致
#     max_iterations = 10000
#     save_interval = 200
#     experiment_name = "distill_post_train" # 实验名称
#     empirical_normalization = True
    
#     # 加载上面的策略配置
#     policy = QuadcopterDistillPostPolicyCfg()
    
#     algorithm = RslRlPpoAlgorithmCfg(
#         value_loss_coef=1.0,
#         use_clipped_value_loss=True,
#         clip_param=0.2,
#         entropy_coef=0.0002,
#         num_learning_epochs=5,
#         num_mini_batches=4, # 根据显存调整
#         learning_rate=1.0e-4,
#         schedule="adaptive",
#         gamma=0.99,
#         lam=0.95,
#         desired_kl=0.01,
#         max_grad_norm=1.0,
#     )

# [新增] 1. 定义一个新的算法配置类，包含论文 Algorithm 1 所需的所有参数
@configclass
class QuadcopterAdaptivePpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """
    适配论文 Performance-Adaptive RL 的配置类
    """
    # 覆盖原有的 schedule
    schedule: str = "performance_adaptive"
    
    # [新增] Algorithm 1 专用超参数
    critic_learning_rate: float = 1.0e-3    # Critic 初始学习率 (LR_V)
    critic_warmup_iterations: int = 50      # Critic 预热步数 (Freeze Actor)
    
    # 自适应调整系数 (对应论文中的常数 c)
    adaptive_c_v: float = 1e-5        # c_V: 用于增加 Actor LR
    adaptive_c_pi: float = 1e-5       # c_pi: 用于减小 Critic LR
    adaptive_c_epsilon: float = 0.01  # c_epsilon: 用于增加 Clip Range
    
    # 边界限制
    lr_max: float = 5.0e-4            # Actor LR 上限
    lr_min: float = 1.0e-6            # Critic LR 下限
    epsilon_max: float = 0.5          # Clip Range 上限

# ... (QuadcopterTeacherRunnerCfg 保持不变，此处省略) ...
# ... (QuadcopterUpperRunnerCfg 保持不变，此处省略) ...
# ... (QuadcopterDistillationPolicyCfg 保持不变，此处省略) ...
# ... (QuadcopterDistillationRunnerCfg 保持不变，此处省略) ...
# ... (QuadcopterDistillPostPolicyCfg 保持不变，此处省略) ...

@configclass
class QuadcopterDistillPostRunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    对应 Distill Post Env 的 Runner 配置 (RL 微调阶段)
    """
    num_steps_per_env = 256
    max_iterations = 10000
    save_interval = 200
    experiment_name = "distill_post_train"
    empirical_normalization = True
    
    # 使用自定义的 Policy 配置 (保持你之前的定义)
    policy = QuadcopterDistillPostPolicyCfg()
    
    # [修改] 2. 使用上面定义的新算法配置类
    algorithm = QuadcopterAdaptivePpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,       # 初始 epsilon
        entropy_coef=0.0002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4, # Actor 初始学习率 (LR_pi)
        schedule="performance_adaptive", # 已经在类定义中默认设置了
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        
        # 下面这些是新类的默认值，你可以在这里覆盖它们
        critic_learning_rate=1.0e-3,
        critic_warmup_iterations=50,
        adaptive_c_v=1e-5,
        adaptive_c_pi=1e-5,
        adaptive_c_epsilon=0.01,
    )