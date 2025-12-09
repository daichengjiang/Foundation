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
class QuadcopterPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 512
    max_iterations = 5000
    save_interval = 25
    experiment_name = "point_ctrl_direct"
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
        num_mini_batches=32,   #4
        learning_rate=1.0e-4,
        schedule="fixed",  #"adaptive"
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
    actor_hidden_dims = []   # 不使用额外的MLP层，直接从post_rnn_dim输出
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
        learning_rate=1e-3,
        max_grad_norm=1.0,
        gradient_length=15,
        class_name="Distillation",
    )