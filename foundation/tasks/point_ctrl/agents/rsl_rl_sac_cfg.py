# rsl_rl_sac_cfg.py
from isaaclab.utils import configclass

@configclass
class SacActorCfg:
    class_name: str = "SACActorModel"
    hidden_dims: list[int] = [128, 128, 128]
    activation: str = "elu"
    obs_normalization: bool = False
    stochastic: bool = True
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"
    state_dependent_std: bool = False

@configclass
class SacCriticCfg:
    class_name: str = "SACCriticModel"
    hidden_dims: list[int] = [128, 128, 128]
    activation: str = "elu"
    obs_normalization: bool = False
    stochastic: bool = False

@configclass
class RslRlSacAlgorithmCfg:
    class_name: str = "SAC"
    replay_buffer_size: int = 1000000
    num_learning_epochs: int = 1
    num_mini_batches: int = 1
    mini_batch_size: int = 4096
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-4
    alpha_learning_rate: float = 1e-4
    gamma: float = 0.99
    tau: float = 0.005
    auto_alpha: bool = True
    alpha: float = 0.05
    policy_frequency: int = 1

@configclass
class RslRlOffPolicyRunnerCfg:
    seed: int = 1
    runner_class_name: str = "OffPolicyRunner"
    actor: SacActorCfg = SacActorCfg()
    critic: SacCriticCfg = SacCriticCfg()
    algorithm: RslRlSacAlgorithmCfg = RslRlSacAlgorithmCfg()
    obs_groups: dict = {"actor": ["policy"], "critic": ["policy"]}
    num_steps_per_env: int = 256
    max_iterations: int = 800
    save_interval: int = 200
    log_interval: int = 1
    experiment_name: str = "single_teacher_sac"
    run_name: str = ""
    logger: str = "wandb"
    resume: bool = False
    load_run: str = "-1"
    load_checkpoint: str = "-1"
    device: str = "cuda:0"
    empirical_normalization: bool = True

@configclass
class QuadcopterTeacherSacRunnerCfg(RslRlOffPolicyRunnerCfg):
    num_steps_per_env = 8  # 每次收集少量步数就赶紧去更新网络
    max_iterations = 2000
    save_interval = 500
    experiment_name = "single_teacher_sac"
    empirical_normalization = True
    
    # 按照论文，隐藏层维度改为 64
    actor = SacActorCfg(hidden_dims=[64, 64, 64], activation="elu", obs_normalization=True)
    critic = SacCriticCfg(hidden_dims=[64, 64, 64], activation="elu", obs_normalization=True)
    
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    
    algorithm = RslRlSacAlgorithmCfg(
        replay_buffer_size=1000000, 
        mini_batch_size=4096,            
        tau=0.005,                  
        auto_alpha=True,     
        alpha=0.05,  # 恢复正常温度     
        
        # 【核心修正】让网络在这一批数据上多学几次！
        num_learning_epochs=4,
        num_mini_batches=8, 
        
        actor_learning_rate=3.0e-4,
        critic_learning_rate=3.0e-4,
        alpha_learning_rate=3.0e-4,
        gamma=0.99,
    )