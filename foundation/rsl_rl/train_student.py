import isaaclab
import torch
import hydra
from omegaconf import DictConfig
from foundation.rsl_rl.distill_runner import DistillRunner
import gymnasium as gym
# Import your env config registration

@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 1. 创建环境 (使用 Isaac Lab 标准流程)
    # 这里的 task name 需要对应你注册的 task
    env = gym.make("point_ctrl_single_dense", cfg=cfg.env) 
    
    # 2. 启动蒸馏 Runner
    # log_dir 指向你的 Foundation/logs/rsl_rl 目录
    runner = DistillRunner(env, log_dir="/home/nv/Foundation/logs/rsl_rl")
    
    # 3. 开始训练
    runner.learn()

if __name__ == "__main__":
    main()