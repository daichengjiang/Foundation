#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import torch
import numpy as np
from isaaclab.app import AppLauncher

def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IsaacLab Evaluation for Quadcopter")
    parser.add_argument("--task", type=str, default="point_ctrl_single_train", help="Task name")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to run in parallel")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the checkpoint (.pt)")
    parser.add_argument("--max_episodes", type=int, default=1000, help="Number of episodes to evaluate before stopping")
    
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args
    return args

def main() -> None:
    args_cli = _parse_arguments()
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    from isaaclab_tasks.utils.hydra import hydra_task_config
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from foundation import tasks
    import gymnasium as gym

    class EvaluationBridge:
        def __init__(self, env, simulation_app, policy, max_episodes):
            self._env = env
            self._sim_app = simulation_app
            self._unwrapped = env.unwrapped
            self._policy = policy
            self._device = self._unwrapped.device
            
            # 统计变量
            self.max_episodes = max_episodes
            self.total_episodes = 0
            self.total_successes = 0
            # 使用环境内部定义的 Enum 方便对比
            self.Outcome = self._unwrapped.EpisodeOutcome

        def run(self):
            print(f"开始评估: 并行环境数={self._unwrapped.num_envs}, 目标总 Episode={self.max_episodes}")
            obs, _ = self._env.get_observations()
            
            while self._sim_app.is_running():
                with torch.inference_mode():
                    actions = self._policy(obs)
                    obs, rewards, dones, infos = self._env.step(actions)
                
                # --- 修改后的统计逻辑 ---
                # 直接从环境的 extras 中读取本步完成的所有 episode 结果
                # upper_env.py 在 _get_dones 中存入了这些数据
                log_data = self._unwrapped.extras.get("log", {})
                outcomes = log_data.get("Metrics/outcome_episodes_per_step", [])

                if len(outcomes) > 0:
                    for outcome in outcomes:
                        self.total_episodes += 1
                        if outcome == self.Outcome.SUCCESS:
                            self.total_successes += 1
                    
                    # 实时打印进度
                    success_rate = (self.total_successes / self.total_episodes) * 100
                    print(f">>> [进度] Ep: {self.total_episodes}/{self.max_episodes} | "
                          f"成功率: {success_rate:.2f}%", end='\r')

                # 达到目标数量停止
                if self.total_episodes >= self.max_episodes:
                    print(f"\n\n{'='*50}")
                    print(f"评估完成！")
                    print(f"总 Episodes: {self.total_episodes}")
                    print(f"总成功数: {self.total_successes}")
                    print(f"最终成功率: {(self.total_successes / self.total_episodes) * 100:.2f}%")
                    print(f"{'='*50}")
                    break

    @hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
    def _launch(env_cfg, agent_cfg: RslRlOnPolicyRunnerCfg):
        # 修改环境配置：增加并行数量
        env_cfg.scene.num_envs = args_cli.num_envs
        
        # 创建环境
        env = gym.make(args_cli.task, cfg=env_cfg)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        
        # 加载模型
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        print(f"正在加载模型权重: {args_cli.model_path}")
        ppo_runner.load(args_cli.model_path)
        
        # 运行评估桥接器
        bridge = EvaluationBridge(
            env=env, 
            simulation_app=simulation_app, 
            policy=ppo_runner.get_inference_policy(device=env.unwrapped.device),
            max_episodes=args_cli.max_episodes
        )
        bridge.run()
        simulation_app.close()

    _launch()

if __name__ == "__main__":
    main()