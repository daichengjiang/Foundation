#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import torch
import numpy as np
import cv2  # 需要安装: pip install opencv-python
import math
from isaaclab.app import AppLauncher

def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IsaacLab Evaluation for Quadcopter")
    parser.add_argument("--task", type=str, default="point_ctrl_single_train", help="Task name")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to visualize (建议不要太大，比如16或25，否则窗口太挤)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the checkpoint (.pt)")
    parser.add_argument("--max_episodes", type=int, default=1000, help="Number of episodes to evaluate before stopping")
    
    # --- 新增：可视化开关 ---
    parser.add_argument("--visual_vis", action="store_true", default=False, help="Enable real-time depth map visualization")
    
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
    from foundation import tasks  # 确保你的任务被注册
    import gymnasium as gym

    class EvaluationBridge:
        def __init__(self, env, simulation_app, policy, max_episodes, visual_vis=False):
            self._env = env
            self._sim_app = simulation_app
            self._unwrapped = env.unwrapped
            self._policy = policy
            self._device = self._unwrapped.device
            self.max_episodes = max_episodes
            self.visual_vis = visual_vis  # 开关状态
            
            # --- 统计变量 ---
            self.total_episodes = 0
            self.total_successes = 0
            self.death_counts = {"COLLISION": 0, "TOO_LOW": 0, "TOO_HIGH": 0, "UNSTABLE": 0, "TIMEOUT": 0, "OTHER": 0}
            self.Outcome = self._unwrapped.EpisodeOutcome

            # --- 可视化配置 ---
            if self.visual_vis:
                self.img_height = 60
                self.img_width = 100
                self.img_size = self.img_height * self.img_width
                
                # 计算网格布局
                self.n_envs = self._unwrapped.num_envs
                self.grid_cols = int(math.ceil(math.sqrt(self.n_envs)))
                self.grid_rows = int(math.ceil(self.n_envs / self.grid_cols))
                print(f"[Visual] 开启深度图可视化: {self.grid_rows}行 x {self.grid_cols}列")
            else:
                self.img_size = 0 # 占位

        def run(self):
            print(f"开始评估: 目标总 Episode={self.max_episodes}, 可视化={self.visual_vis}")
            obs, _ = self._env.get_observations()
            
            # 只有开启可视化才做维度检查
            if self.visual_vis:
                obs_dim = obs.shape[1]
                if obs_dim > self.img_size:
                    print(f"[Info] 自动推断本体感知维度: {obs_dim - self.img_size}, 深度图维度: {self.img_size}")
                else:
                    print(f"[Warning] Obs维度 ({obs_dim}) 小于预期图片大小，可视化可能失败！")
            
            while self._sim_app.is_running():
                with torch.inference_mode():
                    actions = self._policy(obs)
                    obs, rewards, dones, infos = self._env.step(actions)
                
                # --- 1. 可视化逻辑 (带开关) ---
                if self.visual_vis:
                    self.render_depth_grid(obs)

                # --- 2. 统计逻辑 ---
                self.update_stats()

                # 退出条件
                if self.total_episodes >= self.max_episodes:
                    self.print_summary()
                    break
            
            if self.visual_vis:
                cv2.destroyAllWindows()

        def render_depth_grid(self, obs):
            # 1. 提取深度部分 (Batch, H*W)
            depth_flat = obs[:, -self.img_size:] 
            depth_np = depth_flat.cpu().numpy()
            
            # 2. 归一化以便显示
            min_v, max_v = depth_np.min(), depth_np.max()
            if max_v - min_v > 1e-5:
                depth_norm = (depth_np - min_v) / (max_v - min_v)
            else:
                depth_norm = depth_np
            depth_uint8 = (depth_norm * 255).astype(np.uint8)

            # 3. 创建大画布
            canvas_h = self.grid_rows * self.img_height
            canvas_w = self.grid_cols * self.img_width
            canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

            for i in range(self.n_envs):
                row = i // self.grid_cols
                col = i % self.grid_cols
                y1, y2 = row * self.img_height, (row + 1) * self.img_height
                x1, x2 = col * self.img_width, (col + 1) * self.img_width

                img = depth_uint8[i].reshape(self.img_height, self.img_width)
                canvas[y1:y2, x1:x2] = img
                
                # 标注环境号
                label = f"{i}"
                cv2.putText(canvas, label, (x1 + 2, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255), 1)

            # 4. 显示 (放大两倍)
            canvas_display = cv2.resize(canvas, (canvas_w * 2, canvas_h * 2), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Depth View", canvas_display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                sys.exit()

        def update_stats(self):
            log_data = self._unwrapped.extras.get("log", {})
            outcomes = log_data.get("Metrics/outcome_episodes_per_step", [])
            
            if len(outcomes) > 0:
                for outcome in outcomes:
                    self.total_episodes += 1
                    if outcome == self.Outcome.SUCCESS:
                        self.total_successes += 1
                    elif outcome == self.Outcome.COLLISION:
                        self.death_counts["COLLISION"] += 1
                    elif outcome == self.Outcome.TOO_LOW:
                        self.death_counts["TOO_LOW"] += 1
                    elif outcome == self.Outcome.TOO_HIGH:
                        self.death_counts["TOO_HIGH"] += 1
                    elif outcome == self.Outcome.UNSTABLE:
                        self.death_counts["UNSTABLE"] += 1
                    elif outcome == self.Outcome.TIMEOUT:
                        self.death_counts["TIMEOUT"] += 1
                    else:
                        self.death_counts["OTHER"] += 1
                
                success_rate = (self.total_successes / self.total_episodes) * 100
                death_str = ""
                # 如果要简化输出，这里只打印成功率
                print(f">>> [进度] Ep: {self.total_episodes}/{self.max_episodes} | Rate: {success_rate:.1f}%", end='\r')

        def print_summary(self):
            print(f"\n\n{'='*50}")
            print(f"评估完成！最终成功率: {(self.total_successes / self.total_episodes) * 100:.2f}%")
            print("失败原因细分:")
            deaths = self.total_episodes - self.total_successes
            if deaths > 0:
                for k, v in self.death_counts.items():
                    if v > 0:
                        print(f"  - {k:10}: {v} ({v/deaths*100:.1f}%)")
            print(f"{'='*50}")

    @hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
    def _launch(env_cfg, agent_cfg: RslRlOnPolicyRunnerCfg):
        env_cfg.scene.num_envs = args_cli.num_envs
        # 如果开启可视化，强制启用 debug_vis 可能有帮助 (视 upper_env 实现而定)
        if args_cli.visual_vis:
             env_cfg.debug_vis = True

        env = gym.make(args_cli.task, cfg=env_cfg)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        print(f"Loading: {args_cli.model_path}")
        ppo_runner.load(args_cli.model_path)
        
        # 传入 visual_vis 参数
        bridge = EvaluationBridge(
            env, 
            simulation_app, 
            ppo_runner.get_inference_policy(device=env.unwrapped.device), 
            args_cli.max_episodes,
            visual_vis=args_cli.visual_vis
        )
        bridge.run()
        simulation_app.close()

    _launch()

if __name__ == "__main__":
    main()