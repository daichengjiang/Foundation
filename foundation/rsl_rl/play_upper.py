#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import torch
import numpy as np
import cv2  # 需要安装: pip install opencv-python
import math
from isaaclab.app import AppLauncher

# --- 新增：导入数学工具用于四元数转欧拉角 ---
from isaaclab.utils.math import euler_xyz_from_quat

def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IsaacLab Evaluation for Quadcopter")
    parser.add_argument("--task", type=str, default="point_ctrl_single_train", help="Task name")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to visualize (建议16或25)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the checkpoint (.pt)")
    parser.add_argument("--max_episodes", type=int, default=1000, help="Number of episodes to evaluate before stopping")
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
    from foundation import tasks
    import gymnasium as gym

    class EvaluationBridge:
        def __init__(self, env, simulation_app, policy, max_episodes, visual_vis=False):
            self._env = env
            self._sim_app = simulation_app
            self._unwrapped = env.unwrapped
            self._policy = policy
            self._device = self._unwrapped.device
            self.max_episodes = max_episodes
            self.visual_vis = visual_vis
            
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
                
                self.n_envs = self._unwrapped.num_envs
                self.grid_cols = int(math.ceil(math.sqrt(self.n_envs)))
                self.grid_rows = int(math.ceil(self.n_envs / self.grid_cols))
                print(f"[Visual] 开启深度图可视化: {self.grid_rows}行 x {self.grid_cols}列")
            else:
                self.img_size = 0

        def run(self):
            print(f"开始评估: 目标总 Episode={self.max_episodes}, 可视化={self.visual_vis}")
            obs, _ = self._env.get_observations()
            
            if self.visual_vis:
                obs_dim = obs.shape[1]
                if obs_dim > self.img_size:
                    print(f"[Info] Obs维度: {obs_dim}, 深度图维度: {self.img_size}")
                else:
                    print(f"[Warning] Obs维度 ({obs_dim}) 小于预期图片大小，可视化可能失败！")
            
            while self._sim_app.is_running():
                with torch.inference_mode():
                    actions = self._policy(obs)
                    obs, rewards, dones, infos = self._env.step(actions)
                
                # --- 可视化逻辑 ---
                if self.visual_vis:
                    self.render_depth_grid(obs)

                # --- 统计逻辑 ---
                self.update_stats()

                if self.total_episodes >= self.max_episodes:
                    self.print_summary()
                    break
            
            if self.visual_vis:
                cv2.destroyAllWindows()

        def render_depth_grid(self, obs):
            # 1. 获取深度图数据 (GPU -> CPU)
            depth_flat = obs[:, -self.img_size:] 
            depth_np = depth_flat.cpu().numpy()
            
            # 2. 获取姿态数据 (Roll, Pitch, Yaw)
            # 从环境机器人对象中直接读取四元数
            quats = self._unwrapped._robot.data.root_quat_w
            # 转换为欧拉角 (返回值为 tuple of tensors: roll, pitch, yaw)
            roll, pitch, yaw = euler_xyz_from_quat(quats)
            
            # 转换为 CPU 列表以便循环打印
            roll_deg = (roll * 180 / math.pi).cpu().numpy()
            pitch_deg = (pitch * 180 / math.pi).cpu().numpy()
            yaw_deg = (yaw * 180 / math.pi).cpu().numpy()

            # 3. 深度图归一化
            min_v, max_v = depth_np.min(), depth_np.max()
            if max_v - min_v > 1e-5:
                depth_norm = (depth_np - min_v) / (max_v - min_v)
            else:
                depth_norm = depth_np
            depth_uint8 = (depth_norm * 255).astype(np.uint8)

            # 4. 创建画布
            canvas_h = self.grid_rows * self.img_height
            canvas_w = self.grid_cols * self.img_width
            # 使用3通道彩色画布，以便画彩色文字
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

            for i in range(self.n_envs):
                row = i // self.grid_cols
                col = i % self.grid_cols
                y1, y2 = row * self.img_height, (row + 1) * self.img_height
                x1, x2 = col * self.img_width, (col + 1) * self.img_width

                # 填充深度图 (转为伪彩色或灰度)
                img_gray = depth_uint8[i].reshape(self.img_height, self.img_width)
                img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
                canvas[y1:y2, x1:x2] = img_color
                
                # --- 绘制信息 ---
                # A. 环境编号 (左上角)
                label_env = f"Env {i}"
                cv2.putText(canvas, label_env, (x1 + 2, y1 + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1) # 绿色文字
                
                # B. 姿态信息 (左下角，分两行打印以防重叠)
                # Roll & Pitch
                rp_text = f"R:{roll_deg[i]:.1f} P:{pitch_deg[i]:.1f}"
                cv2.putText(canvas, rp_text, (x1 + 2, y2 - 12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1) # 黄色文字
                
                # Yaw
                y_text = f"Y:{yaw_deg[i]:.1f}"
                cv2.putText(canvas, y_text, (x1 + 2, y2 - 3), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

            # 5. 显示 (放大两倍，视觉效果更好)
            canvas_display = cv2.resize(canvas, (canvas_w * 2, canvas_h * 2), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Depth View & Attitude", canvas_display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                sys.exit()

        def update_stats(self):
            # 复用之前的统计代码
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

                # 计算比例
                success_rate = (self.total_successes / self.total_episodes) * 100
                
                # 构造死亡原因显示的字符串
                deaths = self.total_episodes - self.total_successes
                death_info = ""
                if deaths > 0:
                    parts = []
                    mapping = {"碰": "COLLISION", "低": "TOO_LOW", "高": "TOO_HIGH", "晕": "UNSTABLE", "时": "TIMEOUT"}
                    for label, key in mapping.items():
                        if self.death_counts[key] > 0:
                            parts.append(f"{label}:{self.death_counts[key]/deaths*100:.0f}%")
                    death_info = " | " + " ".join(parts)

                print(f">>> [进度] Ep: {self.total_episodes}/{self.max_episodes} | "
                        f"成功率: {success_rate:.2f}%{death_info}          ", end='\r')
        def print_summary(self):
            print(f"\n\n{'='*50}")
            print(f"评估完成！最终成功率: {(self.total_successes / self.total_episodes) * 100:.2f}%")
            print("失败原因细分 (占总失败次数):")
            deaths = self.total_episodes - self.total_successes
            if deaths > 0:
                for k, v in self.death_counts.items():
                    if v > 0:
                        print(f"  - {k:10}: {v} ({v/deaths*100:.1f}%)")
            print(f"{'='*50}")


    @hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
    def _launch(env_cfg, agent_cfg: RslRlOnPolicyRunnerCfg):
        env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.visual_vis:
             env_cfg.debug_vis = True

        env = gym.make(args_cli.task, cfg=env_cfg)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        print(f"Loading: {args_cli.model_path}")
        ppo_runner.load(args_cli.model_path)
        
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