#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import torch
import torch.nn as nn      # [新增] 导入 nn
import copy                # [新增] 导入 copy
import numpy as np
import cv2  # 需要安装: pip install opencv-python
import math
import os
from isaaclab.app import AppLauncher

# --- 手写数学转换函数 ---
def euler_from_matrix(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将旋转矩阵转换为欧拉角 (Roll, Pitch, Yaw) - ZYX顺序
    输入: matrix (shape: [N, 3, 3])
    输出: roll, pitch, yaw (shape: [N])
    """
    r11, r12, r13 = matrix[:, 0, 0], matrix[:, 0, 1], matrix[:, 0, 2]
    r21, r22, r23 = matrix[:, 1, 0], matrix[:, 1, 1], matrix[:, 1, 2]
    r31, r32, r33 = matrix[:, 2, 0], matrix[:, 2, 1], matrix[:, 2, 2]

    # Pitch (y)
    pitch = torch.asin(torch.clamp(-r31, -1.0, 1.0))
    # Roll (x)
    roll = torch.atan2(r32, r33)
    # Yaw (z)
    yaw = torch.atan2(r21, r11)

    return roll, pitch, yaw
# ----------------------

# --- [新增] 实物部署模型封装 Wrapper ---
class UpperActorDeployWrapper(nn.Module):
    def __init__(self, policy, obs_normalizer, device):
        super().__init__()
        # 1. 拷贝归一化器
        if obs_normalizer is not None:
            self.obs_normalizer = copy.deepcopy(obs_normalizer).to(device)
        else:
            self.obs_normalizer = nn.Identity().to(device)

        # 2. 提取参数配置
        self.obs_size = policy.obs_size
        self.obs_history_length = policy.obs_history_length
        self.depth_history_length = policy.depth_history_length
        self.depth_height = policy.depth_height
        self.depth_width = policy.depth_width

        # 3. 提取神经网络模块
        self.actor_cnn = copy.deepcopy(policy.actor_cnn).to(device)
        self.actor_obs_his_mlp = copy.deepcopy(policy.actor_obs_his_mlp).to(device)
        
        # 关键：提取底层的纯 PyTorch RNN/GRU，抛弃 RSL-RL 的 Memory 外壳
        self.actor_rnn = copy.deepcopy(policy.actor_memory.rnn).to(device)
        self.actor_mlp = copy.deepcopy(policy.actor).to(device)

        # 4. 初始化 GRU 隐藏状态 (实物部署 batch_size=1)
        num_layers = self.actor_rnn.num_layers
        hidden_size = self.actor_rnn.hidden_size
        self.hidden_states = torch.zeros(num_layers, 1, hidden_size, device=device)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 1. 观测值归一化
        observations = self.obs_normalizer(observations)

        # 2. 切分数据
        obs = observations[:, :self.obs_size]
        depth_start_idx = self.obs_size + self.obs_history_length * self.obs_size
        depth_obs = observations[:, depth_start_idx:]

        # 3. 处理深度图
        depth_obs = depth_obs.reshape(-1, self.depth_history_length, self.depth_height, self.depth_width)
        depth_features = self.actor_cnn(depth_obs)

        # 4. 特征融合
        if self.obs_history_length != 0:
            obs_history = observations[:, self.obs_size:depth_start_idx]
            obs_history_features = self.actor_obs_his_mlp(obs_history)
            fused_features = torch.cat([obs_history_features, depth_features], dim=-1)
        else:
            fused_features = torch.cat([obs, depth_features], dim=-1)

        # 5. GRU 时序推理 (需增加 sequence length 维度)
        rnn_in = fused_features.unsqueeze(0)
        rnn_out, self.hidden_states = self.actor_rnn(rnn_in, self.hidden_states)
        features = rnn_out.squeeze(0)

        # 6. MLP 输出动作并限幅
        actions_mean = self.actor_mlp(features)
        actions = torch.clamp(actions_mean, -1.0, 1.0)
        return actions
# ------------------------------------

def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IsaacLab Evaluation for Quadcopter")
    parser.add_argument("--task", type=str, default="point_ctrl_single_train", help="Task name")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to visualize")
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
            
            self.total_episodes = 0
            self.total_successes = 0
            self.death_counts = {"COLLISION": 0, "TOO_LOW": 0, "TOO_HIGH": 0, "UNSTABLE": 0, "TIMEOUT": 0, "OTHER": 0}
            self.Outcome = self._unwrapped.EpisodeOutcome
            
            # 构建 Outcome 枚举值到名称的映射字典
            self.outcome_map = {v: k for k, v in self.Outcome.__members__.items()}

            # 存储每个环境上一次的结束状态
            self.n_envs = self._unwrapped.num_envs
            self.last_outcomes = ["Running"] * self.n_envs

            if self.visual_vis:
                self.img_height = 60
                self.img_width = 100
                self.img_size = self.img_height * self.img_width
                
                self.grid_cols = int(math.ceil(math.sqrt(self.n_envs)))
                self.grid_rows = int(math.ceil(self.n_envs / self.grid_cols))
                print(f"[Visual] 开启深度图可视化: {self.grid_rows}行 x {self.grid_cols}列")
                
                self.vis_scale = 3
            else:
                self.img_size = 0

        def run(self):
            print(f"开始评估: 目标总 Episode={self.max_episodes}, 可视化={self.visual_vis}")
            obs, _ = self._env.get_observations()

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
                
                # --- 更新环境结果状态 ---
                reset_idxs = torch.nonzero(dones).flatten().cpu().numpy()
                current_outcomes_list = self._unwrapped.extras.get("log", {}).get("Metrics/outcome_episodes_per_step", [])
                
                if len(reset_idxs) > 0 and len(reset_idxs) == len(current_outcomes_list):
                    for idx, outcome_val in zip(reset_idxs, current_outcomes_list):
                        outcome_name = self.outcome_map.get(outcome_val, "Unknown")
                        self.last_outcomes[idx] = outcome_name

                if self.visual_vis:
                    self.render_depth_grid(obs)

                self.update_stats()

                if self.total_episodes >= self.max_episodes:
                    self.print_summary()
                    break
            
            if self.visual_vis:
                cv2.destroyAllWindows()

        def render_depth_grid(self, obs):
            # 1. 解析姿态
            rot_flat = obs[:, 3:12] 
            rot_mat = rot_flat.view(-1, 3, 3)
            roll, pitch, yaw = euler_from_matrix(rot_mat)
            
            roll_deg = torch.rad2deg(roll).cpu().numpy()
            pitch_deg = torch.rad2deg(pitch).cpu().numpy()
            yaw_deg = torch.rad2deg(yaw).cpu().numpy()

            # 2. 解析深度图
            depth_flat = obs[:, -self.img_size:] 
            depth_np = depth_flat.cpu().numpy()
            
            min_v, max_v = depth_np.min(), depth_np.max()
            if max_v - min_v > 1e-5:
                depth_norm = (depth_np - min_v) / (max_v - min_v)
            else:
                depth_norm = depth_np
            depth_uint8 = (depth_norm * 255).astype(np.uint8)

            # 3. 拼接基础画布
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
            
            # 4. 放大画布
            canvas_large = cv2.resize(canvas, (canvas_w * self.vis_scale, canvas_h * self.vis_scale), interpolation=cv2.INTER_NEAREST)
            canvas_large = cv2.cvtColor(canvas_large, cv2.COLOR_GRAY2BGR)

            # 5. 绘制可视化信息
            for i in range(self.n_envs):
                row = i // self.grid_cols
                col = i % self.grid_cols
                
                base_x = col * self.img_width * self.vis_scale
                base_y = row * self.img_height * self.vis_scale
                h_scaled = self.img_height * self.vis_scale
                
                # --- [A] 左上角: ID ---
                label_id = f"ID:{i}"
                cv2.putText(canvas_large, label_id, (base_x + 5, base_y + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3) 
                cv2.putText(canvas_large, label_id, (base_x + 5, base_y + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # --- [B] 右上角: 上次结束结果 ---
                outcome_text = self.last_outcomes[i]
                
                if outcome_text == "SUCCESS":
                    text_color = (0, 255, 0)
                elif outcome_text == "TIMEOUT":
                    text_color = (0, 255, 255)
                elif outcome_text == "Running":
                    text_color = (200, 200, 200)
                else:
                    text_color = (0, 0, 255)

                font_scale_status = 0.45
                (text_w, text_h), _ = cv2.getTextSize(outcome_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale_status, 1)
                
                status_x = base_x + (self.img_width * self.vis_scale) - text_w - 5
                status_y = base_y + 20

                cv2.putText(canvas_large, outcome_text, (status_x, status_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale_status, (0, 0, 0), 3)
                cv2.putText(canvas_large, outcome_text, (status_x, status_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale_status, text_color, 1)

                # --- [C] 左下角: R/P/Y 条形图 ---
                names = ["R", "P", "Y"]
                values = [roll_deg[i], pitch_deg[i], yaw_deg[i]]
                
                start_x = base_x + 5
                bottom_y = base_y + h_scaled - 10
                line_height = 15
                max_bar_width = 190 
                pixels_per_deg = 2.0 
                
                start_draw_y = bottom_y - (3 * line_height) 

                ruler_base_y = start_draw_y - 4
                bar_zero_x = start_x + 20
                for deg in range(10, 91, 10):
                    px_offset = int(deg * pixels_per_deg)
                    tick_x = bar_zero_x + px_offset
                    if px_offset <= max_bar_width:
                        cv2.line(canvas_large, (tick_x, ruler_base_y), (tick_x, ruler_base_y - 3), (200, 200, 200), 1)
                        text_pos = (tick_x - 10, ruler_base_y - 5)
                        cv2.putText(canvas_large, str(deg), text_pos, 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (220, 220, 220), 1)

                for idx, (name, val) in enumerate(zip(names, values)):
                    curr_y = start_draw_y + idx * line_height
                    color = (0, 255, 0) if val >= 0 else (0, 0, 255)
                    length = max(min(int(abs(val) * pixels_per_deg), max_bar_width), 1)

                    cv2.putText(canvas_large, name, (start_x, curr_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.rectangle(canvas_large, (bar_zero_x, curr_y - 8), (bar_zero_x + length, curr_y + 2), color, -1)
                    cv2.rectangle(canvas_large, (bar_zero_x, curr_y - 8), (bar_zero_x + length, curr_y + 2), (0, 0, 0), 1)

            cv2.imshow("Drone Attitude Monitor", canvas_large)
            
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
            deaths = self.total_episodes - self.total_successes
            if deaths > 0:
                print("失败原因细分:")
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
        
        # =========================================================================
        # [新增] 开始导出实物部署模型 (TorchScript)
        # =========================================================================
        export_device = agent_cfg.device
        print(f"\n[INFO] 正在构建 Upper 实物部署模型...")
        try:
            policy_nn = ppo_runner.alg.policy
            obs_normalizer = getattr(ppo_runner, "obs_normalizer", None)
            
            # 1. 实例化 Wrapper 并置为 eval 模式
            deploy_model = UpperActorDeployWrapper(
                policy=policy_nn, 
                obs_normalizer=obs_normalizer, 
                device=export_device
            )
            deploy_model.eval()
            
            # 2. 从环境中获取准确的观测维度
            obs, _ = env.get_observations()
            total_obs_dim = obs.shape[1]
            print(f"[INFO] 预期实物端输入的单帧观测向量维度 (obs_dim): {total_obs_dim}")
            
            # 3. 构造 batch_size=1 的 dummy input 进行 JIT 追踪编译
            dummy_obs_input = torch.randn(1, total_obs_dim, device=export_device)
            
            with torch.inference_mode():
                trace_model = torch.jit.script(deploy_model, dummy_obs_input)
                
                # 默认保存在当前 checkpoint 的同目录下
                export_dir = os.path.dirname(args_cli.model_path)
                export_path = os.path.join(export_dir, "upper_actor_deploy.pt")
                trace_model.save(export_path)
                
                print(f"[SUCCESS] Upper 部署模型已成功保存至: {export_path}")
                print(f"        -> 实体机(LibTorch)直接加载此文件即可运行")
                print(f"        -> 实物端调用输入形状需严格为: (1, {total_obs_dim})\n")
        except Exception as e:
            print(f"[ERROR] 导出部署模型失败: {e}\n")
        # =========================================================================

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