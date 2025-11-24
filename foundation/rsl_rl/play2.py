# # Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# """Script to play a checkpoint if an RL agent from RSL-RL."""

# """Launch Isaac Sim Simulator first."""

# import argparse
# import sys
# import numpy as np
# from isaaclab.app import AppLauncher

# # local imports
# import cli_args  # isort: skip

# # add argparse arguments
# parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
# parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
# parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
# parser.add_argument(
#     "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
# )
# parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
# parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# parser.add_argument(
#     "--use_pretrained_checkpoint",
#     action="store_true",
#     help="Use the pre-trained checkpoint from Nucleus.",
# )
# parser.add_argument("--realtime", action="store_true", default=False, help="Run in real-time, if possible.")
# # append RSL-RL cli arguments
# cli_args.add_rsl_rl_args(parser)
# # append AppLauncher cli args
# AppLauncher.add_app_launcher_args(parser)
# args_cli, hydra_args = parser.parse_known_args()

# # always enable cameras to record video
# if args_cli.video:
#     args_cli.enable_cameras = True

# # clear out sys.argv for Hydra
# sys.argv = [sys.argv[0]] + hydra_args

# # launch omniverse app
# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app

# """Rest everything follows."""

# import gymnasium as gym
# import os
# import time
# import torch

# from rsl_rl.runners import OnPolicyRunner

# from isaaclab.envs import (
#     DirectMARLEnv,
#     DirectMARLEnvCfg,
#     DirectRLEnvCfg,
#     ManagerBasedRLEnvCfg,
#     multi_agent_to_single_agent,
# )
# from isaaclab.utils.assets import retrieve_file_path
# from isaaclab.utils.dict import print_dict
# from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

# from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

# import isaaclab_tasks  # noqa: F401
# from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
# from isaaclab_tasks.utils.hydra import hydra_task_config
# from e2e_drone import tasks
# import copy
# # PLACEHOLDER: Extension template (do not remove this comment)

# class ActorWrapper(torch.nn.Module):
#     def __init__(self, onpolicyrunner: OnPolicyRunner, device):
#         super().__init__()
#         if hasattr(onpolicyrunner, "obs_normalizer"):
#             self.obs_normalizer = copy.deepcopy(onpolicyrunner.obs_normalizer).to(device)
#         else:
#             self.obs_normalizer = torch.nn.Identity().to(device)
#         self.actor_cnn = copy.deepcopy(onpolicyrunner.alg.policy.actor_cnn).to(device)
#         self.actor_rnn = copy.deepcopy(onpolicyrunner.alg.policy.actor_memory.rnn).to(device)
#         self.actor = copy.deepcopy(onpolicyrunner.alg.policy.actor).to(device)
#         self.hidden_states = torch.zeros(1, 1, 512, device=device)

#     def forward(self, observations):
#         observations = self.obs_normalizer(observations)
#         obs = observations[:, :20]
#         depth_obs = observations[:, 20:].reshape(-1, 1, 60, 80)
#         depth_feature = self.actor_cnn(depth_obs)
#         feature = torch.cat([obs, depth_feature], dim=-1)
#         feature, self.hidden_states = self.actor_rnn(feature.unsqueeze(0), self.hidden_states)
#         actions_mean = self.actor(feature.squeeze(0))
#         actions = torch.clamp(actions_mean, -1.0, 1.0)  # 确保动作在[-1, 1]范围内
#         actions[:, 3] = (actions[:, 3] + 1.0) * 0.5
#         return actions
    
# @hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
# def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
#     """Play with RSL-RL agent."""
#     agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
#     env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
#     env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device
#     env_cfg.sim.use_fabric = not args_cli.disable_fabric if args_cli.disable_fabric is not None else env_cfg.sim.use_fabric
    
#     # specify directory for logging experiments
#     log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
#     log_root_path = os.path.abspath(log_root_path)
#     print(f"[INFO] Loading experiment from directory: {log_root_path}")
#     if args_cli.use_pretrained_checkpoint:
#         resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
#         if not resume_path:
#             print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
#             return
#     elif args_cli.checkpoint:
#         resume_path = retrieve_file_path(args_cli.checkpoint)
#     else:
#         resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

#     log_dir = os.path.dirname(resume_path)

#     # create isaac environment
#     env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

#     # convert to single-agent instance if required by the RL algorithm
#     if isinstance(env.unwrapped, DirectMARLEnv):
#         env = multi_agent_to_single_agent(env)

#     # wrap for video recording
#     if args_cli.video:
#         video_kwargs = {
#             "video_folder": os.path.join(log_dir, "videos", "play"),
#             "step_trigger": lambda step: step == 0,
#             "video_length": args_cli.video_length,
#             "disable_logger": True,
#         }
#         print("[INFO] Recording videos during training.")
#         print_dict(video_kwargs, nesting=4)
#         env = gym.wrappers.RecordVideo(env, **video_kwargs)

#     # wrap around environment for rsl-rl
#     env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

#     print(f"[INFO]: Loading model checkpoint from: {resume_path}")
#     # load previously trained model
#     ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
#     ppo_runner.load(resume_path)

#     # obtain the trained policy for inference
#     policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

#     # extract the neural network module
#     # we do this in a try-except to maintain backwards compatibility.
#     try:
#         # version 2.3 onwards
#         policy_nn = ppo_runner.alg.policy
#     except AttributeError:
#         # version 2.2 and below
#         policy_nn = ppo_runner.alg.actor_critic

#     # export policy to onnx/jit
#     export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
#     # export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
#     # export_policy_as_onnx(
#     #     policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
#     # )

#     obs_input = torch.randn(1, 4820).to(agent_cfg.device)
#     model = ActorWrapper(onpolicyrunner=ppo_runner, device=agent_cfg.device)
#     model.eval()
#     with torch.inference_mode():
#         trace_model = torch.jit.script(model, obs_input)
#         modelfile = log_root_path + "/actor_deploy.pt"
#         trace_model.save(modelfile)
#         print(f"模型已成功保存至 {modelfile}")

#     dt = env.unwrapped.step_dt

#     # reset environment
#     obs, _ = env.get_observations()
#     timestep = 0
#     # simulate environment
#     while simulation_app.is_running():
#         start_time = time.time()
#         # run everything in inference mode
#         with torch.inference_mode():
#             # agent stepping
#             actions = policy(obs)

#             first_obs = obs[0:1]  # 保持batch维度 [1, 4820]
#             deploy_actions = trace_model(first_obs)
            
#             # 比较第一个无人机的动作
#             policy_action = actions[0]      # 第一个无人机的policy动作
#             deploy_action = deploy_actions[0]  # trace_model的输出动作
#             policy_action = torch.clamp(policy_action, -1.0, 1.0)
#             policy_action[3] = (policy_action[3] + 1.0) * 0.5  # 确保动作在[-1, 1]范围内
#             # 计算差异
#             action_diff = torch.abs(policy_action - deploy_action)
#             max_diff = torch.max(action_diff).item()
#             mean_diff = torch.mean(action_diff).item()
            
#             print(f"  Policy Action:  {policy_action}")
#             print(f"  Deploy Action:  {deploy_action}")
#             print(f"  Difference:     {action_diff}")
#             print(f"  Max Diff: {max_diff:.6f}, Mean Diff: {mean_diff:.6f}")
#             print("-" * 50)            
#             # env stepping
#             obs, _, _, _ = env.step(actions)
#         # if timestep % 10 == 0:
#             # print(f"[INFO] Velocity: {np.mean(np.linalg.norm(env.unwrapped._robot.data.root_lin_vel_w.cpu().numpy(), axis=1))}")
#             # rot_matrix_b2w = matrix_from_quat(env.unwrapped._robot.data.root_quat_w)  # Shape: (num_envs, 3, 3)
#             # direction_to_goal_w = env.unwrapped._desired_pos_w - env.unwrapped._robot.data.root_state_w[:, :3]
#             # direction_to_goal_xy_w = direction_to_goal_w[:, :2]
#             # # 计算 R_b2w 的转置（即 R_w2b）
#             # rot_matrix_w2b = rot_matrix_b2w.transpose(1, 2)  # [num_envs, 3, 3]

#             # # 将 direction_to_goal_xy_w 扩展为 3D 向量（z 分量为 0）
#             # direction_to_goal_w_3d = torch.cat([
#             #     direction_to_goal_xy_w,  # [num_envs, 2]
#             #     torch.zeros(env.unwrapped.num_envs, 1, device=direction_to_goal_xy_w.device)  # [num_envs, 1]
#             # ], dim=1)  # [num_envs, 3]

#             # # 转换到机体坐标系：v_b = R_w2b @ v_w
#             # direction_to_goal_b_3d = torch.bmm(rot_matrix_w2b, direction_to_goal_w_3d.unsqueeze(-1)).squeeze(-1)  # [num_envs, 3]

#             # # 提取 xy 分量
#             # direction_to_goal_xy_b = direction_to_goal_b_3d[:, :2]  # [num_envs, 2]

#         if args_cli.video:
#             timestep += 1
#             # Exit the play loop after recording one video
#             if timestep == args_cli.video_length:
#                 break

#         # time delay for real-time evaluation
#         sleep_time = dt - (time.time() - start_time)
#         # if args_cli.realtime and sleep_time > 0:
#         #     time.sleep(sleep_time)
#         # else:
#         #     print("is too slow!!!")

#     # close the simulator
#     env.close()


# if __name__ == "__main__":
#     # run the main function
#     main()
#     # close sim app
#     simulation_app.close()


# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import numpy as np
import csv  # 添加csv模块导入
import random  # 添加random模块导入
from isaaclab.app import AppLauncher
import math

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--realtime", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import random

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from e2e_drone import tasks
import copy
# PLACEHOLDER: Extension template (do not remove this comment)
class ActorWrapper(torch.nn.Module):
    def __init__(self, onpolicyrunner: OnPolicyRunner, device):
        super().__init__()
        if hasattr(onpolicyrunner, "obs_normalizer"):
            self.obs_normalizer = copy.deepcopy(onpolicyrunner.obs_normalizer).to(device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(device)
        self.actor_cnn = copy.deepcopy(onpolicyrunner.alg.policy.actor_cnn).to(device)
        self.actor_rnn = copy.deepcopy(onpolicyrunner.alg.policy.actor_memory.rnn).to(device)
        self.actor = copy.deepcopy(onpolicyrunner.alg.policy.actor).to(device)
        self.hidden_states = torch.zeros(1, 1, 512, device=device)

    def forward(self, observations):
        observations = self.obs_normalizer(observations)
        obs = observations[:, :20]
        depth_obs = observations[:, 20:].reshape(-1, 1, 60, 100)
        depth_feature = self.actor_cnn(depth_obs)
        feature = torch.cat([obs, depth_feature], dim=-1)
        feature, self.hidden_states = self.actor_rnn(feature.unsqueeze(0), self.hidden_states)
        actions_mean = self.actor(feature.squeeze(0))
        actions = torch.clamp(actions_mean, -1.0, 1.0)  # 确保动作在[-1, 1]范围内
        actions[:, 3] = (actions[:, 3] + 1.0) * 0.5
        return actions
    
@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric if args_cli.disable_fabric is not None else env_cfg.sim.use_fabric
    
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # 创建CSV文件用于保存policy_action数据
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11111")
    csv_file_path = os.path.join(log_dir, "policy_actions.csv")
    csv_file = open(csv_file_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    
    # 写入CSV头部
    csv_writer.writerow(['action_0', 'action_1', 'action_2', 'roll', 'pitch', 'yaw'])
    print(f"[INFO] CSV file created at: {csv_file_path}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    # export_policy_as_onnx(
    #     policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    # )

    obs_input = torch.randn(1, 4820).to(agent_cfg.device)
    model = ActorWrapper(onpolicyrunner=ppo_runner, device=agent_cfg.device)
    model.eval()
    with torch.inference_mode():
        trace_model = torch.jit.script(model, obs_input)
        modelfile = log_root_path + "/actor_deploy2.pt"
        trace_model.save(modelfile)
        print(f"模型已成功保存至 {modelfile}")

    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.get_observations()
    # Save first frame observations as npy file
    first_obs_path = os.path.join(log_dir, "first_observations.npy")
    np.save(first_obs_path, obs.cpu().numpy())
    print(f"[INFO] Saved first frame observations to: {first_obs_path}")
    
    # Save first frame observations as txt file
    first_obs_txt_path = os.path.join(log_dir, "first_observations.txt")
    np.savetxt(first_obs_txt_path, obs.cpu().numpy(), fmt='%.6f')
    print(f"[INFO] Saved first frame observations to: {first_obs_txt_path}")
    
    timestep = 0

    _action_history = []
    _action_history_length = 15

    # trace_model = torch.jit.load("/home/zjr/CrazyE2E/logs/rsl_rl/point_ctrl_direct/actor_deploy1.pt")
    # trace_model = trace_model.to("cuda")
    # simulate environment
    try:
        while simulation_app.is_running():
            start_time = time.time()
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)

                _action_history.append(actions.clone())
                if len(_action_history) > _action_history_length:
                    _action_history.pop(0)


                # --- 新增：随机时延窗口扰动 ---
                # 只在历史足够时才扰动，否则用全部历史
                history_len = len(_action_history)
                if history_len >= _action_history_length:
                    # 随机选择窗口起点，范围为[-6, -2]，即倒数第6到倒数第2个
                    window_start = random.randint(-15, -11)
                    window_end = window_start + 10  # 窗口长度为6
                    # Python负索引切片，window_end可以为0（不包含0本身）
                    selected_actions = _action_history[window_start:window_end]
                    stacked = torch.stack(selected_actions, dim=0)
                    mean_action = stacked.mean(dim=0)
                else:
                    # 历史不足6步，直接用全部历史均值
                    stacked = torch.stack(_action_history, dim=0)
                    mean_action = stacked.mean(dim=0)

                first_obs = obs[0:1]  # 保持batch维度 [1, 4820]

                # obs05 = torch.full((1, 4820), 0.5,dtype=torch.float32).to("cuda")
                # print(obs05)

                deploy_actions = trace_model(first_obs)
                
                # 比较第一个无人机的动作
                policy_action = actions[0]      # 第一个无人机的policy动作
                deploy_action = deploy_actions[0]  # trace_model的输出动作
                mean_action_first = mean_action[0]
                
                policy_action = torch.clamp(policy_action, -1.0, 1.0)
                policy_action[3] = (policy_action[3] + 1.0) * 0.5  # 确保动作在[-1, 1]范围内

                # 将policy_action和mean_action保存到CSV文件
                mean_action_cpu = mean_action_first.cpu().numpy()
                

                mean_action_cpu_angle_0 = mean_action_cpu[0] * (30.0)
                mean_action_cpu_angle_1 = mean_action_cpu[1] * (30.0)
                mean_action_cpu_angle_2 = mean_action_cpu[2] * (90.0)

                # 获取当前无人机的姿态角度（弧度转度）
                robot_quat = env.unwrapped._robot.data.root_quat_w[0]  # 第一个无人机的四元数
                
                # 将四元数转换为欧拉角（roll, pitch, yaw）
                # 使用四元数到欧拉角的转换
                qw, qx, qy, qz = robot_quat.cpu().numpy()
                
                # 四元数到欧拉角转换（ZYX顺序）
                # Roll (x-axis rotation)
                roll_rad = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
                # Pitch (y-axis rotation)
                pitch_rad = math.asin(2 * (qw * qy - qz * qx))
                # Yaw (z-axis rotation)
                yaw_rad = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
                
                # 转换为角度
                roll_deg = math.degrees(roll_rad)
                pitch_deg = math.degrees(pitch_rad)
                yaw_deg = math.degrees(yaw_rad)
                
                # 写入CSV：历史平均动作 + 当前姿态角度
                csv_writer.writerow([
                                   f"{mean_action_cpu_angle_0:.6f}",
                                   f"{mean_action_cpu_angle_1:.6f}", 
                                   f"{mean_action_cpu_angle_2:.6f}",
                                   f"{roll_deg:.6f}",
                                   f"{pitch_deg:.6f}", 
                                   f"{yaw_deg:.6f}"])

                # 计算差异
                action_diff = torch.abs(policy_action - deploy_action)
                max_diff = torch.max(action_diff).item()
                mean_diff = torch.mean(action_diff).item()
                
                print(f"  Policy Action:  {policy_action}")
                print(f"  Deploy Action:  {deploy_action}")
                print(f"  Difference:     {action_diff}")
                print(f"  Max Diff: {max_diff:.6f}, Mean Diff: {mean_diff:.6f}")
                print("-" * 50)            
                # env stepping
                obs, _, _, _ = env.step(mean_action)

            if args_cli.video:
                timestep += 1
                # Exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    break

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            # if args_cli.realtime and sleep_time > 0:
            #     time.sleep(sleep_time)
            # else:
            #     print("is too slow!!!")

    except KeyboardInterrupt:
        print("\n[INFO] Program interrupted by user (Ctrl+C)")
        print(f"[INFO] CSV file saved with current data at: {csv_file_path}")
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        print(f"[INFO] CSV file saved with current data at: {csv_file_path}")
    finally:
        # 确保CSV文件被正确关闭
        csv_file.close()
        print(f"[INFO] Policy actions saved to CSV file: {csv_file_path}")
        
        # close the simulator
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()