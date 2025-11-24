# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import numpy as np
from isaaclab.app import AppLauncher

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
# PLACEHOLDER: Extension template (do not remove this comment)
import pandas as pd

import copy    
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
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to jit
    obs_input = torch.randn(1, 6020).to(agent_cfg.device)
    model = ActorWrapper(onpolicyrunner=ppo_runner, device=agent_cfg.device)
    model = model.to(agent_cfg.device)
    model.eval()
    with torch.inference_mode():
        trace_model = torch.jit.script(model, obs_input)
        modelfile = log_root_path + "/actor_deploy.pt"
        trace_model.save(modelfile)
        print(f"模型已成功保存至 {modelfile}")

        for i in range(10):
            print(f"Actions : {trace_model(obs_input)}")

    dt = env.unwrapped.step_dt
    max_steps = 1000
    # reset environment and print first frame observations
    obs, _ = env.get_observations()
    actions_list = []
    timestep = 0
    record_duration = 10
    max_steps = int(record_duration / dt)
    actions = torch.zeros(env.num_envs, 4, device=env.unwrapped.device)
    pre_vel = torch.zeros(env.num_envs, 3, device=env.unwrapped.device)
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            # actions = policy(obs)
            actions[:, 0] = 0.0
            actions[:, 1] = 0.0
            actions[:, 2] = 0.0
            actions[:, 3] = 1.0
            # env stepping
            obs, _, _, _ = env.step(actions)
        acc = (env.unwrapped._robot.data.root_lin_vel_b - pre_vel) / dt
        pre_vel = env.unwrapped._robot.data.root_lin_vel_b.clone()
        if timestep % 10 == 0:
            print(f"Timestep: {timestep}, Acc: {acc[0].cpu().numpy()}")
        # if timestep == max_steps - 1:
        #     df = pd.DataFrame(actions_list, columns=['roll', 'pitch', 'yaw', 'thrust'])
        #     csv_path = os.path.join(log_dir, f"actions_{record_duration}s.csv")
        #     df.to_csv(csv_path, index=False)
        #     print(f"[INFO] Actions saved to: {csv_path}")

        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        
        timestep += 1
        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        # if args_cli.realtime and sleep_time > 0:
        #     time.sleep(sleep_time)
        # else:
        #     print("is too slow!!!")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()