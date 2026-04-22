# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--log_timestamp", type=str, default=None, help="Fixed timestamp folder name.")

# [NEW] 参数筛选所需的命令行参数
parser.add_argument("--override_hidden_dims", type=int, nargs="+", default=None, help="Override actor and critic hidden dims (e.g. 128 128 128)")
parser.add_argument("--override_entropy", type=float, default=None, help="Override entropy coefficient")
parser.add_argument("--override_schedule", type=str, default=None, help="Override learning rate schedule")
parser.add_argument("--override_num_learning_epochs", type=int, default=None, help="Override num learning epochs per iteration")
parser.add_argument("--run_name_suffix", type=str, default=None, help="Suffix for wandb run name")
parser.add_argument("--wandb_project", type=str, default=None, help="WandB project name")

# [NEW] 奖励系数参数 (默认值为 None，表示使用 Config 中的原始值)
parser.add_argument("--reward_coef_position_cost", type=float, default=None, help="Override position cost coef")
parser.add_argument("--reward_coef_orientation_cost", type=float, default=None, help="Override orientation cost coef")
parser.add_argument("--reward_coef_d_action_cost", type=float, default=None, help="Override action smooth cost coef")
parser.add_argument("--reward_coef_termination_penalty", type=float, default=None, help="Override termination penalty")
parser.add_argument("--reward_constant", type=float, default=None, help="Override reward constant")
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
import torch
from datetime import datetime

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
# from on_policy_runner import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from foundation import tasks

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.num_steps_per_env = agent_cfg.num_steps_per_env
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # [NEW] 使用命令行参数覆盖默认配置 (参数搜索核心逻辑)
    if args_cli.override_hidden_dims:
        print(f"[INFO] Overriding Hidden Dims to: {args_cli.override_hidden_dims}")
        agent_cfg.policy.actor_hidden_dims = args_cli.override_hidden_dims
        agent_cfg.policy.critic_hidden_dims = args_cli.override_hidden_dims
        
    if args_cli.override_entropy is not None:
        print(f"[INFO] Overriding Entropy Coef to: {args_cli.override_entropy}")
        agent_cfg.algorithm.entropy_coef = args_cli.override_entropy
        
    if args_cli.override_schedule:
        print(f"[INFO] Overriding Schedule to: {args_cli.override_schedule}")
        agent_cfg.algorithm.schedule = args_cli.override_schedule

    if args_cli.override_num_learning_epochs is not None:
        print(f"[INFO] Overriding Num Learning Epochs to: {args_cli.override_num_learning_epochs}")
        agent_cfg.algorithm.num_learning_epochs = args_cli.override_num_learning_epochs

    # [NEW] 修改 WandB 的 Run Name 和 Experiment Name
    if args_cli.run_name_suffix:
        # 修改 experiment_name，防止污染正常的 single_teacher 文件夹
        agent_cfg.experiment_name = "param_search"
        # 修改 run_name，这样 WandB 上能直接看出参数组合
        agent_cfg.run_name = f"Search_{args_cli.run_name_suffix}"
        

    # [NEW] 覆盖奖励系数
    # 请根据你 teacher_env.py 中 QuadcopterEnvCfg 的实际结构调整以下属性名
    if args_cli.reward_coef_position_cost is not None:
        print(f"[INFO] Overriding Position Cost to: {args_cli.reward_coef_position_cost}")
        # 如果你的参数在 env_cfg.rewards 下，请改为 env_cfg.rewards.xxx.weight
        env_cfg.reward_coef_position_cost = args_cli.reward_coef_position_cost
        
    if args_cli.reward_coef_orientation_cost is not None:
        print(f"[INFO] Overriding Orientation Cost to: {args_cli.reward_coef_orientation_cost}")
        env_cfg.reward_coef_orientation_cost = args_cli.reward_coef_orientation_cost
        
    if args_cli.reward_coef_d_action_cost is not None:
        print(f"[INFO] Overriding Action Cost to: {args_cli.reward_coef_d_action_cost}")
        env_cfg.reward_coef_d_action_cost = args_cli.reward_coef_d_action_cost

    if args_cli.reward_coef_termination_penalty is not None:
        print(f"[INFO] Overriding Termination Penalty to: {args_cli.reward_coef_termination_penalty}")
        env_cfg.reward_coef_termination_penalty = args_cli.reward_coef_termination_penalty
        
    if args_cli.reward_constant is not None:
        print(f"[INFO] Overriding Reward Constant to: {args_cli.reward_constant}")
        env_cfg.reward_constant = args_cli.reward_constant

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    agent_cfg_dict = agent_cfg.to_dict()

    # [修改] WandB 命名逻辑
    if args_cli.log_timestamp:
        # 0. 记录去掉时间戳之前的原始 run_name
        original_run_name = agent_cfg.run_name 
        
        # 1. 构造用于 WandB 的长运行名称
        new_run_name = f"{args_cli.log_timestamp}_{original_run_name}"
        
        # 2. 更新配置字典中的 run_name
        agent_cfg_dict["run_name"] = new_run_name
        agent_cfg.run_name = new_run_name
        
        # 3. 确保 logger 依然是字符串
        if agent_cfg_dict.get("logger") == "wandb":
            agent_cfg_dict["wandb_name"] = new_run_name
            agent_cfg_dict["wandb_id"] = new_run_name
            agent_cfg_dict["wandb_group"] = agent_cfg_dict.get("experiment_name")
        
        # 4. 计算本地日志路径
        local_run_folder = original_run_name
        
        log_dir = os.path.join(log_root_path, args_cli.log_timestamp, local_run_folder)
        
        print(f"[INFO] Using fixed timestamp: {args_cli.log_timestamp}")
        print(f"[INFO] Local Log Dir: {log_dir}")
        
    else:
        # 默认逻辑 (参数搜索通常走这里，或者你可以根据喜好在 select_params.py 里也不传 timestamp)
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)


    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=log_dir, device=agent_cfg.device)
    
    # [NEW] 强制覆盖 Runner 内部的 run_name，确保 WandB 记录正确
    if args_cli.log_timestamp:
        runner.run_name = agent_cfg.run_name
    elif args_cli.run_name_suffix:
         # 如果是参数搜索模式，也确保 Runner 使用带有 Search_ 前缀的名字
        runner.run_name = agent_cfg.run_name

    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or args_cli.checkpoint:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
    if args_cli.init_noise_std:
        runner.load_std(args_cli.init_noise_std)
        print(f"[INFO]: Loading init noise std from: {args_cli.init_noise_std}")


    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()