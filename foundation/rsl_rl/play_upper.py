#!/usr/bin/env python3
"""
IsaacLab runner adapted for quad_point_ctrl_env_single_train.py
适配 Train 环境：移除了脚本端重复的控制器，增加了奖励函数触发以同步状态。
"""

from __future__ import annotations

import argparse
import csv
import math
import struct
import sys
import time
from multiprocessing import shared_memory
from typing import Optional, Tuple

import numpy as np
import torch

from isaaclab.app import AppLauncher

class DepthSharedMemoryWriter:
    """写入深度图到共享内存"""
    _HEADER = struct.Struct("<IId")

    def __init__(self, name: str) -> None:
        self._name = name
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._size = 0
        self._owns_segment = False

    def write(self, depth: np.ndarray, timestamp: float) -> None:
        if depth is None: return
        height, width = depth.shape
        depth_mm = (depth * 1000.0).clip(0, 65535).astype(np.uint16)
        payload_size = width * height * 2
        total_size = self._HEADER.size + payload_size
        self._ensure_segment(total_size)
        if self._shm is None: return
        buf = self._shm.buf
        self._HEADER.pack_into(buf, 0, width, height, timestamp)
        start = self._HEADER.size
        mv = memoryview(buf)[start : start + payload_size]
        mv[:] = depth_mm.tobytes()

    def _ensure_segment(self, required_size: int) -> None:
        if self._shm is not None and self._size >= required_size: return
        self.close()
        try:
            self._shm = shared_memory.SharedMemory(name=self._name, create=True, size=required_size)
            self._size = required_size
            self._owns_segment = True
        except FileExistsError:
            existing = shared_memory.SharedMemory(name=self._name, create=False)
            self._shm = existing
            self._size = existing.size
            self._owns_segment = False

    def close(self) -> None:
        if self._shm is not None:
            self._shm.close()
            if self._owns_segment:
                try: self._shm.unlink()
                except FileNotFoundError: pass
            self._shm = None

def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IsaacLab Evaluation for Train Env")
    parser.add_argument("--task", type=str, default="point_ctrl_single_train", help="必须对应Train环境的Task名")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--drone_id", type=int, default=0)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--reset_log_path", type=str, default="./logs/rsl_rl/data/test.csv")
    parser.add_argument("--reset_log_count", type=int, default=25)
    parser.add_argument("--depth_shm_name", type=str, default="depth_image_shm")
    
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

    class EnvBridge:
        """适配 Train 环境的桥接器"""
        def __init__(self, env, simulation_app, policy, drone_id, reset_log_path, reset_log_count):
            self._env = env
            self._sim_app = simulation_app
            self._unwrapped = env.unwrapped
            self._policy = policy
            self._device = self._unwrapped.device
            self._drone_id = int(drone_id)
            self._step_dt = float(self._unwrapped.step_dt)
            
            self._depth_shm_writer = DepthSharedMemoryWriter(args_cli.depth_shm_name)
            self._reset_log_path = reset_log_path
            self._reset_log_target = int(reset_log_count)
            self._reset_count = 0
            self._csv_header_written = False
            self._traj_buffers = [list() for _ in range(self._unwrapped.num_envs)]
            self._last_states = None

        def run(self):
            self._obs, _ = self._env.get_observations()
            while self._sim_app.is_running():
                self._step_env()

        def _step_env(self):
            with torch.inference_mode():
                # 1. 获取模型输出
                cmd = self._policy(self._obs)
                
                # 2. 调用 Train 环境的 _pre_physics_step
                # 该函数内部已包含控制器计算，会更新环境内的 self._forces 和 self._torques
                _ = self._unwrapped._pre_physics_step(cmd)

                # 3. 物理步进循环
                for _ in range(self._unwrapped.cfg.decimation):
                    self._unwrapped._sim_step_counter += 1
                    # 直接调用环境内部的动作应用逻辑（应用力和扭矩）
                    self._unwrapped._apply_action() 
                    self._unwrapped.scene.write_data_to_sim()
                    self._unwrapped.sim.step(render=False)
                    if self._unwrapped._sim_step_counter % 4 == 0:
                        self._unwrapped.sim.render()
                    self._unwrapped.scene.update(dt=self._unwrapped.physics_dt)

                # 4. 状态同步与奖励触发
                self._unwrapped.episode_length_buf += 1
                self._unwrapped.common_step_counter += 1
                self._unwrapped.reset_terminated[:], self._unwrapped.reset_time_outs[:] = self._unwrapped._get_dones()
                
                self._unwrapped._last_pos_w = self._unwrapped._robot.data.root_state_w[:, :3].clone()
                self._unwrapped._last_actions = self._unwrapped._actions.clone()

                reset_env_ids = (self._unwrapped.reset_terminated | self._unwrapped.reset_time_outs).nonzero(as_tuple=False).squeeze(-1)
                if len(reset_env_ids) > 0:
                    self._unwrapped._reset_idx(reset_env_ids)
                    self._unwrapped.scene.write_data_to_sim()
                    self._unwrapped.sim.forward()
                    self._unwrapped.sim.render()
                
                self._obs, _ = self._env.get_observations()

            # 5. 记录统计数据
            self._cache_state()
            sim_time = float(self._unwrapped._sim_step_counter) * self._step_dt
            self._cache_depth(sim_time)
            reset_flags = self._unwrapped.reset_terminated.detach().cpu().numpy().astype(bool)
            self._record_stats(sim_time, reset_flags)

        def _cache_depth(self, timestamp):
            depth_tensor = self._unwrapped._tiled_camera.data.output.get("depth", None)
            if depth_tensor is not None:
                depth = depth_tensor[self._drone_id, :, :, 0].detach().cpu().numpy().astype(np.float32)
                self._depth_shm_writer.write(depth, timestamp)

        def _cache_state(self):
            states = self._unwrapped._robot.data.root_state_w.detach().cpu().numpy()
            states[:, 10:13] = self._unwrapped._robot.data.root_ang_vel_b.detach().cpu().numpy()
            self._last_states = states

        def _record_stats(self, timestamp, reset_flags):
            if self._last_states is None or self._reset_count >= self._reset_log_target: return
            
            # 使用 Train 环境定义的成功阈值 69.0
            success_threshold = self._unwrapped.cfg.success_threshold
            
            for env_id, state in enumerate(self._last_states):
                pos_x, pos_y, pos_z = state[0], state[1], state[2]
                success = 1.0 if pos_x > success_threshold else 0.0
                self._traj_buffers[env_id].append([timestamp, env_id, pos_x, pos_y, pos_z, state[7], state[8], state[9], success, float(reset_flags[env_id])])
                
                if reset_flags[env_id]:
                    self._write_csv(self._traj_buffers[env_id])
                    self._traj_buffers[env_id].clear()
                    self._reset_count += 1

        def _write_csv(self, rows):
            mode = "w" if not self._csv_header_written else "a"
            with open(self._reset_log_path, mode, newline="") as f:
                writer = csv.writer(f)
                if not self._csv_header_written:
                    writer.writerow(["time", "id", "x", "y", "z", "vx", "vy", "vz", "success", "reset"])
                    self._csv_header_written = True
                writer.writerows(rows)

    @hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
    def _launch(env_cfg, agent_cfg: RslRlOnPolicyRunnerCfg):
        env_cfg.scene.num_envs = args_cli.num_envs
        env = gym.make(args_cli.task, cfg=env_cfg)
        if isinstance(env.unwrapped, DirectMARLEnv): env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        ppo_runner.load(args_cli.model_path)
        
        bridge = EnvBridge(env, simulation_app, ppo_runner.get_inference_policy(device=env.unwrapped.device), args_cli.drone_id, args_cli.reset_log_path, args_cli.reset_log_count)
        bridge.run()
        simulation_app.close()

    _launch()

if __name__ == "__main__":
    main()