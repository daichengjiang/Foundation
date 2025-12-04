# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Quacopter environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="point_ctrl_single_experiment",
    entry_point=f"{__name__}.quad_point_ctrl_env_single_experiment:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quad_point_ctrl_env_single_experiment:QuadcopterEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
    },
)

gym.register(
    id="point_ctrl_single_px4",
    entry_point=f"{__name__}.quad_point_ctrl_env_single_px4:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quad_point_ctrl_env_single_px4:QuadcopterEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
    },
)

gym.register(
    id="point_ctrl_single_rate",
    entry_point=f"{__name__}.quad_point_ctrl_env_single_rate:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quad_point_ctrl_env_single_rate:QuadcopterEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
    },
)

gym.register(
    id="point_ctrl_single_simple",
    entry_point=f"{__name__}.quad_point_ctrl_env_single_simple:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quad_point_ctrl_env_single_simple:QuadcopterEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
    },
)

gym.register(
    id="point_ctrl_single_sparse",
    entry_point=f"{__name__}.quad_point_ctrl_env_single_sparse:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quad_point_ctrl_env_single_sparse:QuadcopterEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
    },
)

gym.register(
    id="point_ctrl_single_dense",
    entry_point=f"{__name__}.quad_point_ctrl_env_single_dense:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quad_point_ctrl_env_single_dense:QuadcopterEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
    },
)

gym.register(
    id="distillation",
    entry_point=f"{__name__}.distillation_env:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.distillation_env:QuadcopterEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterDistillationRunnerCfg",
    },
)