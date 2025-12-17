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
    id="teacher",
    entry_point=f"{__name__}.teacher_env:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.teacher_env:QuadcopterEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
    },
)

gym.register(
    id="offset",
    entry_point=f"{__name__}.offset_env:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.offset_env:QuadcopterEnvCfg",
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