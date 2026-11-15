# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Microduck fixed-ball standing and free-ball balancing (two stages)."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

_STAGES = {
    "Fixed": "MicroduckBallFixed",
    "Balance": "MicroduckBallBalance",
}

for _stage, _name in _STAGES.items():
    gym.register(
        id=f"Isaac-Velocity-Ball-{_stage}-Microduck",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ball_env_cfg:{_name}EnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{_name}PPORunnerCfg",
            "default_agent": "rsl_rl",
        },
    )
