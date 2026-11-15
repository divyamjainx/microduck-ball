# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms of the Microduck-on-a-ball task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .observations import BALL_CFG, FEET_CFG, ball_pos_b

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_off_feet(
    env: ManagerBasedRLEnv,
    max_distance: float,
    robot_cfg: SceneEntityCfg = FEET_CFG,
    ball_cfg: SceneEntityCfg = BALL_CFG,
) -> torch.Tensor:
    """Terminate when the ball centre is further than ``max_distance`` [m] (horizontally) from the middle of the feet.

    This is the "safe zone" around the feet: past it the duck has lost the ball.
    """
    return torch.linalg.norm(ball_pos_b(env, robot_cfg, ball_cfg), dim=1) > max_distance


def base_too_low(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = BALL_CFG,
) -> torch.Tensor:
    """Terminate when the base is less than ``minimum_height`` [m] above the top of the ball (the duck has crouched
    onto it or slid off it)."""
    ball = env.scene[ball_cfg.name]
    top = ball.data.root_pos_w.torch[:, 2] + ball.cfg.spawn.radius
    return env.scene[asset_cfg.name].data.root_pos_w.torch[:, 2] - top < minimum_height


def ball_outside_zone(env: ManagerBasedRLEnv, radius: float, ball_cfg: SceneEntityCfg = BALL_CFG) -> torch.Tensor:
    """Terminate when the ball is further than ``radius`` [m] (horizontally) from the origin of its environment."""
    ball = env.scene[ball_cfg.name]
    offset = ball.data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2]
    return torch.linalg.norm(offset, dim=1) > radius
