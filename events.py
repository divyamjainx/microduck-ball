# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms of the Microduck-on-a-ball task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_duck_on_ball(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice,
    standing_height: float,
    xy_offset: float = 0.01,
    tilt: float = 0.05,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
):
    """Reset the ball and duck to their starting poses and zero their root velocities.

    Place the ball on the ground at the environment origin. Set the duck's base
    `standing_height` metres above the top of the ball, with up to `xy_offset`
    metres of variation along each horizontal axis. Sample roll and pitch between
    `-tilt` and `tilt` radians and yaw between `-pi` and `pi` radians.

    Apply the reset only to the environments selected by `env_ids`.
    """
    robot, ball = env.scene[robot_cfg.name], env.scene[ball_cfg.name]
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    count = len(env_ids)
    origins = env.scene.env_origins[env_ids]
    radius = ball.cfg.spawn.radius
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device).repeat(count, 1)  # x, y, z, w

    # the ball rests on the ground, exactly at the origin
    ball_pose = torch.cat([origins + torch.tensor([0.0, 0.0, radius], device=env.device), identity], dim=-1)
    ball.write_root_pose_to_sim_index(root_pose=ball_pose, env_ids=env_ids)
    ball.write_root_velocity_to_sim_index(root_velocity=torch.zeros(count, 6, device=env.device), env_ids=env_ids)

    # the duck stands on top of it, a few millimetres off-centre and slightly tilted
    angles = math_utils.sample_uniform(-tilt, tilt, (count, 2), env.device)
    yaw = math_utils.sample_uniform(-torch.pi, torch.pi, (count,), env.device)
    quat = math_utils.quat_from_euler_xyz(angles[:, 0], angles[:, 1], yaw)
    offset = math_utils.sample_uniform(-xy_offset, xy_offset, (count, 2), env.device)
    position = origins + torch.cat([offset, torch.full((count, 1), 2 * radius + standing_height, device=env.device)], 1)
    robot.write_root_pose_to_sim_index(root_pose=torch.cat([position, quat], dim=-1), env_ids=env_ids)
    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(count, 6, device=env.device), env_ids=env_ids)
