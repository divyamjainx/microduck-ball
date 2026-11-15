# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms of the Microduck-on-a-ball task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .observations import BALL_CFG, FEET_CFG, ball_pos_b, ball_vel_b, velocity_target_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def upright(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward the base for staying upright: ``exp(-|gravity in the base xy-plane|^2 / std^2)``, in ``(0, 1]``."""
    gravity_xy = env.scene[asset_cfg.name].data.projected_gravity_b.torch[:, :2]
    return torch.exp(-torch.sum(torch.square(gravity_xy), dim=1) / std**2)


def ball_under_feet(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg = FEET_CFG,
    ball_cfg: SceneEntityCfg = BALL_CFG,
) -> torch.Tensor:
    """Reward the ball for staying under the middle of the feet: ``exp(-distance^2 / std^2)``, in ``(0, 1]``."""
    distance_sq = torch.sum(torch.square(ball_pos_b(env, robot_cfg, ball_cfg)), dim=1)
    return torch.exp(-distance_sq / std**2)


def ball_slip_l2(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = FEET_CFG, ball_cfg: SceneEntityCfg = BALL_CFG
) -> torch.Tensor:
    """Return the squared horizontal speed of the ball relative to the duck's base [m^2/s^2]."""
    return torch.sum(torch.square(ball_vel_b(env, robot_cfg, ball_cfg)), dim=1)


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Mechanical power of the joints, ``sum(|torque * velocity|)`` [W]."""
    asset = env.scene[asset_cfg.name]
    torque = asset.data.applied_torque.torch[:, asset_cfg.joint_ids]
    velocity = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(torque * velocity), dim=1)


def foot_height_on_ball(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = FEET_CFG, ball_cfg: SceneEntityCfg = BALL_CFG
) -> torch.Tensor:
    """Return each ankle's signed radial clearance from the ball surface [m].

    Subtract the ball radius from the ankle's distance to the ball centre.
    Negative values place the ankle inside the sphere. The result has shape
    (num_envs, 2).
    """
    robot, ball = env.scene[robot_cfg.name], env.scene[ball_cfg.name]
    offset = robot.data.body_pos_w.torch[:, robot_cfg.body_ids] - ball.data.root_pos_w.torch.unsqueeze(1)
    return torch.linalg.norm(offset, dim=-1) - ball.cfg.spawn.radius


def foot_slip_on_ball(
    env: ManagerBasedRLEnv,
    contact_height: float,
    robot_cfg: SceneEntityCfg = FEET_CFG,
    ball_cfg: SceneEntityCfg = BALL_CFG,
) -> torch.Tensor:
    """Estimate each foot's tangential slip speed relative to the ball [m/s].

    Use ankle velocity and the ball's linear and angular velocities to estimate
    slip. Return zero for ankles whose radial clearance is at or above
    `contact_height`. The result has shape (num_envs, 2).
    """
    robot, ball = env.scene[robot_cfg.name], env.scene[ball_cfg.name]
    offset = robot.data.body_pos_w.torch[:, robot_cfg.body_ids] - ball.data.root_pos_w.torch.unsqueeze(1)
    normal = offset / torch.linalg.norm(offset, dim=-1, keepdim=True)
    surface_velocity = ball.data.root_lin_vel_w.torch.unsqueeze(1) + torch.cross(
        ball.data.root_ang_vel_w.torch.unsqueeze(1).expand_as(offset), offset, dim=-1
    )
    relative = robot.data.body_lin_vel_w.torch[:, robot_cfg.body_ids] - surface_velocity
    tangential = relative - torch.sum(relative * normal, dim=-1, keepdim=True) * normal
    on_ball = foot_height_on_ball(env, robot_cfg, ball_cfg) < contact_height
    return torch.linalg.norm(tangential, dim=-1) * on_ball


def track_lin_vel_xy_world_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward low horizontal velocity error in world coordinates.

    Rotate the commanded velocity by the reference heading and return
    `exp(-squared_error / std**2)`. Both tasks use a zero velocity command,
    so the reward is highest when the duck has no horizontal motion.
    """
    asset = env.scene[asset_cfg.name]
    error = torch.sum(
        torch.square(velocity_target_w(env, command_name) - asset.data.root_lin_vel_w.torch[:, :2]), dim=1
    )
    return torch.exp(-error / std**2)
