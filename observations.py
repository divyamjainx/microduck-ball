# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms of the Microduck-on-a-ball task.

The ball is read straight from the simulator. A real Microduck has no such sensor: it would have to estimate the ball
(for example from the legs' joint feedback or a camera), so a policy trained on these terms is a simulation result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

FEET_CFG = SceneEntityCfg("robot", body_names=["ankle_left", "ankle_right"])
BALL_CFG = SceneEntityCfg("ball")


def feet_center_w(env: ManagerBasedEnv, robot_cfg: SceneEntityCfg = FEET_CFG) -> torch.Tensor:
    """Middle of the two feet in the world frame. Shape is (num_envs, 3)."""
    robot = env.scene[robot_cfg.name]
    return robot.data.body_pos_w.torch[:, robot_cfg.body_ids].mean(dim=1)


def ball_pos_b(
    env: ManagerBasedEnv, robot_cfg: SceneEntityCfg = FEET_CFG, ball_cfg: SceneEntityCfg = BALL_CFG
) -> torch.Tensor:
    """Return the horizontal offset from the ankle midpoint to the ball centre.

    Express the offset in metres in the duck's heading frame, which follows yaw.
    The result has shape (num_envs, 2) and is zero when the ball centre and ankle
    midpoint share the same horizontal position.
    """
    robot, ball = env.scene[robot_cfg.name], env.scene[ball_cfg.name]
    delta = ball.data.root_pos_w.torch - feet_center_w(env, robot_cfg)
    return quat_apply_inverse(yaw_quat(robot.data.root_quat_w.torch), delta)[:, :2]


def ball_vel_b(
    env: ManagerBasedEnv, robot_cfg: SceneEntityCfg = FEET_CFG, ball_cfg: SceneEntityCfg = BALL_CFG
) -> torch.Tensor:
    """Horizontal velocity of the ball relative to the duck's base, in the duck's heading frame [m/s].

    Zero when the ball is carried along with the duck. Shape is (num_envs, 2).
    """
    robot, ball = env.scene[robot_cfg.name], env.scene[ball_cfg.name]
    delta = ball.data.root_lin_vel_w.torch - robot.data.root_lin_vel_w.torch
    return quat_apply_inverse(yaw_quat(robot.data.root_quat_w.torch), delta)[:, :2]


def ball_ang_vel_w(env: ManagerBasedEnv, ball_cfg: SceneEntityCfg = BALL_CFG) -> torch.Tensor:
    """Angular velocity of the ball in the world frame [rad/s]. Shape is (num_envs, 3)."""
    return env.scene[ball_cfg.name].data.root_ang_vel_w.torch


def _heading(env: ManagerBasedEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Yaw angle of the robot base in the world frame [rad]. Shape is (num_envs,)."""
    return euler_xyz_from_quat(env.scene[robot_cfg.name].data.root_quat_w.torch)[2]


def reference_heading(env: ManagerBasedEnv) -> torch.Tensor:
    """Initial episode heading [rad], held fixed for the zero-command balance tasks."""
    if not hasattr(env, "_reference_heading"):
        env._reference_heading = torch.zeros(env.num_envs, device=env.device)
    return env._reference_heading


def reset_reference_heading(env: ManagerBasedEnv, env_ids: torch.Tensor | slice):
    """Store the current yaw as the reference heading for the reset environments."""
    reference_heading(env)[env_ids] = _heading(env)[env_ids]


def velocity_target_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """The commanded horizontal velocity in the world frame [m/s]: the command turned by the reference heading."""
    command = env.command_manager.get_command(command_name)[:, :2]
    heading = reference_heading(env)
    cos, sin = torch.cos(heading), torch.sin(heading)
    return torch.stack([cos * command[:, 0] - sin * command[:, 1], sin * command[:, 0] + cos * command[:, 1]], dim=-1)


def velocity_target_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """The commanded horizontal velocity in the frame of the duck's heading [m/s]. Shape is (num_envs, 2)."""
    target = velocity_target_w(env, command_name)
    heading = _heading(env)
    cos, sin = torch.cos(heading), torch.sin(heading)
    return torch.stack([cos * target[:, 0] + sin * target[:, 1], -sin * target[:, 0] + cos * target[:, 1]], dim=-1)


def velocity_command_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """What the policy is asked for: the target velocity as seen from the duck, and the yaw rate. Shape is (N, 3).

    Both supported tasks command zero velocity, so this term remains zero. It is retained to preserve the trained
    policy input structure; it does not encode heading error for these tasks.
    """
    command = env.command_manager.get_command(command_name)
    return torch.cat([velocity_target_b(env, command_name), command[:, 2:3]], dim=-1)
