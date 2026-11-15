# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate an RSL-RL checkpoint of the zero-command standing and balancing tasks.

Every command is held for a few seconds on all environments. The script reports, per command, the velocity the duck
achieves, the share of episodes that ended early (fell, or lost the ball), how far the ball stays from the middle of
the feet and how far the duck tilts. It is a numerical complement to watching the policy with ``isaaclab play``.

.. code-block:: bash

    uv run --extra isaacsim python \
        source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/microduck_ball/scripts/evaluate.py \
        logs/rsl_rl/microduck_ball_balance/<timestamp>/model_799.pt --task Isaac-Velocity-Ball-Balance-Microduck \
        --commands 0,0,0

Extra ``key=value`` arguments are forwarded to the launcher, e.g. ``physics=isaacsim_physx`` to select the backend used for training.
"""

import argparse
import contextlib
import sys

parser = argparse.ArgumentParser(description="Evaluate a checkpoint of the Microduck-on-a-ball task.")
parser.add_argument("checkpoint", type=str, help="Path to the RSL-RL checkpoint (.pt).")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Ball-Balance-Microduck", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments.")
parser.add_argument("--seconds", type=float, default=10.0, help="Seconds each command is held.")
parser.add_argument(
    "--commands",
    nargs="+",
    default=["0,0,0"],
    help="Commands as 'vx,vy,wz' triples [m/s, m/s, rad/s].",
)
parser.add_argument("--ball_radius", type=float, default=None, help="Ball radius [m] instead of the task's.")
parser.add_argument("--ball_mass", type=float, default=None, help="Ball mass [kg] instead of the task's.")
parser.add_argument("--noise", action="store_true", help="Keep the observation noise of training (off by default).")
args, remaining = parser.parse_known_args()
if args.num_envs <= 0 or args.seconds <= 0:
    parser.error("num_envs and seconds must be positive.")
if args.ball_radius is not None and args.ball_radius <= 0:
    parser.error("ball_radius must be positive.")
if args.ball_mass is not None and args.ball_mass <= 0:
    parser.error("ball_mass must be positive.")
# ``--commands`` takes any number of values: leave trailing ``key=value`` overrides (e.g. physics=...) to the launcher
remaining += [c for c in args.commands if "=" in c]
args.commands = [c for c in args.commands if "=" not in c]
if not args.commands or any(c != "0,0,0" for c in args.commands):
    parser.error("Fixed and Balance support only --commands 0,0,0.")

import torch  # noqa: E402

from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args  # noqa: E402
from isaaclab_rl.entrypoints.backends import play_rsl_rl  # noqa: E402
from isaaclab_rl.entrypoints.common import apply_env_overrides, close_env, create_isaaclab_env  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.core.velocity.config.microduck_ball.ball_env_cfg import (  # noqa: E402
    FOOT_CONTACT_HEIGHT,
    set_ball_radius,
)
from isaaclab_tasks.core.velocity.config.microduck_ball.observations import (  # noqa: E402
    _heading,
    reference_heading,
)
from isaaclab_tasks.core.velocity.config.microduck_ball.rewards import (  # noqa: E402
    foot_height_on_ball,
    foot_slip_on_ball,
)
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402


def main():
    argv = ["--task", args.task, "--num_envs", str(args.num_envs), "--checkpoint", args.checkpoint, *remaining]
    args_cli = play_rsl_rl._parse_args(argv)
    installed = check_rsl_rl_version()
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=True)
    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        agent_cfg = handle_deprecated_rsl_rl_cfg(cli_args.update_rsl_rl_cfg(agent_cfg, args_cli), installed)
        apply_env_overrides(args_cli, env_cfg)
        env_cfg.observations.policy.enable_corruption = args.noise
        if args.ball_radius is not None:
            set_ball_radius(env_cfg.scene, args.ball_radius)
        if args.ball_mass is not None:
            env_cfg.scene.ball.spawn.mass_props.mass = args.ball_mass
        # hold the commands: no standing environments or resampling
        cmd = env_cfg.commands.base_velocity
        cmd.rel_standing_envs = 0.0
        cmd.resampling_time_range = (1e6, 1e6)
        cmd.debug_vis = False
        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        cleanup.callback(lambda: close_env(env))
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(env, agent_cfg)
        runner.load(args.checkpoint)
        torch.set_grad_enabled(False)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        unwrapped = env.unwrapped
        term = unwrapped.command_manager.get_term("base_velocity")
        robot, ball = unwrapped.scene["robot"], unwrapped.scene["ball"]
        feet = [robot.body_names.index(name) for name in ("ankle_left", "ankle_right")]
        foot_cfg = SceneEntityCfg("robot", body_names=["ankle_left", "ankle_right"])
        foot_cfg.resolve(unwrapped.scene)
        ball_cfg = SceneEntityCfg("ball")
        contact_height = FOOT_CONTACT_HEIGHT
        steps = int(args.seconds / unwrapped.step_dt)
        if steps < 2:
            raise ValueError("Evaluation duration must cover at least two environment steps.")
        settle = steps * 3 // 10
        print(
            f"\nEVAL {args.task}, {args.num_envs} envs, {args.seconds:.0f} s per command"
            "\nEVAL command (vx, vy, wz) -> achieved (vx, vy in the reference-heading frame, wz)"
            " | heading drift [deg] | early ends | ball-feet offset [cm] | tilt [deg]"
            " | ball speed rel. to duck [cm/s] | foot lifts per s | inter-foot radial height difference >1 cm [cm]"
            " | foot slip on the ball [cm/s]",
            flush=True,
        )
        for text in args.commands:
            command = torch.tensor([float(v) for v in text.split(",")], device=unwrapped.device)
            env.reset()
            obs = env.get_observations()
            early_ends = torch.zeros(unwrapped.num_envs, dtype=torch.bool, device=unwrapped.device)
            achieved, offset, tilt, slip = [], [], [], []
            lifts, foot_slip, was_on, lift_height = 0.0, [], None, []
            drift = []
            for step in range(steps):
                term.vel_command_b[:] = command
                obs, _, dones, extras = env.step(policy(obs))
                policy.reset(dones)
                early_ends |= dones.bool() & ~extras["time_outs"].bool()
                if step >= settle:
                    # the velocity that the reward tracks: world velocity in the frame of the reference heading
                    ref = reference_heading(unwrapped)
                    cos, sin = torch.cos(ref), torch.sin(ref)
                    world = robot.data.root_lin_vel_w.torch
                    lin = torch.stack(
                        [cos * world[:, 0] + sin * world[:, 1], -sin * world[:, 0] + cos * world[:, 1]], 1
                    )
                    ang = robot.data.root_ang_vel_b.torch
                    achieved.append(torch.cat([lin, ang[:, 2:3]], dim=1).mean(0))
                    error = torch.remainder(_heading(unwrapped) - ref + torch.pi, 2 * torch.pi) - torch.pi
                    drift.append(error.abs().mean())
                    delta = ball.data.root_pos_w.torch[:, :2] - robot.data.body_pos_w.torch[:, feet, :2].mean(1)
                    offset.append(delta.norm(dim=1).mean())
                    tilt.append(
                        torch.acos((-robot.data.projected_gravity_b.torch[:, 2]).clamp(-1.0, 1.0)).rad2deg().mean()
                    )
                    relative = ball.data.root_lin_vel_w.torch[:, :2] - robot.data.root_lin_vel_w.torch[:, :2]
                    slip.append(relative.norm(dim=1).mean())
                    on_ball = foot_height_on_ball(unwrapped, foot_cfg, ball_cfg) < contact_height
                    if was_on is not None:
                        lifts += (was_on & ~on_ball).float().sum(dim=1).mean().item()
                    was_on = on_ball
                    difference = (foot_height_on_ball(unwrapped, foot_cfg, ball_cfg)).diff(dim=1).abs().squeeze(1)
                    lift_height.append(difference[difference > 0.01])
                    speed = foot_slip_on_ball(unwrapped, contact_height, foot_cfg, ball_cfg)
                    foot_slip.append(speed.sum() / on_ball.sum().clamp(min=1))
            heights = torch.cat(lift_height)
            lifted = heights.mean().item() if heights.numel() else 0.0
            v = torch.stack(achieved).mean(0).tolist()
            c = command.tolist()
            print(
                f"EVAL ({c[0]:+.2f}, {c[1]:+.2f}, {c[2]:+.2f}) -> ({v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}) "
                f"| {torch.stack(drift).mean().rad2deg().item():5.1f} "
                f"| {100 * early_ends.float().mean().item():5.1f}% "
                f"| {100 * torch.stack(offset).mean().item():5.2f} | {torch.stack(tilt).mean().item():5.1f} "
                f"| {100 * torch.stack(slip).mean().item():5.1f} "
                f"| {lifts / ((steps - settle - 1) * unwrapped.step_dt):4.2f} | {100 * lifted:4.1f} "
                f"| {100 * torch.stack(foot_slip).mean().item():5.1f}",
                flush=True,
            )


if __name__ == "__main__":
    sys.exit(main())
