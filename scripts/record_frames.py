# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Save a short sequence of Isaac Sim viewport frames (and a contact sheet) of a trained policy with a chosen command.

Handy to inspect balancing without a window, and to pick frames for the video. Needs the Isaac Sim viewport
(``--viz kit``).

.. code-block:: bash

    uv run --extra isaacsim python \
        source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/microduck_ball/scripts/record_frames.py \
        logs/rsl_rl/microduck_ball_balance/<timestamp>/model_<last>.pt --command 0,0,0 --viz kit
"""

import argparse
import contextlib
import os
import sys

parser = argparse.ArgumentParser(description="Save viewport frames of a trained policy.")
parser.add_argument("checkpoint", type=str, help="Path to the RSL-RL checkpoint (.pt).")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Ball-Balance-Microduck", help="Name of the task.")
parser.add_argument("--command", type=str, default="0,0,0", help="Command 'vx,vy,wz' [m/s, m/s, rad/s].")
parser.add_argument("--frames", type=int, default=8, help="Number of frames.")
parser.add_argument("--every", type=int, default=3, help="Environment steps between two frames (1 step = 0.02 s).")
parser.add_argument("--warmup", type=int, default=60, help="Environment steps before the first frame.")
parser.add_argument("--distance", type=float, default=1.1, help="Distance of the camera from the ball [m].")
parser.add_argument("--height", type=float, default=0.34, help="Height of the camera [m].")
parser.add_argument("--out", type=str, default="frames", help="Folder of the frames.")
args, remaining = parser.parse_known_args()
if args.frames <= 0 or args.every <= 0 or args.warmup < 0 or args.distance <= 0:
    parser.error("frames, every, and distance must be positive; warmup must be nonnegative.")
if args.command != "0,0,0":
    parser.error("Fixed and Balance support only --command 0,0,0.")

import torch  # noqa: E402

from isaaclab.app import launch_simulation  # noqa: E402

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
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402


def main():
    argv = ["--task", args.task, "--num_envs", "1", "--checkpoint", args.checkpoint, *remaining]
    args_cli = play_rsl_rl._parse_args(argv)
    installed = check_rsl_rl_version()
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=True)
    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        agent_cfg = handle_deprecated_rsl_rl_cfg(cli_args.update_rsl_rl_cfg(agent_cfg, args_cli), installed)
        apply_env_overrides(args_cli, env_cfg)
        cmd = env_cfg.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_standing_envs = 0.0
        cmd.resampling_time_range = (1e6, 1e6)
        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        cleanup.callback(lambda: close_env(env))
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(env, agent_cfg)
        runner.load(args.checkpoint)
        torch.set_grad_enabled(False)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        unwrapped = env.unwrapped
        term = unwrapped.command_manager.get_term("base_velocity")
        command = torch.tensor([float(v) for v in args.command.split(",")], device=unwrapped.device)

        import omni.kit.app  # noqa: PLC0415
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport  # noqa: PLC0415

        os.makedirs(args.out, exist_ok=True)
        env.reset()
        obs = env.get_observations()
        origin = unwrapped.scene.env_origins[0].tolist()
        unwrapped.sim.set_camera_view(
            eye=(origin[0] + 0.1, origin[1] - args.distance, args.height), target=(origin[0], origin[1], 0.2)
        )
        files = []
        for step in range(args.warmup + args.frames * args.every):
            term.vel_command_b[:] = command
            obs, _, dones, _ = env.step(policy(obs))
            policy.reset(dones)
            since = step - args.warmup
            if since >= 0 and since % args.every == 0:
                for _ in range(4):
                    omni.kit.app.get_app().update()
                files.append(os.path.join(args.out, f"frame_{since // args.every:02d}.png"))
                capture_viewport_to_file(get_active_viewport(), files[-1])
                for _ in range(15):
                    omni.kit.app.get_app().update()
        # one contact sheet of all the frames
        from PIL import Image  # noqa: PLC0415

        images = [Image.open(f).convert("RGB") for f in files]
        width, height = images[0].size
        small = [image.crop((width // 4, 0, 3 * width // 4, height)).resize((320, 360)) for image in images]
        columns = min(4, len(small))
        rows = (len(small) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * 320, rows * 360))
        for i, image in enumerate(small):
            sheet.paste(image, ((i % columns) * 320, (i // columns) * 360))
        sheet.save(os.path.join(args.out, "sheet.png"))
        print(f"[record_frames] {len(files)} frames and sheet.png in {os.path.abspath(args.out)}", flush=True)
        return 0  # ExitStack closes the environment before the simulation app exits.


if __name__ == "__main__":
    sys.exit(main())
