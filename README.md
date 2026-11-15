# Microduck Ball Balance

Two-stage reinforcement learning for standing and balancing on a ball with Microduck. Policies are trained with PPO in Isaac Lab and viewed through Isaac Sim, with browser VS Code and a streamed simulator on NVIDIA Brev.

This is a simulation prototype. Balance playback has been verified in the browser viewer; real-robot deployment is outside the current scope.

## Tasks

| Task ID | Objective | Starting policy |
| --- | --- | --- |
| `Isaac-Velocity-Ball-Fixed-Microduck` | Stand upright on a kinematic ball | Train from scratch |
| `Isaac-Velocity-Ball-Balance-Microduck` | Balance on a freely moving ball with zero velocity commands | Initialize from Fixed |

Each environment contains one robot and one ball. During playback, every environment runs the same policy independently.

## Status

- Both stages have trained checkpoints and numerical evaluations.
- Balance has been viewed successfully through Isaac Sim streaming.
- Fixed playback in the current streaming environment remains to be verified.
- Training has not been repeated across multiple seeds or systematically tuned.
- The training environment and browser viewer use different Isaac Sim versions. Successful playback does not establish identical physics or equivalent performance across versions.

## Control and policy

| Component | Configuration |
| --- | --- |
| Physics backend used for training | PhysX |
| Physics / control frequency | 200 Hz / 50 Hz |
| Policy observations | 55 values |
| Critic observations | 61 values, including privileged simulator information |
| Actions | 14 joint-position action slots; head and neck action scales are zero |
| Actuation | Ideal PD control |
| Actor and critic hidden layers | 256, 128, 128; ELU activation |
| Observation normalization | Enabled |
| Training sequence | 400 Fixed iterations, then 600 additional Balance iterations |

Both tasks share the same observation and action structure. Use a checkpoint trained for the corresponding task with this project's observation order and action scaling. Ball observations come from simulator state.

The PPO configuration uses an adaptive learning rate starting at `0.001`, clipping `0.2`, entropy coefficient `0.005`, discount `0.99`, GAE lambda `0.95`, 24 rollout steps per environment, five learning epochs, and four minibatches. See `agents/rsl_rl_ppo_cfg.py` for the complete configuration.

## Installation layouts

The project is an Isaac Lab task folder, not a standalone simulator installation. Keep the entire folder and its asset paths together.

| Environment | Isaac Lab root | Task directory beneath the root |
| --- | --- | --- |
| Original training installation | `/data/microduck-work/IsaacLab` | `source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/microduck_ball` |
| Working browser viewer | `/workspace/isaaclab` | `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/microduck_ball` |

The training checkout used Isaac Lab release `3.0.0` at commit `44044b0` with Isaac Sim 6.2. The launchable provides Isaac Sim 6.0.1 and a different Isaac Lab source layout. The viewer copy includes API compatibility changes and an explicitly enabled ball rigid body.

The viewer also needs the saved startup fix in `isaaclab_physx/physics/physx_manager.py`: it processes 20 Kit app updates before creating PhysX tensor views. Without it, startup invalidated the robot's physics handles. The startup patch and viewer-specific compatibility changes are separate from this repository. Use the training environment below to reproduce the runs. This is a workaround verified for this setup, not a general Isaac Lab installation requirement.

## Build the runtime assets

This source release regenerates the runtime USD files locally. Place this task folder at the training layout above, inside an Isaac Lab checkout with the Isaac Sim extra available. From the Isaac Lab root, run:

```bash
uv run --extra isaacsim python \
  source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/microduck_ball/scripts/build_asset.py
```

The builder downloads the pinned Microduck model and meshes, converts the robot to USD, and creates the ball wrapper with the NVIDIA ArchVis walnut visual. Use `--local /absolute/path/to/upstream/model-folder` to supply the pinned model and meshes locally, or `--ball /absolute/path/to/ball.usd` to select your own ball visual. If the supplied USD references other files, keep those paths available during playback and training. The script writes `asset/robot_walk/robot_walk.usda` and `asset/ball/ball.usda`. The visual is separate from the sphere collider.

## Training

These commands target the **original training installation**, from its Isaac Lab root. They are not verified training commands for the older browser launchable.

Train Fixed from scratch:

```bash
uv run --extra isaacsim --extra rsl-rl isaaclab train \
  --rl_library rsl_rl \
  --task Isaac-Velocity-Ball-Fixed-Microduck \
  --num_envs 2048 \
  --max_iterations 400 \
  --viz none \
  physics=isaacsim_physx
```

Initialize Balance from the newly trained Fixed checkpoint:

```bash
uv run --extra isaacsim --extra rsl-rl isaaclab train \
  --rl_library rsl_rl \
  --task Isaac-Velocity-Ball-Balance-Microduck \
  --checkpoint /absolute/path/to/fixed_checkpoint.pt \
  --num_envs 2048 \
  --max_iterations 600 \
  --viz none \
  physics=isaacsim_physx
```

Replace the checkpoint path with your actual file. Logs are stored under `logs/rsl_rl/microduck_ball_fixed/` and `logs/rsl_rl/microduck_ball_balance/`.

## Evaluation

On the original training installation:

```bash
uv run --extra isaacsim --extra rsl-rl python \
  source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/microduck_ball/scripts/evaluate.py \
  /absolute/path/to/balance_checkpoint.pt \
  --task Isaac-Velocity-Ball-Balance-Microduck \
  --commands 0,0,0 \
  --seconds 10 \
  --num_envs 128 \
  --noise \
  physics=isaacsim_physx
```

Evaluate Fixed by changing the task and checkpoint. Omit `--noise` for clean observations. The evaluation and frame-capture scripts depend on the original training API and imports; they have not been ported to the browser launchable.

Recorded 10-second evaluations with 128 environments and observation noise:

| Metric | Fixed | Balance |
| --- | ---: | ---: |
| Environments with at least one early termination | 0.8% | 0% |
| Reported tilt | 1.7 degrees | 0.9 degrees |
| Ball-to-feet offset | 5.29 cm | 0.52 cm |
| Heading drift | 8.4 degrees | 33.7 degrees |

These are short evaluations of individual trained checkpoints, not multi-seed robustness estimates. Early termination is measured per environment over the evaluation window, not as a per-episode failure rate. Foot contact and slip measurements use geometric proxies. Balance is stable in these tests but still exhibits heading drift and foot slip.

## Browser playback

The Isaac Launchable on NVIDIA Brev provides browser VS Code and an Isaac Sim viewer. This example assumes the viewer-compatible setup described below, including `xvfb-run`.

Trained checkpoints are stored separately from this repository. Train Fixed from scratch, then use its checkpoint to initialize Balance. For playback or evaluation, pass a checkpoint for the corresponding task.

In the browser VS Code terminal, set `BALANCE_CHECKPOINT` to the checkpoint path **inside** the container and run:

```bash
cd /workspace/isaaclab
BALANCE_CHECKPOINT=/absolute/path/to/balance_checkpoint.pt

xvfb-run -a ./isaaclab.sh play \
  --rl_library rsl_rl \
  --task Isaac-Velocity-Ball-Balance-Microduck \
  --checkpoint "$BALANCE_CHECKPOINT" \
  --num_envs 1 \
  --livestream 2 \
  --viz kit \
  physics=isaacsim_physx
```

Once playback starts, open the VS Code service's base URL with `/viewer/` appended. Keep one viewer tab open. First-time rendering can take longer while shaders compile.

To show multiple ducks, replace `--num_envs 1` with `--num_envs 16`. This uses the same checkpoint and needs no retraining; the multi-environment streamed view has not yet been verified. Zoom out to see the grid.

For Fixed, replace the task with `Isaac-Velocity-Ball-Fixed-Microduck` and pass your Fixed checkpoint to `--checkpoint`. Stop an existing playback with **Ctrl+C** before launching another.

## Project files

```text
microduck_ball/
├── __init__.py                # Task registration
├── ball_env_cfg.py            # Scenes, observations, rewards, and stage settings
├── microduck_cfg.py           # Robot and actuator configuration
├── observations.py           # Policy and privileged observations
├── rewards.py                # Reward functions
├── events.py                 # Reset events
├── terminations.py           # Failure conditions
├── agents/
│   └── rsl_rl_ppo_cfg.py      # PPO networks and training budgets
├── scripts/
│   ├── build_asset.py        # Download and build runtime assets
│   ├── evaluate.py           # Numerical policy evaluation
│   └── record_frames.py      # Rendered frame capture
└── asset/                    # Local robot and ball assets
```

Build the assets before running training or playback. Robot collider contact offsets are authored in the asset definitions and should be preserved with the checkpoints.

## Sources and licensing

The robot simulation model comes from [Pollen Robotics' Microduck RL repository](https://github.com/pollen-robotics/microduck_rl). The asset builder pins its upstream commit. The upstream model repository uses Apache-2.0; preserve applicable model and asset notices.

Python files retain their Isaac Lab BSD-3-Clause copyright headers. The basketball and NVIDIA ArchVis visuals have separate terms. The asset builder retrieves the ArchVis visual during local setup when a local ball asset is unavailable. Include the applicable license texts when redistributing code or assets.
