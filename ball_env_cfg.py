# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-ball standing and free-ball balancing with shared observations and actions."""

import os

from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.utils import PresetCfg

from ... import mdp as velocity_mdp
from . import events, observations, rewards, terminations
from .microduck_cfg import ASSET_DIR, MICRODUCK_CFG, STANDING_HEIGHT

##
# Sizes
##

BALL_RADIUS = 0.12
"""Radius of the ball [m]: a basketball. The key tuning knob of the task.

This radius is part of the trained task configuration. Changing it changes the contact geometry and should be
evaluated separately; it is not a general guarantee of passive stability.
"""
BALL_USD_PATH = os.path.join(ASSET_DIR, "ball", "ball.usda")
"""Unit ball built by ``scripts/build_asset.py``: a visual ball mesh and a sphere collider."""
BALL_MASS = 0.3
"""Mass of the ball [kg] (a real basketball is 0.6 kg)."""
BASE_HEIGHT_ON_BALL = STANDING_HEIGHT + 0.003
"""Height of the base above the top of the ball at the start of an episode: the soles are 3 mm above the ball."""
FOOT_BODIES = ["ankle_left", "ankle_right"]
FOOT_CONTACT_HEIGHT = 0.040
"""A foot whose ankle is closer than this to the surface of the ball counts as standing on it [m]."""


@configclass
class BallSpawnCfg(sim_utils.UsdFileCfg):
    """Spawns the unit ball of :obj:`BALL_USD_PATH`, scaled to ``radius``. Use :func:`set_ball_radius` to change it."""

    radius: float = BALL_RADIUS
    """Radius of the ball [m]."""


def set_ball_radius(scene_cfg: "MicroduckBallSceneCfg", radius: float):
    """Set the ball radius in metres and update its spawn scale and initial centre height."""
    scene_cfg.ball.spawn.radius = radius
    scene_cfg.ball.spawn.scale = (radius, radius, radius)
    scene_cfg.ball.init_state.pos = (0.0, 0.0, radius)


##
# Physics
##


@configclass
class BallPhysicsCfg(PresetCfg):
    """PhysX backend used by both trained tasks."""

    isaacsim_physx = PhysxCfg()
    default = isaacsim_physx


##
# Scene
##


@configclass
class MicroduckBallSceneCfg(InteractiveSceneCfg):
    """Flat ground, the duck and a ball."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = MICRODUCK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=BallSpawnCfg(
            usd_path=BALL_USD_PATH,
            radius=BALL_RADIUS,
            scale=(BALL_RADIUS, BALL_RADIUS, BALL_RADIUS),
            mass_props=sim_utils.MassCfg(mass=BALL_MASS),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, BALL_RADIUS)),
    )
    # a dark, glossy floor that shows the white robot and its reflection; it is only a picture (no collider), 0.5 mm
    # above the ground plane of the terrain, which does the physics
    floor = AssetBaseCfg(
        prim_path="/World/floor",
        spawn=sim_utils.MeshCuboidCfg(
            size=(60.0, 60.0, 0.002),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.025, 0.027, 0.032), roughness=0.08, metallic=0.5
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.0005)),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 0.9)),
    )


##
# MDP
##


@configclass
class CommandsCfg:
    """Zero velocity targets for standing and balancing."""

    base_velocity = velocity_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 8.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=velocity_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Position targets of the 14 servos, as offsets from the standing pose."""

    joint_pos = velocity_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        # the head and neck stay where they are (a free head is a counterweight, and droops); the hips barely yaw
        scale={
            ".*_hip_yaw": 0.1,
            ".*_hip_roll": 0.25,
            ".*_(hip_pitch|knee|ankle)": 0.5,
            "neck_pitch": 0.0,
            "head_.*": 0.0,
        },
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observations of the policy (what a real robot would have, plus the ball) and of the critic."""

    @configclass
    class PolicyCfg(ObsGroup):
        # body orientation and angular velocity: what the IMU measures
        base_ang_vel = ObsTerm(func=velocity_mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(func=velocity_mdp.projected_gravity, noise=Unoise(n_min=-0.03, n_max=0.03))
        velocity_commands = ObsTerm(func=observations.velocity_command_b, params={"command_name": "base_velocity"})
        # joint states: what the servos report
        joint_pos = ObsTerm(func=velocity_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=velocity_mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=velocity_mdp.last_action)
        # the ball, relative to the duck. Read from the simulator: a real robot would have to estimate it.
        ball_pos = ObsTerm(
            func=observations.ball_pos_b,
            params={"robot_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES), "ball_cfg": SceneEntityCfg("ball")},
            noise=Unoise(n_min=-0.005, n_max=0.005),
        )
        ball_vel = ObsTerm(
            func=observations.ball_vel_b,
            params={"robot_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES), "ball_cfg": SceneEntityCfg("ball")},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Extra terms for the critic only."""

        base_lin_vel = ObsTerm(func=velocity_mdp.base_lin_vel)
        ball_ang_vel = ObsTerm(func=observations.ball_ang_vel_w, params={"ball_cfg": SceneEntityCfg("ball")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventsCfg:
    """Reset events and fixed startup contact materials."""

    # startup
    feet_material = EventTerm(
        func=velocity_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    ball_material = EventTerm(
        func=velocity_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    # reset
    reset_duck_on_ball = EventTerm(
        func=events.reset_duck_on_ball,
        mode="reset",
        params={"standing_height": BASE_HEIGHT_ON_BALL, "xy_offset": 0.01, "tilt": 0.05},
    )
    reset_reference_heading = EventTerm(func=observations.reset_reference_heading, mode="reset")
    reset_robot_joints = EventTerm(
        func=velocity_mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.05, 0.05), "velocity_range": (0.0, 0.0)},
    )


@configclass
class RewardsCfg:
    """Stay upright and keep the ball under the feet; track the command; and a few regularisers."""

    # -- task
    upright = RewTerm(func=rewards.upright, weight=1.0, params={"std": 0.25})
    ball_under_feet = RewTerm(
        func=rewards.ball_under_feet,
        weight=1.0,
        params={
            "std": 0.05,
            "robot_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES),
            "ball_cfg": SceneEntityCfg("ball"),
        },
    )
    track_lin_vel_xy_exp = RewTerm(
        func=rewards.track_lin_vel_xy_world_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.2}
    )
    track_ang_vel_z_exp = RewTerm(
        func=velocity_mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": 0.5}
    )
    # -- penalties
    ball_slip = RewTerm(
        func=rewards.ball_slip_l2,
        weight=-0.5,
        params={"robot_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES), "ball_cfg": SceneEntityCfg("ball")},
    )
    ang_vel_xy_l2 = RewTerm(func=velocity_mdp.ang_vel_xy_l2, weight=-0.05)
    action_rate_l2 = RewTerm(func=velocity_mdp.action_rate_l2, weight=-0.02)
    dof_pos_limits = RewTerm(func=velocity_mdp.joint_pos_limits, weight=-1.0)
    energy = RewTerm(
        func=rewards.energy, weight=-0.02, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])}
    )
    terminated = RewTerm(func=velocity_mdp.is_terminated, weight=-5.0)


@configclass
class TerminationsCfg:
    """The episode ends when the duck falls, loses the ball or the time is up."""

    time_out = DoneTerm(func=velocity_mdp.time_out, time_out=True)
    duck_tilted = DoneTerm(func=velocity_mdp.bad_orientation, params={"limit_angle": 0.8})
    duck_too_low = DoneTerm(func=terminations.base_too_low, params={"minimum_height": 0.05})
    ball_off_feet = DoneTerm(
        func=terminations.ball_off_feet,
        params={"max_distance": 0.07, "robot_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES)},
    )
    ball_outside_zone: DoneTerm | None = DoneTerm(func=terminations.ball_outside_zone, params={"radius": 0.6})


##
# Environment
##


@configclass
class MicroduckBallEnvCfg(ManagerBasedRLEnvCfg):
    """Common configuration of the stages (the ball is free, the command is zero)."""

    sim: SimulationCfg = SimulationCfg(physics=BallPhysicsCfg())
    scene: MicroduckBallSceneCfg = MicroduckBallSceneCfg(num_envs=2048, env_spacing=1.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material

    def play_mode(self):
        """Set playback defaults to 16 environments, 20-second episodes, and no policy observation noise."""
        super().play_mode()
        self.scene.num_envs = 16
        self.episode_length_s = 20.0
        self.observations.policy.enable_corruption = False


@configclass
class MicroduckBallFixedEnvCfg(MicroduckBallEnvCfg):
    """Stage 1: the ball is fixed in place (kinematic). The duck stands on a ball that cannot move."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.ball.spawn.rigid_props = sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True)
        # nothing to lose with a fixed ball: no ball-related reward or termination
        self.rewards.ball_under_feet.weight = 0.0
        self.rewards.ball_slip.weight = 0.0
        self.terminations.ball_off_feet = None
        self.terminations.ball_outside_zone = None


@configclass
class MicroduckBallBalanceEnvCfg(MicroduckBallEnvCfg):
    """Stage 2: the ball rolls freely. The duck must stay upright and keep the ball near its feet."""
