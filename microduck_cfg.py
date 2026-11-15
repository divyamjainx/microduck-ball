# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of the Microduck robot (Pollen Robotics / Hugging Face).

Microduck is a 25 cm, 737 g biped with 14 Dynamixel XL330 servos: five per leg (hip yaw, hip roll, hip pitch, knee,
ankle) and four in the neck and head. The USD is not shipped with Isaac Lab: build it once with
``scripts/build_asset.py`` (it downloads the MJCF model from ``pollen-robotics/microduck_rl`` and converts it).
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asset")
"""Included robot and ball asset folder; optionally rebuilt by ``scripts/build_asset.py``."""
MICRODUCK_USD_PATH = os.path.join(ASSET_DIR, "robot_walk", "robot_walk.usda")

# Crouched pose: knees bent, hip and ankle solved so that the soles are flat and the centre of mass is over them. It
# lowers the centre of mass to 0.118 m above the soles (0.138 m with straight legs), below the 0.12 m radius of the
# ball, which makes standing on it far less of a stretch than the straight-legged STAND pose of the upstream model.
HOME_JOINT_POS = {
    "left_hip_yaw": 0.0,
    "left_hip_roll": -0.0873,
    "left_hip_pitch": 0.2043,
    "left_knee": 0.9,
    "left_ankle": 0.6957,
    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.0873,
    "right_hip_pitch": -0.2043,
    "right_knee": -0.9,
    "right_ankle": -0.6957,
    "neck_pitch": 0.3491,
    "head_pitch": 0.3491,
    "head_yaw": 0.0,
    "head_roll": 0.0,
}
"""Default joint positions [rad]. The right leg is the mirror image of the left one."""

STANDING_HEIGHT = 0.0905
"""Height of the base (``trunk_base``) above the lowest point of the soles in :obj:`HOME_JOINT_POS` [m]."""

MICRODUCK_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=MICRODUCK_USD_PATH,
        activate_contact_sensors=True,
        # Contact offsets are authored directly in the robot asset's collider definitions.
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.12),
        joint_pos=HOME_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        # The converted USD carries no usable joint drives, so the position servos are modelled explicitly: a PD law on
        # the target followed by the XL330's torque limit (0.96 N m). Armature, friction and the stiffness are those of
        # the "chosen_actuator" servo class of the MJCF; the stiffness is higher than its identified 0.55 N m/rad, which
        # cannot hold the robot up without the real servo's voltage control (BAM); hardware actuator identification is outside this simulation setup.
        "servos": IdealPDActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=0.96,
            velocity_limit=12.0,
            stiffness=5.0,
            damping=0.1,
            armature=0.0018,
            friction=0.0048,
        ),
    },
)
"""Microduck standing on the ground, base at 0.12 m. Tasks move it with their own reset events."""
