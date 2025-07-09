from __future__ import annotations

from typing import Literal

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class HumanoidCfg(BaseRobotCfg):
    """Config for generic humanoid robot from IsaacGymEnvs."""

    name: str = "humanoid"
    num_joints: int = 21
    fix_base_link: bool = False
    mjcf_path: str = "roboverse_data/assets/isaacgymenvs/assets/mjcf/nv_humanoid.xml"
    urdf_path: str = "roboverse_data/assets/isaacgymenvs/assets/mjcf/nv_humanoid.xml"
    isaacgym_read_mjcf: bool = True
    enabled_gravity: bool = True
    enabled_self_collisions: bool = True

    # Humanoid joints
    actuators: dict[str, BaseActuatorCfg] = {
        # Torso
        "torso_rot": BaseActuatorCfg(),
        "torso_bend": BaseActuatorCfg(),
        "torso_side": BaseActuatorCfg(),
        # Right leg
        "right_hip_rot": BaseActuatorCfg(),
        "right_hip_bend": BaseActuatorCfg(),
        "right_hip_side": BaseActuatorCfg(),
        "right_knee": BaseActuatorCfg(),
        "right_ankle_bend": BaseActuatorCfg(),
        "right_ankle_side": BaseActuatorCfg(),
        # Left leg
        "left_hip_rot": BaseActuatorCfg(),
        "left_hip_bend": BaseActuatorCfg(),
        "left_hip_side": BaseActuatorCfg(),
        "left_knee": BaseActuatorCfg(),
        "left_ankle_bend": BaseActuatorCfg(),
        "left_ankle_side": BaseActuatorCfg(),
        # Right arm
        "right_shoulder_bend": BaseActuatorCfg(),
        "right_shoulder_side": BaseActuatorCfg(),
        "right_elbow": BaseActuatorCfg(),
        # Left arm
        "left_shoulder_bend": BaseActuatorCfg(),
        "left_shoulder_side": BaseActuatorCfg(),
        "left_elbow": BaseActuatorCfg(),
    }

    joint_limits: dict[str, tuple[float, float]] = {
        # Torso
        "torso_rot": (-1.57, 1.57),
        "torso_bend": (-1.0, 1.0),
        "torso_side": (-1.0, 1.0),
        # Right leg
        "right_hip_rot": (-0.5, 0.5),
        "right_hip_bend": (-1.2, 0.6),
        "right_hip_side": (-0.8, 0.4),
        "right_knee": (-2.0, 0.0),
        "right_ankle_bend": (-0.8, 0.8),
        "right_ankle_side": (-0.4, 0.4),
        # Left leg
        "left_hip_rot": (-0.5, 0.5),
        "left_hip_bend": (-1.2, 0.6),
        "left_hip_side": (-0.4, 0.8),
        "left_knee": (-2.0, 0.0),
        "left_ankle_bend": (-0.8, 0.8),
        "left_ankle_side": (-0.4, 0.4),
        # Right arm
        "right_shoulder_bend": (-1.5, 1.5),
        "right_shoulder_side": (-1.2, 0.0),
        "right_elbow": (-2.0, 0.0),
        # Left arm
        "left_shoulder_bend": (-1.5, 1.5),
        "left_shoulder_side": (0.0, 1.2),
        "left_elbow": (-2.0, 0.0),
    }

    default_joint_positions: dict[str, float] = {
        # All joints start at 0
        "torso_rot": 0.0,
        "torso_bend": 0.0,
        "torso_side": 0.0,
        "right_hip_rot": 0.0,
        "right_hip_bend": 0.0,
        "right_hip_side": 0.0,
        "right_knee": 0.0,
        "right_ankle_bend": 0.0,
        "right_ankle_side": 0.0,
        "left_hip_rot": 0.0,
        "left_hip_bend": 0.0,
        "left_hip_side": 0.0,
        "left_knee": 0.0,
        "left_ankle_bend": 0.0,
        "left_ankle_side": 0.0,
        "right_shoulder_bend": 0.0,
        "right_shoulder_side": 0.0,
        "right_elbow": 0.0,
        "left_shoulder_bend": 0.0,
        "left_shoulder_side": 0.0,
        "left_elbow": 0.0,
    }

    control_type: dict[str, Literal["position", "effort"]] = {joint: "effort" for joint in actuators.keys()}

    default_position: tuple[float, float, float] = (0.0, 0.0, 1.34)