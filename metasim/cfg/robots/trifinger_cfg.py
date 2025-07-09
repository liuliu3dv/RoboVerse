from __future__ import annotations

from typing import Literal

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class TrifingerCfg(BaseRobotCfg):
    """Config for Trifinger robot."""

    name: str = "trifinger"
    num_joints: int = 9
    fix_base_link: bool = True
    urdf_path: str = "roboverse_data/assets/isaacgymenvs/assets/trifinger/robot_properties_fingers/urdf/trifinger.urdf"
    mjcf_path: str = "roboverse_data/assets/isaacgymenvs/assets/trifinger/robot_properties_fingers/urdf/trifinger.urdf"
    enabled_gravity: bool = True
    enabled_self_collisions: bool = True

    # Three fingers with 3 joints each
    actuators: dict[str, BaseActuatorCfg] = {
        # Finger 0
        "finger_0_joint_0": BaseActuatorCfg(),
        "finger_0_joint_1": BaseActuatorCfg(),
        "finger_0_joint_2": BaseActuatorCfg(),
        # Finger 1
        "finger_1_joint_0": BaseActuatorCfg(),
        "finger_1_joint_1": BaseActuatorCfg(),
        "finger_1_joint_2": BaseActuatorCfg(),
        # Finger 2
        "finger_2_joint_0": BaseActuatorCfg(),
        "finger_2_joint_1": BaseActuatorCfg(),
        "finger_2_joint_2": BaseActuatorCfg(),
    }

    joint_limits: dict[str, tuple[float, float]] = {
        # Finger 0
        "finger_0_joint_0": (-0.6, 1.2),
        "finger_0_joint_1": (0.0, 1.67),
        "finger_0_joint_2": (-2.7, 2.7),
        # Finger 1
        "finger_1_joint_0": (-0.6, 1.2),
        "finger_1_joint_1": (0.0, 1.67),
        "finger_1_joint_2": (-2.7, 2.7),
        # Finger 2
        "finger_2_joint_0": (-0.6, 1.2),
        "finger_2_joint_1": (0.0, 1.67),
        "finger_2_joint_2": (-2.7, 2.7),
    }

    default_joint_positions: dict[str, float] = {
        # Finger 0
        "finger_0_joint_0": 0.0,
        "finger_0_joint_1": 0.9,
        "finger_0_joint_2": -1.57,
        # Finger 1
        "finger_1_joint_0": 0.0,
        "finger_1_joint_1": 0.9,
        "finger_1_joint_2": -1.57,
        # Finger 2
        "finger_2_joint_0": 0.0,
        "finger_2_joint_1": 0.9,
        "finger_2_joint_2": -1.57,
    }

    control_type: dict[str, Literal["position", "effort"]] = {
        # All joints use position control
        "finger_0_joint_0": "position",
        "finger_0_joint_1": "position",
        "finger_0_joint_2": "position",
        "finger_1_joint_0": "position",
        "finger_1_joint_1": "position",
        "finger_1_joint_2": "position",
        "finger_2_joint_0": "position",
        "finger_2_joint_1": "position",
        "finger_2_joint_2": "position",
    }

    default_position: tuple[float, float, float] = (0.0, 0.0, 0.0)