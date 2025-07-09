from __future__ import annotations

from typing import Literal

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class BallBalanceCfg(BaseRobotCfg):
    """Config for ball balance platform robot."""

    name: str = "ballbalance"
    num_joints: int = 2
    fix_base_link: bool = True
    mjcf_path: str = "roboverse_data/assets/isaacgymenvs/assets/mjcf/balance_bot.xml"
    urdf_path: str = "roboverse_data/assets/isaacgymenvs/assets/mjcf/balance_bot.xml"
    isaacgym_read_mjcf: bool = True
    enabled_gravity: bool = True
    enabled_self_collisions: bool = False

    # Platform tilt actuators
    actuators: dict[str, BaseActuatorCfg] = {
        "platform_tilt_x": BaseActuatorCfg(),
        "platform_tilt_y": BaseActuatorCfg(),
    }

    joint_limits: dict[str, tuple[float, float]] = {
        "platform_tilt_x": (-0.4, 0.4),  # Radians
        "platform_tilt_y": (-0.4, 0.4),  # Radians
    }

    default_joint_positions: dict[str, float] = {
        "platform_tilt_x": 0.0,
        "platform_tilt_y": 0.0,
    }

    control_type: dict[str, Literal["position", "effort"]] = {
        "platform_tilt_x": "position",
        "platform_tilt_y": "position",
    }

    default_position: tuple[float, float, float] = (0.0, 0.0, 0.5)