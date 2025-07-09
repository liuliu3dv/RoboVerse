from __future__ import annotations

from typing import Literal

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class IngenuityCfg(BaseRobotCfg):
    """Config for Ingenuity Mars helicopter robot."""

    name: str = "ingenuity"
    num_joints: int = 4
    fix_base_link: bool = False
    urdf_path: str = "roboverse_data/assets/isaacgymenvs/assets/urdf/ingenuity.urdf"
    mjcf_path: str = "roboverse_data/assets/isaacgymenvs/assets/urdf/ingenuity.urdf"
    enabled_gravity: bool = True
    enabled_self_collisions: bool = False

    # Rotor actuators
    actuators: dict[str, BaseActuatorCfg] = {
        "rotor_1": BaseActuatorCfg(is_ee=False),
        "rotor_2": BaseActuatorCfg(is_ee=False),
        "rotor_3": BaseActuatorCfg(is_ee=False),
        "rotor_4": BaseActuatorCfg(is_ee=False),
    }

    # Rotors have continuous rotation
    joint_limits: dict[str, tuple[float, float]] = {
        "rotor_1": (-1000.0, 1000.0),  # Effectively unlimited
        "rotor_2": (-1000.0, 1000.0),
        "rotor_3": (-1000.0, 1000.0),
        "rotor_4": (-1000.0, 1000.0),
    }

    default_joint_positions: dict[str, float] = {
        "rotor_1": 0.0,
        "rotor_2": 0.0,
        "rotor_3": 0.0,
        "rotor_4": 0.0,
    }

    control_type: dict[str, Literal["position", "effort"]] = {
        "rotor_1": "effort",
        "rotor_2": "effort",
        "rotor_3": "effort",
        "rotor_4": "effort",
    }

    default_position: tuple[float, float, float] = (0.0, 0.0, 1.0)

    # Ingenuity specific parameters
    mass: float = 1.8  # kg
    rotor_radius: float = 0.605  # m
    max_thrust_per_rotor: float = 5.0  # N