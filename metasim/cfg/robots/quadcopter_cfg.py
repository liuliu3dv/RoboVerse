from __future__ import annotations

from typing import Literal

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class QuadcopterCfg(BaseRobotCfg):
    """Config for quadcopter robot."""

    name: str = "quadcopter"
    num_joints: int = 4
    fix_base_link: bool = False
    urdf_path: str = "roboverse_data/assets/isaacgymenvs/assets/urdf/quadcopter.urdf"
    mjcf_path: str = "roboverse_data/assets/isaacgymenvs/assets/urdf/quadcopter.urdf"
    enabled_gravity: bool = True
    enabled_self_collisions: bool = False

    # Rotor actuators
    actuators: dict[str, BaseActuatorCfg] = {
        "rotor_front_left": BaseActuatorCfg(is_ee=False),
        "rotor_front_right": BaseActuatorCfg(is_ee=False),
        "rotor_back_left": BaseActuatorCfg(is_ee=False),
        "rotor_back_right": BaseActuatorCfg(is_ee=False),
    }

    # Rotors have continuous rotation
    joint_limits: dict[str, tuple[float, float]] = {
        "rotor_front_left": (-1000.0, 1000.0),  # Effectively unlimited
        "rotor_front_right": (-1000.0, 1000.0),
        "rotor_back_left": (-1000.0, 1000.0),
        "rotor_back_right": (-1000.0, 1000.0),
    }

    default_joint_positions: dict[str, float] = {
        "rotor_front_left": 0.0,
        "rotor_front_right": 0.0,
        "rotor_back_left": 0.0,
        "rotor_back_right": 0.0,
    }

    control_type: dict[str, Literal["position", "effort"]] = {
        "rotor_front_left": "effort",
        "rotor_front_right": "effort",
        "rotor_back_left": "effort",
        "rotor_back_right": "effort",
    }

    default_position: tuple[float, float, float] = (0.0, 0.0, 1.0)

    # Quadcopter specific parameters
    mass: float = 2.5  # kg
    arm_length: float = 0.25  # m
    max_thrust_per_rotor: float = 10.0  # N