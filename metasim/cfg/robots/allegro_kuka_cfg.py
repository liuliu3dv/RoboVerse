from __future__ import annotations

from typing import Literal

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class AllegroKukaCfg(BaseRobotCfg):
    """Config for Allegro hand mounted on Kuka IIWA arm."""

    name: str = "allegro_kuka"
    num_joints: int = 23  # 7 (Kuka) + 16 (Allegro)
    fix_base_link: bool = True
    urdf_path: str = "roboverse_data/assets/isaacgymenvs/assets/urdf/kuka_allegro_description/kuka_allegro_touch_sensor.urdf"
    mjcf_path: str = "roboverse_data/assets/isaacgymenvs/assets/urdf/kuka_allegro_description/kuka_allegro_touch_sensor.urdf"
    isaacgym_read_mjcf: bool = False
    enabled_gravity: bool = True
    enabled_self_collisions: bool = True

    # Kuka joints
    actuators: dict[str, BaseActuatorCfg] = {
        "iiwa7_joint_1": BaseActuatorCfg(),
        "iiwa7_joint_2": BaseActuatorCfg(),
        "iiwa7_joint_3": BaseActuatorCfg(),
        "iiwa7_joint_4": BaseActuatorCfg(),
        "iiwa7_joint_5": BaseActuatorCfg(),
        "iiwa7_joint_6": BaseActuatorCfg(),
        "iiwa7_joint_7": BaseActuatorCfg(),
        # Allegro joints
        "joint_0.0": BaseActuatorCfg(),
        "joint_1.0": BaseActuatorCfg(),
        "joint_2.0": BaseActuatorCfg(),
        "joint_3.0": BaseActuatorCfg(),
        "joint_4.0": BaseActuatorCfg(),
        "joint_5.0": BaseActuatorCfg(),
        "joint_6.0": BaseActuatorCfg(),
        "joint_7.0": BaseActuatorCfg(),
        "joint_8.0": BaseActuatorCfg(),
        "joint_9.0": BaseActuatorCfg(),
        "joint_10.0": BaseActuatorCfg(),
        "joint_11.0": BaseActuatorCfg(),
        "joint_12.0": BaseActuatorCfg(),
        "joint_13.0": BaseActuatorCfg(),
        "joint_14.0": BaseActuatorCfg(),
        "joint_15.0": BaseActuatorCfg(),
    }

    joint_limits: dict[str, tuple[float, float]] = {
        # Kuka limits
        "iiwa7_joint_1": (-2.97, 2.97),
        "iiwa7_joint_2": (-2.09, 2.09),
        "iiwa7_joint_3": (-2.97, 2.97),
        "iiwa7_joint_4": (-2.09, 2.09),
        "iiwa7_joint_5": (-2.97, 2.97),
        "iiwa7_joint_6": (-2.09, 2.09),
        "iiwa7_joint_7": (-3.05, 3.05),
        # Allegro limits
        "joint_0.0": (-0.47, 0.47),
        "joint_1.0": (-0.196, 1.61),
        "joint_2.0": (-0.174, 1.719),
        "joint_3.0": (-0.227, 1.618),
        "joint_4.0": (-0.47, 0.47),
        "joint_5.0": (-0.196, 1.61),
        "joint_6.0": (-0.174, 1.719),
        "joint_7.0": (-0.227, 1.618),
        "joint_8.0": (-0.47, 0.47),
        "joint_9.0": (-0.196, 1.61),
        "joint_10.0": (-0.174, 1.719),
        "joint_11.0": (-0.227, 1.618),
        "joint_12.0": (0.263, 1.396),
        "joint_13.0": (-0.105, 1.163),
        "joint_14.0": (-0.189, 1.644),
        "joint_15.0": (-0.162, 1.719),
    }

    default_joint_positions: dict[str, float] = {
        # Kuka default positions
        "iiwa7_joint_1": 0.0,
        "iiwa7_joint_2": 0.0,
        "iiwa7_joint_3": 0.0,
        "iiwa7_joint_4": -1.57,
        "iiwa7_joint_5": 0.0,
        "iiwa7_joint_6": 1.57,
        "iiwa7_joint_7": 0.0,
        # Allegro default positions
        "joint_0.0": 0.0,
        "joint_1.0": 0.0,
        "joint_2.0": 0.0,
        "joint_3.0": 0.0,
        "joint_4.0": 0.0,
        "joint_5.0": 0.0,
        "joint_6.0": 0.0,
        "joint_7.0": 0.0,
        "joint_8.0": 0.0,
        "joint_9.0": 0.0,
        "joint_10.0": 0.0,
        "joint_11.0": 0.0,
        "joint_12.0": 0.4,
        "joint_13.0": 0.0,
        "joint_14.0": 0.0,
        "joint_15.0": 0.0,
    }

    control_type: dict[str, Literal["position", "effort"]] = {
        # All joints use position control
        "iiwa7_joint_1": "position",
        "iiwa7_joint_2": "position",
        "iiwa7_joint_3": "position",
        "iiwa7_joint_4": "position",
        "iiwa7_joint_5": "position",
        "iiwa7_joint_6": "position",
        "iiwa7_joint_7": "position",
        "joint_0.0": "position",
        "joint_1.0": "position",
        "joint_2.0": "position",
        "joint_3.0": "position",
        "joint_4.0": "position",
        "joint_5.0": "position",
        "joint_6.0": "position",
        "joint_7.0": "position",
        "joint_8.0": "position",
        "joint_9.0": "position",
        "joint_10.0": "position",
        "joint_11.0": "position",
        "joint_12.0": "position",
        "joint_13.0": "position",
        "joint_14.0": "position",
        "joint_15.0": "position",
    }