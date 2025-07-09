from __future__ import annotations

import logging

from metasim.cfg.checkers import EmptyChecker
from metasim.cfg.control import ControlCfg
from metasim.cfg.objects import RigidObjCfg
from metasim.cfg.robots.trifinger_cfg import TrifingerCfg
from metasim.constants import PhysicStateType, TaskType
from metasim.utils import configclass

from ..base_task_cfg import BaseTaskCfg

log = logging.getLogger(__name__)


@configclass
class TrifingerCfg(BaseTaskCfg):
    name = "isaacgym_envs:Trifinger"
    episode_length = 750
    traj_filepath = None
    task_type = TaskType.TABLETOP_MANIPULATION

    task_meta = {
        "need_update": True,
        "need_sim": True,
        "need_control": True,
        "save_to_file": False,
        "enable_stats": True,
        "asymmetric_obs": False,
        "is_train": True,
    }

    # Rewards
    dist_reward_scale = -0.1
    rot_reward_scale = 0.5
    action_penalty_scale = -0.0001
    reach_goal_bonus = 250.0
    fall_dist = 0.24
    fall_penalty = -50.0
    rotation_eps = 0.1

    # Task settings
    success_tolerance = 0.01
    safety_distance = 0.18
    apply_safety_damping = True
    
    # Reset settings
    reset_position_noise = 0.01
    reset_rotation_noise = 0.0
    reset_dof_pos_noise = 0.2
    reset_dof_vel_noise = 0.0
    reset_target_pose = True

    # Force disturbance settings
    force_scale = 0.0
    force_prob_range = (0.001, 0.1)
    force_decay = 0.99
    force_decay_interval = 0.08

    robot: TrifingerCfg = TrifingerCfg()

    control: ControlCfg = ControlCfg(
        action_scale=1.0,
        action_offset=False,
        torque_limit_scale=1.0
    )

    checker = EmptyChecker()

    objects: list[RigidObjCfg] | None = None

    def __post_init__(self):
        super().__post_init__()

        if self.objects is None:
            self.objects = [
                RigidObjCfg(
                    name="cube",
                    usd_path="roboverse_data/assets/isaacgymenvs/assets/trifinger/objects/urdf/cube_multicolor.urdf",
                    mjcf_path="roboverse_data/assets/isaacgymenvs/assets/trifinger/objects/urdf/cube_multicolor.urdf",
                    urdf_path="roboverse_data/assets/isaacgymenvs/assets/trifinger/objects/urdf/cube_multicolor.urdf",
                    default_position=(0.0, 0.0, 0.0325),
                    default_orientation=(1.0, 0.0, 0.0, 0.0),
                ),
                RigidObjCfg(
                    name="goal",
                    usd_path="roboverse_data/assets/isaacgymenvs/assets/trifinger/objects/urdf/cube_goal_multicolor.urdf",
                    mjcf_path="roboverse_data/assets/isaacgymenvs/assets/trifinger/objects/urdf/cube_goal_multicolor.urdf",
                    urdf_path="roboverse_data/assets/isaacgymenvs/assets/trifinger/objects/urdf/cube_goal_multicolor.urdf",
                    default_position=(0.0, 0.0, 0.0325),
                    default_orientation=(1.0, 0.0, 0.0, 0.0),
                    physics=PhysicStateType.XFORM,
                ),
                RigidObjCfg(
                    name="table",
                    usd_path="roboverse_data/assets/isaacgymenvs/assets/trifinger/robot_properties_fingers/urdf/table_without_border.urdf",
                    mjcf_path="roboverse_data/assets/isaacgymenvs/assets/trifinger/robot_properties_fingers/urdf/table_without_border.urdf",
                    urdf_path="roboverse_data/assets/isaacgymenvs/assets/trifinger/robot_properties_fingers/urdf/table_without_border.urdf",
                    default_position=(0.0, 0.0, -0.019),
                    default_orientation=(1.0, 0.0, 0.0, 0.0),
                ),
            ]

    observation_space = {"shape": [65]}

    randomize = {
        "robot": {
            "trifinger": {
                "joint_qpos": {
                    "type": "uniform",
                    "low": -0.2,
                    "high": 0.2,
                }
            }
        },
        "object": {
            "cube": {
                "position": {
                    "x": [-0.08, 0.08],
                    "y": [-0.08, 0.08],
                    "z": [0.0325, 0.0325],
                },
                "orientation": {
                    "x": [-1.0, 1.0],
                    "y": [-1.0, 1.0],
                    "z": [-1.0, 1.0],
                    "w": [-1.0, 1.0],
                },
            },
            "goal": {
                "position": {
                    "x": [-0.08, 0.08],
                    "y": [-0.08, 0.08],
                    "z": [0.0325, 0.15],
                },
                "orientation": {
                    "x": [-1.0, 1.0],
                    "y": [-1.0, 1.0],
                    "z": [-1.0, 1.0],
                    "w": [-1.0, 1.0],
                }
            },
        },
    }