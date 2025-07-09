from __future__ import annotations

import logging

from metasim.cfg.control import ControlCfg
from metasim.cfg.objects import RigidObjCfg
from metasim.cfg.robots.allegro_kuka_cfg import AllegroKukaCfg
from metasim.constants import PhysicStateType, TaskType
from metasim.utils import configclass

from ..base_task_cfg import BaseTaskCfg

log = logging.getLogger(__name__)


@configclass
class AllegroKukaCfg(BaseTaskCfg):
    name = "isaacgym_envs:AllegroKuka"
    episode_length = 600
    traj_filepath = None
    task_type = TaskType.FULLBODY_MANIPULATION

    task_meta = {
        "need_update": True,
        "need_sim": True,
        "need_control": True,
        "save_to_file": False,
        "enable_stats": True,
        "asymmetric_obs": False,
        "is_train": True,
    }

    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    action_penalty_scale = -0.0002
    reach_goal_bonus = 250.0
    fall_dist = 0.5
    fall_penalty = 0.0
    rot_eps = 0.1
    vel_obs_scale = 0.2
    
    success_tolerance = 0.1
    max_consecutive_successes = 0
    av_factor = 0.1

    reset_position_noise = 0.01
    reset_rotation_noise = 0.0
    reset_dof_pos_noise = 0.2
    reset_dof_vel_noise = 0.0

    force_scale = 0.0
    force_prob_range = (0.001, 0.1)
    force_decay = 0.99
    force_decay_interval = 0.08

    robot: AllegroKukaCfg = AllegroKukaCfg()

    control: ControlCfg = ControlCfg(
        action_scale=1.0,
        action_offset=False,
        torque_limit_scale=1.0,
        dof_speed_scale=20.0
    )

    objects: list[RigidObjCfg] | None = None

    def __post_init__(self):
        super().__post_init__()

        if self.objects is None:
            self.objects = [
                RigidObjCfg(
                    name="cube",
                    usd_path="roboverse_data/assets/isaacgymenvs/assets/urdf/objects/cube_multicolor.urdf",
                    mjcf_path="roboverse_data/assets/isaacgymenvs/assets/urdf/objects/cube_multicolor.urdf",
                    urdf_path="roboverse_data/assets/isaacgymenvs/assets/urdf/objects/cube_multicolor.urdf",
                    default_position=(0.5, 0.0, 0.7),
                    default_orientation=(1.0, 0.0, 0.0, 0.0),
                ),
                RigidObjCfg(
                    name="goal",
                    usd_path="roboverse_data/assets/isaacgymenvs/assets/urdf/objects/cube_multicolor.urdf",
                    mjcf_path="roboverse_data/assets/isaacgymenvs/assets/urdf/objects/cube_multicolor.urdf",
                    urdf_path="roboverse_data/assets/isaacgymenvs/assets/urdf/objects/cube_multicolor.urdf",
                    default_position=(0.5, 0.0, 0.8),
                    default_orientation=(1.0, 0.0, 0.0, 0.0),
                    physics=PhysicStateType.XFORM,
                ),
            ]

    observation_space = {"shape": [263]}

    randomize = {
        "robot": {
            "allegro_kuka": {
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
                    "x": [0.45, 0.55],
                    "y": [-0.05, 0.05],
                    "z": [0.7, 0.7],
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
                    "x": [0.45, 0.55],
                    "y": [-0.05, 0.05],
                    "z": [0.7, 0.9],
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