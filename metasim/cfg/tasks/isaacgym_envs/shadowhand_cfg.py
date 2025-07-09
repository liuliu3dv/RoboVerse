from __future__ import annotations

import logging

from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType, TaskType
from metasim.utils import configclass

from ..base_task_cfg import BaseTaskCfg

log = logging.getLogger(__name__)


@configclass
class ShadowHandCfg(BaseTaskCfg):
    name = "isaacgym_envs:ShadowHand"
    episode_length = 600
    traj_filepath = None
    task_type = TaskType.TABLETOP_MANIPULATION

    object_type = "block"

    reward_scale = 1.0
    clip_observations = 5.0
    clip_actions = 1.0

    task_meta = {
        "need_update": True,
        "need_sim": True,
        "need_control": True,
        "save_to_file": False,
        "enable_stats": True,
        "asymmetric_obs": False,
        "is_train": True,
    }

    success_tolerance = 0.1
    reach_goal_bonus = 250.0
    fall_dist = 0.24
    fall_penalty = 0.0
    rot_eps = 0.1

    vel_obs_scale = 0.2
    force_torque_obs_scale = 10.0
    reset_position_noise = 0.01
    reset_rotation_noise = 0.0
    reset_dof_pos_noise = 0.2
    reset_dof_vel_noise = 0.0

    force_scale = 0.0
    force_prob_range = (0.001, 0.1)
    force_decay = 0.99
    force_decay_interval = 0.08

    objects: list[RigidObjCfg] | None = None

    def __post_init__(self):
        super().__post_init__()

        if self.objects is None:
            self.objects = [
                RigidObjCfg(
                    name="block",
                    usd_path="roboverse_data/assets/isaacgymenvs/assets/urdf/objects/cube_multicolor.urdf",
                    mjcf_path="roboverse_data/assets/isaacgymenvs/assets/urdf/objects/cube_multicolor.urdf",
                    urdf_path="roboverse_data/assets/isaacgymenvs/assets/urdf/objects/cube_multicolor.urdf",
                    default_position=(0.0, -0.39, 0.615),
                    default_orientation=(1.0, 0.0, 0.0, 0.0),
                ),
                RigidObjCfg(
                    name="goal",
                    usd_path="roboverse_data/assets/isaacgymenvs/assets/urdf/objects/cube_multicolor.urdf",
                    mjcf_path="roboverse_data/assets/isaacgymenvs/assets/urdf/objects/cube_multicolor.urdf",
                    urdf_path="roboverse_data/assets/isaacgymenvs/assets/urdf/objects/cube_multicolor.urdf",
                    default_position=(0.0, -0.39, 0.715),
                    default_orientation=(1.0, 0.0, 0.0, 0.0),
                    physics=PhysicStateType.XFORM,
                ),
            ]

    observation_space = {"shape": [211]}

    randomize = {
        "robot": {
            "shadow_hand": {
                "joint_qpos": {
                    "type": "uniform",
                    "low": -0.2,
                    "high": 0.2,
                }
            }
        },
        "object": {
            "block": {
                "position": {
                    "x": [-0.05, 0.05],
                    "y": [-0.44, -0.34],
                    "z": [0.615, 0.615],
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
                    "x": [-0.05, 0.05],
                    "y": [-0.44, -0.34],
                    "z": [0.615, 0.815],
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