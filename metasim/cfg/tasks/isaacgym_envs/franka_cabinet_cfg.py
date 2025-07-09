from __future__ import annotations

import logging

from metasim.cfg.checkers import EmptyChecker
from metasim.cfg.control import ControlCfg
from metasim.cfg.objects import RigidObjCfg
from metasim.cfg.robots.franka_cfg import FrankaCfg
from metasim.constants import TaskType
from metasim.utils import configclass

from ..base_task_cfg import BaseTaskCfg

log = logging.getLogger(__name__)


@configclass
class FrankaCabinetCfg(BaseTaskCfg):
    name = "isaacgym_envs:FrankaCabinet"
    episode_length = 500
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

    # Rewards
    dist_reward_scale = 2.0
    rot_reward_scale = 0.5
    around_handle_reward_scale = 10.0
    open_reward_scale = 7.5
    action_penalty_scale = -0.0001
    finger_dist_reward_scale = 10.0
    
    # Task settings
    grip_limit = 10.0
    cabinet_open_angle_limit = 1.57  # 90 degrees
    success_tolerance = 0.3
    
    # Reset settings
    reset_position_noise = 0.1
    reset_rotation_noise = 0.25
    reset_dof_pos_noise = 0.1
    reset_dof_vel_noise = 0.0
    reset_cabinet_noise = 0.25

    # Force disturbance settings
    force_scale = 0.0
    force_prob_range = (0.001, 0.1)
    force_decay = 0.99
    force_decay_interval = 0.08

    robot: FrankaCfg = FrankaCfg()

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
                    name="cabinet",
                    usd_path="roboverse_data/assets/isaacgymenvs/assets/urdf/sektion_cabinet_model/urdf/sektion_cabinet.urdf",
                    mjcf_path="roboverse_data/assets/isaacgymenvs/assets/urdf/sektion_cabinet_model/urdf/sektion_cabinet.urdf",
                    urdf_path="roboverse_data/assets/isaacgymenvs/assets/urdf/sektion_cabinet_model/urdf/sektion_cabinet.urdf",
                    default_position=(0.5, 0.0, 0.4),
                    default_orientation=(0.0, 0.0, 0.0, 1.0),
                ),
            ]

    observation_space = {"shape": [37]}

    randomize = {
        "robot": {
            "franka": {
                "joint_qpos": {
                    "type": "uniform",
                    "low": -0.1,
                    "high": 0.1,
                },
                "pos": {
                    "type": "gaussian",
                    "mean": [0.0, 0.0, 0.0],
                    "std": [0.1, 0.1, 0.0],
                }
            }
        },
        "object": {
            "cabinet": {
                "joint_qpos": {
                    "type": "uniform",
                    "low": -0.25,
                    "high": 0.25,
                },
            },
        },
    }