from __future__ import annotations

import logging

from metasim.cfg.checkers import EmptyChecker
from metasim.cfg.control import ControlCfg
from metasim.cfg.objects import RigidObjCfg
from metasim.cfg.robots.ingenuity_cfg import IngenuityCfg
from metasim.constants import TaskType
from metasim.utils import configclass

from ..base_task_cfg import BaseTaskCfg

log = logging.getLogger(__name__)


@configclass
class IngenuityCfg(BaseTaskCfg):
    name = "isaacgym_envs:Ingenuity"
    episode_length = 400
    traj_filepath = None
    task_type = TaskType.LOCOMOTION

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
    target_dist_scale = 2.0
    target_rot_scale = 0.5
    pos_error_scale = -0.1
    orient_error_scale = -0.1
    lin_vel_scale = -0.05
    ang_vel_scale = -0.05
    action_penalty_scale = -0.001
    crash_penalty = -100.0

    # Task settings
    target_height = 10.0
    min_height = 0.2
    max_height = 20.0
    max_tilt_angle = 0.5
    gravity_mars = 3.71  # Mars gravity
    
    # Reset settings
    reset_position_noise = 0.5
    reset_rotation_noise = 0.1
    reset_velocity_noise = 0.2

    # Force disturbance settings (wind)
    force_scale = 1.0
    force_prob_range = (0.5, 0.8)
    force_decay = 0.99
    force_decay_interval = 0.08

    robot: IngenuityCfg = IngenuityCfg()

    control: ControlCfg = ControlCfg(
        action_scale=1.0,
        action_offset=False,
        torque_limit_scale=1.0
    )

    checker = EmptyChecker()

    objects: list[RigidObjCfg] = []

    observation_space = {"shape": [29]}

    randomize = {
        "robot": {
            "ingenuity": {
                "pos": {
                    "type": "gaussian",
                    "mean": [0.0, 0.0, 1.0],
                    "std": [0.5, 0.5, 0.1],
                },
                "rot": {
                    "type": "gaussian",
                    "mean": [0.0, 0.0, 0.0, 1.0],
                    "std": [0.1, 0.1, 0.1, 0.0],
                },
                "linear_velocity": {
                    "type": "uniform",
                    "low": -0.2,
                    "high": 0.2,
                },
                "angular_velocity": {
                    "type": "uniform",
                    "low": -0.1,
                    "high": 0.1,
                },
            }
        }
    }