from __future__ import annotations

import logging

from metasim.cfg.checkers import EmptyChecker
from metasim.cfg.control import ControlCfg
from metasim.cfg.objects import RigidObjCfg
from metasim.cfg.robots.ballbalance_cfg import BallBalanceCfg
from metasim.constants import TaskType
from metasim.utils import configclass

from ..base_task_cfg import BaseTaskCfg

log = logging.getLogger(__name__)


@configclass
class BallBalanceCfg(BaseTaskCfg):
    name = "isaacgym_envs:BallBalance"
    episode_length = 300
    traj_filepath = None
    task_type = TaskType.OTHERS

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
    rot_reward_scale = 0.1
    vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.05
    action_penalty_scale = -0.001

    # Task settings
    ball_radius = 0.1
    max_platform_angle = 0.2
    success_tolerance = 0.02
    
    # Reset settings
    reset_position_noise = 0.1
    reset_dof_pos_noise = 0.0
    reset_dof_vel_noise = 0.0
    reset_ball_pos_noise = 0.05
    reset_ball_vel_noise = 0.1

    # Force disturbance settings
    force_scale = 0.0
    force_prob_range = (0.001, 0.1)
    force_decay = 0.99
    force_decay_interval = 0.08

    robot: BallBalanceCfg = BallBalanceCfg()

    control: ControlCfg = ControlCfg(
        action_scale=0.5,
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
                    name="ball",
                    usd_path="roboverse_data/assets/isaacgymenvs/assets/urdf/ball.urdf",
                    mjcf_path="roboverse_data/assets/isaacgymenvs/assets/urdf/ball.urdf",
                    urdf_path="roboverse_data/assets/isaacgymenvs/assets/urdf/ball.urdf",
                    default_position=(0.0, 0.0, 0.65),
                    default_orientation=(1.0, 0.0, 0.0, 0.0),
                ),
            ]

    observation_space = {"shape": [18]}

    randomize = {
        "robot": {
            "ballbalance": {
                "joint_qpos": {
                    "type": "uniform",
                    "low": 0.0,
                    "high": 0.0,
                }
            }
        },
        "object": {
            "ball": {
                "position": {
                    "x": [-0.05, 0.05],
                    "y": [-0.05, 0.05],
                    "z": [0.65, 0.65],
                },
                "velocity": {
                    "x": [-0.1, 0.1],
                    "y": [-0.1, 0.1],
                    "z": [0.0, 0.0],
                },
            },
        },
    }