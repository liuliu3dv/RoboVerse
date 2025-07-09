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
class FrankaCubeStackCfg(BaseTaskCfg):
    name = "isaacgym_envs:FrankaCubeStack"
    episode_length = 500
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
    dist_reward_scale = 1.0
    lift_reward_scale = 5.0
    align_reward_scale = 2.0
    stack_reward_scale = 10.0
    action_penalty_scale = -0.0001
    
    # Task settings
    cube_size = 0.05
    success_tolerance = 0.01
    max_stack_height = 0.15
    table_height = 0.4
    
    # Reset settings
    reset_position_noise = 0.02
    reset_rotation_noise = 0.0
    reset_dof_pos_noise = 0.1
    reset_dof_vel_noise = 0.0

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
                    name="cube1",
                    usd_path="roboverse_data/assets/isaacgymenvs/assets/urdf/objects/cube_multicolor.urdf",
                    mjcf_path="roboverse_data/assets/isaacgymenvs/assets/urdf/objects/cube_multicolor.urdf",
                    urdf_path="roboverse_data/assets/isaacgymenvs/assets/urdf/objects/cube_multicolor.urdf",
                    default_position=(0.4, -0.1, 0.425),
                    default_orientation=(1.0, 0.0, 0.0, 0.0),
                ),
                RigidObjCfg(
                    name="cube2",
                    usd_path="roboverse_data/assets/isaacgymenvs/assets/urdf/objects/cube_multicolor.urdf",
                    mjcf_path="roboverse_data/assets/isaacgymenvs/assets/urdf/objects/cube_multicolor.urdf",
                    urdf_path="roboverse_data/assets/isaacgymenvs/assets/urdf/objects/cube_multicolor.urdf",
                    default_position=(0.4, 0.1, 0.425),
                    default_orientation=(1.0, 0.0, 0.0, 0.0),
                ),
                RigidObjCfg(
                    name="table",
                    usd_path="roboverse_data/assets/isaacgymenvs/assets/urdf/table_wide.urdf",
                    mjcf_path="roboverse_data/assets/isaacgymenvs/assets/urdf/table_wide.urdf",
                    urdf_path="roboverse_data/assets/isaacgymenvs/assets/urdf/table_wide.urdf",
                    default_position=(0.5, 0.0, 0.2),
                    default_orientation=(1.0, 0.0, 0.0, 0.0),
                ),
            ]

    observation_space = {"shape": [44]}

    randomize = {
        "robot": {
            "franka": {
                "joint_qpos": {
                    "type": "uniform",
                    "low": -0.1,
                    "high": 0.1,
                }
            }
        },
        "object": {
            "cube1": {
                "position": {
                    "x": [0.35, 0.45],
                    "y": [-0.15, -0.05],
                    "z": [0.425, 0.425],
                },
            },
            "cube2": {
                "position": {
                    "x": [0.35, 0.45],
                    "y": [0.05, 0.15],
                    "z": [0.425, 0.425],
                },
            },
        },
    }