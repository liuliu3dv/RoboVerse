from __future__ import annotations

import logging

from metasim.cfg.checkers import EmptyChecker
from metasim.cfg.control import ControlCfg
from metasim.cfg.objects import RigidObjCfg
from metasim.cfg.robots.humanoid_cfg import HumanoidCfg
from metasim.constants import TaskType
from metasim.utils import configclass

from ..base_task_cfg import BaseTaskCfg

log = logging.getLogger(__name__)


@configclass
class HumanoidAMPCfg(BaseTaskCfg):
    name = "isaacgym_envs:HumanoidAMP"
    episode_length = 300
    traj_filepath = None
    task_type = TaskType.LOCOMOTION

    task_meta = {
        "need_update": True,
        "need_sim": True,
        "need_control": True,
        "save_to_file": False,
        "enable_stats": True,
        "asymmetric_obs": True,
        "is_train": True,
    }

    # AMP specific settings
    motion_file = "roboverse_data/assets/isaacgymenvs/assets/amp/motions/amp_humanoid_walk.npy"
    enable_discriminator = True
    disc_obs_dim = 66
    disc_hidden_dims = [1024, 512]
    disc_learning_rate = 5e-5
    amp_diversity_bonus = 0.01
    style_reward_scale = 1.0
    task_reward_scale = 0.5
    disc_reward_scale = 5.0
    amp_obs_demo_buffer_size = 50000

    # Task rewards
    lin_vel_scale = 2.0
    ang_vel_scale = 0.25
    dof_pos_scale = 1.0
    dof_vel_scale = 0.1
    height_reward_scale = 1.0
    heading_weight = 0.5
    up_weight = 0.1

    # Costs
    actions_cost_scale = 0.01
    energy_cost_scale = 0.05
    joints_at_limit_cost_scale = 0.25
    death_cost = -10.0
    termination_height = 0.5

    # Motion settings
    power_scale = 1.0
    dof_vel_scale = 0.1
    contact_force_scale = 0.1
    
    # Reset settings
    reset_position_noise = 0.1
    reset_rotation_noise = 0.1
    reset_dof_pos_noise = 0.2
    reset_dof_vel_noise = 0.1
    reference_state_initialization = True
    reference_state_init_prob = 0.85

    # Force disturbance settings
    force_scale = 0.0
    force_prob_range = (0.001, 0.1)
    force_decay = 0.99
    force_decay_interval = 0.08

    robot: HumanoidCfg = HumanoidCfg()

    control: ControlCfg = ControlCfg(
        action_scale=0.5,
        action_offset=False,
        torque_limit_scale=1.0
    )

    checker = EmptyChecker()

    objects: list[RigidObjCfg] = []

    observation_space = {"shape": [108], "asymmetric": [198]}

    randomize = {
        "robot": {
            "humanoid": {
                "pos": {
                    "type": "gaussian",
                    "mean": [0.0, 0.0, 1.34],
                    "std": [0.0, 0.0, 0.0],
                },
                "rot": {
                    "type": "gaussian",
                    "mean": [0.0, 0.0, 0.0, 1.0],
                    "std": [0.0, 0.0, 0.0, 0.0],
                },
                "joint_qpos": {
                    "type": "uniform",
                    "low": -0.2,
                    "high": 0.2,
                },
                "joint_qvel": {
                    "type": "uniform",
                    "low": -0.1,
                    "high": 0.1,
                },
            }
        }
    }