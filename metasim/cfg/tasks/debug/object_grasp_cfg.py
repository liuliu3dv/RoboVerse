from __future__ import annotations

import torch

from metasim.cfg.objects import RigidObjCfg
from metasim.cfg.tasks.base_task_cfg import BaseTaskCfg
from metasim.constants import BenchmarkType, PhysicStateType, TaskType
from metasim.utils import configclass
from metasim.utils.state import TensorState


def near_object(states: TensorState, robot_name: str | None = None) -> torch.Tensor:
    ee_pos = states.robots[robot_name].body_state[:, states.robots[robot_name].body_names.index("panda_hand"), :3]
    object_pos = states.objects["bbq_sauce"].root_state[:, :3]
    distances = torch.norm(ee_pos - object_pos, dim=1)
    return -distances  # Negative distance as reward


def object_lift(states: TensorState, robot_name: str | None = None) -> torch.Tensor:
    object_pos = states.objects["bbq_sauce"].root_state[:, :3]
    return object_pos[:, 2]


@configclass
class ObjectGraspingCfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.DEBUG
    task_type = TaskType.TABLETOP_MANIPULATION
    can_tabletop = True
    episode_length = 100
    objects = [
        RigidObjCfg(
            name="bbq_sauce",
            scale=(1.5, 1.5, 1.5),
            physics=PhysicStateType.RIGIDBODY,
            usd_path="get_started/example_assets/bbq_sauce/usd/bbq_sauce.usd",
            urdf_path="get_started/example_assets/bbq_sauce/urdf/bbq_sauce.urdf",
            mjcf_path="get_started/example_assets/bbq_sauce/mjcf/bbq_sauce.xml",
        ),
    ]
    traj_filepath = "metasim/cfg/tasks/debug/object_grasping_franka_v2.json"
    reward_functions = [near_object, object_lift]
    reward_weights = [0.00, 0.99]
    ## TODO: add a empty checker to suppress warning
