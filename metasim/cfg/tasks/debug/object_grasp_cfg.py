from __future__ import annotations

import torch

from metasim.cfg.objects import PrimitiveCubeCfg
from metasim.cfg.tasks.base_task_cfg import BaseTaskCfg
from metasim.constants import BenchmarkType, PhysicStateType, TaskType
from metasim.utils import configclass
from metasim.utils.state import TensorState


def near_object(states: TensorState, robot_name: str | None = None) -> torch.Tensor:
    ee_pos_l = states.robots[robot_name].body_state[
        :, states.robots[robot_name].body_names.index("panda_leftfinger"), :3
    ]
    ee_pos_r = states.robots[robot_name].body_state[
        :, states.robots[robot_name].body_names.index("panda_rightfinger"), :3
    ]
    object_pos = states.objects["cube"].root_state[:, :3]
    ee_pos = torch.mean(torch.stack([ee_pos_l, ee_pos_r], dim=1), dim=1)
    tcp_to_obj_dist = torch.norm(ee_pos - object_pos, dim=1)
    tcp_to_obj_dist = torch.clamp(tcp_to_obj_dist, min=0.05, max=float("inf"))
    reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
    return reaching_reward


def grasp_success(states: TensorState, robot_name: str | None = None) -> torch.Tensor:
    ee_pos_l = states.robots[robot_name].body_state[
        :, states.robots[robot_name].body_names.index("panda_leftfinger"), :3
    ]
    ee_pos_r = states.robots[robot_name].body_state[
        :, states.robots[robot_name].body_names.index("panda_rightfinger"), :3
    ]
    object_pos = states.objects["cube"].root_state[:, :3]

    # fingers相对距离 & 物体在中间
    finger_dist = torch.norm(ee_pos_l - ee_pos_r, dim=1)
    center_to_obj = torch.norm((ee_pos_l + ee_pos_r) / 2 - object_pos, dim=1)

    grasped = (finger_dist < 0.06) & (center_to_obj < 0.04)
    return grasped


def object_lift(states: TensorState, robot_name: str | None = None) -> torch.Tensor:
    object_pos = states.objects["cube"].root_state[:, :3]
    grasped = grasp_success(states, robot_name)
    return (object_pos[:, 2] * 50) * grasped


@configclass
class ObjectGraspingCfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.DEBUG
    task_type = TaskType.TABLETOP_MANIPULATION
    can_tabletop = True
    episode_length = 100
    objects = [
        PrimitiveCubeCfg(
            name="cube",
            size=(0.04, 0.04, 0.04),
            color=[1.0, 0.0, 0.0],
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    cameras = []
    traj_filepath = "metasim/cfg/tasks/debug/object_grasping_franka_v2.json"
    reward_functions = [near_object, object_lift, grasp_success]
    reward_weights = [0.1, 0.9, 0.1]
    ## TODO: add a empty checker to suppress warning
