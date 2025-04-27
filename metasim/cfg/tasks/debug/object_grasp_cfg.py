from __future__ import annotations

import torch
from loguru import logger as log

from metasim.cfg.objects import ArticulationObjCfg, PrimitiveCubeCfg
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
    ee_rot_l_quat = states.robots[robot_name].body_state[
        :, states.robots[robot_name].body_names.index("panda_leftfinger"), 3:7
    ]
    ee_rot_r_quat = states.robots[robot_name].body_state[
        :, states.robots[robot_name].body_names.index("panda_rightfinger"), 3:7
    ]
    tip_offset = torch.tensor([0.0, 0.0, 0.045]).to(ee_pos_l.device)
    # tip_offset = torch.tensor([0.0, 0.0, 0.10312]).to(ee_pos_l.device)
    import pytorch3d.transforms as T

    ee_rot_l_mat = T.quaternion_to_matrix(ee_rot_l_quat)
    ee_rot_r_mat = T.quaternion_to_matrix(ee_rot_r_quat)
    # import pdb; pdb.set_trace()
    tip_offset_l_with_rot = torch.matmul(ee_rot_l_mat, tip_offset)
    tip_offset_r_with_rot = torch.matmul(ee_rot_r_mat, tip_offset)
    tip_pos_l = ee_pos_l + tip_offset_l_with_rot
    tip_pos_r = ee_pos_r + tip_offset_r_with_rot
    object_pos = states.objects["object"].root_state[:, :3]
    ee_pos = torch.mean(torch.stack([tip_pos_l, tip_pos_r], dim=1), dim=1)
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
    ee_rot_l_quat = states.robots[robot_name].body_state[
        :, states.robots[robot_name].body_names.index("panda_leftfinger"), 3:7
    ]
    ee_rot_r_quat = states.robots[robot_name].body_state[
        :, states.robots[robot_name].body_names.index("panda_rightfinger"), 3:7
    ]
    tip_offset = torch.tensor([0.0, 0.0, 0.045]).to(ee_pos_l.device)
    # tip_offset = torch.tensor([0.0, 0.0, 0.10312]).to(ee_pos_l.device)
    import pytorch3d.transforms as T

    ee_rot_l_mat = T.quaternion_to_matrix(ee_rot_l_quat)
    ee_rot_r_mat = T.quaternion_to_matrix(ee_rot_r_quat)
    # import pdb; pdb.set_trace()
    tip_offset_l_with_rot = torch.matmul(ee_rot_l_mat, tip_offset)
    tip_offset_r_with_rot = torch.matmul(ee_rot_r_mat, tip_offset)
    tip_pos_l = ee_pos_l + tip_offset_l_with_rot
    tip_pos_r = ee_pos_r + tip_offset_r_with_rot
    object_pos = states.objects["object"].root_state[:, :3]

    finger_dist = torch.norm(tip_pos_l - tip_pos_r, dim=1)
    center_to_obj = torch.norm((tip_pos_l + tip_pos_r) / 2 - object_pos, dim=1)

    grasped = (finger_dist < 0.07) & (center_to_obj < 0.02)
    log.info(f"grasped: {grasped}")
    return grasped


# def object_lift(states: TensorState, robot_name: str | None = None) -> torch.Tensor:
#     object_pos = states.objects["object"].root_state[:, :3]
#     grasped = grasp_success(states, robot_name)
#     print(f"object_pos: {object_pos[:, 2]}")
#     return (object_pos[:, 2] * 100) * grasped + object_pos[:, 2]


def object_lift(states: TensorState, robot_name: str | None = None) -> torch.Tensor:
    object_pos = states.objects["object"].root_state[:, :3]
    grasped = grasp_success(states, robot_name)
    log.info(f"object_pos: {object_pos[:, 2]}")
    return (object_pos[:, 2] - 0.02) * 100 * grasped


def object_shift(states: TensorState, robot_name: str | None = None) -> torch.Tensor:
    object_pos = states.objects["object"].root_state[:, :3]
    # grasped = grasp_success(states, robot_name)
    # print(f"object_pos: {object_pos[:, 2]}")
    # shift is xy distance
    shift = torch.norm(object_pos[:, :2] - torch.tensor([0.5, 0.0]).to(object_pos.device), dim=1)
    # print("shift: ", shift)
    log.info(f"shift: {shift}")
    return -shift * 10


@configclass
class ObjectGraspingCfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.DEBUG
    task_type = TaskType.TABLETOP_MANIPULATION
    can_tabletop = True
    episode_length = 100
    objects = [
        PrimitiveCubeCfg(
            name="object",
            size=(0.04, 0.04, 0.04),
            color=[1.0, 0.0, 0.0],
            physics=PhysicStateType.RIGIDBODY,
        ),
        # PrimitiveSphereCfg(
        #     name="finger_tip_l",
        #     radius=0.01,
        #     color=[1.0, 0.0, 0.0],
        #     physics=PhysicStateType.XFORM,
        # ),
        # PrimitiveSphereCfg(
        #     name="finger_tip_r",
        #     radius=0.01,
        #     color=[0.0, 0.0, 1.0],
        #     physics=PhysicStateType.XFORM,
        # ),
    ]
    cameras = []
    traj_filepath = "metasim/cfg/tasks/debug/object_grasping_franka_v2.json"
    reward_functions = [near_object, object_lift, object_shift, grasp_success]
    reward_weights = [1, 1, 0.1, 1]
    ## TODO: add a empty checker to suppress warning


@configclass
class CustObjectGraspingCfg(BaseTaskCfg):
    source_benchmark = BenchmarkType.DEBUG
    task_type = TaskType.TABLETOP_MANIPULATION
    can_tabletop = True
    episode_length = 100
    objects = [
        ArticulationObjCfg(
            name="scene",
            fix_base_link=True,
            # physics=PhysicStateType.RIGIDBODY,
            usd_path="metasim/cfg/tasks/debug/cust_scene/Collected_World0/World0.usd",
            urdf_path="metasim/cfg/tasks/debug/cust_scene/office_pour_righthand_coke_0000/scene.urdf",
            mjcf_path="metasim/cfg/tasks/debug/cust_scene/scene.xml",
        ),
        ArticulationObjCfg(
            name="object",
            fix_base_link=True,
            # physics=PhysicStateType.RIGIDBODY,
            usd_path="metasim/cfg/tasks/debug/cust_scene/Collected_World0/World0.usd",
            urdf_path="metasim/cfg/tasks/debug/cust_scene/office_pour_righthand_coke_0000/meshes/coke/coke.urdf",
            mjcf_path="metasim/cfg/tasks/debug/cust_scene/scene.xml",
        ),
    ]
    cameras = []
    traj_filepath = "metasim/cfg/tasks/debug/cust_object_grasping_franka_v2.json"
    reward_functions = [near_object, object_lift, grasp_success]
    reward_weights = [0.1, 0.9, 0.1]
    ## TODO: add a empty checker to suppress warning
