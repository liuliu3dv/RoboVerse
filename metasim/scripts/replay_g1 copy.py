import os
import torch
import importlib
import rootutils
from pathlib import Path
from loguru import logger as log
from rich.logging import RichHandler
rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])
import json
import numpy as np
import pypose as pp
import pyroki as pk
from glob import glob
from pyroki import Robot
from yourdfpy import URDF
from loguru import logger as log
from tqdm.rich import tqdm, trange
from rich.logging import RichHandler
from metasim.utils.setup_util import get_robot, get_task
import third_party.pyroki.examples.pyroki_snippets as pks
from metasim.utils.demo_util.loader import load_traj_file, save_traj_file
# from metasim.utils.kinematics_utils import ee_pose_from_tcp_pose, tcp_pose_from_ee_pose
from metasim.utils import is_camel_case, is_snake_case, to_camel_case, to_snake_case
from tqdm.rich import tqdm_rich as tqdm
import tyro
from get_started.utils import ObsSaver

from metasim.cfg.robots.base_robot_cfg import BaseRobotCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import SimType
from metasim.sim import BaseSimHandler, EnvWrapper
from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import get_robot, get_sim_env_class, get_task
from metasim.utils.state import state_tensor_to_nested
from metasim.utils.tensor_util import tensor_to_cpu
from scipy.spatial.transform import Rotation as R
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])
NUM_SEED = 20

###########################################################
## Global Variables
###########################################################
global global_step, tot_success, tot_give_up
tot_success = 0
tot_give_up = 0
global_step = 0
#########################################
### Add command line arguments
#########################################
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import tyro
from metasim.cfg.render import RenderCfg
from metasim.constants import PhysicStateType, SimType
from metasim.cfg.randomization import RandomizationCfg
from metasim.cfg.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg

@dataclass
class Args:
    random: RandomizationCfg
    """Domain randomization options"""
    render: RenderCfg
    """Renderer options"""
    task: str = "CloseBox"
    """Task name"""
    robot: str = "franka"
    """Robot name"""
    num_envs: int = 1
    """Number of parallel environments, find a proper number for best performance on your machine"""
    sim: Literal["isaaclab", "mujoco", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3"] = "isaaclab"
    """Simulator backend"""
    demo_start_idx: int | None = None
    """The index of the first demo to collect, None for all demos"""
    max_demo_idx: int | None = None
    """Maximum number of demos to collect, None for all demos"""
    retry_num: int = 0
    """Number of retries for a failed demo"""
    headless: bool = False
    """Run in headless mode"""
    table: bool = True
    """Try to add a table"""
    tot_steps_after_success: int = 20
    """Maximum number of steps to collect after success, or until run out of demo"""
    split: Literal["train", "val", "test", "all"] = "all"
    """Split to collect"""
    cust_name: str | None = None
    """Custom name for the dataset"""
    scene: str | None = None
    """Scene name"""
    run_all: bool = False
    """Rollout all trajectories, overwrite existing demos"""
    run_unfinished: bool = False
    """Rollout unfinished trajectories"""
    run_failed: bool = False
    """Rollout unfinished and failed trajectories"""

    def __post_init__(self):

        if self.random.table and not self.table:
            log.warning("Cannot enable table randomization without a table, disabling table randomization")
            self.random.table = False

        if self.max_demo_idx is None:
            self.max_demo_idx = math.inf

        if self.demo_start_idx is None:
            self.demo_start_idx = 0

        log.info(f"Args: {self}")


args = tyro.cli(Args)

import pypose as pp
def wxyz2xyzw(poses):
    # poses: (N, 7) → [tx, ty, tz, qw, qx, qy, qz]
    t = poses[..., 0:3]  # (N, 3)
    q = poses[..., 3:7]  # (N, 4) in (w, x, y, z)
    # SciPy expects (x, y, z, w)
    poses = pp.SE3(np.concatenate([t, q[..., [1]], q[..., [2]], q[..., [3]], q[..., [0]]], axis=-1)) # (N, 3, 3)
    # Build transformation matrices (N, 4, 4)
    return poses

def xyzw2wxyz(poses):
    # poses: (N, 7) → [tx, ty, tz, qx, qy, qz, qw]
    t = poses[..., 0:3]  # (N, 3)
    q = poses[..., 3:7]  # (N, 4) in (x, y, z, w)
    poses = np.concatenate([t, q[..., [3]], q[..., [0]], q[..., [1]], q[..., [2]]], axis=-1)
    return poses

def get_urdf(robot_name: str) -> str:
    """Get the robot cfg instance from the robot name.

    Args:
        robot_name: The name of the robot.

    Returns:
        The robot cfg instance.
    """
    if is_camel_case(robot_name):
        RobotName = robot_name
    elif is_snake_case(robot_name):
        RobotName = to_camel_case(robot_name)
    else:
        raise ValueError(f"Invalid robot name: {robot_name}, should be in either camel case or snake case")
    module = importlib.import_module("metasim.cfg.robots")
    robot_cls = getattr(module, f"{RobotName}Cfg")
    urdf = URDF.load("/home/xyc/RoboVerse/roboverse_data/robots/franka/urdf/franka_panda.urdf")
    return urdf


def get_pk_robot(urdf) -> Robot:
    return pk.Robot.from_urdf(urdf)

init_states = [
    {
        "objects": {
            "cube": {
                "pos": torch.tensor([0, 0, 0]),
                "rot": torch.tensor([0, 0.0, 0.0, 1.0]),
            },
            "sphere": {
                "pos": torch.tensor([0., 0, 0.0]),
                "rot": torch.tensor([.0, 0.0, 0.0, 1.0]),
            },
            "bbq_sauce": {
                "pos": torch.tensor([0.7, -0.3, 0.14]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
            "box_base": {
                "pos": torch.tensor([0.5, 0.2, 0.1]),
                "rot": torch.tensor([0.0, 0.7071, 0.0, 0.7071]),
                "dof_pos": {"box_joint": 0.0},
            },
        },
        "robots": {
            "g1": {
                "pos": torch.tensor([0, 0, 0.]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0]),

            },
        },
    }
]
def main():

    global global_step, tot_success, tot_give_up
    handler_class = get_sim_env_class(SimType("isaaclab"))
    task = get_task("CloseBox")()
    robot = get_robot("g1_hand")
    robot_franka = get_robot("franka")
    camera = PinholeCameraCfg(data_types=["rgb", "depth"],pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))
    objects = [
        PrimitiveCubeCfg(
            name="cube",
            size=(0.1, 0.1,0.1),
            color=[1.0, 0.0, 0.0],
            physics=PhysicStateType.RIGIDBODY,
        ),
        PrimitiveSphereCfg(
            name="sphere",
            radius=0.1,
            color=[0.0, 0.0, 1.0],
            physics=PhysicStateType.RIGIDBODY,
        ),
        RigidObjCfg(
        name="bbq_sauce",
        scale=(2, 2, 2),
        physics=PhysicStateType.RIGIDBODY,
        usd_path="get_started/example_assets/bbq_sauce/usd/bbq_sauce.usd",
        urdf_path="get_started/example_assets/bbq_sauce/urdf/bbq_sauce.urdf",
        mjcf_path="get_started/example_assets/bbq_sauce/mjcf/bbq_sauce.xml",
        ),
        ArticulationObjCfg(
        name="box_base",
        fix_base_link=True,
        usd_path="get_started/example_assets/box_base/usd/box_base.usd",
        urdf_path="get_started/example_assets/box_base/urdf/box_base_unique.urdf",
        mjcf_path="get_started/example_assets/box_base/mjcf/box_base_unique.mjcf",
    ),

    ]
    scenario = ScenarioCfg(
        task=task,
        robots=[robot],
        scene=args.scene,
        # objects=objects,
        cameras=[camera],
        random=args.random,
        try_add_table=args.table,
        render=args.render,
        split=args.split,
        sim=args.sim,
        headless=args.headless,
        num_envs=args.num_envs,
    )

    env = handler_class(scenario)
    init_states, all_actions, all_states = get_traj(task, robot_franka, env.handler)
    init_states_franka = np.concatenate([init_states[0]['robots']['franka']['pos'],
                                        init_states[0]['robots']['franka']['rot']])

    init_states_franka_se3 = wxyz2xyzw(init_states_franka)

    print("Init states:", init_states[0]['robots']['franka'])
    init_states[0]["robots"]["g1"] = {
                "pos": torch.tensor([0, 0., 0.]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0]),

            }
    init_states_g1 = np.concatenate([init_states[0]['robots']['g1']['pos'],
                                        init_states[0]['robots']['g1']['rot']])
    init_states_g1_se3 = pp.SE3(np.array(init_states_g1, copy=False))
    # init_states[0]["objects"] = {
    #         "cube": {
    #             "pos": torch.tensor([0, 0., 0.]),
    #             "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
    #         },
    #         "sphere": {
    #             "pos": torch.tensor([0, -0., 0.]),
    #             "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
    #         },
    #         "bbq_sauce": {
    #             "pos": torch.tensor([0.7, -0.3, 0.14]),
    #             "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
    #         },
    #         "box_base": {
    #             "pos": torch.tensor([0.5, 0.2, 0.1]),
    #             "rot": torch.tensor([0.0, 0.7071, 0.0, 0.7071]),
    #             "dof_pos": {"box_joint": 0.0},
    #         },
    #     }
    # 环境复位
    # obs, extras = env.reset()
    # 准备录像保存器
    obs, extras = env.reset(states=init_states)
    obs_saver = ObsSaver(video_path=f"/home/xyc/RoboVerse/metasim/scripts/replay_g1/replay_{args.sim}.mp4")
    # obs_saver.add(obs)
    src_robot_urdf = URDF.load("/home/xyc/RoboVerse/roboverse_data/robots/franka_with_gripper_extension/urdf/franka_with_gripper_extensions.urdf")
    src_robot = get_pk_robot(src_robot_urdf)
    tgt_robot_urdf = URDF.load("/home/xyc/RoboVerse/roboverse_data/robots/g1/urdf/g1_29dof_with_hand.urdf")

    tgt_robot = get_pk_robot(tgt_robot_urdf)
    # load jason file

    meta_file = "/home/xyc/RoboVerse/roboverse_demo/demo_isaaclab/CloseBox-Level0/robot-franka/demo_0000/metadata.json"
    with open(meta_file, 'r') as f:
        metadata = json.load(f)
    robot_joint = np.array(metadata["joint_qpos"])
    # src_robot_urdf = URDF.load(robot_franka.urdf_path)
    # src_robot = get_pk_robot(src_robot_urdf)
    # all_actions00 = all_actions[0]
    # robot_joint = np.array([list(action["franka"]['dof_pos_target'].values()) for action in all_actions00])
    print("Robot joint shape:", robot_joint.shape)
    robot_joint_names = src_robot.joints.actuated_names
    robot_links_names = src_robot.links.names
    print("Robot joint names:", len(robot_joint_names))
    print("Robot links names:", robot_joint.shape)
    robot_pose = src_robot.forward_kinematics(robot_joint)

    # robot_pose = robot_pose - robot_pose[:,[0],:]
    # robot_init =
    robot_se3 = wxyz2xyzw(robot_pose)
    # robot_se3 = robot_se3 @ init_states_franka_se3[None, None, :] @ init_states_g1_se3.Inv()
    robot_se3 = robot_se3 @ init_states_franka_se3[None, None, :]
    robot_final_pose = xyzw2wxyz(robot_se3)

    # robot_se3[..., 0], robot_se3[...,1] = -robot_se3[..., 0], -robot_se3[..., 1]
    franka_g1 = {
    'right_hand_palm_link': 'panda_hand',
    'waist_yaw_link': 'panda_link0'
    }
    humanoid_hand_names = ['right_hand_palm_link',
                            'right_ankle_pitch_link',
                            'left_ankle_pitch_link']
    humanoid_hand_names = ['right_hand_palm_link',
                           'waist_yaw_link',
                           'right_ankle_pitch_link',
                           'left_ankle_pitch_link']
    target_link_names = ["right_hand_palm_link",
                        "left_rubber_hand",
                        "waist_yaw_link",
                        "right_ankle_pitch_link",
                        "left_ankle_pitch_link"]
    target_link_names = ["right_hand_palm_link"]
    right_ankle_pose = np.array([0, -0.16, -0.75, 1, 0, 0, 0])
    left_ankle_pose = np.array([0, 0.16, -0.75, 1, 0, 0, 0])
    right_hand_pose = np.array([0.26, -0.3, 0.20, 1, 0, 0, 0])
    left_hand_pose = np.array([0.30, 0.3, 0.15, 1, 0, 0, 0])
    waist_pose = np.array([0, 0, 0, 1, 0, 0, 0])

    inds = [robot_links_names.index(franka_g1[name]) for name in humanoid_hand_names[:2]]
    solutions = []
    for i in range(robot_pose.shape[0]):
        solution = pks.solve_ik_with_multiple_targets(
            robot=tgt_robot,
            target_link_names=target_link_names,
            # target_positions=np.array([robot_pose[i, inds[0], :3],
            #                         robot_pose[i, inds[1], :3]]),
            # target_wxyzs=np.array([robot_pose[i, inds[0], 3:],
            #                     robot_pose[i, inds[1], 3:]]),
            target_positions=np.array([robot_final_pose[i, inds[0], :3]]),
            target_wxyzs=np.array([robot_final_pose[i, inds[0], 3:]]),
            # target_positions=np.array([robot_se3[i, inds[0], :3],
            #                            left_hand_pose[:3],
            #                            robot_se3[i, inds[1], :3],
            #                            right_ankle_pose[:3],
            #                            left_ankle_pose[:3]]),
            # target_wxyzs=np.array([robot_se3[i, inds[0], 3:],
            #                        left_hand_pose[3:],
            #                        robot_se3[i, inds[1], 3:],
            #                        right_ankle_pose[3:],
            #                        left_ankle_pose[3:]]),
        )
        print([robot_final_pose[i, inds[0], :3]])
        print([robot_final_pose[i, inds[0], 3:]])
        solutions.append(solution)

    for step, solution in enumerate(solutions):
        robot_obj = scenario.robots[0]
        actions = [
            {
                "g1": {
                    "dof_pos_target": dict(zip(robot_obj.actuators.keys(), solution))
                }
            }
            for i_env in range(args.num_envs)
        ]
        # print(solution)
        # 执行动作
        obs, reward, success, time_out, extras = env.step(actions)

        # 第一步额外执行多步以稳定环境
        # if step == 0:
        #     for _ in range(50):
        #         obs, _, _, _, _ = env.step(actions)
        obs_saver.add(obs)

    obs_saver.save()

if __name__=="__main__":
    main()
