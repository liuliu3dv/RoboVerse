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
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])
NUM_SEED = 20

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

import multiprocessing as mp

from transforms3d import quaternions, affines

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
    urdf = URDF.load("./roboverse_data/robots/franka/urdf/franka_panda.urdf")
    return urdf


def get_pk_robot(urdf) -> Robot:
    return pk.Robot.from_urdf(urdf)


def run_task(task_name: str):
    # main() + task
    args.task = task_name
    main()


def main():

    handler_class = get_sim_env_class(SimType("isaaclab"))

    cur_task = args.task
    print(cur_task)
    task = get_task(cur_task)()
    robot_franka_dst = get_robot("franka")
    camera = PinholeCameraCfg(
        data_types=["rgb", "depth"],
        pos=(1.5, -1.5, 1.5),
        look_at=(0.0, 0.0, 0.0)
    )

    scenario = ScenarioCfg(
        task=task,
        robots=[robot_franka_dst],
        scene=args.scene,
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
    init_states, all_actions, all_states = get_traj(task, robot_franka_dst, env.handler)
    # pos = [-0.3400, -0.4600,  0.2400]
    # rot = [1., 0., 0., 0.]
    obs, extras = env.reset(states=init_states)
    obs_saver = ObsSaver(video_path=f"./humanoid_retargeting_v2/output/replay_franka_gt_from_isaaclab_calvin_actions/{cur_task}/replay_{args.sim}.mp4")

    src_robot_urdf = URDF.load("./roboverse_data/robots/franka_with_gripper_extension/urdf/franka_with_gripper_extensions.urdf")
    src_robot = get_pk_robot(src_robot_urdf)
    tgt_robot_urdf = URDF.load("./roboverse_data/robots/franka_with_gripper_extension/urdf/franka_with_gripper_extensions.urdf")
    tgt_robot = get_pk_robot(tgt_robot_urdf)

    robot_joint = all_actions[0]
    robot_joint_list = []
    for action in robot_joint:
        joint_angle = action['franka']['dof_pos_target']
        robot_joint_list.append(list(joint_angle.values()))
    robot_joint_array = np.array(robot_joint_list)

    # robot_pose = src_robot.forward_kinematics(robot_joint_array)  # forward obtain ee related to baselink, thus can directly ik
    # isaaclab return world coordinate, need transfer
    src_robot_links_names = ["basebody1_link", "basebody2_link", "cable1_link", "cable2_link", "camera_link", "fake_flange", "finger_link1_2", "finger_link1_3", "finger_link1_4", "finger_link2_2", "finger_link2_3", "finger_link2_4", "flange", "panda_grasptarget", "panda_hand", "panda_leftfinger", "panda_link0", "panda_link1", "panda_link2", "panda_link3", "panda_link4", "panda_link5", "panda_link6", "panda_link7", "panda_link8", "panda_rightfinger"]
    # src_robot_links_names = src_robot.links.names

    franka2franka = {'panda_hand': 'panda_hand', 'panda_link0': 'panda_link0'}
    target_link_names = ["panda_hand"]
    inds = [src_robot_links_names.index(franka2franka[name]) for name in target_link_names]

    robot_pose = np.load("/home/RoboVerse_Humanoid/humanoid_retargeting_v2/data/open_drawer_a_001.npy")
    robot_root_state = np.load("/home/RoboVerse_Humanoid/humanoid_retargeting_v2/data/open_drawer_a_001_root_state.npy")
    robot_pose = np.squeeze(robot_pose)  #

    robot_pose = robot_pose[:,:,0:7]
    robot_root_state = robot_root_state[:,0:7]


    root_poses = np.tile(robot_root_state, (64, 1))  # 复制成 (64, 7)
    root_positions = root_poses[:, :3]  # 根链接的位置 (64, 3)
    root_quaternions = root_poses[:, 3:]  # 根链接的旋转四元数 (64, 4)

    # 创建一个存储相对位姿的数组 (64, 26, 7)
    relative_poses = np.zeros((root_poses.shape[0], 26, 7))

    # 遍历每个其他的 link（从第1个到第26个）
    for link_index in range(robot_pose.shape[1]):
        # 获取当前 link 的世界位姿
        link_poses = robot_pose[:, link_index, :]  # 当前link的世界位姿 (64, 7)

        # 提取位置和旋转四元数
        link_positions = link_poses[:, :3]  # link的位置 (64, 3)
        link_quaternions = link_poses[:, 3:]  # link的旋转四元数 (64, 4)

        # 计算相对位置：link位置 - 根链接位置
        relative_positions = link_positions - root_positions

        # 计算相对旋转：link旋转四元数 * 根链接旋转四元数的共轭
        # 旋转四元数的逆变换是共轭操作
        relative_quaternions = np.zeros((root_quaternions.shape[0], 4))
        for i in range(root_quaternions.shape[0]):
            root_quaternions[i] = quaternions.qinverse(root_quaternions[i])
        for i in range(root_quaternions.shape[0]):
            relative_quaternions[i] = quaternions.qmult(root_quaternions[i], link_quaternions[i])


        # 将相对位姿存入对应位置
        relative_poses[:, link_index, :3] = relative_positions  # 位置
        relative_poses[:, link_index, 3:] = relative_quaternions  # 旋转

    solutions = []
    for i in range(robot_pose.shape[0]):
        solution = pks.solve_ik_with_multiple_targets(
            robot=tgt_robot,
            target_link_names=target_link_names,
            target_positions=np.array([robot_pose[i, inds[0], 0:3]]),
            target_wxyzs=np.array([robot_pose[i, inds[0], 3:7]]),
        )
        solutions.append(solution)

    # robot_arm_pos_list = []
    # robot_arm_name = None
    # solutions = robot_joint_list
    for solution in solutions:
        robot_obj = scenario.robots[0]
        actions = [
            {
                robot_obj.name: {
                    "dof_pos_target": dict(zip(robot_obj.actuators.keys(), solution))
                }
            }
            for _ in range(args.num_envs)
        ]
        obs, reward, success, time_out, extras = env.step(actions)
        obs_saver.add(obs)
        # body_names = obs.robots['franka'].body_names
        # body_state = obs.robots['franka'].body_state  # [x,y,z,wxyz]
        # robot_arm_pos_list.append(body_state)
        # robot_arm_name = body_names


    # data_list = [tensor.cpu().numpy() for tensor in robot_arm_pos_list]
    # data_array = np.array(data_list)
    # np.save('./humanoid_retargeting_v2/data/open_drawer_a_001.npy', data_array)

    # import json
    # with open('./humanoid_retargeting_v2/data/franka_body_names.json', 'w') as json_file:
    #     json.dump(robot_arm_name, json_file)

    obs_saver.save()
    env.close()


if __name__ == "__main__":
    task_list = [

        "open_drawer_a"
    ]

    mp.set_start_method("spawn", force=True)

    # multiprocess
    for task in task_list:
        p = mp.Process(target=run_task, args=(task,))
        p.start()
        p.join()  # wait until last task end
