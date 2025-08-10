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

import multiprocessing as mp

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

    # cur_task = args.task
    cur_task = "close_fridge"

    task = get_task(cur_task)()
    robot_g1 = get_robot("g1")
    robot_franka = get_robot("franka")
    camera = PinholeCameraCfg(data_types=["rgb", "depth"],pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))

    scenario = ScenarioCfg(
        task=task,
        robots=[robot_g1],
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

    # print(task.traj_path)
    # print(k)
    env = handler_class(scenario)
    init_states, all_actions, all_states = get_traj(task, robot_franka, env.handler)

    src_robot_urdf = URDF.load("./roboverse_data/robots/franka_with_gripper_extension/urdf/franka_with_gripper_extensions.urdf")
    src_robot = get_pk_robot(src_robot_urdf)
    tgt_robot_urdf = URDF.load("./roboverse_data/robots/g1/urdf/g1_29dof_lock_waist_rev_1_0_modified.urdf")
    tgt_robot = get_pk_robot(tgt_robot_urdf)
    src_robot.joints.actuated_names
    robot_joint = all_actions[0]
    robot_joint_list = []
    for index, action in enumerate(robot_joint):
        joint_angle = action['franka']['dof_pos_target']
        robot_joint_list.append(list(joint_angle.values()))
    robot_joint_array = np.array(robot_joint_list)

    robot_pose = src_robot.forward_kinematics(robot_joint_array)  # [247, 26, 7] # robotic arm has 26 links
    robot_links_names = src_robot.links.names

    init_states[0]["robots"]["g1"] = {
                "pos": torch.tensor([0, 0., 0.2]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
    }

    # 准备录像保存器
    obs, extras = env.reset(states=init_states)
    obs_saver = ObsSaver(video_path=f"./humanoid_retargeting/replay_g1_10_actions_modified_ik/{cur_task}/replay_{args.sim}.mp4")

    franka_g1 = {
    'right_rubber_hand': 'panda_hand',
    'waist_yaw_link': 'panda_link0'
    }

    humanoid_hand_names = ['right_rubber_hand',
                           'waist_yaw_link']
    target_link_names = ["right_rubber_hand",
                        "waist_yaw_link"]

    right_ankle_pose = np.array([0, -0.16, -0.75, 1, 0, 0, 0])
    # left_ankle_pose = np.array([0, 0.16, -0.75, 1, 0, 0, 0])
    # right_hand_pose = np.array([0.26, -0.3, 0.20, 1, 0, 0, 0])
    # left_hand_pose = np.array([0.30, 0.3, 0.15, 1, 0, 0, 0])
    waist_pose = np.array([0, 0, 0, 1, 0, 0, 0])


    inds = [robot_links_names.index(franka_g1[name]) for name in humanoid_hand_names[:2]]


    # solutions = []
    # for i in range(robot_pose.shape[0]):  # iterate on 156 frames
    #     solution = pks.solve_ik_with_multiple_targets(
    #         robot=tgt_robot,
    #         target_link_names=target_link_names,

    #         target_positions=np.array([robot_pose[i, inds[0], 4:],
    #                                     waist_pose[:3]]),
    #         target_wxyzs=np.array([robot_pose[i, inds[0], :4],
    #                                waist_pose[3:]]),
    #     )

    #     solutions.append(solution)

    # 初始化 previous_solution 以便平滑
    batch_size = args.num_envs
    prev_solution = None

    solutions = []
    for i in range(robot_pose.shape[0]):  # iterate on trajectory frames
        # 当前 humanoid 关节状态（上一次的解）用于保持未约束部位
        if prev_solution is None:
            current_joint_values = np.zeros((batch_size, len(tgt_robot.joints.names)))
        else:
            current_joint_values = prev_solution

        # 前向运动学获取当前 humanoid 所有关节的 link pose
        current_links_pose = tgt_robot.forward_kinematics(current_joint_values)
        link_names = tgt_robot.links.names

        # 获取当前 humanoid 的腰位置
        waist_idx = link_names.index("waist_yaw_link")
        waist_pos = current_links_pose[waist_idx, 4:]
        waist_rot = current_links_pose[waist_idx, :4]

        # 目标 link 名称：只锁右手+腰
        target_link_names = ["right_rubber_hand", "waist_yaw_link"]

        # 目标位置与姿态
        target_positions = np.array([
            robot_pose[i, inds[0], 4:],  # franka ee 对应 humanoid 右手
            waist_pos                    # 保持腰部原位置
        ])
        target_wxyzs = np.array([
            robot_pose[i, inds[0], :4],  # franka ee 对应 humanoid 右手旋转
            waist_rot                    # 保持腰部原旋转
        ])

        # IK 解算
        solution = pks.solve_ik_with_multiple_targets(
            robot=tgt_robot,
            target_link_names=target_link_names,
            target_positions=target_positions,
            target_wxyzs=target_wxyzs,
            # 如果支持权重，可加：
            target_weights=np.array([1.0, 0.3])
        )

        # 平滑关节输出
        if prev_solution is None:
            smoothed_solution = solution
        else:
            alpha = 0.2
            smoothed_solution = alpha * np.array(solution) + (1 - alpha) * np.array(prev_solution)

        prev_solution = smoothed_solution
        solutions.append(smoothed_solution)

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

        # 执行动作
        obs, reward, success, time_out, extras = env.step(actions)
        obs_saver.add(obs)

    obs_saver.save()
    env.close()

if __name__ == "__main__":
    # task_list = [
    #     # "basketball_in_hoop",
    #     # # # "beat_the_buzz",  # has bug
    #     # "block_pyramid",
    #     # "change_clock",
    #     "close_fridge",
    #     # "empty_dishwasher",
    #     # "insert_onto_square_peg",
    #     # "lamp_on",
    #     # "light_bulb_in",
    #     # "meat_on_grill",
    #     # "open_box",
    #     # # # "reach_and_drag" # bug
    #     # # # "take_cup_out_from_cabinet"  # AttributeError: 'RigidObject' object has no attribute '_data'. Did you mean: 'data'?
    #     # "play_jenga"
    # ]

    # mp.set_start_method("spawn", force=True)

    # # multiprocess
    # for task in task_list:
    #     p = mp.Process(target=run_task, args=(task,))
    #     p.start()
    #     p.join()  # wait until last task end

    main()
