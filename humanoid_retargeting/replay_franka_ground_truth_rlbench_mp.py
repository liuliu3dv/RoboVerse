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
    obs, extras = env.reset(states=init_states)
    obs_saver = ObsSaver(video_path=f"./humanoid_retargeting/replay_franka_ground_truth_actions/replay_{cur_task}_{args.sim}.mp4")

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

    # robot_pose = src_robot.forward_kinematics(robot_joint_array)
    src_robot_links_names = src_robot.links.names
    franka2franka = {'panda_hand': 'panda_hand', 'panda_link0': 'panda_link0'}
    target_link_names = ["panda_hand"]
    inds = [src_robot_links_names.index(franka2franka[name]) for name in target_link_names]

    # solutions = []
    # for i in range(robot_pose.shape[0]):
    #     solution = pks.solve_ik_with_multiple_targets(
    #         robot=tgt_robot,
    #         target_link_names=target_link_names,
    #         target_positions=np.array([robot_pose[i, inds[0], 4:]]),
    #         target_wxyzs=np.array([robot_pose[i, inds[0], 0:4]]),
    #     )
    #     solutions.append(solution)

    solutions = robot_joint_list
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

    obs_saver.save()
    env.close()


if __name__ == "__main__":
    # task_list = [
    #     # "basketball_in_hoop",
    #     # "beat_the_buzz",  # has bug
    #     # "block_pyramid",
    #     # "change_clock",
    #     # "close_fridge",
    #     # "empty_dishwasher",
    #     # "insert_onto_square_peg",
    #     # "lamp_on",
    #     # "light_bulb_in",
    #     # "meat_on_grill",
    #     # "open_box",
    #     # "reach_and_drag" # bug
    #     # "take_cup_out_from_cabinet"  # AttributeError: 'RigidObject' object has no attribute '_data'. Did you mean: 'data'?
    #     "play_jenga"
    # ]

    base_path = "./roboverse_data/trajs/rlbench/"
    task_list = os.listdir(base_path)
    task_list.sort()
    task_list = task_list[1:] # remove readme.md

    mp.set_start_method("spawn", force=True)

    bug_tasks_list = []
    # multiprocess
    for task in task_list:
        try:
            p = mp.Process(target=run_task, args=(task,))
            p.start()
            p.join()  # wait until last task end
        except Exception as e:
            bug_tasks_list.append(task)
            log.error(f"Task {task} failed with error: {e}")
    print(bug_tasks_list, file=open("error.txt", "w"))
    # main()
