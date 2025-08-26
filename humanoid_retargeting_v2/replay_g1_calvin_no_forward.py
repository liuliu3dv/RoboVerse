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
import os
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
    headless: bool = True
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


import torch
import numpy as np

def main():
    link = "panda_link3"
    # global global_step, tot_success, tot_give_up
    handler_class = get_sim_env_class(SimType("isaaclab"))
    # task = get_task("open_box")()

    cur_task = args.task
    task = get_task(cur_task)()

    robot_franka_src = get_robot("franka")
    robot_franka_dst = get_robot("g1_hand")

    # camera = PinholeCameraCfg(data_types=["rgb", "depth"],pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))
    camera = PinholeCameraCfg(data_types=["rgb", "depth"],pos=(2.0, -2.0, 2.0), look_at=(0.0, 0.0, 0.0))

    scenario = ScenarioCfg(
        # TODO retarget task
        task=task,
        robots=[robot_franka_dst],
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
        humanoid=True
    )

    env = handler_class(scenario)
    init_states, _, _ = get_traj(task, robot_franka_src, env.handler)

    tgt_robot_urdf = URDF.load(robot_franka_dst.urdf_path)
    tgt_robot = get_pk_robot(tgt_robot_urdf)

    src_robot_links_names = ["panda_hand", "panda_leftfinger", "panda_link0", "panda_link1", "panda_link2", "panda_link3", "panda_link4", "panda_link5", "panda_link6", "panda_link7", "panda_rightfinger"]
    franka_g1 = {
    'right_hand_palm_link': 'panda_link7',
    'waist_yaw_link': 'panda_link0',
    'right_elbow_link': link,
    "right_hand_index_1_link":"panda_leftfinger", #up
    "right_hand_middle_1_link":"panda_rightfinger", #down
    }

    target_link_names = ["right_hand_palm_link",
                        "left_hand_palm_link",
                        "waist_yaw_link",
                        "right_ankle_pitch_link",
                        "left_ankle_pitch_link",
                        "right_elbow_link",
                        "right_hand_index_1_link",
                        "right_hand_middle_1_link"
                        ]
    # right_hand_pose = np.array([0.0, -0.3, 0.0, 1, 0, 0, 0])
    # This is the relative pos related to waist
    left_hand_pose = np.array([0.0, 0.3, 0.0, 1, 0, 0, 0])
    right_ankle_pose = np.array([0, -0.16, -0.75, 1, 0, 0, 0])
    left_ankle_pose = np.array([0, 0.16, -0.75, 1, 0, 0, 0])
    waist_pose = np.array([0, 0, 0, 1, 0, 0, 0])

    inds = [src_robot_links_names.index(franka_g1[name]) for name in franka_g1.keys()]

    init_states[0]["robots"]["g1"] = {
                "pos": torch.tensor([-0.3400, -0.4600,  0.2400]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                # "dof_pos": init_dof_pos
            }

    # 环境复位
    # obs, extras = env.reset()
    # 准备录像保存器
    obs, extras = env.reset(states=init_states)
    obs_saver = ObsSaver(video_path=f"./humanoid_retargeting_v2/output/replay_g1_calvin_actions_debug/{cur_task}/replay_{args.sim}.mp4")

    # load isaaclab中的data
    # [T, 1, 11, 13]
    traj_data = np.load("/home/RoboVerse_Humanoid/humanoid_retargeting_v2/data/open_drawer_a_001.npy")
    traj_data = np.squeeze(traj_data)

    solutions = []
    for i in range(traj_data.shape[0]):  # iterate on 156 frames
        solution = pks.solve_ik_with_multiple_targets(
            robot=tgt_robot,
            target_link_names=target_link_names,
            target_positions=np.array([
                                       # right_hand_pose[:3],
                                       traj_data[i, inds[0], :3],
                                       left_hand_pose[:3],
                                       waist_pose[:3],
                                       right_ankle_pose[:3],
                                       left_ankle_pose[:3],
                                       traj_data[i, inds[2], :3],
                                       traj_data[i, inds[3], :3],
                                       traj_data[i, inds[4], :3],
                                       ]),
            target_wxyzs=np.array([
                                   # right_hand_pose[3:],
                                   traj_data[i, inds[0], 3:7],
                                   left_hand_pose[3:7],
                                   waist_pose[3:7],
                                   right_ankle_pose[3:7],
                                   left_ankle_pose[3:7],
                                   traj_data[i, inds[2], 3:7],
                                   traj_data[i, inds[3], 3:7],
                                   traj_data[i, inds[4], 3:7],
                                   ]),
        )
        # print(robot_pose[i, inds[0], 4:])
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

        obs, reward, success, time_out, extras = env.step(actions)
        obs_saver.add(obs)

    obs_saver.save()
    env.close()

# if __name__=="__main__":
#     main()

if __name__ == "__main__":

    # base_path = "/home/RoboVerse_Humanoid/roboverse_data/trajs/calvin/"
    # task_list = os.listdir(base_path)
    # task_list.sort()
    # mp.set_start_method("spawn", force=True)

    # print(task_list)
    # print(bk)

    # bug_tasks_list = []
    # # multiprocess
    # for task in task_list:
    #     try:
    #         p = mp.Process(target=run_task, args=(task,))
    #         p.start()
    #         p.join()  # wait until last task end
    #     except Exception as e:
    #         bug_tasks_list.append(task)
    #         log.error(f"Task {task} failed with error: {e}")
    # print(bug_tasks_list, file=open("calvin_error.txt", "w"))
    # # main()


#  ['close_drawer_a', '
#  lift_blue_block_drawer_a',
# 'lift_blue_block_slider_a',
# 'lift_blue_block_table_a',
# 'lift_pink_block_drawer_a',
# 'lift_pink_block_slider_a',
# 'lift_pink_block_table_a',
# 'lift_red_block_drawer_a',
# 'lift_red_block_slider_a',
#  'lift_red_block_table_a',
#  'move_slider_left_a',
#  'move_slider_right_a',
#  'open_drawer_a',
#  'place_in_drawer_a',
#  'place_in_slider_a',
#  'push_blue_block_left_a',
#  'push_blue_block_right_a',
#  'push_into_drawer_a',
#  'push_pink_block_left_a',
#  'push_pink_block_right_a',
#  'push_red_block_left_a',
#  'push_red_block_right_a',
#  'rotate_blue_block_left_a',
#  'rotate_blue_block_right_a',
#  'rotate_pink_block_left_a',
#  'rotate_pink_block_right_a',
#  'rotate_red_block_left_a',
#  'rotate_red_block_right_a',
#  'stack_block_a',
#  'unstack_block_a']
    task_list = [
        "open_drawer_a"
    ]

    mp.set_start_method("spawn", force=True)

    # multiprocess
    for task in task_list:
        p = mp.Process(target=run_task, args=(task,))
        p.start()
        p.join()  # wait until last task end
