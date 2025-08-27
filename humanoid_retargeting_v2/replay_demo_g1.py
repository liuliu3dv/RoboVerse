from __future__ import annotations

import logging
from loguru import logger as log
from rich.logging import RichHandler
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

import time
from typing import Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import imageio as iio
import numpy as np
import tyro
from loguru import logger as log
from numpy.typing import NDArray
from rich.logging import RichHandler

import rootutils
rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

from torchvision.utils import make_grid, save_image
from tyro import MISSING

from metasim.cfg.randomization import RandomizationCfg
from metasim.cfg.render import RenderCfg
from metasim.cfg.robots.base_robot_cfg import BaseRobotCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import SimType
from metasim.sim import HybridSimEnv
from metasim.utils import configclass
from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import get_sim_env_class, get_task, get_robot
from metasim.utils.state import TensorState
from metasim.utils import is_camel_case, is_snake_case, to_camel_case, to_snake_case


import pyroki as pk
from glob import glob
from pyroki import Robot
from yourdfpy import URDF
import importlib
import torch
import third_party.pyroki.examples.pyroki_snippets as pks

logging.addLevelName(5, "TRACE")
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


@configclass
class Args:
    task: str = MISSING
    robot: str = "franka"
    scene: str | None = None
    render: RenderCfg = RenderCfg()
    random: RandomizationCfg = RandomizationCfg()

    ## Handlers
    sim: Literal["isaaclab", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx"] = "isaaclab"
    renderer: Literal["isaaclab", "isaacgym", "genesis", "pybullet", "mujoco", "sapien2", "sapien3"] | None = None

    ## Others
    num_envs: int = 1
    try_add_table: bool = True
    object_states: bool = False
    split: Literal["train", "val", "test", "all"] = "all"
    headless: bool = False

    ## Only in args
    save_image_dir: str | None = "tmp"
    save_video_path: str | None = None
    stop_on_runout: bool = False

    def __post_init__(self):
        log.info(f"Args: {self}")


args = tyro.cli(Args)


###########################################################
## Utils
###########################################################
def get_actions(all_actions, action_idx: int, num_envs: int, robot: BaseRobotCfg):
    envs_actions = all_actions[:num_envs]
    actions = [
        env_actions[action_idx] if action_idx < len(env_actions) else env_actions[-1] for env_actions in envs_actions
    ]
    return actions


def get_states(all_states, action_idx: int, num_envs: int):
    envs_states = all_states[:num_envs]
    states = [env_states[action_idx] if action_idx < len(env_states) else env_states[-1] for env_states in envs_states]
    return states


def get_runout(all_actions, action_idx: int):
    runout = all([action_idx >= len(all_actions[i]) for i in range(len(all_actions))])
    return runout

def get_pk_robot(urdf) -> Robot:
    return pk.Robot.from_urdf(urdf)

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

class ObsSaver:
    """Save the observations to images or videos."""

    def __init__(self, image_dir: str | None = None, video_path: str | None = None):
        """Initialize the ObsSaver."""
        self.image_dir = image_dir
        self.video_path = video_path
        self.images: list[NDArray] = []

        self.image_idx = 0

    def add(self, state: TensorState):
        """Add the observation to the list."""
        if self.image_dir is None and self.video_path is None:
            return

        try:
            rgb_data = next(iter(state.cameras.values())).rgb
            image = make_grid(rgb_data.permute(0, 3, 1, 2) / 255, nrow=int(rgb_data.shape[0] ** 0.5))  # (C, H, W)
        except Exception as e:
            log.error(f"Error adding observation: {e}")
            return

        if self.image_dir is not None:
            os.makedirs(self.image_dir, exist_ok=True)
            save_image(image, os.path.join(self.image_dir, f"rgb_{self.image_idx:04d}.png"))
            self.image_idx += 1

        image = image.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
        image = (image * 255).astype(np.uint8)
        self.images.append(image)

    def save(self):
        """Save the images or videos."""
        if self.video_path is not None and self.images:
            log.info(f"Saving video of {len(self.images)} frames to {self.video_path}")
            os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
            iio.mimsave(self.video_path, self.images, fps=30)


###########################################################
## Main
###########################################################
def main():
    camera = PinholeCameraCfg(pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))
    link = "panda_link3"
    cur_task = "close_box"
    task = get_task(cur_task)()

    robot_franka_src = get_robot("franka")
    robot_franka_dst = get_robot("g1_hand")


    scenario = ScenarioCfg(
        task=task,
        robots=[robot_franka_dst],
        scene=args.scene,
        cameras=[camera],
        random=args.random,
        render=args.render,
        sim=args.sim,
        renderer=args.renderer,
        num_envs=args.num_envs,
        try_add_table=args.try_add_table,
        object_states=args.object_states,
        split=args.split,
        headless=args.headless,
    )

    num_envs: int = scenario.num_envs

    tic = time.time()
    if scenario.renderer is None:
        log.info(f"Using simulator: {scenario.sim}")
        env_class = get_sim_env_class(SimType(scenario.sim))
        env = env_class(scenario)
    else:
        log.info(f"Using simulator: {scenario.sim}, renderer: {scenario.renderer}")
        env_class_render = get_sim_env_class(SimType(scenario.renderer))
        env_render = env_class_render(scenario)  # Isaaclab must launch right after import
        env_class_physics = get_sim_env_class(SimType(scenario.sim))
        env_physics = env_class_physics(scenario)  # Isaaclab must launch right after import
        env = HybridSimEnv(env_physics, env_render)
    toc = time.time()
    log.trace(f"Time to launch: {toc - tic:.2f}s")

    ## Data
    tic = time.time()

    # assert os.path.exists(scenario.task.traj_filepath), (
    #     f"Trajectory file: {scenario.task.traj_filepath} does not exist."
    # )
    # init_states, all_actions, all_states = get_traj(
    #     scenario.task, scenario.robots[0], env.handler
    # )  # XXX: only support one robot

    init_states, all_actions, all_states = get_traj(task, robot_franka_src, env.handler)

    toc = time.time()
    log.trace(f"Time to load data: {toc - tic:.2f}s")


    ## Retargeting from Franka to G1
    src_robot_urdf = URDF.load("./roboverse_data/robots/franka_with_gripper_extension/urdf/franka_with_gripper_extensions.urdf")
    src_robot = get_pk_robot(src_robot_urdf)
    tgt_robot_urdf = URDF.load(robot_franka_dst.urdf_path)
    tgt_robot = get_pk_robot(tgt_robot_urdf)
    robot_joint = all_actions[0]

    robot_joint_list = []
    for index, action in enumerate(robot_joint):
        joint_angle = action['franka']['dof_pos_target']
        robot_joint_list.append(list(joint_angle.values()))
    robot_joint_array = np.array(robot_joint_list)
    robot_pose = src_robot.forward_kinematics(robot_joint_array)  # [247, 26, 7]
    src_robot_links_names = src_robot.links.names
    franka_g1 = {
    'right_hand_palm_link': 'panda_link7',
    'waist_yaw_link': 'panda_link0',
    'right_elbow_link': link,
    "right_hand_index_1_link":"finger_link1_4", #up
    "right_hand_middle_1_link":"finger_link2_4", #down
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
    right_hand_pose = np.array([0.26, -0.3, 0.20, 1, 0, 0, 0])
    left_hand_pose = np.array([0.0, 0.3, 0.0, 1, 0, 0, 0])
    right_ankle_pose = np.array([0, -0.16, -0.75, 1, 0, 0, 0])
    left_ankle_pose = np.array([0, 0.16, -0.75, 1, 0, 0, 0])
    waist_pose = np.array([0, 0, 0, 1, 0, 0, 0])
    inds = [src_robot_links_names.index(franka_g1[name]) for name in franka_g1.keys()]
    # print("inds:", inds)
    solutions = []
    for i in range(robot_pose.shape[0]):  # iterate on 156 frames
        solution = pks.solve_ik_with_multiple_targets(
            robot=tgt_robot,
            target_link_names=target_link_names,
            target_positions=np.array([
                                    #    right_hand_pose[:3],
                                       robot_pose[i, inds[0], 4:],
                                       left_hand_pose[:3],
                                       waist_pose[:3],
                                       right_ankle_pose[:3],
                                       left_ankle_pose[:3],
                                       robot_pose[i, inds[2], 4:],
                                       robot_pose[i, inds[3], 4:],
                                       robot_pose[i, inds[4], 4:],
                                       ]),
            target_wxyzs=np.array([
                                # right_hand_pose[3:],
                                   robot_pose[i, inds[0], :4],
                                   left_hand_pose[3:],
                                   waist_pose[3:],
                                   right_ankle_pose[3:],
                                   left_ankle_pose[3:],
                                   robot_pose[i, inds[2], :4],
                                   robot_pose[i, inds[3], :4],
                                   robot_pose[i, inds[4], :4],
                                   ]),
        )
        # print(robot_pose[i, inds[0], 4:])
        solutions.append(solution)
    init_dof_pos = {}
    for i in range(len(tgt_robot.joints.actuated_names)):
        init_dof_pos[tgt_robot.joints.actuated_names[i]] = solutions[0][i]

    init_states[0]["robots"]["g1"] = {
                "pos": torch.tensor([-0.263, -0.0053, -0.]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0]),
                "dof_pos": init_dof_pos
            }


    ########################################################
    ## Main
    ########################################################

    obs_saver = ObsSaver(image_dir=args.save_image_dir, video_path=args.save_video_path)

    ## Reset before first step
    tic = time.time()
    obs, extras = env.reset(states=init_states[:num_envs])
    toc = time.time()
    log.trace(f"Time to reset: {toc - tic:.2f}s")
    obs_saver.add(obs)

    ## Main loop
    # step = 0
    # while True:
    #     log.debug(f"Step {step}")
    #     tic = time.time()
    #     if scenario.object_states:
    #         ## TODO: merge states replay into env.step function
    #         if all_states is None:
    #             raise ValueError("All states are None, please check the trajectory file")
    #         states = get_states(all_states, step, num_envs)
    #         env.handler.set_states(states)
    #         env.handler.refresh_render()
    #         obs = env.handler.get_states()

    #         ## XXX: hack
    #         success = env.handler.task.checker.check(env.handler)
    #         if success.any():
    #             log.info(f"Env {success.nonzero().squeeze(-1).tolist()} succeeded!")
    #         if success.all():
    #             break

    #     else:
    #         actions = get_actions(all_actions, step, num_envs, scenario.robots[0])
    #         obs, reward, success, time_out, extras = env.step(actions)

    #         if success.any():
    #             log.info(f"Env {success.nonzero().squeeze(-1).tolist()} succeeded!")

    #         if time_out.any():
    #             log.info(f"Env {time_out.nonzero().squeeze(-1).tolist()} timed out!")

    #         if success.all() or time_out.all():
    #             break

    #     toc = time.time()
    #     log.trace(f"Time to step: {toc - tic:.2f}s")

    #     tic = time.time()
    #     obs_saver.add(obs)
    #     toc = time.time()
    #     log.trace(f"Time to save obs: {toc - tic:.2f}s")
    #     step += 1

    #     if args.stop_on_runout and get_runout(all_actions, step):
    #         log.info("Run out of actions, stopping")
    #         break

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
        obs_saver.add(obs)

    obs_saver.save()
    env.close()


if __name__ == "__main__":
    main()
