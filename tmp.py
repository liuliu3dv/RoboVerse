"""This script is used to test the static scene."""

from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import os

import imageio.v3 as iio
import numpy as np
import rootutils
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


from metasim.cfg.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.cfg.robots.base_robot_cfg import BaseRobotCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import PhysicStateType, SimType
from metasim.sim import BaseSimHandler, EnvWrapper
from metasim.utils import configclass
from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import get_robot, get_sim_env_class, get_task


def get_actions(all_actions, action_idx: int, num_envs: int, robot: BaseRobotCfg):
    envs_actions = all_actions[:num_envs]
    actions = [
        env_actions[action_idx] if action_idx < len(env_actions) else env_actions[-1] for env_actions in envs_actions
    ]
    return actions


@configclass
class Args:
    task: str = "close_box"
    robot: str = "franka"
    sim: str = "isaaclab"
    num_envs: int = 1
    headless: bool = False


args = tyro.cli(Args)

# initialize scenario
scenario = ScenarioCfg(
    task=args.task,
    robots=[args.robot],
    try_add_table=False,
    sim=args.sim,
    headless=args.headless,
    num_envs=args.num_envs,
)

# add cameras
scenario.cameras = [PinholeCameraCfg(width=256, height=256, pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))]

# add objects
scenario.objects = [
    PrimitiveCubeCfg(
        name="cube",
        size=(0.1, 0.1, 0.1),
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


log.info(f"Using simulator: {args.sim}")
env_class = get_sim_env_class(SimType(args.sim))
env = env_class(scenario)

task = get_task(args.task)()
robot = get_robot(args.robot)
assert os.path.exists(task.traj_filepath), f"Trajectory file does not exist: {task.traj_filepath}"
init_states, all_actions, all_states = get_traj(task, robot, env.handler)


def save_obs(obs, step: int):
    rgb = next(iter(obs.cameras.values())).rgb[0].cpu().numpy()
    depth = next(iter(obs.cameras.values())).depth[0].cpu().numpy().squeeze(-1)
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
    iio.imwrite("tmp_rgb.png", rgb)
    np.savez(
        f"tmp_metadata_{step}.npz",
        rgb=rgb,
        depth=depth_normalized,
        depth_min=depth.min().item(),
        depth_max=depth.max().item(),
        intrinsics=next(iter(obs.cameras.values())).intrinsics[0].cpu().numpy(),
        cam_pos=next(iter(obs.cameras.values())).pos[0].cpu().numpy(),
        cam_quat_world=next(iter(obs.cameras.values())).quat_world[0].cpu().numpy(),
        cam_quat_ros=next(iter(obs.cameras.values())).quat_ros[0].cpu().numpy(),
        cam_quat_opengl=next(iter(obs.cameras.values())).quat_opengl[0].cpu().numpy(),
    )


obs, extras = env.reset(states=init_states)

for i in range(10):
    pos = (1, -0.5 + 0.1 * i, 1)
    look_at = (0.0, 0.0, 0.0)
    actions = get_actions(all_actions, i, 1, scenario.robots[0])
    env.handler.set_camera_pose(pos, look_at)
    obs, _, _, _, _ = env.step(actions)
    save_obs(obs, i)

breakpoint()
