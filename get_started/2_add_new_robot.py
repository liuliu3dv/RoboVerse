"""This script is used to test the static scene."""

from __future__ import annotations

from typing import Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import rootutils
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

import time

from get_started.utils import ObsSaver
from metasim.cfg.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.cfg.robots import FrankaShadowHandLeftCfg, FrankaShadowHandRightCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import PhysicStateType, SimType
from metasim.utils import configclass
from metasim.utils.setup_util import get_sim_env_class


@configclass
class Args:
    """Arguments for the static scene."""

    ## Handlers
    sim: Literal["isaaclab", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx"] = "isaaclab"

    ## Others
    num_envs: int = 1
    headless: bool = False

    def __post_init__(self):
        """Post-initialization configuration."""
        log.info(f"Args: {self}")


args = tyro.cli(Args)

# initialize scenario
scenario = ScenarioCfg(
    robots=[FrankaShadowHandLeftCfg(enabled_gravity=False), FrankaShadowHandRightCfg(enabled_gravity=False)],
    try_add_table=False,
    sim=args.sim,
    headless=args.headless,
    num_envs=args.num_envs,
)

# add cameras
scenario.cameras = [PinholeCameraCfg(name="camera_0", pos=(2.0, -1.0, 1.49), look_at=(0.0, -0.5, 0.89))]

# add objects
scenario.objects = [
    RigidObjCfg(
        name="cube",
        scale=(1, 1, 1),
        physics=PhysicStateType.RIGIDBODY,
        urdf_path="roboverse_data/assets/bidex/objects/urdf/cube_multicolor.urdf",
        usd_path="roboverse_data/assets/bidex/objects/usd/cube_multicolor.usd",
        default_density=500.0,
        use_vhacd=True,
    ),
]


log.info(f"Using simulator: {args.sim}")
env_class = get_sim_env_class(SimType(args.sim))
env = env_class(scenario)

init_states = [
    {
        "objects": {
            "cube": {
                "pos": torch.tensor([0.0, -0.37, 0.86]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
        },
        "robots": {
            "franka_shadow_left": {
                "pos": torch.tensor([0.0, -1.336, 0.0]),
                "rot": torch.tensor([0.7071, 0, 0, 0.7071]),
                "dof_pos": {
                    "panda_joint1": 0.0,
                    "panda_joint2": -0.785398,
                    "panda_joint3": 0.0,
                    "panda_joint4": -2.356194,
                    "panda_joint5": 0.0,
                    "panda_joint6": 3.1415926,
                    "panda_joint7": -2.356194,
                    "WRJ2": 0.0,
                    "WRJ1": 0.0,
                    "FFJ4": 0.0,
                    "FFJ3": 0.0,
                    "FFJ2": 0.0,
                    "FFJ1": 0.0,
                    "MFJ4": 0.0,
                    "MFJ3": 0.0,
                    "MFJ2": 0.0,
                    "MFJ1": 0.0,
                    "RFJ4": 0.0,
                    "RFJ3": 0.0,
                    "RFJ2": 0.0,
                    "RFJ1": 0.0,
                    "LFJ5": 0.0,
                    "LFJ4": 0.0,
                    "LFJ3": 0.0,
                    "LFJ2": 0.0,
                    "LFJ1": 0.0,
                    "THJ5": 0.0,
                    "THJ4": 0.0,
                    "THJ3": 0.0,
                    "THJ2": 0.0,
                    "THJ1": 0.0,
                },
            },
            "franka_shadow_right": {
                "pos": torch.tensor([0.0, 0.336, 0.0]),
                "rot": torch.tensor([0.7071, 0, 0, -0.7071]),
                "dof_pos": {
                    "panda_joint1": 0.0,
                    "panda_joint2": -0.785398,
                    "panda_joint3": 0.0,
                    "panda_joint4": -2.356194,
                    "panda_joint5": 0.0,
                    "panda_joint6": 3.1415928,
                    "panda_joint7": -2.356194,
                    "WRJ2": 0.0,
                    "WRJ1": 0.0,
                    "FFJ4": 0.0,
                    "FFJ3": 0.0,
                    "FFJ2": 0.0,
                    "FFJ1": 0.0,
                    "MFJ4": 0.0,
                    "MFJ3": 0.0,
                    "MFJ2": 0.0,
                    "MFJ1": 0.0,
                    "RFJ4": 0.0,
                    "RFJ3": 0.0,
                    "RFJ2": 0.0,
                    "RFJ1": 0.0,
                    "LFJ5": 0.0,
                    "LFJ4": 0.0,
                    "LFJ3": 0.0,
                    "LFJ2": 0.0,
                    "LFJ1": 0.0,
                    "THJ5": 0.0,
                    "THJ4": 0.0,
                    "THJ3": 0.0,
                    "THJ2": 0.0,
                    "THJ1": 0.0,
                },
            },
        },
    }
]
num_robo_dof = {}
for robot in scenario.robots:
    num_robo_dof[robot.name] = robot.num_joints
dof_num = 0
for robot in scenario.robots:
    dof_num += num_robo_dof[robot.name]
joint_low_limits = {}
joint_high_limits = {}
for robot in scenario.robots:
    joint_low_limits[robot.name] = torch.zeros(num_robo_dof[robot.name], device="cuda:0")
    joint_high_limits[robot.name] = torch.zeros(num_robo_dof[robot.name], device="cuda:0")
    for i, joint_name in enumerate(robot.joint_limits.keys()):
        if robot.actuators[joint_name].fully_actuated:
            joint_low_limits[robot.name][i] = robot.joint_limits[joint_name][0]
            joint_high_limits[robot.name][i] = robot.joint_limits[joint_name][1]
            i += 1
obs, extras = env.reset(states=init_states)
for robot in scenario.robots:
    robot.update_state(obs)
os.makedirs("get_started/output", exist_ok=True)


## Main loop
obs_saver = ObsSaver(video_path=f"get_started/output/2_add_new_robot_{args.sim}.mp4")
obs_saver.add(obs)
start_time = time.time()
step = 0
for _ in range(100):
    log.debug(f"Step {step}")
    # actions = [
    #     {
    #         robot.name: {
    #             "dof_pos_target": {
    #                 joint_name: (
    #                     torch.rand(1).item() * (robot.joint_limits[joint_name][1] - robot.joint_limits[joint_name][0])
    #                     + robot.joint_limits[joint_name][0]
    #                 )
    #                 for joint_name in robot.joint_limits.keys()
    #                 if robot.actuators[joint_name].fully_actuated
    #             }
    #         }
    #         for robot in scenario.robots
    #     }
    #     for _ in range(scenario.num_envs)
    # ]
    left_pos_err = torch.zeros((args.num_envs, 3), device="cuda:0")
    left_pos_err[:, 1] = 0.0
    left_pos_err[:, 2] = 0.0
    left_rot_err = torch.zeros((args.num_envs, 3), device="cuda:0")
    left_dpose = torch.cat([left_pos_err, left_rot_err], -1).unsqueeze(-1)
    left_targets = scenario.robots[0].control_arm_ik(left_dpose, args.num_envs, "cuda:0")

    right_pos_err = torch.zeros((args.num_envs, 3), device="cuda:0")
    right_pos_err[:, 1] = 0.0
    right_pos_err[:, 2] = 0.0
    right_rot_err = torch.zeros((args.num_envs, 3), device="cuda:0")
    right_dpose = torch.cat([right_pos_err, right_rot_err], -1).unsqueeze(-1)
    right_targets = scenario.robots[1].control_arm_ik(right_dpose, args.num_envs, "cuda:0")

    left_ft_pos_err = torch.zeros((args.num_envs, 5, 3), device="cuda:0")
    left_ft_pos_err[..., 1] = 0.03
    right_ft_pos_err = torch.zeros((args.num_envs, 5, 3), device="cuda:0")
    right_ft_pos_err[..., 1] = -0.03
    left_ft_rot_err = torch.zeros((args.num_envs, 5, 3), device="cuda:0")
    right_ft_rot_err = torch.zeros((args.num_envs, 5, 3), device="cuda:0")
    left_dof_pos = scenario.robots[0].control_hand_ik(left_ft_pos_err, left_ft_rot_err)
    right_dof_pos = scenario.robots[1].control_hand_ik(right_ft_pos_err, right_ft_rot_err)

    actions = torch.zeros((args.num_envs, dof_num), device="cuda:0")
    num_dof = 0
    for robot in scenario.robots:
        arm_dof_idx = [i + num_dof for i in robot.arm_dof_idx]
        hand_dof_idx = [i + num_dof for i in robot.hand_dof_idx]
        actions[:, arm_dof_idx] = left_targets if robot.name == "franka_shadow_left" else right_targets
        actions[:, hand_dof_idx] = left_dof_pos if robot.name == "franka_shadow_left" else right_dof_pos
        num_dof += num_robo_dof[robot.name]
    obs, reward, success, time_out, extras = env.step(actions)
    for robot in scenario.robots:
        robot.update_state(obs)
    obs_saver.add(obs)
    step += 1
    if step % 10 == 0:
        log.info(f"Step {step}, Time Elapsed: {time.time() - start_time:.2f} seconds")
        start_time = time.time()

obs_saver.save()
