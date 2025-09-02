"""This script is used to test the static scene."""

from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import math
import os
from typing import Literal

import rootutils
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


from metasim.constants import PhysicStateType, SimType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils import configclass
from metasim.utils.obs_utils import ObsSaver
from metasim.utils.setup_util import get_sim_handler_class


@configclass
class Args:
    """Arguments for the static scene."""

    robot: str = "franka"

    ## Handlers
    sim: Literal["isaaclab", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco","mjx"] = "mujoco"

    ## Others
    num_envs: int = 1
    headless: bool = False
    solver: Literal["curobo", "pyroki"] = "pyroki"

    def __post_init__(self):
        """Post-initialization configuration."""
        log.info(f"Args: {self}")


args = tyro.cli(Args)

if args.solver == "curobo":
    from curobo.types.math import Pose

    from metasim.utils.kinematics_utils import get_curobo_models

elif args.solver == "pyroki":
    from metasim.utils.kinematics_pyroki import get_pyroki_model

# initialize scenario
scenario = ScenarioCfg(
    robots=[args.robot],
    simulator=args.sim,
    headless=args.headless,
    num_envs=args.num_envs,
)

# add cameras
scenario.cameras = [PinholeCameraCfg(width=1024, height=1024, pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))]

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
        usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/usd/bbq_sauce.usd",
        urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/urdf/bbq_sauce.urdf",
        mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/mjcf/bbq_sauce.xml",
    ),
    ArticulationObjCfg(
        name="box_base",
        fix_base_link=True,
        usd_path="roboverse_data/assets/rlbench/close_box/box_base/usd/box_base.usd",
        urdf_path="roboverse_data/assets/rlbench/close_box/box_base/urdf/box_base_unique.urdf",
        mjcf_path="roboverse_data/assets/rlbench/close_box/box_base/mjcf/box_base_unique.mjcf",
    ),
]

log.info(f"Using simulator: {args.sim}")
env_class = get_sim_handler_class(SimType(args.sim))
env = env_class(scenario)
env.launch()

if args.robot == "franka":
    robot_dict = {
        "franka": {
            "pos": torch.tensor([0.0, 0.0, 0.0]),
            "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "dof_pos": {
                "panda_joint1": 0.0,
                "panda_joint2": -0.785398,
                "panda_joint3": 0.0,
                "panda_joint4": -2.356194,
                "panda_joint5": 0.0,
                "panda_joint6": 1.570796,
                "panda_joint7": 0.785398,
                "panda_finger_joint1": 0.04,
                "panda_finger_joint2": 0.04,
            },
        }
    }
elif args.robot == "kinova_gen3_robotiq_2f85":
    robot_dict = {
        "kinova_gen3_robotiq_2f85": {
            "pos": torch.tensor([0.0, 0.0, 0.0]),
            "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "dof_pos": {
                "joint_1": 0.0,
                "joint_2": math.pi / 6,
                "joint_3": 0.0,
                "joint_4": math.pi / 2,
                "joint_5": 0.0,
                "joint_6": 0.0,
                "joint_7": 0.0,
                "finger_joint": 0.0,
            },
        }
    }
else:
    robot_dict = {}

init_states = [
    {
        "objects": {
            "cube": {
                "pos": torch.tensor([0.3, -0.2, 0.05]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
            "sphere": {
                "pos": torch.tensor([0.4, -0.6, 0.05]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
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
        "robots": robot_dict,
    }
    for _ in range(args.num_envs)
]


robot = scenario.robots[0]

if args.solver == "curobo":
    *_, robot_ik = get_curobo_models(robot)
    curobo_n_dof = len(robot_ik.robot_config.cspace.joint_names)
    ee_n_dof = len(robot.gripper_open_q)
elif args.solver == "pyroki":
    robot_ik = get_pyroki_model(robot)

env.set_states(init_states)
obs = env.get_states(mode="dict")
os.makedirs("get_started/output", exist_ok=True)

## Main loop
obs_saver = ObsSaver(video_path=f"get_started/output/4_motion_planning_{args.sim}.mp4")
obs_saver.add(obs)

step = 0
robot_joint_limits = scenario.robots[0].joint_limits
for step in range(200):
    log.debug(f"Step {step}")
    states = env.get_states()

    if scenario.robots[0].name == "franka":
        x_target = 0.3 + 0.1 * (step / 100)
        y_target = 0.5 - 0.5 * (step / 100)
        z_target = 0.6 - 0.2 * (step / 100)
        # Randomly assign x/y/z target for each env
        ee_pos_target = torch.zeros((args.num_envs, 3), device="cuda:0")
        for i in range(args.num_envs):
            if i % 3 == 0:
                ee_pos_target[i] = torch.tensor([x_target, 0.0, 0.6], device="cuda:0")
            elif i % 3 == 1:
                ee_pos_target[i] = torch.tensor([0.3, y_target, 0.6], device="cuda:0")
            else:
                ee_pos_target[i] = torch.tensor([0.3, 0.0, z_target], device="cuda:0")
        ee_quat_target = torch.tensor(
            [[0.0, 1.0, 0.0, 0.0]] * args.num_envs,
            device="cuda:0",
        )
    elif scenario.robots[0].name == "kinova_gen3_robotiq_2f85":
        ee_pos_target = torch.tensor([[0.2 + 0.2 * (step / 100), 0.0, 0.4]], device="cuda:0").repeat(args.num_envs, 1)
        ee_quat_target = torch.tensor(
            [[0.0, 0.0, 1.0, 0.0]] * args.num_envs,
            device="cuda:0",
        )

    if args.solver == "curobo":
        curr_robot_q = states.robots[robot.name].joint_pos.cuda()
        seed_config = curr_robot_q[:, :curobo_n_dof].unsqueeze(1).tile([1, robot_ik._num_seeds, 1])
        result = robot_ik.solve_batch(Pose(ee_pos_target, ee_quat_target), seed_config=seed_config)
        ik_succ = result.success.squeeze(1)
        q = torch.zeros((scenario.num_envs, robot.num_joints), device="cuda:0")
        q[ik_succ, :curobo_n_dof] = result.solution[ik_succ, 0].clone()
        q[:, -ee_n_dof:] = 0.04
    elif args.solver == "pyroki":
        q_list = []
        for i_env in range(args.num_envs):
            q_tensor = robot_ik.solve_ik(ee_pos_target[i_env], ee_quat_target[i_env])
            q_list.append(q_tensor)
        q = torch.stack(q_list, dim=0)

    actions = [
        {robot.name: {"dof_pos_target": dict(zip(robot.actuators.keys(), q[i_env].tolist()))}}
        for i_env in range(scenario.num_envs)
    ]

    env.set_dof_targets(actions)
    env.simulate()
    obs = env.get_states(mode="dict")
    # obs, reward, success, time_out, extras = env.step(actions)

    obs_saver.add(obs)
    step += 1

obs_saver.save()
