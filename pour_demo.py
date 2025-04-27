"""This script is used to test the static scene."""

from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import math
from typing import Literal

import rootutils
import torch
import tyro
from curobo.types.math import Pose
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


from metasim.cfg.objects import FluidObjCfg, PrimitiveCubeCfg, PrimitiveFrameCfg, RigidObjCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.constants import PhysicStateType, SimType
from metasim.utils import configclass
from metasim.utils.kinematics_utils import get_curobo_models
from metasim.utils.math import quat_apply, quat_inv, quat_mul
from metasim.utils.setup_util import get_sim_env_class


@configclass
class Args:
    """Arguments for the static scene."""

    robot: str = "franka"
    sim: Literal["isaaclab", "isaacgym", "genesis", "pyrep", "pybullet", "sapien", "sapien3", "mujoco", "blender"] = (
        "isaaclab"
    )
    num_envs: int = 1
    headless: bool = False

    def __post_init__(self):
        """Post-initialization configuration."""
        log.info(f"Args: {self}")


args = tyro.cli(Args)
scenario = ScenarioCfg(
    robot=args.robot,
    try_add_table=False,
    sim=args.sim,
    headless=args.headless,
    num_envs=args.num_envs,
)
scenario.objects = [
    PrimitiveCubeCfg(
        name="table",
        size=(0.7366, 1.4732, 0.0254),
        color=(0.8, 0.4, 0.2),
        physics=PhysicStateType.GEOM,
    ),
    PrimitiveCubeCfg(
        name="wall",
        size=(0.05, 2.0, 2.0),
        color=(1.0, 1.0, 1.0),
        physics=PhysicStateType.GEOM,
    ),
    RigidObjCfg(
        name="cup1",
        usd_path="/home/fs/cod/IsaacLabPouringExtension/Tall_Glass_5.usd",
        physics=PhysicStateType.RIGIDBODY,
        scale=0.01,
        default_position=(0.3, 0.5, 0.6943 + 0.0127),
    ),
    FluidObjCfg(
        name="water",
        numParticlesX=10,
        numParticlesY=10,
        numParticlesZ=15,
        density=0.0,
        particle_mass=0.0001,
        particleSpacing=0.005,
        viscosity=0.1,
        default_position=(0.3, 0.3, 0.6943 + 0.0127 + 0.03),
    ),
    RigidObjCfg(
        name="cup2",
        usd_path="/home/fs/cod/IsaacLabPouringExtension/Tall_Glass_5.usd",
        physics=PhysicStateType.RIGIDBODY,
        scale=0.01,
        default_position=(0.3, 0.3, 0.6943 + 0.0127),
    ),
    PrimitiveFrameCfg(name="frame", scale=0.1, base_link=("kinova_gen3_robotiq_2f85", "end_effector_link")),
]

env_class = get_sim_env_class(SimType(args.sim))
env = env_class(scenario)

init_states = [
    {
        "objects": {
            "table": {
                "pos": torch.tensor([0.3683, 0.1234, 0.6943]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
            "wall": {
                "pos": torch.tensor([0.7616, 0.1234, 1.0]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
            "cup1": {},
            "cup2": {},
        },
        "robots": {
            "kinova_gen3_robotiq_2f85": {
                "pos": torch.tensor([-0.05, 0.05, 1.6891]),
                "rot": torch.tensor([0.2706, -0.65328, -0.65328, -0.2706]),
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
            },
        },
    }
    for _ in range(args.num_envs)
]


robot = scenario.robot
*_, robot_ik = get_curobo_models(robot)
curobo_n_dof = len(robot_ik.robot_config.cspace.joint_names)
ee_n_dof = len(robot.gripper_release_q)

env.reset(states=init_states)


def reach_target_try(ee_pos: torch.Tensor, ee_quat: torch.Tensor):
    """Reach the target position and orientation."""
    states = env.handler.get_states()
    curr_robot_q = states.robots[robot.name].joint_pos.cuda()

    seed_config = curr_robot_q[:, :curobo_n_dof].unsqueeze(1).tile([1, robot_ik._num_seeds, 1])
    ee_pos_target_global = torch.tensor(ee_pos, device="cuda").repeat(args.num_envs, 1)
    ee_quat_target_global = torch.tensor(ee_quat, device="cuda").repeat(args.num_envs, 1)
    robot_base_pos = states.robots[robot.name].root_state[:, :3].cuda()
    robot_base_quat = states.robots[robot.name].root_state[:, 3:7].cuda()
    ee_pos_target_local = quat_apply(
        quat_inv(robot_base_quat),
        ee_pos_target_global - robot_base_pos,
    )
    ee_quat_target_local = quat_mul(quat_inv(robot_base_quat), ee_quat_target_global)
    result = robot_ik.solve_batch(Pose(ee_pos_target_local, ee_quat_target_local), seed_config=seed_config)

    q = torch.zeros((scenario.num_envs, robot.num_joints), device="cuda")
    ik_succ = result.success.squeeze(1)
    q[ik_succ, :curobo_n_dof] = result.solution[ik_succ, 0].clone()
    q[:, -ee_n_dof:] = 0.04
    actions = [
        {"dof_pos_target": dict(zip(robot.actuators.keys(), q[i_env].tolist()))} for i_env in range(scenario.num_envs)
    ]

    env.step(actions)


def reach_target_dedicated(ee_pos: torch.Tensor, ee_quat: torch.Tensor, atol: float = 0.05):
    """Reach the target position and orientation."""
    states = env.handler.get_states()
    ee_idx = states.robots[robot.name].body_names.index(env.handler.robot.ee_body_name)
    cur_ee_pos = states.robots[robot.name].body_state[:, ee_idx, :3]
    cur_ee_quat = states.robots[robot.name].body_state[:, ee_idx, 3:7]

    while not torch.allclose(cur_ee_pos, ee_pos, atol=atol) or not torch.allclose(cur_ee_quat, ee_quat, atol=atol):
        log.debug(f"Cur pos: {cur_ee_pos}")
        log.debug(f"Cur quat: {cur_ee_quat}")
        log.debug(f"Target pos: {ee_pos}")
        log.debug(f"Target quat: {ee_quat}")
        log.debug(f"pos close: {torch.allclose(cur_ee_pos, ee_pos, atol=atol)}")
        log.debug(f"quat close: {torch.allclose(cur_ee_quat, ee_quat, atol=atol)}")

        reach_target_try(ee_pos, ee_quat)

        states = env.handler.get_states()
        ee_idx = states.robots[robot.name].body_names.index(env.handler.robot.ee_body_name)
        cur_ee_pos = states.robots[robot.name].body_state[:, ee_idx, :3]
        cur_ee_quat = states.robots[robot.name].body_state[:, ee_idx, 3:7]


reach_target_dedicated(torch.tensor([[0.0, 0.75, 1.9]]), torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
reach_target_dedicated(torch.tensor([[0.0, 0.75, 1.5]]), torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
