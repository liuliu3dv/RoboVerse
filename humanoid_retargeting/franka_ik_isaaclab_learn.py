"""This script is used to test the static scene."""

from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import math
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.3'
from typing import Literal

import rootutils
import torch
import tyro
# from curobo.types.math import Pose
from loguru import logger as log
from rich.logging import RichHandler

# Set project path
rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

from get_started.utils import ObsSaver
from metasim.cfg.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.cfg.robots import G1Cfg
from metasim.constants import PhysicStateType, SimType
from metasim.utils import configclass
# from metasim.utils.kinematics_utils import get_curobo_models
from metasim.utils.setup_util import get_sim_env_class

import numpy as np
import pyroki as pk
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf
from yourdfpy import URDF
import third_party.pyroki.examples.pyroki_snippets as pks

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 末端执行器名称，PyRoki需要明确
END_EFFECTOR_LINK_NAME = "panda_hand"  # 确认对应你的URDF
G1_CFG = G1Cfg()
# New args parser
@configclass
class Args:
    """Arguments for the static scene."""

    robot: str = "franka"

    ## Handlers
    sim: Literal["isaaclab", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco"] = "isaaclab"

    ## Others
    num_envs: int = 1
    headless: bool = False

    def __post_init__(self):
        """Post-initialization configuration."""
        log.info(f"Args: {self}")


args = tyro.cli(Args)

# initialize scenario
scenario = ScenarioCfg(
    robots=[args.robot],
    try_add_table=False,
    sim=args.sim,
    headless=args.headless,
    num_envs=args.num_envs,
)

# add cameras
# scenario.cameras = [PinholeCameraCfg(width=1024, height=1024, pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))]
# scenario.cameras = [PinholeCameraCfg(width=1024, height=1024, pos=(2.0, -2.0, 2.0), look_at=(0.0, 0.0, 0.0))]
scenario.cameras = [PinholeCameraCfg(width=1280, height=720, pos=(6.0, -6.0, 6.0), look_at=(0.0, 0.0, 0.0))]

# add objects
scenario.objects = [
    PrimitiveCubeCfg(
        name="cube",
        size=(0.1, 0.1, 0.1),
        color=[1.0, 0.0, 0.0],
        physics=PhysicStateType.RIGIDBODY,
    ),
    PrimitiveCubeCfg(
        name="cube_2",
        size=(0.1, 0.1, 0.1),
        color=[1.0, 0.0, 0.0],
        physics=PhysicStateType.RIGIDBODY,
    ),

    ArticulationObjCfg(
        name="mars_table",
        fix_base_link=True,
        scale=(2.0, 2.0, 2.0),
        usd_path="./humanoid_retargeting/assets/table/table.usd",
        urdf_path="./humanoid_retargeting/assets/table/table.urdf",
        # usd_path="/home/RoboVerse/humanoid_retargeting/assets/box_table/usd/box_table.usd",
        # urdf_path="/home/RoboVerse/humanoid_retargeting/assets/box_table/urdf/box_table.urdf",
        # mjcf_path="get_started/example_assets/box_base/mjcf/box_base_unique.mjcf",
    ),

    ArticulationObjCfg(
        name="mars_table_2",
        fix_base_link=True,
        scale=(2.0, 2.0, 2.0),
        usd_path="./humanoid_retargeting/assets/table/table.usd",
        urdf_path="./humanoid_retargeting/assets/table/table.urdf",
        # usd_path="/home/RoboVerse/humanoid_retargeting/assets/box_table/usd/box_table.usd",
        # urdf_path="/home/RoboVerse/humanoid_retargeting/assets/box_table/urdf/box_table.urdf",
        # mjcf_path="get_started/example_assets/box_base/mjcf/box_base_unique.mjcf",
    )


]


log.info(f"Using simulator: {args.sim}")
env_class = get_sim_env_class(SimType(args.sim))
env = env_class(scenario)

init_states = [
    {
        "objects": {
            "cube": {
                "pos": torch.tensor([-1.5, 0.0, 1.0]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
            "cube_2": {
                "pos": torch.tensor([1.5, 0.0, 1.0]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },

            "mars_table": {
                # "pos": torch.tensor([0.7, 0.3, 0.2]),
                "pos": torch.tensor([-1.5, 0.0, 0.0]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                "dof_pos": {"table_joint": 0.0},
            },
            "mars_table_2": {
                # "pos": torch.tensor([0.7, 0.3, 0.2]),
                "pos": torch.tensor([1.5, 0.0, 0.0]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                "dof_pos": {"table_joint": 0.0},
            },
        },
        "robots": {
            "franka": {
                # "pos": torch.tensor([0.0, 0.0, 0.0]),
                "pos": torch.tensor([-1.5, 0.4, 0.9]),
                "rot": torch.tensor([1.0, 0.0, 0.0, -0.7071]),
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
            },
        },
    }
    for _ in range(args.num_envs)
]

# urdf_path = "roboverse_data/robots/franka/urdf/franka_panda.urdf"
# urdf = URDF.load(urdf_path)

urdf = load_robot_description("panda_description")
target_link_name = "panda_hand"
robot = pk.Robot.from_urdf(urdf)

# Set up visualizer.
server = viser.ViserServer()
# base_frame = server.scene.add_frame("/base", show_axes=False)
# base_frame.position = np.array([0.0, 0.0, 0.7])
# urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
server.scene.add_grid("/ground", width=2, height=2)
# waist_yaw_frame = server.scene.add_frame("/waist_yaw_link", show_axes=False)
urdf_vis = ViserUrdf(server, urdf, root_node_name="/waist_yaw_link")
# waist_yaw_frame.position = np.array([0.0, 0.0, 0.7])

# 环境复位
obs, extras = env.reset(states=init_states)
os.makedirs("get_started/output", exist_ok=True)

# 准备录像保存器
# obs_saver = ObsSaver(video_path=f"get_started/output/4_motion_planning_{args.sim}.mp4")
obs_saver = ObsSaver(video_path=f"humanoid_retargeting/output/test_{args.sim}.mp4")
obs_saver.add(obs)


step = 0
robot_joint_limits = scenario.robots[0].joint_limits
for step in range(200):
    log.debug(f"Step {step}")
    states = env.handler.get_states()
    robot_name = scenario.robots[0].name  # 例如 'franka'
    curr_robot_q = states.robots[robot_name].joint_pos.cuda()

    curr_robot_q_np = curr_robot_q.cpu().numpy()  # shape: (num_envs, dof)
    # -1.5, 0.4, 0.9
    if scenario.robots[0].name == "franka":
        x_target = 0.1 + 0.3 * (step / 100)
        y_target = 0.4 - 0.3 * (step / 100)
        z_target = 0.7 - 0.2 * (step / 100)
        # Randomly assign x/y/z target for each env
        ee_pos_target = torch.zeros((args.num_envs, 3), device=DEVICE)
        for i in range(args.num_envs):
            if i % 3 == 0:
                ee_pos_target[i] = torch.tensor([x_target, 0.0, 0.6], device=DEVICE)
            elif i % 3 == 1:
                ee_pos_target[i] = torch.tensor([0.3, y_target, 0.6], device=DEVICE)
            else:
                ee_pos_target[i] = torch.tensor([0.3, 0.0, z_target], device=DEVICE)
        ee_quat_target = torch.tensor(
            [[0.0, 1.0, 0.0, 0.0]] * args.num_envs,
            device=DEVICE,
        )

    q_list = []
    for i_env in range(args.num_envs):
        pos = ee_pos_target[i_env].detach().cpu().numpy().reshape(3)
        quat = ee_quat_target[i_env].detach().cpu().numpy().reshape(4)
        # 8 dim, 8 joint?
        # array, 8 dim, float32
        solution = pks.solve_ik(
            robot,
            END_EFFECTOR_LINK_NAME,
            target_wxyz=quat,
            target_position=pos,
        )

        # q_list.append(solution)  # solution 是 np.ndarray，不要再 `.q`
        # 8+2 = 10 dim?
        q_list = np.concatenate([solution, [0.04, 0.04]])  # 手动加上两个夹爪关节值

    # 转为 tensor 并移动到 CUDA (如果有)

    # 准备动作字典
    robot_obj = scenario.robots[0]
    # frank arm has joints named:
    # ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7', 'panda_finger_joint1', 'panda_finger_joint2']
    # q_list = array
    # directly use dict?
    actions = [
        {
            args.robot: {
                "dof_pos_target": dict(zip(robot_obj.actuators.keys(), q_list)),
            },
        }
        for i_env in range(args.num_envs)
    ]

    # 执行动作
    obs, reward, success, time_out, extras = env.step(actions)

    # 第一步额外执行多步以稳定环境
    if step == 0:
        for _ in range(10):
            obs, _, _, _, _ = env.step(actions)

    # 保存观测
    obs_saver.add(obs)

obs_saver.save()
