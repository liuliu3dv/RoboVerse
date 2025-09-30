"""This script is used to test the static scene."""

from __future__ import annotations

from typing import Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass


import rootutils
import torch
import tyro
from curobo.types.math import Pose
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


from metasim.constants import PhysicStateType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.objects import (
    ArticulationObjCfg,
    PrimitiveCubeCfg,
    PrimitiveSphereCfg,
    RigidObjCfg,
)
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils import configclass
from metasim.utils.kinematics import get_curobo_models
from metasim.utils.obs_utils import ObsSaver
from metasim.utils.setup_util import get_handler
from metasim.utils.state import state_tensor_to_nested


@configclass
class Args:
    """Arguments for the static scene."""

    robot: str = "franka"

    ## Handlers
    sim: Literal[
        "isaacsim",
        "isaacgym",
        "isaaclab",
        "genesis",
        "pybullet",
        "sapien2",
        "sapien3",
        "mujoco",
    ] = "mujoco"

    ## Others
    num_envs: int = 1
    headless: bool = True  # Use viser for visualization, not simulator's viewer

    def __post_init__(self):
        """Post-initialization configuration."""
        log.info(f"Args: {self}")


args = tyro.cli(Args)

# initialize scenario
scenario = ScenarioCfg(
    robots=[args.robot],
    simulator=args.sim,
    headless=args.headless,
    num_envs=args.num_envs,
)

# add cameras
scenario.cameras = [
    PinholeCameraCfg(
        name="camera",
        width=1024,
        height=1024,
        pos=(1.5, -1.5, 1.5),
        look_at=(0.0, 0.0, 0.0),
    )
]

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
handler = get_handler(scenario)

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
        "robots": {
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
            },
        },
    }
]

robot = scenario.robots[0]
*_, robot_ik = get_curobo_models(robot)
curobo_n_dof = len(robot_ik.robot_config.cspace.joint_names)
ee_n_dof = len(robot.gripper_open_q)

handler.set_states(init_states * scenario.num_envs)
obs = handler.get_states(mode="tensor")


# ========================================================================
# viser visuallization
# ========================================================================
from get_started.viser.viser_util import ViserVisualizer


def extract_states_from_init(obs, handler, key):
    """
    obs: TensorState
    handler: simulator handler
    key: "objects" or "robots"
    Return: dict[name] = {"pos": ..., "rot": ..., "dof_pos": ...}
    """
    env_states = state_tensor_to_nested(handler, obs)
    result = {}
    if env_states and len(env_states) > 0:
        state = env_states[0]
        if key in state:
            for name, item in state[key].items():
                state_dict = {}
                if "pos" in item and item["pos"] is not None:
                    state_dict["pos"] = (
                        item["pos"].cpu().numpy().tolist() if hasattr(item["pos"], "cpu") else list(item["pos"])
                    )
                if "rot" in item and item["rot"] is not None:
                    state_dict["rot"] = (
                        item["rot"].cpu().numpy().tolist() if hasattr(item["rot"], "cpu") else list(item["rot"])
                    )
                if "dof_pos" in item and item["dof_pos"] is not None:
                    state_dict["dof_pos"] = item["dof_pos"]
                result[name] = state_dict
    return result


# initialize the viser server
visualizer = ViserVisualizer(port=8080)
visualizer.add_grid()
visualizer.add_frame("/world_frame")

obs_saver = ObsSaver(video_path=f"get_started/output/4_motion_planning_{args.sim}.mp4")
obs_saver.add(obs)

# extract states from objects and robots
default_object_states = extract_states_from_init(obs, handler, "objects")
default_robot_states = extract_states_from_init(obs, handler, "robots")

# visualize all objects and robots
visualizer.visualize_scenario_items(scenario.objects, default_object_states)
visualizer.visualize_scenario_items(scenario.robots, default_robot_states)

log.info("Viser has been initialized, visit http://localhost:8080 to view the scene!")

# scene info string
scene_info = ["The scene includes:"]
for obj in scenario.objects:
    scene_info.append(f"  • {obj.name} ({type(obj).__name__})")
for robot in scenario.robots:
    scene_info.append(f"  • {robot.name} ({type(robot).__name__})")

# print for debugging
log.info("\n".join(scene_info))

# Enable camera controls
# The camera can be controlled via the GUI sliders and can be manually dragged
visualizer.enable_camera_controls(
    initial_position=[1.5, -1.5, 1.5],
    render_width=1024,
    render_height=1024,
    look_at_position=[0, 0, 0],
    initial_fov=71.28,
)

step = 0
robot_joint_limits = scenario.robots[0].joint_limits
for step in range(200):
    log.debug(f"Step {step}")
    states = handler.get_states(mode="tensor")
    curr_robot_q = states.robots[robot.name].joint_pos.cuda()

    seed_config = curr_robot_q[:, :curobo_n_dof].unsqueeze(1).tile([1, robot_ik._num_seeds, 1])
    if scenario.robots[0].name == "franka":
        x_target = 0.3 + 0.1 * (step / 100)
        y_target = 0.5 - 0.5 * (step / 100)
        z_target = 0.6 - 0.2 * (step / 100)
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

    result = robot_ik.solve_batch(Pose(ee_pos_target, ee_quat_target), seed_config=seed_config)

    q = torch.zeros((scenario.num_envs, robot.num_joints), device="cuda:0")
    ik_succ = result.success.squeeze(1)
    q[ik_succ, :curobo_n_dof] = result.solution[ik_succ, 0].clone()
    q[:, -ee_n_dof:] = 0.04
    robot = scenario.robots[0]
    actions = [
        {robot.name: {"dof_pos_target": dict(zip(robot.actuators.keys(), q[i_env].tolist()))}}
        for i_env in range(scenario.num_envs)
    ]

    handler.set_dof_targets(actions)
    handler.simulate()
    obs = handler.get_states(mode="tensor")

    if step == 0:
        for _ in range(50):
            handler.simulate()
            obs = handler.get_states(mode="tensor")

    obs_saver.add(obs)
    step += 1

    object_states = extract_states_from_init(obs, handler, "objects")
    robot_states = extract_states_from_init(obs, handler, "robots")

    for name, state in object_states.items():
        visualizer.update_item_pose(name, state)
    for name, state in robot_states.items():
        visualizer.update_item_pose(name, state)

    visualizer.refresh_camera_view()

obs_saver.save()

while True:
    pass