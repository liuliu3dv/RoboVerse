"""This script is used to test the static scene."""

from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import os

import imageio as iio
import numpy as np
import rootutils
import tyro
from loguru import logger as log
from rich.logging import RichHandler
from torchvision.utils import make_grid

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

from metasim.cfg.objects import FluidObjCfg, PrimitiveCubeCfg, RigidObjCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import PhysicStateType, SimType
from metasim.utils import configclass
from metasim.utils.setup_util import get_sim_env_class
from metasim.utils.state import TensorState


class ObsSaver:
    def __init__(self, video_path: str | None = None):
        self.video_path = video_path
        self.images: list[np.array] = []

        self.image_idx = 0

    def add(self, state: TensorState):
        try:
            rgb_data = next(iter(state.cameras.values())).rgb
            image = make_grid(rgb_data.permute(0, 3, 1, 2) / 255, nrow=int(rgb_data.shape[0] ** 0.5))  # (C, H, W)
        except Exception as e:
            log.error(f"Error adding observation: {e}")
            return

        image = image.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
        image = (image * 255).astype(np.uint8)
        self.images.append(image)

    def save(self):
        if self.video_path is not None and self.images:
            log.info(f"Saving video of {len(self.images)} frames to {self.video_path}")
            os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
            iio.mimsave(self.video_path, self.images, fps=30)


@configclass
class Args:
    robot: str = "franka"
    sim: str = "isaaclab"
    num_envs: int = 1
    headless: bool = False

    def __post_init__(self):
        """Post-initialization configuration."""
        log.info(f"Args: {self}")


args = tyro.cli(Args)

# initialize scenario
scenario = ScenarioCfg(
    robot=args.robot,
    try_add_table=False,
    sim=args.sim,
    headless=args.headless,
    num_envs=args.num_envs,
)

# add cameras
scenario.cameras = [PinholeCameraCfg(width=1024, height=1024, pos=(2, 0, 1), look_at=(0, 0, 0))]

# add objects
scenario.objects = [
    PrimitiveCubeCfg(
        name="cube",
        size=(0.1, 0.1, 0.1),
        color=[1.0, 0.0, 0.0],
        default_position=(0.3, -0.3, 0.05),
        physics=PhysicStateType.RIGIDBODY,
    ),
    RigidObjCfg(
        name="glass",
        usd_path="/home/fs/cod/IsaacLabPouringExtension/Tall_Glass_5.usd",
        scale=0.01,
        default_position=(0.61, -0.1, 0.0),
        physics=PhysicStateType.RIGIDBODY,
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
        default_position=(0.61, -0.1, 0.03),
    ),
]


log.info(f"Using simulator: {args.sim}")
env_class = get_sim_env_class(SimType(args.sim))
env = env_class(scenario)

init_states = [
    {
        "objects": {},
        "robots": {},
    }
]
obs_saver = ObsSaver(video_path="./tmp.mp4")
obs, extras = env.reset(states=init_states)
obs_saver.add(obs)
robot_joint_limits = scenario.robot.joint_limits

for _ in range(100):
    actions = [
        {"dof_pos_target": {joint_name: 0 for joint_name in robot_joint_limits.keys()}}
        for _ in range(scenario.num_envs)
    ]
    obs, _, _, _, _ = env.step(actions)
    obs_saver.add(obs)

obs_saver.save()
