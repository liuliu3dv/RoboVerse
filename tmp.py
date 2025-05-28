# ruff: noqa: D101 D103

from __future__ import annotations

import logging
import os
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
from torchvision.utils import make_grid, save_image
from tyro import MISSING

from metasim.cfg.render import RenderCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import SimType
from metasim.utils import configclass
from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import get_sim_env_class
from metasim.utils.state import TensorState

logging.addLevelName(5, "TRACE")
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


@configclass
class Args:
    task: str = MISSING
    robot: str = "franka"
    scene: str | None = None
    render: RenderCfg = RenderCfg()
    width: int = 300
    height: int = 200

    ## Handlers
    sim: Literal["isaaclab", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx"] = "isaaclab"

    ## Others
    num_envs: int = 1
    try_add_table: bool = True
    split: Literal["train", "val", "test", "all"] = "all"
    headless: bool = False

    ## Only in args
    save_image_dir: str | None = "tmp"
    save_video_path: str | None = None

    def __post_init__(self):
        log.info(f"Args: {self}")


args = tyro.cli(Args)


def get_states(all_states, action_idx: int, num_envs: int):
    envs_states = all_states[:1] * num_envs
    states = [env_states[action_idx] if action_idx < len(env_states) else env_states[-1] for env_states in envs_states]
    return states


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
    camera = PinholeCameraCfg(pos=(2.0, -2.0, 2.6), look_at=(0.0, 0.0, 1.0), width=args.width, height=args.height)
    scenario = ScenarioCfg(
        task=args.task,
        robot=args.robot,
        scene=args.scene,
        cameras=[camera],
        render=args.render,
        sim=args.sim,
        num_envs=args.num_envs,
        try_add_table=args.try_add_table,
        split=args.split,
        headless=args.headless,
    )

    num_envs: int = scenario.num_envs

    tic = time.time()
    log.info(f"Using simulator: {scenario.sim}")
    env_class = get_sim_env_class(SimType(scenario.sim))
    env = env_class(scenario)
    toc = time.time()
    log.trace(f"Time to launch: {toc - tic:.2f}s")

    ## Data
    assert os.path.exists(scenario.task.traj_filepath), (
        f"Trajectory file: {scenario.task.traj_filepath} does not exist."
    )
    init_states, _, all_states = get_traj(scenario.task, scenario.robot, env.handler)

    ########################################################
    ## Main
    ########################################################

    obs_saver = ObsSaver(image_dir=args.save_image_dir, video_path=args.save_video_path)

    ## Reset before first step
    obs, extras = env.reset(states=init_states[:1] * num_envs)
    obs_saver.add(obs)

    ## Main loop
    step = 0
    while True:
        log.debug(f"Step {step}")
        ## TODO: merge states replay into env.step function
        if all_states is None:
            raise ValueError("All states are None, please check the trajectory file")
        states = get_states(all_states, step, num_envs)
        env.handler.set_states(states)
        env.handler.refresh_render()
        obs = env.handler.get_states()

        obs_saver.add(obs)
        step += 1

        if step >= len(all_states[0]):
            log.info("Run out of actions, stopping")
            break

    obs_saver.save()
    env.close()


if __name__ == "__main__":
    main()
