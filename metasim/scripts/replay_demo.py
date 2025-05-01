from __future__ import annotations

import json
import logging
import os
import time
from shutil import copy
from typing import Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import imageio as iio
import numpy as np
import open3d as o3d
import rootutils
import tyro
from loguru import logger as log
from numpy.typing import NDArray
from rich.logging import RichHandler
from torchvision.utils import make_grid, save_image
from tyro import MISSING

logging.addLevelName(5, "TRACE")
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])
rootutils.setup_root(__file__, pythonpath=True)

from metasim.cfg.objects import BaseObjCfg, PrimitiveCubeCfg, RigidObjCfg
from metasim.cfg.randomization import RandomizationCfg
from metasim.cfg.render import RenderCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import SimType
from metasim.sim import HybridSimEnv
from metasim.utils import configclass
from metasim.utils.demo_util import get_traj
from metasim.utils.io_util import write_16bit_depth_video
from metasim.utils.setup_util import get_sim_env_class
from metasim.utils.state import TensorState


@configclass
class Args:
    task: str = MISSING
    robot: str = "franka"
    scene: str | None = None
    render: RenderCfg = RenderCfg()
    random: RandomizationCfg = RandomizationCfg()

    ## Handlers
    sim: Literal[
        "isaaclab", "isaacgym", "genesis", "pyrep", "pybullet", "sapien", "sapien2", "sapien3", "mujoco", "blender"
    ] = "isaaclab"
    renderer: (
        Literal[
            "isaaclab", "isaacgym", "genesis", "pyrep", "pybullet", "sapien", "sapien2", "sapien3", "mujoco", "blender"
        ]
        | None
    ) = None

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
def get_actions(all_actions, action_idx: int, num_envs: int):
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


class ObsSaver:
    """Save the observations to images or videos."""

    def __init__(self, image_dir: str | None = None, video_path: str | None = None):
        """Initialize the ObsSaver."""
        self.image_dir = image_dir
        self.video_path = video_path
        self.images: list[NDArray] = []
        self.depths: list[NDArray] = []

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

        depth_data = next(iter(state.cameras.values())).depth
        self.depths.append(depth_data[0].cpu().numpy())

    def save(self):
        """Save the images or videos."""
        if self.video_path is not None and self.images:
            log.info(f"Saving video of {len(self.images)} frames to {self.video_path}")
            os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
            iio.mimsave(self.video_path, self.images, fps=30)
            iio.mimsave(self.video_path.replace(".mp4", "_depth_uint8.mp4"), self.depths, fps=30)
            write_16bit_depth_video(self.video_path.replace(".mp4", "_depth_uint16.mkv"), self.depths, fps=30)


def export_mesh(obj: BaseObjCfg):
    if isinstance(obj, PrimitiveCubeCfg):
        cube = o3d.geometry.TriangleMesh.create_box(width=obj.size[0], height=obj.size[1], depth=obj.size[2])
        cube.translate([-obj.size[0] / 2, -obj.size[1] / 2, -obj.size[2] / 2])
        cube.paint_uniform_color(obj.color)
        o3d.io.write_triangle_mesh(f"{obj.name}.obj", cube)
    elif isinstance(obj, RigidObjCfg):
        mesh_path = obj.mesh_path
        copy(mesh_path, f"{obj.name}.obj")
    else:
        log.warning(f"Unsupported object type: {type(obj)}")


###########################################################
## Main
###########################################################
def main():
    camera = PinholeCameraCfg(pos=(0.8, -0.8, 1.3), look_at=(0.0, 0.0, 0.5), width=1920, height=1080)
    scenario = ScenarioCfg(
        task=args.task,
        robot=args.robot,
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

    for obj in scenario.task.objects:
        export_mesh(obj)

    args.save_video_path = f"./{args.task}_video.mp4"

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
    assert os.path.exists(scenario.task.traj_filepath), (
        f"Trajectory file: {scenario.task.traj_filepath} does not exist."
    )
    init_states, all_actions, all_states = get_traj(scenario.task, scenario.robot, env.handler)
    toc = time.time()
    log.trace(f"Time to load data: {toc - tic:.2f}s")

    ########################################################
    ## Main
    ########################################################

    obs_saver = ObsSaver(image_dir=args.save_image_dir, video_path=args.save_video_path)

    ## Reset before first step
    tic = time.time()
    init_states[0]["objects"]["table"]["dof_pos"]["base__switch"] = 0.0
    obs, extras = env.reset(states=init_states[:num_envs])
    toc = time.time()
    log.trace(f"Time to reset: {toc - tic:.2f}s")
    obs_saver.add(obs)

    ## Main loop
    step = 0
    data = []
    while True:
        log.debug(f"Step {step}")
        tic = time.time()
        if scenario.object_states:
            ## TODO: merge states replay into env.step function
            if all_states is None:
                raise ValueError("All states are None, please check the trajectory file")
            states = get_states(all_states, step, num_envs)
            env.handler.set_states(states)
            env.handler.refresh_render()
            obs = env.handler.get_states()

            ## XXX: hack
            success = env.handler.task.checker.check(env.handler)
            if success.any():
                log.info(f"Env {success.nonzero().squeeze(-1).tolist()} succeeded!")
            if success.all():
                break

        else:
            actions = get_actions(all_actions, step, num_envs)
            obs, reward, success, time_out, extras = env.step(actions)

            if success.any():
                log.info(f"Env {success.nonzero().squeeze(-1).tolist()} succeeded!")

            if time_out.any():
                log.info(f"Env {time_out.nonzero().squeeze(-1).tolist()} timed out!")

            if success.all() or time_out.all():
                break

        toc = time.time()
        log.trace(f"Time to step: {toc - tic:.2f}s")

        tic = time.time()
        obs_saver.add(obs)
        toc = time.time()
        log.trace(f"Time to save obs: {toc - tic:.2f}s")
        step += 1

        if args.stop_on_runout and get_runout(all_actions, step):
            log.info("Run out of actions, stopping")
            break

        depth = obs.cameras["camera0"].depth
        depth_min = depth.min()
        depth_max = depth.max()
        camera_pos = obs.cameras["camera0"].pos
        camera_quat_opengl = obs.cameras["camera0"].quat_opengl
        camera_quat_ros = obs.cameras["camera0"].quat_ros
        camera_quat_world = obs.cameras["camera0"].quat_world
        camera_intrinsics = obs.cameras["camera0"].intrinsics

        data_dict = {
            "depth_min": depth_min.item(),
            "depth_max": depth_max.item(),
            "camera_pos": camera_pos.tolist(),
            "camera_quat_opengl": camera_quat_opengl.tolist(),
            "camera_quat_ros": camera_quat_ros.tolist(),
            "camera_quat_world": camera_quat_world.tolist(),
            "camera_intrinsics": camera_intrinsics.tolist(),
        }
        for obj in scenario.objects:
            data_dict[f"{obj.name}_pos"] = obs.objects[obj.name].root_state[:, :3].tolist()
            data_dict[f"{obj.name}_quat"] = obs.objects[obj.name].root_state[:, 3:7].tolist()
        data.append(data_dict)
    obs_saver.save()

    with open(f"{args.task}_data.json", "w") as f:
        json.dump(data, f)

    env.close()


if __name__ == "__main__":
    main()
