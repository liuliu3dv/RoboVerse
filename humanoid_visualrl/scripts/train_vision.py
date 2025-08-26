"""This script is used to test the static scene."""

from __future__ import annotations

from typing import Literal

import rootutils
import torch
import tyro
from metasim.scenario.cameras import PinholeCameraCfg

from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils import configclass

# for vanilla rsl_rl
# from rsl_rl.runners.on_policy_runner import OnPolicyRunner


# for humanoid_visualrl
from humanoid_visualrl.actor_critic.on_policy_runner import OnPolicyRunner

from humanoid_visualrl.cfg.humanoidVisualRLVisionCfg import BaseTableHumanoidTaskCfg
from humanoid_visualrl.wrapper.walking_wrapper_cnn import WalkingWrapperCNN as TaskWrapper
from humanoid_visualrl.utils.utils import get_log_dir

if __name__ == "__main__":

    @configclass
    class Args:
        """Arguments for the static scene."""

        robot: str = "g1"
        sim: Literal["isaacsim"] = "isaacsim" # only support isaacsim
        num_envs: int = 1
        headless: bool = False
        num_learning_iterations: int = 10000

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
    scenario.lights = []

    # add cameras
    # egocentric camera

    from scipy.spatial.transform import Rotation as R

    quat_xyzw = R.from_euler("xyz", [0, 60, 0], degrees=True).as_quat()
    quat = (quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2])  # convert to wxyz
    translation = (0.1, 0.0, 0.9)
    scenario.cameras = [
        PinholeCameraCfg(
            name="camera_first_person",
            width=64,
            height=48,
            pos=(1.5, -1.5, 1.5),
            look_at=(0.0, 0.0, 0.0),
            mount_to="g1",
            mount_link="torso_link",
            mount_pos=translation,
            mount_quat=quat,
        ),
    ]
    # add objects
    scenario.objects = []

    task_cfg = BaseTableHumanoidTaskCfg()

    # task assign and override
    scenario.sim_params = task_cfg.sim_params
    scenario.decimation = task_cfg.decimation
    scenario.render_interval = scenario.decimation
    scenario.task = task_cfg
    scenario.env_spacing = task_cfg.env_spacing

    log.info(f"Using simulator: {args.sim}")
    env = TaskWrapper(scenario)
    device = torch.device("cuda")
    log_dir = get_log_dir(args, scenario)
    ppo_runner = OnPolicyRunner(
        env=env,
        train_cfg=env.train_cfg,
        device=device,
        log_dir=log_dir,
    )
    ppo_runner.learn(num_learning_iterations=args.num_learning_iterations)
