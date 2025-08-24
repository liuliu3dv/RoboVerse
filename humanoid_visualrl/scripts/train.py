"""This script is used to test the static scene."""

from __future__ import annotations

from typing import Literal

import rootutils
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils import configclass

from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from humanoid_visualrl.cfg.humanoidVisualRLCfg import BaseTableHumanoidTaskCfg
from humanoid_visualrl.wrapper.walking_wrapper import WalkingWrapper as TaskWrapper

if __name__ == "__main__":

    @configclass
    class Args:
        """Arguments for the static scene."""

        robot: str = "g1"
        sim: Literal["isaacsim"] = "isaacsim"
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
    # scenario.cameras = [PinholeCameraCfg(width=1024, height=1024, pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))]
    scenario.cameras = []
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
    log_dir = "outputs/humanoid_visualrl"
    ppo_runner = OnPolicyRunner(
        env=env,
        train_cfg=env.train_cfg,
        device=device,
        log_dir=log_dir,
    )
    ppo_runner.learn(num_learning_iterations=args.num_learning_iterations)
