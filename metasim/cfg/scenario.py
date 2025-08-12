"""Sub-module containing the scenario configuration."""

from __future__ import annotations

from typing import Literal

from loguru import logger as log

from metasim.utils.configclass import configclass
from metasim.utils.hf_util import FileDownloader
from metasim.utils.setup_util import get_robot, get_scene, get_task

from .checkers import BaseChecker, EmptyChecker
from .control import ControlCfg
from .lights import BaseLightCfg, CylinderLightCfg, DistantLightCfg
from .objects import BaseObjCfg
from .randomization import RandomizationCfg
from .render import RenderCfg
from .robots.base_robot_cfg import BaseRobotCfg
from .scenes.base_scene_cfg import SceneCfg
from .sensors import BaseCameraCfg, BaseSensorCfg
from .simulator_params import SimParamCfg
from .tasks.base_task_cfg import BaseTaskCfg


@configclass
class ScenarioCfg:
    """Scenario configuration."""

    task: BaseTaskCfg | None = None  # This item should be removed?
    """None means no task specified"""
    robots: list[BaseRobotCfg] = []
    scene: SceneCfg | None = None
    """None means no scene"""
    lights: list[BaseLightCfg] = [DistantLightCfg()]
    objects: list[BaseObjCfg] = []
    cameras: list[BaseCameraCfg] = []
    sensors: list[BaseSensorCfg] = []
    checker: BaseChecker = EmptyChecker()
    render: RenderCfg = RenderCfg()
    random: RandomizationCfg = RandomizationCfg()
    sim_params: SimParamCfg = SimParamCfg()
    control: ControlCfg = ControlCfg()

    ## Handlers
    sim: Literal["isaaclab", "isaacgym", "sapien2", "sapien3", "genesis", "pybullet", "mujoco"] = "isaaclab"
    renderer: Literal["isaaclab", "isaacgym", "sapien2", "sapien3", "genesis", "pybullet", "mujoco"] | None = None

    ## Others
    num_envs: int = 1
    decimation: int = 1
    episode_length: int = 10000000  # never timeout
    try_add_table: bool = True
    object_states: bool = False
    split: Literal["train", "val", "test", "all"] = "all"
    headless: bool = False
    """environment spacing for parallel environments"""
    env_spacing: float = 1.0

    def __post_init__(self):
        """Post-initialization configuration."""
        ### Align configurations
        if (self.random.scene or self.scene is not None) and self.try_add_table:
            log.warning("try_add_table is set to False because scene randomization is enabled or a scene is specified")
            self.try_add_table = False

        ### Randomization level configuration
        if self.random.level >= 1:
            self.lights = [
                CylinderLightCfg(pos=(-2.0, -2.0, 4.0), intensity=5e4),
                CylinderLightCfg(pos=(-2.0, 2.0, 4.0), intensity=5e4),
                CylinderLightCfg(pos=(2.0, -2.0, 4.0), intensity=5e4),
                CylinderLightCfg(pos=(2.0, 2.0, 4.0), intensity=5e4),
            ]

        ### Parse task and robot
        for i, robot in enumerate(self.robots):
            if isinstance(robot, str):
                self.robots[i] = get_robot(robot)
        if isinstance(self.task, str):
            TaskCls = get_task(self.task)
            ### Instantiate TaskCls with robots if supported
            if "robots" in TaskCls.__dataclass_fields__.keys():
                self.task = TaskCls(robots=self.robots)
            else:
                self.task = TaskCls()
        if isinstance(self.scene, str):
            self.scene = get_scene(self.scene)

        ### Simulator parameters overvide by task
        self.sim_params = self.task.sim_params if self.task is not None else self.sim_params
        ### Control parameters overvide by task
        self.control = self.task.control if self.task is not None else self.control
        ### spacing of parallel environments
        self.env_spacing = self.task.env_spacing if self.task is not None else self.env_spacing
        ### Randomization vervide by task
        if self.task is not None and self.task.random is not None:
            self.random = self.task.random

        FileDownloader(self).do_it()
