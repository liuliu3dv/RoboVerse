from __future__ import annotations

import gymnasium as gym

from metasim.scenario.objects import ArticulationObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.base import BaseTaskEnv
from metasim.task.registry import register_task
from roboverse_pack.robots.franka_with_gripper_extension_cfg import FrankaWithGripperExtensionCfg


@register_task("calvin.base_table")
class BaseCalvinTableTask(BaseTaskEnv):
    scenario = ScenarioCfg(
        robots=[
            FrankaWithGripperExtensionCfg(
                name="franka",
                default_position=[-0.34, -0.46, 0.24],
                default_orientation=[1, 0, 0, 0],
                default_joint_positions={
                    "panda_joint1": -1.21779206,
                    "panda_joint2": 1.03987646,
                    "panda_joint3": 2.11978261,
                    "panda_joint4": -2.34205014,
                    "panda_joint5": -0.87015947,
                    "panda_joint6": 1.64119353,
                    "panda_joint7": 0.55344866,
                    "panda_finger_joint1": 0.04,
                    "panda_finger_joint2": 0.04,
                },
                fix_base_link=True,
                urdf_path="roboverse_data/assets/calvin/franka_panda/panda_longer_finger.urdf",
                usd_path=None,
                mjcf_path=None,
                mjx_mjcf_path=None,
            )
        ],
        objects=[
            ArticulationObjCfg(
                name="table",
                default_position=[0, 0, 0],
                default_orientation=[1, 0, 0, 0],
                fix_base_link=True,
                urdf_path="roboverse_data/assets/calvin/calvin_table_A/urdf/calvin_table_A.urdf",
            )
        ],
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _action_space(self):
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=float)
