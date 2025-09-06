from __future__ import annotations

from metasim.scenario.objects import ArticulationObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .rl_bench import RLBenchTask


@register_task("rlbench.push_button", "push_button", "franka.push_button")
class PushButtonTask(RLBenchTask):
    max_episode_steps = 200
    scenario = ScenarioCfg(
        objects=[
            ArticulationObjCfg(
                name="push_button_target",
                usd_path="roboverse_data/assets/rlbench/push_button/push_button_target/usd/push_button_target.usd",
            ),
        ],
        robots=["franka"],
    )
    traj_filepath = "roboverse_data/trajs/rlbench/push_buttonv2/franka_v2.pkl.gz"
    # TODO: add checker
