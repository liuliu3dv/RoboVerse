from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.tasks.debug.reach_cfg import ReachOriginCfg
from metasim.constants import SimType
from metasim.utils.setup_util import get_sim_env_class

if __name__ == "__main__":

    env_class = get_sim_env_class(SimType("mujoco"))
    reach_cfg = ReachOriginCfg()

    scenario = ScenarioCfg(
        task=reach_cfg,
        num_envs=1,
        robots=["franka"],
        sim="mujoco",
    )

    env = env_class(scenario)

    print(env.action_space.sample())
