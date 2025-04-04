from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from metasim.cfg.scenario import ScenarioCfg
from metasim.constants import SimType
from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import get_robot, get_sim_env_class, get_task


class Sb3EnvWrapperMujoco(gym.Env):
    """Wraps MetaSim environment to be compatible with Gymnasium API."""

    def __init__(self, scenario: ScenarioCfg, **kwargs):
        super().__init__(**kwargs)

        # Create the base environment
        assert scenario.num_envs == 1, "MuJoCo only supports 1 environment"
        env_class = get_sim_env_class(SimType.MUJOCO)
        self.env = env_class(scenario)
        self.robot = scenario.robot

        self.init_states, _, _ = get_traj(scenario.task, scenario.robot, self.env.handler)

        # Set up action space based on robot joint limits
        joint_limits = self.robot.joint_limits
        action_low = []
        action_high = []
        for joint_name in joint_limits.keys():
            action_low.append(joint_limits[joint_name][0])
            action_high.append(joint_limits[joint_name][1])

        self._action_low = np.array(action_low, dtype=np.float32)
        self._action_high = np.array(action_high, dtype=np.float32)

        self.action_space = spaces.Box(low=-1, high=1, shape=self._action_low.shape, dtype=np.float32)

        initial_obs, _ = self.reset()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=initial_obs.shape, dtype=np.float32)

        self.max_episode_steps = self.env.handler.task.episode_length

    def normalize_action(self, action):
        return 2 * (action - self._action_low) / (self._action_high - self._action_low) - 1

    def unnormalize_action(self, action):
        return (action + 1) / 2 * (self._action_high - self._action_low) + self._action_low

    def reset(self, *, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        metasim_observation, info = self.env.reset(self.init_states)
        observation = self.get_humanoid_observation()
        return observation, {}

    def step(self, action):
        """Execute one time step within the environment."""
        # Convert numpy action array to MetaSim format
        joint_names = list(self.robot.joint_limits.keys())
        action_dict = [
            {
                "dof_pos_target": {
                    joint_name: float(pos)
                    # for joint_name, pos in zip(joint_names, action)
                    for joint_name, pos in zip(joint_names, self.unnormalize_action(action))
                }
            }
        ]

        metasim_observation, metasim_reward, terminated, truncated, info = self.env.step(action_dict)

        # Convert torch tensors to numpy arrays
        observation = self.get_humanoid_observation()

        reward = metasim_reward.numpy().item()  # TODO: mujoco only support 1 env

        # observation = observation.numpy()
        # reward = reward.numpy().item()
        terminated = terminated.numpy().item()
        truncated = truncated.numpy().item()

        return observation, reward, terminated, truncated, {}

    def get_humanoid_observation(self):
        envstates = self.env.handler.get_states()
        gym_observation = self.env.handler.task.humanoid_obs_flatten_func(envstates)
        gym_observation = gym_observation[0]  # TODO: mujoco only support 1 env
        return gym_observation

    def render(self):
        """Render the environment."""
        pass

    def close(self):
        """Clean up environment resources."""
        self.env.close()

    def seed(self, seed=None):
        np.random.seed(seed)


def check_metasim_env(sim_type: SimType = SimType.MUJOCO, num_envs: int = 1):
    """Create and check a MetaSim environment."""
    # Create test environment
    task = get_task("Slide")
    robot = get_robot("h1")
    scenario = ScenarioCfg(task=task, robot=robot)

    # Create wrapped environment
    env = Sb3EnvWrapperMujoco(scenario, sim_type, num_envs)

    # Run environment checker
    from gymnasium.utils.env_checker import check_env

    check_env(env, skip_render_check=True)

    return env


if __name__ == "__main__":
    from gymnasium.envs import register

    from metasim.cfg.scenario import ScenarioCfg
    from metasim.constants import SimType
    from metasim.utils.demo_util import get_traj
    from metasim.utils.setup_util import get_robot, get_sim_env_class, get_task

    task = get_task("humanoidbench:Walk")
    robot = get_robot("h1")
    scenario = ScenarioCfg(task=task, robot=robot)
    num_envs = 1
    sim_type = SimType.MUJOCO
    register(
        id="metasim-hb-wrapper",
        entry_point="metasim.scripts.RL.sb3_wrapper_mujoco:Sb3EnvWrapperMujoco",
        kwargs={"scenario": scenario, "sim_type": sim_type, "num_envs": num_envs},
    )

    env = gym.make("metasim-hb-wrapper", headless=False)
    ob, _ = env.reset()
    print(f"ob_space = {env.observation_space}, ob = {ob.shape}")
    print(f"ac_space = {env.action_space.shape}")
    for i in range(1000):
        env.render()
