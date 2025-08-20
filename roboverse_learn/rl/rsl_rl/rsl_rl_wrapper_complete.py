"""Complete wrapper for rsl_rl 2.3.10, align OnPolicyRunner in rsl_rl with metasim."""

from __future__ import annotations

import torch
from loguru import logger as log
from rsl_rl.env import VecEnv

from metasim.scenario.scenario import ScenarioCfg
from metasim.constants import SimType

from metasim.utils.setup_util import get_sim_handler_class
from metasim.utils.state import list_state_to_tensor


class RslRlWrapperComplete(VecEnv):
    """
    Complete wrapper for Metasim environments to be compatible with rsl_rl OnPolicyRunner.

    This wrapper provides full compatibility with rsl_rl's OnPolicyRunner by implementing
    all required methods and maintaining proper observation buffers and episode tracking.
    """

    def __init__(self, scenario: ScenarioCfg):
        super().__init__()

        # Validate simulator type
        if SimType(scenario.simulator) not in [SimType.ISAACSIM, SimType.ISAACGYM, SimType.ISAACLAB, SimType.GENESIS]:
            raise NotImplementedError(
                f"RslRlWrapperComplete now only supports {SimType.ISAACGYM}, but got {scenario.simulator}"
            )

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Using device: {self.device}")

        # Parse configuration
        self._parse_cfg(scenario)

        # Load simulator handler
        env_class = get_sim_handler_class(SimType(scenario.simulator))
        self.env = env_class(scenario)
        self.env.launch()

        # Get initial states
        self._get_init_states(scenario)
        self.env.set_states(self.init_states)

        # Initialize observation buffers and tracking
        self._init_observation_buffers()

        # Initial observation update
        self._update_observation_buffers()

    def _parse_cfg(self, scenario: ScenarioCfg):
        """Parse scenario configuration and extract training parameters."""
        self.scenario = scenario
        self.robot = scenario.robots[0]
        self.num_envs = scenario.num_envs
        self.num_obs = scenario.task.num_observations
        self.num_actions = scenario.task.num_actions
        self.num_privileged_obs = scenario.task.num_privileged_obs
        self.max_episode_length = scenario.task.max_episode_length
        self.cfg = scenario.task

        # Convert PPO config to dictionary
        from metasim.utils.dict import class_to_dict
        self.train_cfg = class_to_dict(scenario.task.ppo_cfg)

    def _init_observation_buffers(self):
        """Initialize observation buffers for rsl_rl compatibility."""
        # Initialize observation buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float32)
        self.privileged_obs_buf = torch.zeros((self.num_envs, self.num_privileged_obs), device=self.device, dtype=torch.float32)

        # Initialize extra observations dictionary (required by rsl_rl)
        self.extra_obs_buf = {
            "observations": {
                "critic": self.privileged_obs_buf,  # For PPO training
                # Add other observation types as needed (e.g., "teacher" for distillation)
            }
        }

        # Initialize episode tracking
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def _get_init_states(self, scenario):
        """Get initial states from the scenario configuration."""
        init_states_list = getattr(scenario.task, 'init_states', None)
        if init_states_list is None:
            raise AttributeError(f"'task cfg' has no attribute 'init_states', please add it in your scenario config!")

        # Replicate initial states if needed
        if len(init_states_list) < self.num_envs:
            init_states_list = (
                init_states_list * (self.num_envs // len(init_states_list))
                + init_states_list[: self.num_envs % len(init_states_list)]
            )
        else:
            init_states_list = init_states_list[: self.num_envs]

        self.init_states = init_states_list

        # Convert to tensor states for IsaacGym
        if scenario.simulator == SimType.ISAACGYM:
            self.init_states = list_state_to_tensor(self.env.handler, init_states_list, device=self.device)

    def _update_observation_buffers(self):
        """Update observation buffers from the environment."""
        try:
            # Get observations from the environment
            obs = self.env.get_observations()

            # Update observation buffers based on return format
            if isinstance(obs, (list, tuple)):
                if len(obs) == 2:
                    # Environment returns (obs, privileged_obs)
                    self.obs_buf = obs[0].to(self.device)
                    self.privileged_obs_buf = obs[1].to(self.device)
                else:
                    # Environment returns single observation
                    self.obs_buf = obs[0].to(self.device)
                    self.privileged_obs_buf = self.obs_buf.clone()
            else:
                # Environment returns single observation tensor
                self.obs_buf = obs.to(self.device)
                self.privileged_obs_buf = self.obs_buf.clone()

            # Update extra observations for rsl_rl compatibility
            self.extra_obs_buf["observations"]["critic"] = self.privileged_obs_buf

        except Exception as e:
            log.error(f"Error updating observation buffers: {e}")
            # Fallback: use zero observations
            self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float32)
            self.privileged_obs_buf = torch.zeros((self.num_envs, self.num_privileged_obs), device=self.device, dtype=torch.float32)
            self.extra_obs_buf["observations"]["critic"] = self.privileged_obs_buf

    def get_observations(self):
        """Get current observations and extra observations for rsl_rl."""
        # Update observation buffers from environment
        self._update_observation_buffers()
        return self.obs_buf, self.extra_obs_buf

    def get_privileged_observations(self):
        """Get privileged observations for critic."""
        return self.privileged_obs_buf

    def step(self, actions):
        """Step the environment with actions and return (obs, rewards, dones, infos)."""
        # Convert actions to appropriate format for the environment
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()

        try:
            # Step the environment
            obs, rewards, dones, infos = self.env.step(actions)

            # Convert to tensors and move to device
            if not isinstance(obs, torch.Tensor):
                obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
            if not isinstance(rewards, torch.Tensor):
                rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
            if not isinstance(dones, torch.Tensor):
                dones = torch.tensor(dones, device=self.device, dtype=torch.bool)

            # Update episode length buffer
            self.episode_length_buf += 1

            # Check for episode termination
            episode_terminated = (self.episode_length_buf >= self.max_episode_length)
            self.reset_buf = dones | episode_terminated

            # Reset environments that are done
            if self.reset_buf.any():
                reset_env_ids = torch.where(self.reset_buf)[0]
                self.reset(reset_env_ids)

            # Prepare infos dictionary
            if infos is None:
                infos = {}

            # Add observations to infos for rsl_rl compatibility
            infos["observations"] = self.extra_obs_buf["observations"]

            # Add episode info if available
            if "episode" in infos:
                infos["episode"] = infos["episode"]

            return obs, rewards, dones, infos

        except Exception as e:
            log.error(f"Error in environment step: {e}")
            # Return default values on error
            obs = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float32)
            rewards = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
            dones = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
            infos = {"observations": self.extra_obs_buf["observations"]}
            return obs, rewards, dones, infos

    def get_visual_observations(self):
        """Get visual observations if available."""
        if hasattr(self.env, 'get_visual_observations'):
            return self.env.get_visual_observations()
        else:
            raise NotImplementedError("Visual observations not available in this environment")

    def reset(self, env_ids=None):
        """Reset state in the env and buffer in the wrapper."""
        if env_ids is None:
            env_ids = list(range(self.env.num_envs))
        elif isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.cpu().numpy().tolist()

        try:
            # Reset in the environment
            self.env.reset(env_ids)

            # Reset episode length buffer for reset environments
            if isinstance(env_ids, list):
                env_ids_tensor = torch.tensor(env_ids, device=self.device)
            else:
                env_ids_tensor = env_ids
            self.episode_length_buf[env_ids_tensor] = 0
            self.reset_buf[env_ids_tensor] = False

            # Update observation buffers after reset
            self._update_observation_buffers()

        except Exception as e:
            log.error(f"Error in environment reset: {e}")

    @property
    def unwrapped(self):
        """Return the unwrapped environment."""
        return self.env

    def close(self):
        """Close the environment."""
        if hasattr(self.env, 'close'):
            self.env.close()

    def render(self, mode='human'):
        """Render the environment."""
        if hasattr(self.env, 'render'):
            return self.env.render(mode)
        else:
            raise NotImplementedError("Render not available in this environment")

    def seed(self, seed=None):
        """Set random seed for the environment."""
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)
        else:
            log.warning("Environment does not support seeding")

    def get_env_info(self):
        """Get environment information."""
        return {
            "num_envs": self.num_envs,
            "num_obs": self.num_obs,
            "num_actions": self.num_actions,
            "num_privileged_obs": self.num_privileged_obs,
            "max_episode_length": self.max_episode_length,
            "device": str(self.device)
        }
