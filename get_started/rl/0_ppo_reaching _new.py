"""Train PPO for reaching task using RLTaskEnv."""

from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

from dataclasses import dataclass
from typing import Literal

import numpy as np
import rootutils
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

# Ensure reaching tasks are registered exactly once from the canonical module
from get_started.utils import ObsSaver
from scenario_cfg.scenario import ScenarioCfg
from tasks.registry import load_task


@dataclass
class Args:
    """Arguments for training PPO."""

    task: str = "reach_origin"
    robot: str = "franka"
    num_envs: int = 128
    sim: Literal["isaaclab", "isaacgym", "mujoco", "genesis", "mjx"] = "mjx"


args = tyro.cli(Args)


def _to_np(x):
    """Convert torch tensors to CPU numpy, else asarray."""
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class VecEnvWrapper(VecEnv):
    """Bridge Gymnasium VectorEnv to SB3 VecEnv."""

    def __init__(self, gym_vec_env):
        """Initialize the wrapper."""
        self.gym_vec = gym_vec_env
        super().__init__(
            num_envs=gym_vec_env.num_envs,
            observation_space=gym_vec_env.observation_space,
            action_space=gym_vec_env.action_space,
        )
        self._last_obs = None

    def reset(self):
        """Reset all envs and return observations."""
        obs, _ = self.gym_vec.reset()
        obs = _to_np(obs)
        self._last_obs = obs.copy()
        return obs

    def step_async(self, actions):
        """Send actions to the vectorized env."""
        self.actions = actions

    def step_wait(self):
        """Step the envs and adapt outputs for SB3."""
        obs, rewards, terminated, truncated, infos = self.gym_vec.step(self.actions)

        obs = _to_np(obs)
        rewards = _to_np(rewards).reshape(self.num_envs)
        terminated = _to_np(terminated).astype(bool).reshape(self.num_envs)
        truncated = _to_np(truncated).astype(bool).reshape(self.num_envs)
        dones = np.logical_or(terminated, truncated)

        if isinstance(infos, list):
            out_infos = []
            n = len(infos) if infos is not None else 0
            for i in range(self.num_envs):
                info_i = dict(infos[i]) if (infos is not None and i < n and infos[i] is not None) else {}
                info_i["TimeLimit.truncated"] = bool(truncated[i] and not terminated[i])
                if dones[i] and self._last_obs is not None:
                    info_i["terminal_observation"] = np.array(self._last_obs[i], copy=True)
                out_infos.append(info_i)
        else:
            base = dict(infos) if infos is not None else {}
            out_infos = []
            for i in range(self.num_envs):
                info_i = dict(base)
                info_i["TimeLimit.truncated"] = bool(truncated[i] and not terminated[i])
                if dones[i] and self._last_obs is not None:
                    info_i["terminal_observation"] = np.array(self._last_obs[i], copy=True)
                out_infos.append(info_i)

        self._last_obs = obs.copy()
        return obs, rewards, dones, out_infos

    def close(self):
        """Close the underlying envs."""
        self.gym_vec.close()

    def get_attr(self, attr_name, indices=None):
        """Get an attribute of the environment."""
        if indices is None:
            indices = list(range(self.num_envs))
        return [getattr(self.env.handler, attr_name)] * len(indices)

    def set_attr(self, attr_name, value, indices=None):
        """Not implemented."""
        raise NotImplementedError

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Not implemented."""
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices=None):
        """Not implemented."""
        raise NotImplementedError


def train_ppo():
    """Train PPO for reaching task using RLTaskEnv."""

    # Create scenario configuration
    scenario = ScenarioCfg(
        robots=[args.robot],
        simulator=args.sim,
        num_envs=args.num_envs,
        headless=False,
        cameras=[],
    )

    # Create RLTaskEnv via registry
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rl_env = load_task(args.task, scenario, device=device)

    # Create VecEnv wrapper for SB3
    env = VecEnvWrapper(rl_env)

    log.info(f"Created environment with {env.num_envs} environments")
    log.info(f"Observation space: {env.observation_space}")
    log.info(f"Action space: {env.action_space}")
    log.info(f"Max episode steps: {rl_env.max_episode_steps}")

    # PPO configuration
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Start training
    model.learn(total_timesteps=1_000_000)

    # Save the model

    model.save(f"get_started/output/rl/0_ppo_reaching{args.sim}")

    env.close()

    # Inference and Save Video
    # Create new environment for inference
    scenario_inference = ScenarioCfg(
        robots=[args.robot],
        simulator=args.sim,
        num_envs=1,
        headless=True,
        cameras=[],
    )

    rl_env_inference = load_task(args.task, scenario_inference, device=device)
    env_inference = VecEnvWrapper(rl_env_inference)

    obs_saver = ObsSaver(video_path=f"get_started/output/rl/0_ppo_reaching_{args.sim}.mp4")

    # load the model
    model = PPO.load(f"get_started/output/rl/0_ppo_reaching_{args.sim}")

    # inference
    obs = env_inference.reset()
    obs_orin = rl_env_inference.env.get_states()
    obs_saver.add(obs_orin)

    for _ in range(100):
        actions, _ = model.predict(obs, deterministic=True)
        env_inference.step_async(actions)
        obs, _, _, _ = env_inference.step_wait()

        obs_orin = rl_env_inference.env.get_states()
        obs_saver.add(obs_orin)

    obs_saver.save()
    env_inference.close()


if __name__ == "__main__":
    train_ppo()
