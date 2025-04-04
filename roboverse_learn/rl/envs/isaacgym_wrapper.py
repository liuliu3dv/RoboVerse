from __future__ import annotations

from isaacgym import gymapi, gymtorch

ISAACGYM_AVAILABLE = True

import random
import time
from contextlib import contextmanager

import numpy as np
import torch

from metasim.cfg.scenario import ScenarioCfg
from metasim.sim.isaacgym.isaacgym import IsaacgymHandler


@contextmanager
def timing_context(name: str, verbose: bool = False, timing_dict: dict | None = None):
    """Context manager for timing code blocks"""
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    if verbose:
        print(f"{name}: {elapsed_time:.4f}s")
    if timing_dict is not None:
        timing_dict[name] = timing_dict.get(name, 0.0) + elapsed_time


class IsaacGymWrapper:
    """Wrapper around IsaacgymHandler for RL algorithms."""

    def __init__(self, scenario: ScenarioCfg, num_envs: int = 1, headless: bool = False, seed: int | None = None):
        """Initialize the wrapper with an IsaacgymHandler."""
        if not ISAACGYM_AVAILABLE:
            raise ImportError("IsaacGym is not installed. Please install it to use IsaacGymWrapper.")

        # Set seed if provided
        if seed is not None:
            self.set_seed(seed)

        scenario.num_envs = num_envs
        scenario.headless = headless
        self.handler = IsaacgymHandler(scenario)

        # Store configuration
        self.headless = headless
        self.num_envs = num_envs
        self._task = scenario.task
        self._robot = scenario.robot
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Additional variables for RL
        self.max_episode_length = 30 if not scenario.task.episode_length else scenario.task.episode_length
        self.reset_buffer = torch.zeros(self.num_envs, 1, device=self.device)
        self.timestep_buffer = torch.zeros(self.num_envs, 1, device=self.device)

        # Add success tracking
        self.success_buffer = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Add timing tracking
        self.verbose = False
        self.step_timings = {}
        self.total_steps = 0
        self.total_time = 0.0

        self.rgb_buffers = [[] for _ in range(self.num_envs)]
        self.max_rgb_buffer_size = self.handler.task.episode_length
        self.writer = None
        self.global_step = 0

    def set_seed(self, seed):
        """Set random seed for reproducibility."""
        if seed == -1 and torch.cuda.is_available():
            seed = torch.randint(0, 10000, (1,))[0].item()
        if seed != -1:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        return seed

    def launch(self) -> None:
        """Launch the simulation."""
        self.handler.launch()

        # Set up end effector tracking
        self.setup_tracking()

    def setup_tracking(self):
        """Set up tracking for end effectors and other objects."""
        # Pre-compute end effector indices for each environment
        self._ee_indices = []
        for env_idx in range(self.num_envs):
            ee_idx = self.handler.gym.find_actor_rigid_body_index(
                self.handler._envs[env_idx], self.handler._robot_handles[env_idx], "panda_hand", gymapi.DOMAIN_SIM
            )
            self._ee_indices.append(ee_idx)
        self._ee_indices = torch.tensor(self._ee_indices, device=self.device)

        # Get box joint indices
        self._box_joint_indices = []
        for env_idx in range(self.num_envs):
            box_handle = self.handler.gym.find_actor_handle(self.handler._envs[env_idx], "box_base")
            box_joint_idx = (
                self.handler.gym.get_actor_joint_count(self.handler._envs[env_idx], box_handle) - 1
            )  # Assuming last joint
            self._box_joint_indices.append(box_joint_idx)
        self._box_joint_indices = torch.tensor(self._box_joint_indices, device=self.device)

    def get_observation(self) -> dict[str, torch.Tensor]:
        """Get observations from the environment."""
        # Get observations from the handler
        handler_obs = self.handler.get_observation()

        # Extract joint positions from DOF states
        dof_state = self.handler._dof_states.clone()
        joint_positions = dof_state[:, 0]

        dofs_per_env = joint_positions.shape[0] // self.num_envs
        joint_positions = joint_positions.reshape(self.num_envs, dofs_per_env)

        # Create observation dictionary with joint positions
        observation = {
            "obs": joint_positions.clone(),
        }

        # Check if RGB images are available in the handler's observation
        if "rgb" in handler_obs and handler_obs["rgb"] is not None:
            try:
                # Get RGB tensors for all environments
                rgb_tensors = []

                # If handler_obs["rgb"] is already a list/tensor with multiple environments
                if isinstance(handler_obs["rgb"], (list, torch.Tensor)) and len(handler_obs["rgb"]) == self.num_envs:
                    # Use the existing RGB tensors directly
                    rgb_tensors = [tensor.to(self.device) for tensor in handler_obs["rgb"]]
                else:
                    # For each environment, try to get its RGB image
                    for env_id in range(self.num_envs):
                        if env_id < len(handler_obs["rgb"]):
                            rgb_tensors.append(handler_obs["rgb"][env_id].to(self.device))
                        else:
                            # If no image for this env, use the first environment's image
                            rgb_tensors.append(handler_obs["rgb"][0].to(self.device))

                # Stack all RGB tensors into a single tensor for better efficiency
                observation["rgb"] = torch.stack(rgb_tensors, dim=0).float()
            except Exception as e:
                print(f"Warning: Could not process RGB images from handler: {e}")

        return observation

    def get_reward(self) -> torch.Tensor:
        """Calculate rewards based on task configuration."""
        # Get current states from the environment
        states = self.handler.get_states()

        # Case 1: Task has a direct reward_fn
        if hasattr(self.handler.task, "reward_fn"):
            return self.handler.task.reward_fn(states).to(self.device)

        # Case 2: Task is a BaseRLTaskCfg with reward functions and weights
        from metasim.cfg.tasks.base_task_cfg import BaseRLTaskCfg

        if isinstance(self.handler.task, BaseRLTaskCfg):
            final_reward = torch.zeros(self.handler.num_envs, device=self.device)
            for reward_func, reward_weight in zip(self.handler.task.reward_functions, self.handler.task.reward_weights):
                # Apply each reward function to the states and add the weighted result
                reward_component = reward_func(self.handler.get_states()) * reward_weight
                final_reward += reward_component
            return final_reward

        return torch.zeros(self.handler.num_envs, device=self.device)

    def get_termination(self) -> torch.Tensor:
        """Get termination states of all environments."""
        if hasattr(self.handler.task, "termination_fn"):
            return self.handler.task.termination_fn(self.handler.get_states())
        return torch.full((self.handler.num_envs,), False, device=self.device)

    def get_success(self) -> torch.Tensor:
        """Get success states of all environments."""
        return self.success_buffer.clone()

    def get_timeout(self) -> torch.Tensor:
        """Check if environments have timed out."""
        timeout = self.timestep_buffer >= self.max_episode_length
        return timeout

    def step(
        self, action: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """Step the environment forward."""
        step_start = time.time()
        self.total_steps += 1

        if self.verbose:
            print(f"\n=== Step {self.total_steps} ===")

        self.timestep_buffer += 1

        TIME_STEPS_TO_RUN = 3
        for substep in range(TIME_STEPS_TO_RUN):
            if self.verbose:
                print(f"  Substep {substep + 1}/{TIME_STEPS_TO_RUN}")

            # Step the physics
            with timing_context("physics_simulation", self.verbose, self.step_timings):
                self.handler.gym.simulate(self.handler.sim)
                self.handler.gym.fetch_results(self.handler.sim, True)

            # Refresh tensors
            with timing_context("tensor_refresh", self.verbose, self.step_timings):
                self.handler.gym.refresh_rigid_body_state_tensor(self.handler.sim)
                self.handler.gym.refresh_actor_root_state_tensor(self.handler.sim)
                self.handler.gym.refresh_dof_state_tensor(self.handler.sim)
                self.handler.gym.refresh_jacobian_tensors(self.handler.sim)
                self.handler.gym.refresh_mass_matrix_tensors(self.handler.sim)

            # Deploy actions
            with timing_context("action_deployment", self.verbose, self.step_timings):
                action_input = torch.zeros_like(self.handler._dof_states[:, 0].clone())
                action_array_all = action.cpu().numpy()
                robot_dim = action_array_all.shape[1] if len(action_array_all.shape) > 1 else action_array_all.shape[0]

                # Validate shape assumptions and compute chunk_size per environment
                assert action_input.shape[0] % self.num_envs == 0, (
                    "Mismatch between action input shape and number of environments"
                )
                chunk_size = action_input.shape[0] // self.num_envs

                # Reshape action_input and deploy
                action_input_2d = action_input.view(self.num_envs, chunk_size)
                env_actions_tensor = torch.as_tensor(action_array_all, dtype=torch.float32, device=self.device)

                # Handle both batched and single actions
                if len(env_actions_tensor.shape) > 1:
                    action_input_2d[:, chunk_size - robot_dim :] = env_actions_tensor
                else:
                    for i in range(self.num_envs):
                        action_input_2d[i, chunk_size - robot_dim :] = env_actions_tensor

                if substep == 0:
                    self.handler.gym.set_dof_position_target_tensor(
                        self.handler.sim, gymtorch.unwrap_tensor(action_input_2d)
                    )

            # Update viewer
            with timing_context("graphics_update", self.verbose, self.step_timings):
                if not self.headless and self.handler.viewer is not None:
                    self.handler.gym.step_graphics(self.handler.sim)
                    self.handler.gym.draw_viewer(self.handler.viewer, self.handler.sim, False)
                    self.handler.gym.sync_frame_time(self.handler.sim)

        # Get observation and compute rewards
        with timing_context("get_observation", self.verbose, self.step_timings):
            observation = self.get_observation()

        with timing_context("compute_rewards", self.verbose, self.step_timings):
            reward = self.get_reward().to(self.device)
            success = self.get_success().to(self.device)
            timeout = self.get_timeout().to(self.device)
            termination = self.get_termination().to(self.device)

        # Handle resets for timed out environments
        with timing_context("handle_timeouts", self.verbose, self.step_timings):
            timeout_indices = torch.nonzero(timeout.squeeze(1)).squeeze(1).tolist()
            if timeout_indices:
                self.reset_idx(timeout_indices)
                updated_obs = self.get_observation()
                obs_tensor = observation["obs"]
                obs_tensor[timeout_indices] = updated_obs["obs"][timeout_indices]
                observation["obs"] = obs_tensor

        dones = timeout.squeeze(1) | success | termination

        # Update timing statistics
        step_time = time.time() - step_start
        self.total_time += step_time

        if self.verbose:
            print(f"Step {self.total_steps} completed in {step_time:.4f}s")
            if success.any():
                success_envs = torch.nonzero(success).squeeze(-1).tolist()
                print(f"Success in environments: {success_envs}")
            if timeout.any():
                timeout_envs = torch.nonzero(timeout).squeeze(-1).tolist()
                print(f"Timeout in environments: {timeout_envs}")
            if termination.any():
                termination_envs = torch.nonzero(termination).squeeze(-1).tolist()
                print(f"Termination in environments: {termination_envs}")
            print("=" * 30 + "\n")

        reward = reward.unsqueeze(1)
        return (
            observation,
            reward,
            dones,
            timeout,
            None,
        )

    def reset_handler(self, env_ids: list[int] | None = None):
        ## TODO: remove this method
        if env_ids is None:
            env_ids = list(range(self.handler.num_envs))
        self.handler._episode_length_buf = [0 for _ in range(self.handler.num_envs)]
        self.handler.checker.reset(self.handler, env_ids=env_ids)
        self.handler.simulate()
        obs = self.handler.get_observation()
        return obs, None

    def reset(self) -> dict[str, torch.Tensor]:
        """Reset all environments."""
        # Call the handler's reset method which uses configurations
        self.reset_handler()

        # Reset internal state tracking buffers
        self.reset_buffer = torch.ones_like(self.reset_buffer)
        self.timestep_buffer = torch.zeros_like(self.timestep_buffer)
        self.success_buffer = torch.zeros_like(self.success_buffer)

        # Get observation after reset
        return self.get_observation()

    def reset_idx(self, env_ids: list[int]) -> None:
        """Reset specific environments to initial configuration."""
        # Call the handler's reset method with specific environment IDs
        self.reset_handler(env_ids=env_ids)

        # Reset internal state tracking only for specified environments
        self.reset_buffer[env_ids] = torch.ones_like(self.reset_buffer[env_ids])
        self.timestep_buffer[env_ids] = torch.zeros_like(self.timestep_buffer[env_ids])
        self.success_buffer[env_ids] = torch.zeros_like(self.success_buffer[env_ids])

    def render(self) -> None:
        """Render the environment."""
        self.handler.render()

    def close(self) -> None:
        """Close the environment."""
        self.handler.close()

    def set_verbose(self, verbose: bool) -> None:
        """Enable or disable verbose timing output"""
        self.verbose = verbose

    def print_timing_stats(self) -> None:
        """Print timing statistics"""
        if self.total_steps == 0:
            return

        print("=== Timing Statistics ===")
        print(f"Total steps: {self.total_steps}")
        print(f"Total time: {self.total_time:.4f}s")
        print(f"Average time per step: {self.total_time / self.total_steps:.4f}s")
        print("\nBreakdown by operation:")
        for op, time_in_secs in self.step_timings.items():
            avg_time = time_in_secs / self.total_steps
            percentage = (time_in_secs / self.total_time) * 100
            print(f"{op:30s}: {avg_time:.4f}s ({percentage:.1f}%)")
        print("=====================")
