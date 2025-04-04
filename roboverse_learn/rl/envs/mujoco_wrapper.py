from __future__ import annotations

import multiprocessing as mp
import random
import time
from contextlib import contextmanager
from multiprocessing import Pipe, Process, connection

import numpy as np
import torch
from gym import spaces

from metasim.cfg.scenario import ScenarioCfg
from metasim.sim.mujoco.mujoco import MujocoHandler


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


def worker_process(
    rank: int,
    scenario: ScenarioCfg,
    parent_conn: connection.Connection,
    child_conn: connection.Connection,
    headless: bool = True,
):
    """Worker process function for managing an individual MujocoHandler instance"""
    parent_conn.close()
    scenario.num_envs = 1
    scenario.headless = headless
    handler = MujocoHandler(scenario)
    handler.launch()

    # Important: Initialize these by getting state once at the beginning
    # This ensures all workers have joint_names and action_dim
    states = handler.get_states()[0]  # Single env
    robot_name = handler._robot.name

    # Get joint names from robot
    robot_joint_names = list(states[robot_name]["dof_pos"].keys())

    # Also collect joint names from articulated objects
    articulated_joint_names = []
    articulated_objects = {}

    # Identify articulated objects and their joints
    for obj_name, obj_state in states.items():
        if obj_name != robot_name and "dof_pos" in obj_state:
            articulated_objects[obj_name] = list(obj_state["dof_pos"].keys())
            for joint_name in obj_state["dof_pos"].keys():
                articulated_joint_names.append(f"{obj_name}_{joint_name}")

    # Combine robot joints and articulated object joints
    joint_names = robot_joint_names + articulated_joint_names
    action_dim = len(robot_joint_names)  # Action dimension is just robot joints

    # Send an initialization confirmation to the parent
    child_conn.send(("init_done", (joint_names, action_dim, robot_joint_names, articulated_objects)))

    while True:
        try:
            cmd, data = child_conn.recv()

            if cmd == "step":
                action = data
                # Only use robot joints for the action
                formatted_action = [{"dof_pos_target": {}}]

                # Only apply actions to robot joints
                for i, joint_name in enumerate(robot_joint_names):
                    if i < len(action):  # Safety check
                        formatted_action[0]["dof_pos_target"][joint_name] = float(action[i])

                handler._episode_length_buf += 1
                handler.set_dof_targets(robot_name, formatted_action)
                handler.simulate()
                obs = handler.get_observation()
                success = handler.checker.check(handler)
                reward = handler.scenario.task.reward_fn(handler.get_states())
                timeout = torch.tensor([handler._episode_length_buf >= scenario.episode_length])
                extra = {}

                # Get joint states for observation
                states = handler.get_states()[0]  # Single env
                joint_positions = []

                # Extract joint positions from robot
                for joint_name in robot_joint_names:
                    joint_positions.append(states[robot_name]["dof_pos"][joint_name])

                # Extract joint positions from articulated objects
                for obj_name, obj_joints in articulated_objects.items():
                    if obj_name in states and "dof_pos" in states[obj_name]:
                        for joint_name in obj_joints:
                            if joint_name in states[obj_name]["dof_pos"]:
                                joint_positions.append(states[obj_name]["dof_pos"][joint_name])
                            else:
                                joint_positions.append(0.0)  # Default value if joint not found

                # Format observation for RL
                obs_dict = {
                    "obs": torch.tensor(joint_positions, dtype=torch.float32),
                    "rgb": obs["rgb"],
                    "depth": obs["depth"],
                }

                # Convert to tensors for consistency
                if not isinstance(reward, torch.Tensor):
                    reward = torch.tensor(reward if reward is not None else 0.0, dtype=torch.float32)
                if not isinstance(success, torch.Tensor):
                    success = torch.tensor(success if success is not None else False, dtype=torch.bool)
                if not isinstance(timeout, torch.Tensor):
                    timeout = torch.tensor(timeout if timeout is not None else False, dtype=torch.bool)

                # Return observation, reward, done, info
                done = success or timeout
                child_conn.send((obs_dict, reward, done, timeout, success, extra))

            elif cmd == "reset":
                # Reset the environment
                # obs, extra = handler.reset()
                handler._episode_length_buf = 0
                handler.simulate()
                obs = handler.get_observation()
                handler.checker.reset(handler)
                extra = {}

                # Create a new state dictionary that follows the expected format for set_states
                state_dict = {}

                # Set up robot state with default values from config
                if hasattr(scenario.robot, "name"):
                    robot_name = scenario.robot.name
                    state_dict[robot_name] = {}

                    # Set default position if available
                    if hasattr(scenario.robot, "default_position"):
                        state_dict[robot_name]["pos"] = list(scenario.robot.default_position)

                    # Set default joint positions if available
                    if hasattr(scenario.robot, "default_joint_positions"):
                        state_dict[robot_name]["dof_pos"] = {}
                        for joint_name, default_pos in scenario.robot.default_joint_positions.items():
                            state_dict[robot_name]["dof_pos"][joint_name] = default_pos

                # Set up articulated object states with default values
                if hasattr(scenario, "objects"):
                    for obj_config in scenario.objects:
                        if hasattr(obj_config, "name"):
                            obj_name = obj_config.name
                            state_dict[obj_name] = {}

                            # Set default position if available
                            if hasattr(obj_config, "default_position"):
                                state_dict[obj_name]["pos"] = list(obj_config.default_position)

                            # Set default joint positions if available
                            if hasattr(obj_config, "default_joint_positions"):
                                state_dict[obj_name]["dof_pos"] = {}
                                for joint_name, default_pos in obj_config.default_joint_positions.items():
                                    state_dict[obj_name]["dof_pos"][joint_name] = default_pos

                # Apply the state directly using handler.set_states
                handler.set_states([state_dict])

                # Now get the observation AFTER applying the default state
                states = handler.get_states()[0]  # Single env
                joint_positions = []

                # Extract joint positions from robot
                for joint_name in robot_joint_names:
                    joint_positions.append(states[robot_name]["dof_pos"][joint_name])

                # Extract joint positions from articulated objects
                for obj_name, obj_joints in articulated_objects.items():
                    if obj_name in states and "dof_pos" in states[obj_name]:
                        for joint_name in obj_joints:
                            if joint_name in states[obj_name]["dof_pos"]:
                                joint_positions.append(states[obj_name]["dof_pos"][joint_name])
                            else:
                                joint_positions.append(0.0)  # Default value if joint not found

                # Get the updated observation with RGB and depth if available
                updated_obs = handler.get_observation()

                # Create observation dictionary with consistent format
                obs_dict = {
                    "obs": torch.tensor(joint_positions, dtype=torch.float32),
                }

                # Include RGB and depth if available
                if "rgb" in updated_obs:
                    obs_dict["rgb"] = updated_obs["rgb"]
                if "depth" in updated_obs:
                    obs_dict["depth"] = updated_obs["depth"]

                # Send the observation dictionary
                child_conn.send(obs_dict)

            elif cmd == "get_info":
                # Get environment information (joints, observation space, action space, etc.)
                child_conn.send((joint_names, action_dim))

            elif cmd == "set_states":
                states = data
                handler.set_states(states)
                child_conn.send(True)

            elif cmd == "close":
                handler.close()
                child_conn.send(True)
                break

            elif cmd == "seed":
                # We'll implement a minimal seed operation
                random.seed(data)
                np.random.seed(data)
                child_conn.send(True)

            elif cmd == "render":
                img = handler.render()
                child_conn.send(img)

            else:
                print(f"Unknown command: {cmd}")

        except (EOFError, KeyboardInterrupt):
            break

    child_conn.close()


class MujocoWrapper:
    """Wrapper for MujocoHandler that supports multiple environments with multiprocessing."""

    def __init__(
        self,
        scenario: ScenarioCfg,
        num_envs: int = 1,
        headless: bool = True,
        seed: int | None = None,
        rgb_observation: bool = False,
    ):
        """Initialize the wrapper with multiple MujocoHandler instances."""
        self.scenario = scenario
        self.num_envs = num_envs
        self.headless = headless
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Worker processes and pipes
        self.processes = []
        self.parent_conns = []
        self.child_conns = []

        # Environment information
        self._task = scenario.task
        self._robot = scenario.robot

        # Additional variables for RL
        self.max_episode_length = 30 if not hasattr(self._task, "episode_length") else self._task.episode_length
        self.reset_buffer = torch.zeros(self.num_envs, 1, device=self.device)
        self.timestep_buffer = torch.zeros(self.num_envs, 1, device=self.device)

        # Success tracking
        self.success_buffer = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Timing tracking
        self.verbose = False
        self.step_timings = {}
        self.total_steps = 0
        self.total_time = 0.0

        # To be initialized during launch
        self.observation_space = None
        self.action_space = None
        self.joint_names = None
        self.action_dim = None
        self.robot_joint_names = None
        self.articulated_objects = None

        self.rgb_observation = False

        # Set seed if provided
        if seed is not None:
            self.set_seed(seed)

    def launch(self) -> None:
        """Launch the simulation with multiple processes."""
        # Create processes for each environment
        mp.set_start_method("spawn", force=True)
        for i in range(self.num_envs):
            parent_conn, child_conn = Pipe()
            process = Process(
                target=worker_process, args=(i, self.scenario, parent_conn, child_conn, self.headless), daemon=True
            )
            self.processes.append(process)
            self.parent_conns.append(parent_conn)
            self.child_conns.append(child_conn)
            process.start()

        # Wait for initialization from all workers
        init_results = []
        for conn in self.parent_conns:
            cmd, data = (
                conn.recv()
            )  # Should receive ('init_done', (joint_names, action_dim, robot_joint_names, articulated_objects))
            if cmd == "init_done":
                init_results.append(data)
            else:
                print(f"Warning: Expected 'init_done' but got '{cmd}'")
                # Use dummy data if initialization failed
                init_results.append(([], 0, [], {}))

        # Verify all environments have the same joint names and action dimensions
        if len(init_results) > 0:
            self.joint_names, self.action_dim, self.robot_joint_names, self.articulated_objects = init_results[0]

            # Check that all environments have the same configuration
            for i, (names, dim, robot_names, artic_objs) in enumerate(init_results[1:], 1):
                if names != self.joint_names or dim != self.action_dim:
                    print(f"Warning: Environment {i} has different configuration")
                    print(f"  Expected: {len(self.joint_names)} joints, {self.action_dim} actions")
                    print(f"  Got: {len(names)} joints, {dim} actions")

        # Set up observation and action spaces
        if self.joint_names:
            obs_dim = len(self.joint_names)
            if self.rgb_observation:
                self.observation_space = spaces.Dict({
                    "obs": spaces.Box(
                        low=0,
                        high=255,
                        shape=(3, self.scenario.camera.height, self.scenario.camera.width),
                        dtype=np.float32,
                    ),
                })
            else:
                self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)

            print(f"Environment launched with {self.num_envs} workers")
            print(f"Observation dimension: {obs_dim}")
            print(f"Action dimension: {self.action_dim}")
            print(f"Joint names: {self.joint_names}")
        else:
            print("Warning: Failed to initialize environments properly. No joint information available.")

    def step(
        self, action: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step all environments forward."""
        step_start = time.time()
        self.total_steps += 1

        if self.verbose:
            print(f"\n=== Step {self.total_steps} ===")

        self.timestep_buffer += 1

        # Convert action tensor to list of dictionaries for each environment
        if len(action.shape) == 1:  # Single action for one environment
            action = action.unsqueeze(0)  # Add batch dimension

        # Convert actions to CPU numpy for multiprocessing
        action_np = action.detach().cpu().numpy()

        # Send step command to all worker processes
        with timing_context("send_actions", self.verbose, self.step_timings):
            for i, conn in enumerate(self.parent_conns):
                # Only send the action values - the worker will map them to the correct robot joints
                conn.send(("step", action_np[i].tolist()))

        # Collect results from all worker processes
        with timing_context("collect_results", self.verbose, self.step_timings):
            results = [conn.recv() for conn in self.parent_conns]

        # Process results
        obs_list, reward_list, done_list, timeout_list, success_list, info_list = zip(*results)

        # Update tracking buffers
        self.success_buffer = torch.stack([result[4] for result in results]).to(self.device)

        # Combine observations
        with timing_context("process_observations", self.verbose, self.step_timings):
            combined_obs = {
                "obs": torch.stack([obs["obs"] for obs in obs_list]).to(self.device),
                "rgb": torch.stack([obs["rgb"] for obs in obs_list]).float().squeeze(1).to(self.device),
                "depth": torch.stack([obs["depth"] for obs in obs_list]).float().squeeze(1).to(self.device),
            }

        # Combine rewards and done flags
        rewards = torch.tensor([float(r) for r in reward_list], device=self.device)
        dones = torch.tensor([bool(d) for d in done_list], device=self.device)
        timeouts = torch.tensor([bool(t) for t in timeout_list], device=self.device)

        # Handle resets for environments that are done
        with timing_context("handle_resets", self.verbose, self.step_timings):
            done_indices = torch.nonzero(dones).squeeze(-1).tolist()
            if done_indices:
                self.reset_idx(done_indices)

        # Update timing statistics
        step_time = time.time() - step_start
        self.total_time += step_time

        if self.verbose:
            print(f"Step {self.total_steps} completed in {step_time:.4f}s")
            if torch.any(self.success_buffer):
                success_envs = torch.nonzero(self.success_buffer).squeeze(-1).tolist()
                print(f"Success in environments: {success_envs}")
            if torch.any(timeouts):
                timeout_envs = torch.nonzero(timeouts).squeeze(-1).tolist()
                print(f"Timeout in environments: {timeout_envs}")
            print("=" * 30 + "\n")

        info = {"success": self.success_buffer}
        rewards = rewards.unsqueeze(0).T
        timeouts = timeouts.unsqueeze(0).T
        return combined_obs, rewards, dones, timeouts, info

    def reset(self) -> dict[str, torch.Tensor]:
        """Reset all environments."""
        with timing_context("reset_all", self.verbose, self.step_timings):
            # Send reset command to all worker processes
            for conn in self.parent_conns:
                conn.send(("reset", None))

            # Collect results with proper handling for consistent format
            obs_list = []
            for conn in self.parent_conns:
                try:
                    # Receive the observation dictionary from the worker
                    obs_dict = conn.recv()

                    # Make sure we actually got a dictionary with 'obs' key
                    if isinstance(obs_dict, dict) and "obs" in obs_dict:
                        obs_list.append(obs_dict)
                    else:
                        # If we got something else (like tuple of joint_names, action_dim),
                        # create a default observation
                        print(f"Warning: Received unexpected data format from worker: {type(obs_dict)}")
                        default_obs = {"obs": torch.zeros(len(self.joint_names), dtype=torch.float32)}
                        obs_list.append(default_obs)
                except Exception as e:
                    print(f"Error receiving observation from worker: {e}")
                    # Create default observation if there's an error
                    default_obs = {"obs": torch.zeros(len(self.joint_names), dtype=torch.float32)}
                    obs_list.append(default_obs)

            # Reset internal state
            self.reset_buffer = torch.ones_like(self.reset_buffer)
            self.timestep_buffer = torch.zeros_like(self.timestep_buffer)
            self.success_buffer = torch.zeros_like(self.success_buffer)

            # Combine observations
            obs_tensors = [obs["obs"] for obs in obs_list]
            rgb_tensors = [obs["rgb"] for obs in obs_list]
            depth_tensors = [obs["depth"] for obs in obs_list]
            combined_obs = {
                "obs": torch.stack(obs_tensors).to(self.device),
                "rgb": torch.stack(rgb_tensors).float().squeeze(1).to(self.device),
                "depth": torch.stack(depth_tensors).float().squeeze(1).to(self.device),
            }

        return combined_obs

    def reset_idx(self, env_ids: list[int]) -> None:
        """Reset specific environments."""
        reset_obses = {}
        reset_obses["obs"] = []
        reset_obses["rgb"] = []
        reset_obses["depth"] = []
        with timing_context("reset_specific", self.verbose, self.step_timings):
            # Send reset command to specified environments
            for idx in env_ids:
                self.parent_conns[idx].send(("reset", None))

            # Collect results with the same error handling as the main reset
            for i, idx in enumerate(env_ids):
                try:
                    # Receive the observation
                    obs_dict = self.parent_conns[idx].recv()
                    reset_obses["obs"].append(obs_dict["obs"])
                    if "rgb" in obs_dict:
                        reset_obses["rgb"].append(obs_dict["rgb"])
                    if "depth" in obs_dict:
                        reset_obses["depth"].append(obs_dict["depth"])
                except Exception as e:
                    print(f"Error receiving observation from worker {idx}: {e}")

            # Reset timestep buffer for these environments
            self.timestep_buffer[env_ids] = torch.zeros_like(self.timestep_buffer[env_ids])
            self.success_buffer[env_ids] = torch.zeros_like(self.success_buffer[env_ids])

        return reset_obses

    def set_states(self, states: list[dict]) -> None:
        """Set the states of all environments."""
        assert len(states) == self.num_envs, "Number of states must match number of environments"

        for i, conn in enumerate(self.parent_conns):
            conn.send(("set_states", [states[i]]))  # Wrap in list for handler

        # Wait for all environments to complete
        [conn.recv() for conn in self.parent_conns]

    def render(self, env_idx: int = 0) -> np.ndarray:
        """Render a specific environment."""
        self.parent_conns[env_idx].send(("render", None))
        return self.parent_conns[env_idx].recv()

    def close(self) -> None:
        """Close all environments."""
        for conn in self.parent_conns:
            conn.send(("close", None))

        # Wait for all environments to close with timeout
        for conn in self.parent_conns:
            try:
                conn.recv(timeout=1.0)  # Add 1 second timeout
            except:
                pass  # Continue even if timeout occurs

        # Terminate processes
        for process in self.processes:
            process.terminate()
            process.join(timeout=1.0)  # Add timeout to join

    def set_seed(self, seed: int) -> None:
        """Set random seed for all environments."""
        if seed == -1 and torch.cuda.is_available():
            seed = torch.randint(0, 10000, (1,))[0].item()
        if seed != -1:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # Set seed for each environment
            for i, conn in enumerate(self.parent_conns):
                conn.send(("seed", seed + i))

            # Wait for all environments to be seeded
            for conn in self.parent_conns:
                try:
                    conn.recv(timeout=1.0)  # Add timeout
                except:
                    pass  # Continue even if timeout occurs

        return seed

    def set_verbose(self, verbose: bool) -> None:
        """Enable or disable verbose timing output."""
        self.verbose = verbose

    def print_timing_stats(self) -> None:
        """Print timing statistics."""
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
