"""Ant task wrapper for IsaacGymEnvs.

This wrapper encapsulates the task-specific logic for the Ant
locomotion task, providing cleaner separation between the
task implementation and the RL infrastructure.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from gymnasium import spaces

from metasim.utils.math import (
    euler_xyz_from_quat,
    normalize,
    quat_inv,
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
)
from roboverse_learn.rl.task_registry import register_task_wrapper
from roboverse_learn.rl.task_wrapper import IsaacGymEnvsTaskWrapper

log = logging.getLogger(__name__)


@register_task_wrapper("isaacgym_envs:AntIsaacGym")
@register_task_wrapper("mujoco:AntMujoco")
class AntTaskWrapper(IsaacGymEnvsTaskWrapper):
    """Task wrapper for Ant locomotion.

    This task involves controlling an 8-DOF ant robot to walk
    towards a target location while maintaining balance.
    """

    def __init__(self, env, cfg, sim_type: str = "isaacgym"):
        super().__init__(env, cfg, sim_type)

        # Task-specific parameters from config
        self.dof_vel_scale = getattr(cfg, "dof_vel_scale", 0.2)
        self.contact_force_scale = getattr(cfg, "contact_force_scale", 0.1)
        self.power_scale = getattr(cfg, "power_scale", 1.0)
        self.heading_weight = getattr(cfg, "heading_weight", 0.5)
        self.up_weight = getattr(cfg, "up_weight", 0.1)
        self.actions_cost_scale = getattr(cfg, "actions_cost_scale", 0.005)
        self.energy_cost_scale = getattr(cfg, "energy_cost_scale", 0.05)
        self.joints_at_limit_cost_scale = getattr(cfg, "joints_at_limit_cost_scale", 0.1)
        self.death_cost = getattr(cfg, "death_cost", -2.0)
        self.termination_height = getattr(cfg, "termination_height", 0.31)
        self.initial_height = getattr(cfg, "initial_height", 0.55)

        # Observation and action dimensions
        self.num_obs = 60
        self.num_actions = 8  # Ant has 8 joints

        # Initialize buffers
        self.initialize_buffers()

        # Task-specific state
        self._targets = None
        self._potentials = None
        self._prev_potentials = None
        self._up_vec = None
        self._heading_vec = None
        self._inv_start_rot = None
        self._basis_vec0 = None
        self._basis_vec1 = None
        self._actions = None
        self._joint_gears = None
        self._dof_limits_lower = None
        self._dof_limits_upper = None

        # MuJoCo-specific buffers (for parallel environments)
        self._mujoco_targets = None
        self._mujoco_potentials = None
        self._mujoco_prev_potentials = None
        self._mujoco_actions = None
        self._mujoco_inv_start_rot = None
        self._mujoco_basis_vec0 = None
        self._mujoco_basis_vec1 = None

        # Initialize task-specific buffers
        self._init_task_buffers()

    def _init_task_buffers(self):
        """Initialize task-specific buffers."""
        deg_to_rad = 3.14159 / 180.0

        # DOF limits for Ant
        self._dof_limits_lower = torch.tensor(
            [
                -40 * deg_to_rad,
                30 * deg_to_rad,
                -40 * deg_to_rad,
                -100 * deg_to_rad,
                -40 * deg_to_rad,
                -100 * deg_to_rad,
                -40 * deg_to_rad,
                30 * deg_to_rad,
            ],
            dtype=torch.float32,
            device=self.device,
        )

        self._dof_limits_upper = torch.tensor(
            [
                40 * deg_to_rad,
                100 * deg_to_rad,
                40 * deg_to_rad,
                -30 * deg_to_rad,
                40 * deg_to_rad,
                -30 * deg_to_rad,
                40 * deg_to_rad,
                100 * deg_to_rad,
            ],
            dtype=torch.float32,
            device=self.device,
        )

    def _build_observation_space(self) -> spaces.Space:
        """Build observation space for Ant task."""
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32)

    def _build_action_space(self) -> spaces.Space:
        """Build action space for Ant control."""
        return spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32)

    def get_observation_isaacgym(self, states) -> torch.Tensor:
        """Extract observations from IsaacGym states.

        Handles the TensorState format specific to IsaacGym.
        """
        tensor_state = self._get_tensor_state(states)

        # Get robot state
        robot_state = tensor_state.robots.get("ant")
        if robot_state is None:
            log.error("Missing ant robot in state")
            return self.obs_buf

        # Get batch size from actual tensor
        batch_size = robot_state.root_state.shape[0]

        # Extract root states
        torso_pos = robot_state.root_state[:, 0:3]
        torso_rot = robot_state.root_state[:, 3:7]
        lin_vel = robot_state.root_state[:, 7:10]
        ang_vel = robot_state.root_state[:, 10:13]

        # Joint states
        dof_pos = robot_state.joint_pos
        dof_vel = robot_state.joint_vel

        # Initialize buffers if needed
        if self._targets is None or self._targets.shape[0] != batch_size:
            self._targets = torch.tensor([1000.0, 0.0, 0.0], dtype=torch.float32, device=self.device).repeat((
                batch_size,
                1,
            ))

        if self._potentials is None or self._potentials.shape[0] != batch_size:
            dt = 0.01667
            self._potentials = torch.zeros(batch_size, dtype=torch.float32, device=self.device) - 1000.0 / dt
            self._prev_potentials = self._potentials.clone()

        if self._inv_start_rot is None or self._inv_start_rot.shape[0] != batch_size:
            start_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
            self._inv_start_rot = quat_inv(start_rot).repeat((batch_size, 1))

        if self._basis_vec0 is None or self._basis_vec0.shape[0] != batch_size:
            self._basis_vec0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device).repeat((
                batch_size,
                1,
            ))
            self._basis_vec1 = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device).repeat((
                batch_size,
                1,
            ))

        # Compute target-related values
        to_target = self._targets - torso_pos
        to_target[:, 2] = 0.0

        # Compute heading and up projections
        torso_quat, up_proj, heading_proj, up_vec, heading_vec = self.compute_heading_and_up(
            torso_rot,
            self._inv_start_rot,
            to_target,
            self._basis_vec0,
            self._basis_vec1,
            2,  # up_idx
        )

        # Compute rotations
        vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = self.compute_rot(
            torso_quat, lin_vel, ang_vel, self._targets, torso_pos
        )

        # Scale DOF positions
        dof_pos_scaled = (2.0 * dof_pos - self._dof_limits_upper - self._dof_limits_lower) / (
            self._dof_limits_upper - self._dof_limits_lower
        )

        # Get sensor forces (contact forces)
        sensor_forces = torch.zeros((batch_size, 24), dtype=torch.float32, device=self.device)
        if hasattr(robot_state, "sensor_forces") and robot_state.sensor_forces is not None:
            # Flatten and take first 24 values
            sensor_forces = robot_state.sensor_forces.view(batch_size, -1)[:, :24]

        # Get previous actions
        if self._actions is None or self._actions.shape[0] != batch_size:
            self._actions = torch.zeros((batch_size, self.num_actions), dtype=torch.float32, device=self.device)

        # Assemble observation
        obs = torch.cat(
            [
                torso_pos[:, 2:3],  # 1 - height
                vel_loc,  # 3 - local velocity
                angvel_loc,  # 3 - local angular velocity
                yaw,  # 1 - yaw angle
                roll,  # 1 - roll angle
                angle_to_target,  # 1 - angle to target
                up_proj,  # 1 - up projection
                heading_proj,  # 1 - heading projection
                dof_pos_scaled,  # 8 - scaled joint positions
                dof_vel * self.dof_vel_scale,  # 8 - scaled joint velocities
                sensor_forces * self.contact_force_scale,  # 24 - scaled contact forces
                self._actions,  # 8 - previous actions
            ],
            dim=-1,
        )

        self.obs_buf[:] = obs
        return self.obs_buf

    def get_observation_mujoco(self, states) -> np.ndarray:
        """Extract observations from MuJoCo states with support for parallel environments."""
        batch_size = len(states)
        obs = np.zeros((batch_size, self.num_obs))

        # Initialize MuJoCo buffers if needed
        if self._mujoco_targets is None or len(self._mujoco_targets) != batch_size:
            self._init_mujoco_buffers(batch_size)

        # Vectorized extraction for better performance
        torso_positions = np.zeros((batch_size, 3))
        torso_rotations = np.zeros((batch_size, 4))
        lin_velocities = np.zeros((batch_size, 3))
        ang_velocities = np.zeros((batch_size, 3))
        dof_positions = np.zeros((batch_size, 8))
        dof_velocities = np.zeros((batch_size, 8))
        sensor_forces = np.zeros((batch_size, 24))

        for i, state in enumerate(states):
            # Extract robot state
            robot_state = state.get("robots", {}).get("ant", {})

            # Get position and orientation
            torso_positions[i] = robot_state.get("pos", [0.0, 0.0, self.initial_height])
            torso_rotations[i] = robot_state.get("rot", [1.0, 0.0, 0.0, 0.0])
            lin_velocities[i] = robot_state.get("lin_vel", robot_state.get("vel", [0.0, 0.0, 0.0]))
            ang_velocities[i] = robot_state.get("ang_vel", [0.0, 0.0, 0.0])

            # Get joint states
            if "joint_qpos" in robot_state:
                dof_positions[i] = robot_state["joint_qpos"]
            elif "dof_pos" in robot_state:
                dof_positions[i] = list(robot_state["dof_pos"].values())

            if "joint_qvel" in robot_state:
                dof_velocities[i] = robot_state["joint_qvel"]
            elif "dof_vel" in robot_state:
                dof_velocities[i] = list(robot_state["dof_vel"].values())

            # Get sensor forces if available
            if "sensor_forces" in robot_state:
                forces = np.array(robot_state["sensor_forces"]).flatten()
                sensor_forces[i, : min(24, len(forces))] = forces[:24]

        # Compute target-related values
        to_target = self._mujoco_targets - torso_positions
        to_target[:, 2] = 0.0

        # Compute heading and up projections (numpy version)
        torso_quat, up_proj, heading_proj = self._compute_heading_and_up_numpy(
            torso_rotations, self._mujoco_inv_start_rot, to_target, self._mujoco_basis_vec0, self._mujoco_basis_vec1
        )

        # Compute rotations
        vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = self._compute_rot_numpy(
            torso_quat, lin_velocities, ang_velocities, self._mujoco_targets, torso_positions
        )

        # Scale DOF positions
        dof_limits_range = self._dof_limits_upper.cpu().numpy() - self._dof_limits_lower.cpu().numpy()
        dof_limits_sum = self._dof_limits_upper.cpu().numpy() + self._dof_limits_lower.cpu().numpy()
        dof_pos_scaled = (2.0 * dof_positions - dof_limits_sum) / dof_limits_range

        # Assemble observations
        obs[:, 0] = torso_positions[:, 2]  # height
        obs[:, 1:4] = vel_loc  # local velocity
        obs[:, 4:7] = angvel_loc  # local angular velocity
        obs[:, 7] = yaw
        obs[:, 8] = roll
        obs[:, 9] = angle_to_target
        obs[:, 10] = up_proj
        obs[:, 11] = heading_proj
        obs[:, 12:20] = dof_pos_scaled  # scaled joint positions
        obs[:, 20:28] = dof_velocities * self.dof_vel_scale  # scaled joint velocities
        obs[:, 28:52] = sensor_forces * self.contact_force_scale  # scaled contact forces
        obs[:, 52:60] = self._mujoco_actions  # previous actions

        return obs

    def compute_reward_isaacgym(self, states, actions, next_states) -> torch.Tensor:
        """Compute rewards for IsaacGym states."""
        tensor_state = self._get_tensor_state(next_states)

        # Get robot state
        robot_state = tensor_state.robots.get("ant")
        if robot_state is None:
            return self.reward_buf

        # Extract states
        torso_pos = robot_state.root_state[:, 0:3]
        torso_rot = robot_state.root_state[:, 3:7]
        lin_vel = robot_state.root_state[:, 7:10]
        ang_vel = robot_state.root_state[:, 10:13]
        dof_pos = robot_state.joint_pos
        dof_vel = robot_state.joint_vel

        batch_size = torso_pos.shape[0]

        # Initialize targets if needed
        if self._targets is None or self._targets.shape[0] != batch_size:
            self._targets = torch.tensor([1000.0, 0.0, 0.0], device=torso_pos.device).repeat((batch_size, 1))

        # Compute to target
        to_target = self._targets - torso_pos
        to_target[:, 2] = 0.0

        # Update potentials
        dt = 0.01667
        if self._prev_potentials is not None:
            self._prev_potentials = self._potentials.clone()
        self._potentials = -torch.norm(to_target, p=2, dim=-1) / dt

        if self._prev_potentials is None:
            self._prev_potentials = self._potentials.clone()

        # Initialize rotation buffers if needed
        if self._inv_start_rot is None or self._inv_start_rot.shape[0] != batch_size:
            start_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=torso_pos.device)
            self._inv_start_rot = quat_inv(start_rot).repeat((batch_size, 1))

        if self._basis_vec0 is None or self._basis_vec0.shape[0] != batch_size:
            self._basis_vec0 = torch.tensor([1.0, 0.0, 0.0], device=torso_pos.device).repeat((batch_size, 1))
            self._basis_vec1 = torch.tensor([0.0, 0.0, 1.0], device=torso_pos.device).repeat((batch_size, 1))

        # Compute heading and up
        torso_quat, up_proj, heading_proj, up_vec, heading_vec = self.compute_heading_and_up(
            torso_rot, self._inv_start_rot, to_target, self._basis_vec0, self._basis_vec1, 2
        )

        # Heading reward
        heading_proj_1d = heading_proj.squeeze(-1) if heading_proj.dim() > 1 else heading_proj
        heading_weight_tensor = torch.ones_like(heading_proj_1d) * self.heading_weight
        heading_reward = torch.where(
            heading_proj_1d > 0.8, heading_weight_tensor, self.heading_weight * heading_proj_1d / 0.8
        )

        # Up reward
        up_proj_1d = up_proj.squeeze(-1) if up_proj.dim() > 1 else up_proj
        up_reward = torch.zeros_like(heading_reward)
        up_reward = torch.where(up_proj_1d > 0.93, up_reward + self.up_weight, up_reward)

        # Convert actions to tensor if needed
        if isinstance(actions, list) and len(actions) > 0 and isinstance(actions[0], dict):
            actions_tensor = torch.zeros((batch_size, self.num_actions), device=torso_pos.device)
            for i, act in enumerate(actions):
                if isinstance(act, dict):
                    robot_name = next(iter(act.keys()))
                    if "dof_pos_target" in act[robot_name]:
                        joint_actions = act[robot_name]["dof_pos_target"]
                        # Map joint names to indices
                        action_values = [
                            joint_actions.get(f"hip_{(j // 2) + 1}", 0.0)
                            if j % 2 == 0
                            else joint_actions.get(f"ankle_{(j // 2) + 1}", 0.0)
                            for j in range(8)
                        ]
                        actions_tensor[i] = torch.tensor(action_values, device=torso_pos.device)
        elif isinstance(actions, torch.Tensor):
            actions_tensor = actions.to(torso_pos.device)
        else:
            actions_tensor = torch.zeros((batch_size, self.num_actions), device=torso_pos.device)

        # Store actions for next observation
        self._actions = actions_tensor.clone()

        # Action and energy costs
        actions_cost = torch.sum(actions_tensor**2, dim=-1)
        electricity_cost = torch.sum(torch.abs(actions_tensor * dof_vel), dim=-1)

        # Joint limit cost
        dof_pos_scaled = (2.0 * dof_pos - self._dof_limits_upper - self._dof_limits_lower) / (
            self._dof_limits_upper - self._dof_limits_lower
        )
        dof_at_limit_cost = torch.sum(dof_pos_scaled > 0.99, dim=-1).float()

        # Alive reward
        alive_reward = torch.ones_like(self._potentials) * 0.5

        # Progress reward
        progress_reward = self._potentials - self._prev_potentials

        # Total reward
        total_reward = (
            progress_reward
            + alive_reward
            + up_reward
            + heading_reward
            - self.actions_cost_scale * actions_cost
            - self.energy_cost_scale * electricity_cost
            - dof_at_limit_cost * self.joints_at_limit_cost_scale
        )

        # Death penalty
        total_reward = torch.where(
            torso_pos[:, 2] < self.termination_height, torch.ones_like(total_reward) * self.death_cost, total_reward
        )

        # Ensure reward is 1D
        if total_reward.dim() > 1:
            total_reward = total_reward.squeeze(-1)

        self.reward_buf[:] = total_reward
        return self.reward_buf

    def check_termination_isaacgym(self, states) -> torch.Tensor:
        """Check termination conditions for IsaacGym."""
        tensor_state = self._get_tensor_state(states)
        robot_state = tensor_state.robots.get("ant")

        if robot_state is None:
            return self.reset_buf

        # Terminate if ant falls below threshold height
        torso_height = robot_state.root_state[:, 2]
        terminations = torso_height < self.termination_height

        self.reset_buf[:] = terminations.float()
        return self.reset_buf

    def reset_task_isaacgym(self, env_ids: list[int] | None = None):
        """Reset task-specific state for IsaacGym."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Reset potentials for specified environments
        if self._potentials is not None:
            dt = 0.01667
            self._potentials[env_ids] = -1000.0 / dt
            if self._prev_potentials is not None:
                self._prev_potentials[env_ids] = self._potentials[env_ids]

        # Reset actions
        if self._actions is not None:
            self._actions[env_ids] = 0

    def _get_observation_generic(self, states) -> np.ndarray | torch.Tensor:
        """Fallback observation extraction."""
        # Try to determine state format and dispatch accordingly
        if hasattr(states, "__class__") and states.__class__.__name__ == "TensorState":
            return self.get_observation_isaacgym(states)
        elif isinstance(states, (list, tuple)) and len(states) > 0 and isinstance(states[0], dict):
            return self.get_observation_mujoco(states)
        else:
            log.warning(f"Unknown state format: {type(states)}")
            return self.obs_buf if hasattr(self, "obs_buf") else np.zeros((1, self.num_obs))

    def compute_reward_mujoco(self, states, actions, next_states) -> np.ndarray:
        """Compute rewards for MuJoCo states."""
        batch_size = len(next_states)
        rewards = np.zeros(batch_size)

        # Extract states for all environments
        torso_positions = np.zeros((batch_size, 3))
        torso_rotations = np.zeros((batch_size, 4))
        lin_velocities = np.zeros((batch_size, 3))
        dof_positions = np.zeros((batch_size, 8))
        dof_velocities = np.zeros((batch_size, 8))

        for i, state in enumerate(next_states):
            robot_state = state.get("robots", {}).get("ant", {})
            torso_positions[i] = robot_state.get("pos", [0.0, 0.0, self.initial_height])
            torso_rotations[i] = robot_state.get("rot", [1.0, 0.0, 0.0, 0.0])
            lin_velocities[i] = robot_state.get("lin_vel", robot_state.get("vel", [0.0, 0.0, 0.0]))

            if "joint_qpos" in robot_state:
                dof_positions[i] = robot_state["joint_qpos"]
            elif "dof_pos" in robot_state:
                dof_positions[i] = list(robot_state["dof_pos"].values())

            if "joint_qvel" in robot_state:
                dof_velocities[i] = robot_state["joint_qvel"]
            elif "dof_vel" in robot_state:
                dof_velocities[i] = list(robot_state["dof_vel"].values())

        # Initialize buffers if needed
        if self._mujoco_targets is None or len(self._mujoco_targets) != batch_size:
            self._init_mujoco_buffers(batch_size)

        # Compute to target
        to_target = self._mujoco_targets - torso_positions
        to_target[:, 2] = 0.0

        # Update potentials
        dt = 0.01667
        self._mujoco_prev_potentials = self._mujoco_potentials.copy()
        self._mujoco_potentials = -np.linalg.norm(to_target, axis=1) / dt

        # Compute heading and up
        torso_quat, up_proj, heading_proj = self._compute_heading_and_up_numpy(
            torso_rotations, self._mujoco_inv_start_rot, to_target, self._mujoco_basis_vec0, self._mujoco_basis_vec1
        )

        # Heading reward
        heading_reward = np.where(heading_proj > 0.8, self.heading_weight, self.heading_weight * heading_proj / 0.8)

        # Up reward
        up_reward = np.where(up_proj > 0.93, self.up_weight, 0.0)

        # Extract actions
        if isinstance(actions, np.ndarray):
            actions_array = actions
        elif isinstance(actions, list):
            actions_array = np.zeros((batch_size, self.num_actions))
            for i, act in enumerate(actions):
                if isinstance(act, dict):
                    robot_name = next(iter(act.keys()))
                    if "dof_pos_target" in act[robot_name]:
                        joint_actions = act[robot_name]["dof_pos_target"]
                        action_values = [
                            joint_actions.get(f"hip_{(j // 2) + 1}", 0.0)
                            if j % 2 == 0
                            else joint_actions.get(f"ankle_{(j // 2) + 1}", 0.0)
                            for j in range(8)
                        ]
                        actions_array[i] = action_values
                elif isinstance(act, (list, np.ndarray)):
                    actions_array[i] = act
        else:
            actions_array = np.zeros((batch_size, self.num_actions))

        # Store actions for next observation
        self._mujoco_actions = actions_array.copy()

        # Action and energy costs
        actions_cost = np.sum(actions_array**2, axis=1)
        electricity_cost = np.sum(np.abs(actions_array * dof_velocities), axis=1)

        # Joint limit cost
        dof_limits_range = self._dof_limits_upper.cpu().numpy() - self._dof_limits_lower.cpu().numpy()
        dof_limits_sum = self._dof_limits_upper.cpu().numpy() + self._dof_limits_lower.cpu().numpy()
        dof_pos_scaled = (2.0 * dof_positions - dof_limits_sum) / dof_limits_range
        dof_at_limit_cost = np.sum(dof_pos_scaled > 0.99, axis=1)

        # Alive reward
        alive_reward = 0.5

        # Progress reward
        progress_reward = self._mujoco_potentials - self._mujoco_prev_potentials

        # Total reward
        rewards = (
            progress_reward
            + alive_reward
            + up_reward
            + heading_reward
            - self.actions_cost_scale * actions_cost
            - self.energy_cost_scale * electricity_cost
            - dof_at_limit_cost * self.joints_at_limit_cost_scale
        )

        # Death penalty
        rewards = np.where(torso_positions[:, 2] < self.termination_height, self.death_cost, rewards)

        return rewards

    def _compute_reward_generic(self, states, actions, next_states):
        """Fallback reward computation."""
        if hasattr(states, "__class__") and states.__class__.__name__ == "TensorState":
            return self.compute_reward_isaacgym(states, actions, next_states)
        elif isinstance(states, (list, tuple)) and len(states) > 0 and isinstance(states[0], dict):
            return self.compute_reward_mujoco(states, actions, next_states)
        else:
            # Simple implementation for other simulators
            return np.zeros(len(actions))

    def check_termination_mujoco(self, states) -> np.ndarray:
        """Check termination conditions for MuJoCo."""
        batch_size = len(states)
        terminations = np.zeros(batch_size, dtype=bool)

        for i, state in enumerate(states):
            robot_state = state.get("robots", {}).get("ant", {})
            torso_pos = robot_state.get("pos", [0.0, 0.0, self.initial_height])

            # Terminate if ant falls below threshold height
            terminations[i] = torso_pos[2] < self.termination_height

        return terminations

    def _check_termination_generic(self, states):
        """Generic termination check."""
        if hasattr(states, "__class__") and states.__class__.__name__ == "TensorState":
            return self.check_termination_isaacgym(states)
        elif isinstance(states, (list, tuple)) and len(states) > 0 and isinstance(states[0], dict):
            return self.check_termination_mujoco(states)
        else:
            return False

    def reset_task_mujoco(self, env_ids: list[int] | None = None):
        """Reset task-specific state for MuJoCo."""
        if self._mujoco_potentials is None:
            return  # Buffers not initialized yet

        if env_ids is None:
            # Reset all environments
            batch_size = len(self._mujoco_potentials)
            env_ids = list(range(batch_size))

        # Reset potentials for specified environments
        dt = 0.01667
        for idx in env_ids:
            if idx < len(self._mujoco_potentials):
                self._mujoco_potentials[idx] = -1000.0 / dt
                self._mujoco_prev_potentials[idx] = self._mujoco_potentials[idx]
                self._mujoco_actions[idx] = 0

    def _reset_task_generic(self, env_ids: list[int] | None = None):
        """Generic task reset."""
        if self.sim_type == "mujoco":
            self.reset_task_mujoco(env_ids)
        elif self.sim_type == "isaacgym":
            self.reset_task_isaacgym(env_ids)

    def compute_heading_and_up(self, torso_rotation, inv_start_rot, to_target, vec0, vec1, up_idx):
        """Compute heading and up projections."""
        num_envs = torso_rotation.shape[0]
        target_dirs = normalize(to_target)

        torso_quat = quat_mul(torso_rotation, inv_start_rot)
        up_vec = quat_rotate(torso_quat, vec1).view(num_envs, 3)
        heading_vec = quat_rotate(torso_quat, vec0).view(num_envs, 3)
        up_proj = up_vec[:, up_idx].unsqueeze(-1)
        heading_proj = torch.bmm(heading_vec.view(num_envs, 1, 3), target_dirs.view(num_envs, 3, 1)).view(num_envs, 1)

        return torso_quat, up_proj, heading_proj, up_vec, heading_vec

    def compute_rot(self, torso_quat, velocity, ang_velocity, targets, torso_positions):
        """Compute rotation-related values."""
        vel_loc = quat_rotate_inverse(torso_quat, velocity)
        angvel_loc = quat_rotate_inverse(torso_quat, ang_velocity)

        roll, pitch, yaw = euler_xyz_from_quat(torso_quat)

        walk_target_angle = torch.atan2(targets[:, 2] - torso_positions[:, 2], targets[:, 0] - torso_positions[:, 0])
        angle_to_target = walk_target_angle - yaw

        # Ensure outputs have correct shape (add dimension if needed)
        if roll.dim() == 1:
            roll = roll.unsqueeze(-1)
        if pitch.dim() == 1:
            pitch = pitch.unsqueeze(-1)
        if yaw.dim() == 1:
            yaw = yaw.unsqueeze(-1)
        if angle_to_target.dim() == 1:
            angle_to_target = angle_to_target.unsqueeze(-1)

        return vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target

    def _init_mujoco_buffers(self, batch_size: int):
        """Initialize buffers for MuJoCo parallel environments."""
        # Targets - fixed location for ant to walk towards
        self._mujoco_targets = np.array([[1000.0, 0.0, 0.0]] * batch_size)

        # Potentials for progress tracking
        dt = 0.01667
        self._mujoco_potentials = np.full(batch_size, -1000.0 / dt)
        self._mujoco_prev_potentials = self._mujoco_potentials.copy()

        # Previous actions
        self._mujoco_actions = np.zeros((batch_size, self.num_actions))

        # Rotation buffers
        start_rot = np.array([1.0, 0.0, 0.0, 0.0])
        self._mujoco_inv_start_rot = np.array([self._quat_inv_numpy(start_rot) for _ in range(batch_size)])
        self._mujoco_basis_vec0 = np.array([[1.0, 0.0, 0.0]] * batch_size)
        self._mujoco_basis_vec1 = np.array([[0.0, 0.0, 1.0]] * batch_size)

    def _compute_heading_and_up_numpy(self, torso_rotations, inv_start_rot, to_target, vec0, vec1):
        """Numpy version of compute_heading_and_up for MuJoCo."""
        batch_size = torso_rotations.shape[0]

        # Normalize target directions
        target_dirs = to_target / (np.linalg.norm(to_target, axis=1, keepdims=True) + 1e-8)

        # Compute torso quaternions
        torso_quat = np.array([self._quat_mul_numpy(torso_rotations[i], inv_start_rot[i]) for i in range(batch_size)])

        # Rotate vectors
        up_vec = np.array([self._quat_rotate_numpy(torso_quat[i], vec1[i]) for i in range(batch_size)])
        heading_vec = np.array([self._quat_rotate_numpy(torso_quat[i], vec0[i]) for i in range(batch_size)])

        # Projections
        up_proj = up_vec[:, 2]  # z component is up
        heading_proj = np.sum(heading_vec * target_dirs, axis=1)

        return torso_quat, up_proj, heading_proj

    def _compute_rot_numpy(self, torso_quat, velocity, ang_velocity, targets, torso_positions):
        """Numpy version of compute_rot for MuJoCo."""
        batch_size = torso_quat.shape[0]

        # Local velocities
        vel_loc = np.array([self._quat_rotate_inverse_numpy(torso_quat[i], velocity[i]) for i in range(batch_size)])
        angvel_loc = np.array([
            self._quat_rotate_inverse_numpy(torso_quat[i], ang_velocity[i]) for i in range(batch_size)
        ])

        # Extract Euler angles
        roll, pitch, yaw = self._euler_from_quat_numpy(torso_quat)

        # Angle to target
        walk_target_angle = np.arctan2(targets[:, 2] - torso_positions[:, 2], targets[:, 0] - torso_positions[:, 0])
        angle_to_target = walk_target_angle - yaw

        return vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target

    def _quat_inv_numpy(self, q):
        """Numpy quaternion inverse."""
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def _quat_mul_numpy(self, q1, q2):
        """Numpy quaternion multiplication."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])

    def _quat_rotate_numpy(self, q, v):
        """Rotate vector by quaternion (numpy)."""
        qvec = np.array([0.0, v[0], v[1], v[2]])
        q_conj = self._quat_inv_numpy(q)
        rotated = self._quat_mul_numpy(self._quat_mul_numpy(q, qvec), q_conj)
        return rotated[1:]

    def _quat_rotate_inverse_numpy(self, q, v):
        """Inverse rotate vector by quaternion (numpy)."""
        q_inv = self._quat_inv_numpy(q)
        return self._quat_rotate_numpy(q_inv, v)

    def _euler_from_quat_numpy(self, quats):
        """Extract Euler angles from quaternions (numpy)."""
        batch_size = quats.shape[0]
        roll = np.zeros(batch_size)
        pitch = np.zeros(batch_size)
        yaw = np.zeros(batch_size)

        for i in range(batch_size):
            w, x, y, z = quats[i]

            # Roll (x-axis rotation)
            sinr_cosp = 2 * (w * x + y * z)
            cosr_cosp = 1 - 2 * (x * x + y * y)
            roll[i] = np.arctan2(sinr_cosp, cosr_cosp)

            # Pitch (y-axis rotation)
            sinp = 2 * (w * y - z * x)
            pitch[i] = np.arcsin(np.clip(sinp, -1.0, 1.0))

            # Yaw (z-axis rotation)
            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            yaw[i] = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw
