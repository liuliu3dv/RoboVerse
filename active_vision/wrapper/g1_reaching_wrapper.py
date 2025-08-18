"""A humanoid base wrapper for skillBench tasks"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Callable

import numpy as np
import torch

from active_vision.cfg.training_scenario_cfg import BaseTableHumanoidTaskCfg
from active_vision.utils.utils import *
from active_vision.utils.utils import (
    get_body_reindexed_indices_from_substring,
    get_joint_reindexed_indices_from_substring,
)
from metasim.scenario.scenario import ScenarioCfg
from metasim.types import TensorState
from metasim.utils.math import quat_rotate_inverse
from roboverse_learn.rl.rsl_rl.rsl_rl_wrapper import RslRlWrapper


class HumanoidBaseWrapper(RslRlWrapper):
    """Wraps Metasim environments to be compatible with rsl_rl OnPolicyRunner.

    Note that rsl_rl is designed for parallel training fully on GPU, with robust support for Isaac Gym and Isaac Lab.
    """

    def __init__(self, scenario: ScenarioCfg):
        super().__init__(scenario)
        self.use_vision = scenario.task.use_vision
        self.up_axis_idx = 2

        self._parse_rigid_body_indices(scenario.robots[0])
        self._parse_joint_indices(scenario.robots[0])
        self._parse_actuation_cfg(scenario)
        self._prepare_reward_function(scenario.task)
        self._init_buffers()

    def _parse_rigid_body_indices(self, robot):
        """Parse rigid body indices from robot cfg."""
        feet_names = robot.feet_links
        knee_names = robot.knee_links
        elbow_names = robot.elbow_links
        wrist_names = robot.wrist_links

        # get sorted indices for specific body links
        self.feet_indices = get_body_reindexed_indices_from_substring(
            self.env, robot.name, feet_names, device=self.device
        )
        self.knee_indices = get_body_reindexed_indices_from_substring(
            self.env, robot.name, knee_names, device=self.device
        )
        self.elbow_indices = get_body_reindexed_indices_from_substring(
            self.env, robot.name, elbow_names, device=self.device
        )
        self.wrist_indices = get_body_reindexed_indices_from_substring(
            self.env, robot.name, wrist_names, device=self.device
        )

        # attach to cfg for reward computation.
        self.cfg.feet_indices = self.feet_indices
        self.cfg.knee_indices = self.knee_indices
        self.cfg.elbow_indices = self.elbow_indices
        self.cfg.wrist_indices = self.wrist_indices
        self.cfg.torso_indices = self.torso_indices
        self.cfg.termination_contact_indices = self.termination_contact_indices
        self.cfg.penalised_contact_indices = self.penalised_contact_indices

    def _parse_joint_indices(self, robot):
        """Parse joint indices."""
        left_yaw_roll_names = robot.left_yaw_roll_joints
        right_yaw_roll_names = robot.right_yaw_roll_joints
        upper_body_names = robot.upper_body_joints
        self.cfg.left_yaw_roll_joint_indices = get_joint_reindexed_indices_from_substring(
            self.env, robot.name, left_yaw_roll_names, device=self.device
        )
        self.cfg.right_yaw_roll_joint_indices = get_joint_reindexed_indices_from_substring(
            self.env, robot.name, right_yaw_roll_names, device=self.device
        )
        self.cfg.upper_body_joint_indices = get_joint_reindexed_indices_from_substring(
            self.env, robot.name, upper_body_names, device=self.device
        )

    def _parse_cfg(self, scenario):
        super()._parse_cfg(scenario)
        self.dt = scenario.decimation * scenario.sim_params.dt
        self.command_ranges = scenario.task.command_ranges
        self.num_commands = scenario.task.command_dim

    def _parse_actuation_cfg(self, scenario):
        """Parse default joint positions and torque limits from cfg."""
        torque_limits = scenario.robots[0].torqwecue_limits
        sorted_joint_names = sorted(torque_limits.keys())
        sorted_limits = [torque_limits[name] for name in sorted_joint_names]
        self.cfg.torque_limits = (
            torch.tensor(sorted_limits, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            * scenario.control.torque_limit_scale
        )

        default_joint_pos = scenario.robots[0].default_joint_positions
        sorted_joint_pos = [default_joint_pos[name] for name in sorted_joint_names]
        self.cfg.default_joint_pd_target = (
            torch.tensor(sorted_joint_pos, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )

    def _init_buffers(self):
        """Init all buffer for reward computation."""
        self.base_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.gravity_vec = torch.tensor(
            self.get_axis_params(-1.0, self.up_axis_idx), device=self.device, dtype=torch.float32
        ).repeat((
            self.num_envs,
            1,
        ))
        self.forward_vec = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device).repeat((
            self.num_envs,
            1,
        ))

        self.common_step_counter = 0
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # TODO read obs from cfg and auto concatenate
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(
                self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float
            )
        else:
            self.privileged_obs_buf = None

        self.contact_forces = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.commands = torch.zeros(
            self.num_envs,
            self.cfg.commands.num_commands,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.extras = {}
        self.commands_scale = torch.tensor(
            [
                self.cfg.normalization.obs_scales.lin_vel,
                self.cfg.normalization.obs_scales.lin_vel,
                self.cfg.normalization.obs_scales.ang_vel,
            ],
            device=self.device,
            requires_grad=False,
        )
        self.feet_air_time = torch.zeros(
            self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_contacts = torch.zeros(
            self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.base_lin_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_ang_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # store globally for reset update and pass to obs and privileged_obs
        self.actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )

        # history buffer for reward computation
        self.last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_dof_vel = torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        self.last_root_vel = torch.zeros(self.num_envs, 6, device=self.device, requires_grad=False)

        self.last_feet_z = 0.05 * torch.ones(
            self.num_envs, len(self.feet_indices), device=self.device, requires_grad=False
        )

        self.feet_pos = torch.zeros((self.num_envs, len(self.feet_indices), 3), device=self.device, requires_grad=False)
        self.feet_height = torch.zeros((self.num_envs, len(self.feet_indices)), device=self.device, requires_grad=False)

        self.rand_push_force = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.rand_push_torque = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

        # TODO add history manager, read from config.
        self.obs_history = deque(maxlen=self.cfg.frame_stack)
        self.critic_history = deque(maxlen=self.cfg.c_frame_stack)
        for _ in range(self.cfg.frame_stack):
            self.obs_history.append(
                torch.zeros(self.num_envs, self.cfg.num_single_obs, dtype=torch.float, device=self.device)
            )
        for _ in range(self.cfg.c_frame_stack):
            self.critic_history.append(
                torch.zeros(self.num_envs, self.cfg.single_num_privileged_obs, dtype=torch.float, device=self.device)
            )

        self.env_frictions = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)  # TODO now set 0
        self.body_mass = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device, requires_grad=False)

    def _get_phase(
        self,
    ):
        cycle_time = self.cfg.reward_cfg.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase

    def _update_history(self, tensor_state: TensorState):
        """Update history buffer at the the of the frame, called after reset"""
        # we should always make a copy here
        # check whether torch.clone is necessary
        self.last_last_actions[:] = torch.clone(self.last_actions[:])
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = tensor_state.robots[self.robot.name].joint_vel_target
        self.last_root_vel[:] = tensor_state.robots[self.robot.name].root_state[:, 3:6]

    def _prepare_reward_function(self, task: BaseTableHumanoidTaskCfg):
        """Prepares a list of reward functions, which will be called to compute the total reward."""
        self.reward_scales = task.reward_weights
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = "reward_" + name
            self.reward_functions.append(self.get_reward_fn(name, task.reward_functions))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()
        }
        self.episode_metrics = {name: 0 for name in self.reward_scales.keys()}

    def compute_reward(self, envstate):
        """Compute all the reward from the states provided."""
        self.rew_buf[:] = 0

        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew_func_return = self.reward_functions[i](envstate, self.robot.name, self.cfg)
            if isinstance(rew_func_return, tuple):
                unscaled_rew, metric = rew_func_return
                self.episode_metrics[name] = metric.mean().item()
            else:
                unscaled_rew = rew_func_return
            rew = unscaled_rew * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        if self.cfg.reward_cfg.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)

    def _get_gait_phase(self):
        """Add phase into states."""
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        # Double support phase
        stance_mask[torch.abs(sin_pos) < 0.1] = 1
        return stance_mask

    def _compute_observations(self, envstate):
        """Compute observations and priviledged observation."""
        raise NotImplementedError

    def _update_refreshed_tensors(self, tensor_state: TensorState):
        """Update tensors from are refreshed tensors after physics step."""
        self.base_quat[:] = tensor_state.robots[self.robot.name].root_state[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(
            self.base_quat, tensor_state.robots[self.robot.name].root_state[:, 7:10]
        )
        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, tensor_state.robots[self.robot.name].root_state[:, 10:13]
        )
        # print(self.base_ang_vel)
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)

    def _post_physics_step(self, env_states):
        """After physics step, compute reward, get obs and privileged_obs, resample command."""
        # update episode length from env_wrapper
        self.episode_length_buf = self.env.episode_length_buf_tensor
        self.common_step_counter += 1

        self._post_physics_step_callback()
        # update refreshed tensors from simulaor
        self._update_refreshed_tensors(env_states)
        # prepare all the states for reward computation
        self._parse_state_for_reward(env_states)
        # compute the reward
        self.compute_reward(env_states)
        # reset envs
        reset_env_idx = self.reset_buf.nonzero(as_tuple=False).flatten().tolist()
        self.reset(reset_env_idx)

        # compute obs for actor,  privileged_obs for critic network
        self._compute_observations(env_states)
        self._update_history(env_states)

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf

    def update_command_curriculum(self, env_ids):
        """Implements a curriculum of increasing commands."""
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if (
            torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length
            > 0.8 * self.reward_scales["tracking_lin_vel"]
        ):
            self.command_ranges.lin_vel_x[0] = np.clip(
                self.command_ranges.lin_vel_x[0] - 0.5, -self.cfg.commands.max_curriculum, 0.0
            )
            self.command_ranges.lin_vel_x[1] = np.clip(
                self.command_ranges.lin_vel_x[1] + 0.5, 0.0, self.cfg.commands.max_curriculum
            )

    def clip_actions(self, actions):
        """Clip actions based on cfg."""
        clip_action_limit = self.cfg.normalization.clip_actions
        return torch.clip(actions, -clip_action_limit, clip_action_limit).to(self.device)

    def _pre_physics_step(self, actions):
        """Apply action smoothing and wrap actions as dict before physics step."""
        delay = torch.rand((self.num_envs, 1), device=self.device)
        actions = (1 - delay) * actions.to(self.device) + delay * self.actions
        clipped_actions = self.clip_actions(actions)
        self.actions = clipped_actions
        return self.actions

    def _physics_step(self, action_dict):
        env_states, _, terminated, time_out, _ = self.env.step(action_dict)
        self.reset_buf = terminated | time_out
        return env_states

    def step(self, actions):
        action_dict = self._pre_physics_step(actions)
        env_states = self._physics_step(action_dict)
        obs, privileged_obs, rewards = self._post_physics_step(env_states)
        return obs, privileged_obs, rewards, self.reset_buf, self.extras

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        if len(env_ids) == 0:
            return
        _, _ = self.env.set_states(self.init_states, env_ids)
        self._resample_commands(env_ids)

        # reset state buffer in the wrapper
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.last_last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.feet_air_time[env_ids] = 0.0
        self.base_quat[env_ids] = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(len(env_ids), 1)
        )
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.cfg.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0

        # log metrics
        self.extras["episode_metrics"] = deepcopy(self.episode_metrics)

        # reset env handler state buffer
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

    def _resample_commands(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges.lin_vel_x[0],
            self.command_ranges.lin_vel_x[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges.lin_vel_y[0],
            self.command_ranges.lin_vel_y[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges.heading[0],
                self.command_ranges.heading[1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges.ang_vel_yaw[0],
                self.command_ranges.ang_vel_yaw[1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _post_physics_step_callback(self):
        """Callback called before computing terminations, rewards, and observations
        Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots.
        """
        env_ids = (
            (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        if len(env_ids) > 0:
            self._resample_commands(env_ids)

    def reward_wrist_pos(env_states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg):
        wrist_pos = env_states.robots[robot_name].body_state[:, cfg.wrist_indices, :7]
        # TODO make it get from extra
        wrist_pos_diff = (
            wrist_pos[:, :, :3] - env_states.robots[robot_name].extra["ref_wrist_pos"][:, :, :3]
        )  # [num_envs, 2, 3], two hands, position only
        wrist_pos_diff = torch.flatten(wrist_pos_diff, start_dim=1)
        wrist_pos_error = torch.mean(torch.abs(wrist_pos_diff), dim=1)
        return torch.exp(-4 * wrist_pos_error), wrist_pos_error

    @staticmethod
    def get_reward_fn(target: str, reward_functions: list[Callable]) -> Callable:
        fn = next((f for f in reward_functions if f.__name__ == target), None)
        if fn is None:
            raise KeyError(f"No reward function named '{target}'")
        return fn

    @staticmethod
    def get_axis_params(value, axis_idx, x_value=0.0, dtype=np.float64, n_dims=3):
        """Construct arguments to `Vec` according to axis index."""
        zs = np.zeros((n_dims,))
        assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
        zs[axis_idx] = 1.0
        params = np.where(zs == 1.0, value, zs)
        params[0] = x_value
        return list(params.astype(dtype))

    @staticmethod
    def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
        angles %= 2 * np.pi
        angles -= 2 * np.pi * (angles > np.pi)
        return angles

    # only for reaching

    def _compute_observations(self, tensor_states: TensorState) -> None:
        """Add observation into states"""
        phase = self._get_phase()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        self.command_input = torch.cat((sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)
        self.command_input_wo_clock = self.commands[:, :3] * self.commands_scale

        q = (
            tensor_states.robots[self.robot.name].joint_pos - self.cfg.default_joint_pd_target
        ) * self.cfg.normalization.obs_scales.dof_pos
        dq = tensor_states.robots[self.robot.name].joint_pos * self.cfg.normalization.obs_scales.dof_vel

        wrist_pos = tensor_states.robots[self.robot.name].body_state[:, self.wrist_indices, :7]
        diff = wrist_pos - self.ref_wrist_pos

        ref_wrist_pos_obs = torch.flatten(self.ref_wrist_pos, start_dim=1)  # [num_envs, 14]
        wrist_pos_obs = torch.flatten(wrist_pos, start_dim=1)  # [num_envs, 14]
        diff_obs = torch.flatten(diff, start_dim=1)  # [num_envs, 14]

        self.privileged_obs_buf = torch.cat(
            (
                ref_wrist_pos_obs,  # 14
                wrist_pos_obs,  # 14
                q,  # |A|
                dq,  # |A|
                self.actions,  # |A|
                diff_obs,
                self.base_lin_vel * self.cfg.normalization.obs_scales.lin_vel,  # 3
                self.base_ang_vel * self.cfg.normalization.obs_scales.ang_vel,  # 3
                self.base_euler_xyz * self.cfg.normalization.obs_scales.quat,  # 3
                self.rand_push_force[:, :2],  # 3
                self.rand_push_torque,  # 3
                self.env_frictions,  # 1
                self.body_mass / 30.0,  # 1
            ),
            dim=-1,
        )

        obs_buf = torch.cat(
            (
                diff_obs,  # 3
                q,  # |A|
                dq,  # |A|
                self.actions,
                self.base_ang_vel * self.cfg.normalization.obs_scales.ang_vel,  # 3
                self.base_euler_xyz * self.cfg.normalization.obs_scales.quat,  # 3
            ),
            dim=-1,
        )

        obs_now = obs_buf.clone()
        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)
        obs_buf_all = torch.stack([self.obs_history[i] for i in range(self.obs_history.maxlen)], dim=1)
        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.c_frame_stack)], dim=1)
        self.privileged_obs_buf = torch.clip(
            self.privileged_obs_buf, -self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations
        )

    def update_target_wp(self, reset_env_ids):
        # self.target_wp_i specifies which seq to use for each env, and self.target_wp_j specifies the timestep in the seq
        self.ref_wrist_pos = (
            self.target_wp[self.target_wp_i, self.target_wp_j] + self.ori_wrist_pos
        )  # [num_envs, 2, 7], two hands
        self.delayed_obs_target_wp = self.target_wp[
            self.target_wp_i, torch.maximum(self.target_wp_j - self.delayed_obs_target_wp_steps_int, torch.tensor(0))
        ]
        resample_i = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.common_step_counter % self.target_wp_update_steps_int == 0:
            self.target_wp_j += 1
            wp_eps_end_bool = self.target_wp_j >= self.num_wp
            self.target_wp_j = torch.where(wp_eps_end_bool, torch.zeros_like(self.target_wp_j), self.target_wp_j)
            resample_i[wp_eps_end_bool.nonzero(as_tuple=False).flatten()] = True
            self.target_wp_update_steps_int = sample_int_from_float(self.target_wp_update_steps)
            self.delayed_obs_target_wp_steps_int = sample_int_from_float(self.delayed_obs_target_wp_steps)
        if self.cfg.human.resample_on_env_reset:
            self.target_wp_j[reset_env_ids] = 0
            resample_i[reset_env_ids] = True
        self.target_wp_i = torch.where(
            resample_i, torch.randint(0, self.num_pairs, (self.num_envs,), device=self.device), self.target_wp_i
        )
