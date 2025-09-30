"""Base class for humanoid tasks."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Literal

import torch
from loguru import logger as log
from rich.logging import RichHandler

from metasim.cfg.objects import ArticulationObjCfg, PrimitiveCubeCfg, RigidObjCfg
from metasim.cfg.robots import (
    FrankaAllegroHandLeftCfg,
    FrankaAllegroHandRightCfg,
    FrankaShadowHandLeftCfg,
    FrankaShadowHandRightCfg,
)
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.cfg.sensors.contact import ContactForceSensorCfg
from metasim.cfg.tasks.base_task_cfg import BaseRLTaskCfg, SimParamCfg
from metasim.constants import BenchmarkType, PhysicStateType, TaskType
from metasim.types import EnvState
from metasim.utils import configclass, math
from metasim.utils.bidex_util import randomize_rotation
from metasim.utils.state import ObjectState, RobotState, TensorState

logging.addLevelName(5, "TRACE")
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

########################################################
## Constants adapted from DexterousHands/bidexhands/tasks/shadow_hand_over.py
########################################################


@configclass
class PushBlockCfg(BaseRLTaskCfg):
    """class for push block tasks."""

    source_benchmark = BenchmarkType.DEXBENCH
    task_type = TaskType.TABLETOP_MANIPULATION
    is_testing = False
    episode_length = 125
    traj_filepath = "roboverse_data/trajs/bidex/ShadowHandPushBlock/v2/initial_state_v2.json"
    device = "cuda:0"
    num_envs = None
    obs_type = "state"
    use_prio = True
    current_object_type = "cube"
    current_robot_type = "shadow"
    max_agg_bodies = 56
    max_agg_shapes = 48
    objects_cfg = {
        "cube": RigidObjCfg(
            name="cube",
            scale=(1, 1, 1),
            physics=PhysicStateType.RIGIDBODY,
            urdf_path="roboverse_data/assets/bidex/objects/urdf/cube_multicolor.urdf",
            default_density=100.0,
            use_vhacd=True,
            friction=0.25,
        ),
        "table": PrimitiveCubeCfg(
            name="table",
            size=(0.6, 0.6, 0.6),
            disable_gravity=True,
            fix_base_link=True,
            flip_visual_attachments=True,
            friction=0.2,
            color=[0.8, 0.8, 0.8],
            physics=PhysicStateType.RIGIDBODY,
        ),
    }
    objects = []
    robots = []
    decimation = 1
    env_spacing = 1.5
    sim_params = SimParamCfg(
        dt=1.0 / 60.0,
        contact_offset=0.002,
        num_velocity_iterations=0,
        bounce_threshold_velocity=0.2,
        num_threads=4,
        use_gpu_pipeline=True,
        use_gpu=True,
        substeps=2,
        friction_correlation_distance=0.025,
        friction_offset_threshold=0.04,
    )
    arm_translation_scale = 0.04
    arm_orientation_scale = 0.25
    hand_translation_scale = 0.02
    hand_orientation_scale = 0.25
    right_goal_pos = None  # Placeholder for goal position, to be set later, shape (num_envs, 3)
    left_goal_pos = None  # Placeholder for goal position, to be set later, shape (num_envs, 3)
    sensors = []
    vel_obs_scale: float = 0.2  # Scale for velocity observations
    force_torque_obs_scale: float = 10.0  # Scale for force and torque observations
    sim: Literal["isaaclab", "isaacgym", "genesis", "pyrep", "pybullet", "sapien", "sapien3", "mujoco", "blender"] = (
        "isaacgym"
    )
    action_penalty_scale = 0
    reach_goal_bonus = 250.0
    reset_position_noise = 0.0
    reset_dof_pos_noise = 0.0
    leave_penalty = 5.0

    def set_objects(self) -> None:
        """Set the objects for the shadow hand push block task."""
        self.objects.append(self.objects_cfg["table"])
        self.objects.append(self.objects_cfg[self.current_object_type].replace(name=f"{self.current_object_type}_1"))
        self.objects.append(self.objects_cfg[self.current_object_type].replace(name=f"{self.current_object_type}_2"))
        if self.current_robot_type == "shadow":
            self.robots = [
                FrankaShadowHandRightCfg(
                    use_vhacd=False, robot_controller="dof_pos", isaacgym_read_mjcf=True, name="right_hand"
                ),
                FrankaShadowHandLeftCfg(
                    use_vhacd=False, robot_controller="dof_pos", isaacgym_read_mjcf=True, name="left_hand"
                ),
            ]
            self.robot_init_state = {
                "right_hand": {
                    "pos": torch.tensor([-1.0, 0.2, 0.0]),
                    "rot": torch.tensor([1, 0, 0, 0]),
                    "dof_pos": {
                        "FFJ1": 0.0,
                        "FFJ2": 0.0,
                        "FFJ3": 0.0,
                        "FFJ4": 0.0,
                        "LFJ1": 0.0,
                        "LFJ2": 0.0,
                        "LFJ3": 0.0,
                        "LFJ4": 0.0,
                        "LFJ5": 0.0,
                        "MFJ1": 0.0,
                        "MFJ2": 0.0,
                        "MFJ3": 0.0,
                        "MFJ4": 0.0,
                        "RFJ1": 0.0,
                        "RFJ2": 0.0,
                        "RFJ3": 0.0,
                        "RFJ4": 0.0,
                        "THJ1": 0.0,
                        "THJ2": 0.0,
                        "THJ3": 0.0,
                        "THJ4": 0.0,
                        "THJ5": 0.0,
                        "WRJ1": 0.0,
                        "WRJ2": 0.0,
                        "panda_joint1": 0.0,
                        "panda_joint2": -0.4116,
                        "panda_joint3": 0.0,
                        "panda_joint4": -2.0366,
                        "panda_joint5": -0.02386,
                        "panda_joint6": 3.1105,
                        "panda_joint7": 0.76586,
                    },
                },
                "left_hand": {
                    "pos": torch.tensor([-1.0, -0.2, 0.0]),
                    "rot": torch.tensor([1, 0, 0, 0]),
                    "dof_pos": {
                        "FFJ1": 0.0,
                        "FFJ2": 0.0,
                        "FFJ3": 0.0,
                        "FFJ4": 0.0,
                        "LFJ1": 0.0,
                        "LFJ2": 0.0,
                        "LFJ3": 0.0,
                        "LFJ4": 0.0,
                        "LFJ5": 0.0,
                        "MFJ1": 0.0,
                        "MFJ2": 0.0,
                        "MFJ3": 0.0,
                        "MFJ4": 0.0,
                        "RFJ1": 0.0,
                        "RFJ2": 0.0,
                        "RFJ3": 0.0,
                        "RFJ4": 0.0,
                        "THJ1": 0.0,
                        "THJ2": 0.0,
                        "THJ3": 0.0,
                        "THJ4": 0.0,
                        "THJ5": 0.0,
                        "WRJ1": 0.0,
                        "WRJ2": 0.0,
                        "panda_joint1": 0.0,
                        "panda_joint2": -0.4116,
                        "panda_joint3": 0.0,
                        "panda_joint4": -2.0366,
                        "panda_joint5": -0.02386,
                        "panda_joint6": 3.1105,
                        "panda_joint7": 0.76586,
                    },
                },
            }
        elif self.current_robot_type == "allegro":
            self.robots = [
                FrankaAllegroHandRightCfg(use_vhacd=False, robot_controller="dof_pos", name="right_hand"),
                FrankaAllegroHandLeftCfg(use_vhacd=False, robot_controller="dof_pos", name="left_hand"),
            ]
            self.robot_init_state = {
                "right_hand": {
                    "pos": torch.tensor([-1.0, 0.2, 0.0]),
                    "rot": torch.tensor([1, 0, 0, 0]),
                    "dof_pos": {
                        "joint_0": 0.0,
                        "joint_1": 0.0,
                        "joint_2": 0.0,
                        "joint_3": 0.0,
                        "joint_4": 0.0,
                        "joint_5": 0.0,
                        "joint_6": 0.0,
                        "joint_7": 0.0,
                        "joint_8": 0.0,
                        "joint_9": 0.0,
                        "joint_10": 0.0,
                        "joint_11": 0.0,
                        "joint_12": 0.0,
                        "joint_13": 0.0,
                        "joint_14": 0.0,
                        "joint_15": 0.0,
                        "panda_joint1": 0.0,
                        "panda_joint2": -0.4116,
                        "panda_joint3": 0.0,
                        "panda_joint4": -2.0366,
                        "panda_joint5": -0.02386,
                        "panda_joint6": 3.1105,
                        "panda_joint7": 0.76586,
                    },
                },
                "left_hand": {
                    "pos": torch.tensor([0.0, -1.13, 0.0]),
                    "rot": torch.tensor([0.7071, 0, 0, 0.7071]),
                    "dof_pos": {
                        "joint_0": 0.0,
                        "joint_1": 0.0,
                        "joint_2": 0.0,
                        "joint_3": 0.0,
                        "joint_4": 0.0,
                        "joint_5": 0.0,
                        "joint_6": 0.0,
                        "joint_7": 0.0,
                        "joint_8": 0.0,
                        "joint_9": 0.0,
                        "joint_10": 0.0,
                        "joint_11": 0.0,
                        "joint_12": 0.0,
                        "joint_13": 0.0,
                        "joint_14": 0.0,
                        "joint_15": 0.0,
                        "panda_joint1": 0.0,
                        "panda_joint2": -0.4116,
                        "panda_joint3": 0.0,
                        "panda_joint4": -2.0366,
                        "panda_joint5": -0.02386,
                        "panda_joint6": 3.1105,
                        "panda_joint7": 0.76586,
                    },
                },
            }
        self.step_actions_shape = 0
        for robot in self.robots:
            self.step_actions_shape += robot.num_joints
        self.action_shape = 0
        for robot in self.robots:
            if robot.robot_controller == "ik":
                self.action_shape += 6 + 6 * robot.num_fingertips
            elif robot.robot_controller == "dof_pos":
                self.action_shape += 6 + robot.num_actuated_joints - robot.num_arm_joints
        for name in self.robots[0].fingertips:
            r_name = "right" + name
            self.sensors.append(
                ContactForceSensorCfg(base_link=(self.robots[0].name, name), source_link=None, name=r_name)
            )
        for name in self.robots[1].fingertips:
            l_name = "left" + name
            self.sensors.append(
                ContactForceSensorCfg(base_link=(self.robots[0].name, name), source_link=None, name=l_name)
            )

    def set_init_states(self) -> None:
        """Set the initial states for the shadow hand push block task."""
        self.state_shape = 0
        for robot in self.robots:
            self.state_shape += robot.observation_shape
            self.state_shape += robot.num_fingertips * 6  # fingertip forces
        self.state_shape += self.action_shape
        if self.use_prio:
            self.state_shape += 32
        self.obs_shape = {
            "state": (self.state_shape,),
        }
        if self.obs_type == "state":
            self.cameras = []
            if not self.use_prio:
                raise ValueError("State observation type requires proprioception to be enabled.")
        elif self.obs_type == "rgb":
            assert hasattr(self, "img_h") and hasattr(self, "img_w"), "Image height and width must be set."
            self.cameras = [
                PinholeCameraCfg(
                    name="camera_0",
                    width=self.img_w,
                    height=self.img_h,
                    pos=(-1.35, -1.0, 1.05),
                    look_at=(0.0, -0.75, 0.5),
                )
            ]  # TODO
            self.obs_shape["rgb"] = (3 * self.img_h * self.img_w,)
        self.init_right_goal_pos = torch.tensor(
            [0.2, 0.2, 0.625], dtype=torch.float, device=self.device
        )  # Initial right goal position, shape (3,)
        self.init_left_goal_pos = torch.tensor(
            [0.2, -0.2, 0.625], dtype=torch.float, device=self.device
        )  # Initial right goal position, shape (3,)
        self.init_states = {
            "objects": {
                "table": {
                    "pos": torch.tensor([0.2, 0.0, 0.3]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                },
                f"{self.current_object_type}_1": {
                    "pos": torch.tensor([0.0, 0.2, 0.625]),
                    "rot": torch.tensor([0.5, 0.5, 0.5, -0.5]),
                },
                f"{self.current_object_type}_2": {
                    "pos": torch.tensor([0.0, -0.2, 0.625]),
                    "rot": torch.tensor([0.5, 0.5, 0.5, -0.5]),
                },
            },
            "robots": self.robot_init_state,
        }
        self.robot_dof_default_pos = {}
        self.robot_dof_default_pos_cpu = {}
        for robot in self.robots:
            self.robot_dof_default_pos[robot.name] = torch.tensor(
                list(self.init_states["robots"][robot.name]["dof_pos"].values()),
                dtype=torch.float,
                device=self.device,
            )
            self.robot_dof_default_pos_cpu[robot.name] = self.robot_dof_default_pos[robot.name].cpu()

    def scale_action_fn(self, actions: torch.Tensor) -> torch.Tensor:
        """Scale actions to the range of the action space.

        Args:
            actions (torch.Tensor): Actions in the range of [-1, 1], shape (num_envs, num_actions).
        """
        step_actions = torch.zeros((self.num_envs, self.step_actions_shape), device=self.device)
        actions_start = 0
        step_actions_start = 0
        for robot in self.robots:
            dpose = actions[:, actions_start : actions_start + 6]
            dpose[:, :3] = dpose[:, :3] * self.arm_translation_scale
            dpose[:, 3:] = dpose[:, 3:] * self.arm_orientation_scale
            arm_dof_targets = robot.control_arm_ik(dpose, dpose.shape[0], dpose.device)
            step_actions[:, step_actions_start + robot.arm_dof_idx] = arm_dof_targets
            actions_start += 6
            if robot.robot_controller == "ik":
                ft_action = actions[:, actions_start : actions_start + 6 * robot.num_fingertips].view(
                    self.num_envs, robot.num_fingertips, 6
                )
                ft_pos = ft_action[:, :, :3] * self.hand_translation_scale
                ft_rot = ft_action[:, :, 3:] * self.hand_orientation_scale
                hand_dof_pos = robot.control_hand_ik(ft_pos, ft_rot)
                step_actions[:, step_actions_start + robot.hand_dof_idx] = hand_dof_pos
                actions_start += 6 * robot.num_fingertips
                step_actions_start += robot.num_joints
            elif robot.robot_controller == "dof_pos":
                hand_dof_pos = robot.scale_hand_action(
                    actions[:, actions_start : actions_start + robot.num_actuated_joints - robot.num_arm_joints]
                )
                step_actions[:, step_actions_start + robot.hand_dof_idx] = hand_dof_pos
                actions_start += robot.num_actuated_joints - robot.num_arm_joints
                step_actions_start += robot.num_joints

        return step_actions

    def observation_fn(self, envstates: list[EnvState], actions: torch.Tensor, device=None) -> torch.Tensor:
        """Observation function for shadow hand over tasks.

        Args:
            envstates (list[EnvState]): List of environment states to process.
            actions (torch.Tensor): Actions taken by the agents in the environment, shape (num_envs, num_actions).
            device (torch.device | None): The device to use for the observations. If None, use the task's device.

        Returns:
            Compute the observations of all environment. The observation is composed of three parts:
            the state values of the left and right hands, and the information of objects and target.
            The state values of the left and right hands were the same for each task, including hand
            joint and finger positions, velocity, and force information. The detail 430-dimensional
            observational space as shown in below:

            Index       Description
            right robot proceptual observation
            right robot fingertip forces
            right robot actions
            left  robot proceptual observation
            left  robot fingertip forces
            left  robot actions
            block1 pose
            block1 linear velocity
            block1 angle velocity
            block2 pose
            block2 linear velocity
            block2 angle velocity
            left goal position
            right goal position
            visual observation, currently RGB image (3 x 256 x 256)
        """
        if device is None:
            device = self.device
        num_envs = envstates.robots[self.robots[0].name].root_state.shape[0]
        if self.num_envs is None:
            self.num_envs = num_envs
        if self.left_goal_pos is None:
            self.left_goal_pos = (
                torch.tensor(self.init_left_goal_pos, dtype=torch.float, device=self.device)
                .view(1, -1)
                .repeat(num_envs, 1)
            )
        if self.right_goal_pos is None:
            self.right_goal_pos = (
                torch.tensor(self.init_right_goal_pos, dtype=torch.float, device=self.device)
                .view(1, -1)
                .repeat(num_envs, 1)
            )
        obs = {
            "state": torch.zeros((num_envs, self.state_shape), dtype=torch.float32, device=device),
        }
        state_obs = obs["state"]
        t = 0
        state_obs[:, : self.robots[0].observation_shape] = self.robots[0].observation()
        t += self.robots[0].observation_shape
        for name in self.robots[0].fingertips:
            # shape: (num_envs, 3) + (num_envs, 3) => (num_envs, 6)
            r_name = "right" + name
            force = envstates.sensors[r_name].force  # (num_envs, 3)
            torque = envstates.sensors[r_name].torque  # (num_envs, 3)
            state_obs[:, t : t + 6] = torch.cat([force, torque], dim=1) * self.force_torque_obs_scale  # (num_envs, 6)
            t += 6
        state_obs[:, t : t + self.action_shape // 2] = actions[:, : self.action_shape // 2]  # actions for right hand
        t += self.action_shape // 2
        state_obs[:, t : t + self.robots[1].observation_shape] = self.robots[1].observation()
        t += self.robots[1].observation_shape
        for name in self.robots[1].fingertips:
            # shape: (num_envs, 3) + (num_envs, 3) => (num_envs, 6)
            l_name = "left" + name
            force = envstates.sensors[l_name].force
            torque = envstates.sensors[l_name].torque
            state_obs[:, t : t + 6] = torch.cat([force, torque], dim=1) * self.force_torque_obs_scale  # (num_envs, 6)
            t += 6
        state_obs[:, t : t + self.action_shape // 2] = actions[:, self.action_shape // 2 :]  # actions for left hand
        t += self.action_shape // 2
        if self.use_prio:
            state_obs[:, t : t + 13] = envstates.objects[f"{self.current_object_type}_1"].root_state
            state_obs[:, t + 10 : t + 13] *= self.vel_obs_scale  # object angvel
            t += 13
            state_obs[:, t : t + 13] = envstates.objects[f"{self.current_object_type}_2"].root_state
            state_obs[:, t + 10 : t + 13] *= self.vel_obs_scale  # object angvel
            t += 13
            state_obs[:, t : t + 3] = self.right_goal_pos  # object 1 goal position
            t += 3
            state_obs[:, t : t + 3] = self.left_goal_pos  # object 2 goal position
            t += 3
        obs["state"] = state_obs
        if self.obs_type == "rgb":
            obs["rgb"] = (
                envstates.cameras["camera_0"].rgb.permute(0, 3, 1, 2) / 255.0
            )  # (num_envs, H, W, 3) -> (num_envs, 3, H, W)
        return obs

    def reward_fn(
        self,
        envstates: list[EnvState],
        actions,
        reset_buf,
        reset_goal_buf,
        episode_length_buf,
        success_buf,
    ) -> torch.Tensor:
        """Compute the reward of all environment. The core function is compute_hand_reward.

        Args:
            envstates (list[EnvState]): States of the environment
            actions (tensor): Actions of agents in the all environment, shape (num_envs, num_actions)
            episode_length_buf (torch.Tensor): The episode length buffer, shape (num_envs,)
            reset_buf (torch.Tensor): The reset buffer of all environments at this time, shape (num_envs,)
            reset_goal_buf (torch.Tensor): The reset goal buffer of all environments at this time, shape (num_envs,)
            success_buf (torch.Tensor): The success buffer of all environments at this time, shape (num_envs,)
            success_rate (float): The success rate of the environment, used to determine the success condition

        Returns:
            reward (torch.Tensor): The reward of all environments at this time, shape (num_envs,)
            reset_buf (torch.Tensor): The reset buffer of all environments at this time, shape (num_envs,)
            reset_goal_buf (torch.Tensor): The reset goal buffer of all environments at this time, shape (num_envs,)
            success_buf (torch.Tensor): The success buffer of all environments at this time, shape (num_envs,)
        """
        right_object_pos = envstates.objects[f"{self.current_object_type}_1"].root_state[:, :3]
        left_object_pos = envstates.objects[f"{self.current_object_type}_2"].root_state[:, :3]
        right_hand_reward, right_hand_dist = self.robots[0].reward(right_object_pos)
        left_hand_reward, left_hand_dist = self.robots[1].reward(left_object_pos)

        # right hand fingertip positions and rotations
        (reward, reset_buf, reset_goal_buf, success_buf) = compute_task_reward(
            reset_buf=reset_buf,
            reset_goal_buf=reset_goal_buf,
            episode_length_buf=episode_length_buf,
            success_buf=success_buf,
            max_episode_length=self.episode_length,
            right_object_pos=right_object_pos,
            left_object_pos=left_object_pos,
            right_target_pos=self.right_goal_pos,
            left_target_pos=self.left_goal_pos,
            right_hand_reward=right_hand_reward,
            left_hand_reward=left_hand_reward,
            right_hand_dist=right_hand_dist,
            left_hand_dist=left_hand_dist,
            action_penalty_scale=self.action_penalty_scale,
            actions=actions,
            reach_goal_bonus=self.reach_goal_bonus,
            leave_penalty=self.leave_penalty,
        )
        return reward, reset_buf, reset_goal_buf, success_buf

    def goal_reset_fn(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reset the goal position and rotation for the environment.

        Args:
            env_ids (torch.Tensor): The reset goal buffer of all environments at this time, shape (num_envs_to_reset,).
        """
        pass

    def reset_init_pose_fn(self, init_states: list[EnvState], env_ids: torch.Tensor) -> list[EnvState]:
        """Reset the initial pose of the environment.

        Args:
            init_states (list[EnvState]): States of the environment
            env_ids (torch.Tensor): The indices of the environments to reset.

        Returns:
            reset_state: The updated states of the environment after resetting.
        """
        if self.reset_dof_pos_noise == 0.0 and self.reset_position_noise == 0.0:
            # If no noise is applied, return the initial states directly
            return deepcopy(init_states)
        if isinstance(init_states, list):
            reset_state = deepcopy(init_states)
            num_shadow_hand_dofs = self.shadow_hand_dof_lower_limits.shape[0]
            x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device="cpu").repeat((len(env_ids), 1))
            y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device="cpu").repeat((len(env_ids), 1))

            # generate random values
            rand_floats = math.torch_rand_float(-1.0, 1.0, (len(env_ids), num_shadow_hand_dofs + 5), device="cpu")

            new_object_rot = randomize_rotation(rand_floats[:, 3], rand_floats[:, 4], x_unit_tensor, y_unit_tensor)

            robot_dof_default_pos = self.robot_dof_default_pos_cpu[self.robots[0].name]
            delta_max = self.shadow_hand_dof_upper_limits_cpu - robot_dof_default_pos
            delta_min = self.shadow_hand_dof_lower_limits_cpu - robot_dof_default_pos

            for i, env_id in enumerate(env_ids):
                # reset object
                for obj_name in reset_state[env_id]["objects"].keys():
                    reset_state[env_id]["objects"][obj_name]["pos"][:3] += (
                        self.reset_position_noise * rand_floats[i, :3]
                    )

                # reset shadow hand
                for robot_name in reset_state[env_id]["robots"].keys():
                    rand_delta = delta_min + (delta_max - delta_min) * rand_floats[i, 5 : 5 + num_shadow_hand_dofs]
                    dof_pos = robot_dof_default_pos + self.reset_dof_pos_noise * rand_delta
                    reset_state[env_id]["robots"][robot_name]["dof_pos"] = {
                        name: dof_pos[j].item()
                        for j, name in enumerate(reset_state[env_id]["robots"][robot_name]["dof_pos"].keys())
                    }

            return reset_state
        elif isinstance(init_states, TensorState):
            reset_state = deepcopy(init_states)  # in sorted order
            num_shadow_hand_dofs = self.shadow_hand_dof_lower_limits.shape[0]
            x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((len(env_ids), 1))
            y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((len(env_ids), 1))

            # generate random values
            rand_floats = math.torch_rand_float(-1.0, 1.0, (len(env_ids), num_shadow_hand_dofs + 5), device=self.device)

            new_object_rot = randomize_rotation(rand_floats[:, 3], rand_floats[:, 4], x_unit_tensor, y_unit_tensor)
            for obj_id, obj in enumerate(self.objects):
                root_state = reset_state.objects[obj.name].root_state
                root_state[env_ids, :3] += self.reset_position_noise * rand_floats[:, :3]
                obj_state = ObjectState(
                    root_state=root_state,
                )
                if isinstance(obj, ArticulationObjCfg):
                    joint_pos = reset_state.objects[obj.name].joint_pos
                    obj_state.joint_pos = joint_pos
                reset_state.objects[obj.name] = obj_state

            for robot_id, robot in enumerate(self.robots):
                robot_dof_default_pos = self.robot_dof_default_pos[robot.name][self.joint_reindex]
                delta_max = self.shadow_hand_dof_upper_limits[self.joint_reindex] - robot_dof_default_pos
                delta_min = self.shadow_hand_dof_lower_limits[self.joint_reindex] - robot_dof_default_pos
                rand_delta = delta_min + (delta_max - delta_min) * rand_floats[:, 5 : 5 + num_shadow_hand_dofs]
                dof_pos = robot_dof_default_pos + self.reset_dof_pos_noise * rand_delta
                joint_pos = reset_state.robots[robot.name].joint_pos
                joint_pos[env_ids, :] = dof_pos
                robot_state = RobotState(
                    root_state=reset_state.robots[robot.name].root_state,
                    joint_pos=joint_pos,
                )
                reset_state.robots[robot.name] = robot_state

            return reset_state
        else:
            raise Exception("Unsupported state type, must be EnvState or TensorState")

    def update_state(self, envstates: TensorState):
        """Update the observation of the environment."""
        for robot in self.robots:
            robot.update_state(envstates)


@torch.jit.script
def compute_task_reward(
    reset_buf,
    reset_goal_buf,
    episode_length_buf,
    success_buf,
    max_episode_length: float,
    right_object_pos,
    left_object_pos,
    right_target_pos,
    left_target_pos,
    right_hand_reward,
    left_hand_reward,
    right_hand_dist,
    left_hand_dist,
    action_penalty_scale: float,
    actions,
    reach_goal_bonus: float,
    leave_penalty: float,
):
    """Compute the reward of all environment.

    Args:
        reset_buf (tensor): The reset buffer of all environments at this time, shape (num_envs,)

        reset_goal_buf (tensor): The reset goal buffer of all environments at this time, shape (num_envs,)

        episode_length_buf (tensor): The porgress buffer of all environments at this time, shape (num_envs,)

        success_buf (tensor): The success buffer of all environments at this time, shape (num_envs,)

        max_episode_length (float): The max episode length in this environment

        right_object_pos (tensor): The position of the right object, shape (num_envs, 3)

        left_object_pos (tensor): The position of the left object, shape (num_envs, 3)

        right_target_pos (tensor): The target position of the right object, shape (num_envs, 3)

        left_target_pos (tensor): The target position of the left object, shape (num_envs, 3)

        right_hand_reward (tensor): The reward from the right hand, shape (num_envs,)

        left_hand_reward (tensor): The reward from the left hand, shape (num_envs,)

        right_hand_dist (tensor): The distance from the right hand to the right object, shape (num_envs,)

        left_hand_dist (tensor): The distance from the left hand to the left object, shape (num_envs,)

        action_penalty_scale (float): The scale of the action penalty

        actions (tensor): The action buffer of all environments at this time

        reach_goal_bonus (float): The reward given when the object reaches the goal

        leave_penalty (float): The penalty for leaving the goal area

    """
    # Distance from the hand to the object
    left_goal_dist = torch.norm(left_target_pos - left_object_pos, p=2, dim=-1)
    right_goal_dist = torch.norm(right_target_pos - right_object_pos, p=2, dim=-1)

    action_penalty = torch.sum(actions**2, dim=-1)

    # No goal reset
    goal_resets = torch.zeros_like(reset_buf, dtype=torch.float32)

    # # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    up_rew = torch.zeros_like(right_hand_reward)
    up_rew = 5 - 5 * left_goal_dist - 5 * right_goal_dist
    reward = right_hand_reward + left_hand_reward + up_rew

    left_success = torch.abs(left_goal_dist) <= 0.1
    right_success = torch.abs(right_goal_dist) <= 0.1

    success = left_success & right_success
    success_buf = torch.where(
        success_buf == 0,
        torch.where(
            success,
            torch.ones_like(success_buf),
            torch.zeros_like(success_buf),
        ),
        success_buf,
    )

    # reward = torch.where(left_success == 1, reward + reach_goal_bonus // 2, reward)
    # reward = torch.where(right_success == 1, reward + reach_goal_bonus // 2, reward)
    reward = torch.where(success, reward + reach_goal_bonus, reward)
    # Check env termination conditions, including maximum success number
    resets = torch.where(right_hand_dist >= 0.24, torch.ones_like(reset_buf), reset_buf)
    resets = torch.where(left_hand_dist >= 0.24, torch.ones_like(resets), resets)

    # penalty = (left_hand_finger_dist >= 1.2) | (right_hand_finger_dist >= 1.2)
    # reward = torch.where(penalty, reward - leave_penalty, reward)

    # Reset because of terminate or fall or success
    resets = torch.where(episode_length_buf >= max_episode_length, torch.ones_like(resets), resets)
    resets = torch.where(success_buf >= 1, torch.ones_like(resets), resets)
    return reward, resets, goal_resets, success_buf
