"""Base class for humanoid tasks."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Literal

import torch
from loguru import logger as log
from rich.logging import RichHandler

from metasim.cfg.objects import ArticulationObjCfg, RigidObjCfg
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
class CatchAbreastCfg(BaseRLTaskCfg):
    """class for bidex catch abreast tasks."""

    source_benchmark = BenchmarkType.DEXBENCH
    task_type = TaskType.TABLETOP_MANIPULATION
    is_testing = False
    episode_length = 150
    traj_filepath = "roboverse_data/trajs/bidex/ShadowHandCatchAbreast/v2/initial_state_v2.json"
    device = "cuda:0"
    num_envs = None
    obs_type = "state"
    use_prio = True
    current_object_type = "egg"
    current_robot_type = "shadow"
    objects_cfg = {
        "cube": RigidObjCfg(
            name="cube",
            scale=(1, 1, 1),
            physics=PhysicStateType.RIGIDBODY,
            urdf_path="roboverse_data/assets/bidex/objects/urdf/cube_multicolor.urdf",
            usd_path="roboverse_data/assets/bidex/objects/usd/cube_multicolor.usd",
            default_density=500.0,
            use_vhacd=True,
        ),
        "egg": RigidObjCfg(
            name="egg",
            scale=(1, 1, 1),
            physics=PhysicStateType.RIGIDBODY,
            mjcf_path="roboverse_data/assets/bidex/open_ai_assets/mjcf/hand/egg.xml",
            usd_path="roboverse_data/assets/bidex/open_ai_assets/usd/hand/egg.usd",
            isaacgym_read_mjcf=True,  # Use MJCF for IsaacGym
            use_vhacd=True,
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
    dt = sim_params.dt  # Simulation time step
    transition_scale = 0.5
    orientation_scale = 1.0
    goal_pos = None  # Placeholder for goal position, to be set later, shape (num_envs, 3)
    goal_rot = None  # Placeholder for goal rotation, to be set later, shape (num_envs, 4)
    fingertips = ["robot0_ffdistal", "robot0_mfdistal", "robot0_rfdistal", "robot0_lfdistal", "robot0_thdistal"]
    sensors = []
    vel_obs_scale: float = 0.2  # Scale for velocity observations
    force_torque_obs_scale: float = 10.0  # Scale for force and torque observations
    sim: Literal["isaaclab", "isaacgym", "genesis", "pyrep", "pybullet", "sapien", "sapien3", "mujoco", "blender"] = (
        "isaacgym"
    )
    arm_translation_scale: float = 0.005
    arm_orientation_scale: float = 0.05
    hand_translation_scale: float = 0.02
    hand_orientation_scale: float = 0.25 * torch.pi
    dist_reward_scale = 50.0
    action_penalty_scale = 0
    success_tolerance = 0.1
    reach_goal_bonus = 500.0
    throw_bonus = 10.0
    fall_penalty = 0.0
    leave_penalty = 0.0  # Penalty for leaving the base
    reset_position_noise = 0.0
    reset_dof_pos_noise = 0.0

    def set_objects(self) -> None:
        """Set the objects for the catch abreast task."""
        self.objects.append(self.objects_cfg[self.current_object_type])
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
                    "pos": torch.tensor([0.45, -0.55, 0.0]),
                    "rot": torch.tensor([0, 0, 0, 1]),
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
                        "panda_joint7": -2.35619,
                    },
                },
                "left_hand": {
                    "pos": torch.tensor([0.45, -1.15, 0.0]),
                    "rot": torch.tensor([0, 0, 0, 1]),
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
                        "panda_joint7": -2.35619,
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
                    "pos": torch.tensor([0.23, -0.55, 0.0]),
                    "rot": torch.tensor([0, 0, 0, 1]),
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
                        "joint_14": 1.64,
                        "joint_15": 0.0,
                        "panda_joint1": 0.0,
                        "panda_joint2": -0.4116,
                        "panda_joint3": 0.0,
                        "panda_joint4": -2.0366,
                        "panda_joint5": -0.02386,
                        "panda_joint6": 3.1105,
                        "panda_joint7": -2.35619,
                    },
                },
                "left_hand": {
                    "pos": torch.tensor([0.23, -1.15, 0.0]),
                    "rot": torch.tensor([0, 0, 0, 1]),
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
                        "joint_14": 1.64,
                        "joint_15": 0.0,
                        "panda_joint1": 0.0,
                        "panda_joint2": -0.4116,
                        "panda_joint3": 0.0,
                        "panda_joint4": -2.0366,
                        "panda_joint5": -0.02386,
                        "panda_joint6": 3.1105,
                        "panda_joint7": -2.35619,
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
        """Set the initial states for the shadow hand over task."""
        self.state_shape = 0
        for robot in self.robots:
            self.state_shape += robot.observation_shape
            self.state_shape += robot.num_fingertips * 6  # fingertip forces
        self.state_shape += self.action_shape
        if self.use_prio:
            self.state_shape += 24
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
                    name="camera_0", width=self.img_w, height=self.img_h, pos=(0.9, -1.0, 1.3), look_at=(0.0, -0.5, 0.6)
                )
            ]
            self.obs_shape["rgb"] = (
                3,
                self.img_h,
                self.img_w,
            )
        self.init_goal_pos = torch.tensor(
            [-0.37, -1.15, 0.85], dtype=torch.float32, device=self.device
        )  # Initial goal position, shape (3,)
        self.init_goal_rot = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device
        )  # Initial goal rotation, shape (4,)
        """Set the initial states for the shadow hand over task."""
        self.init_states = {
            "objects": {
                self.current_object_type: {
                    "pos": torch.tensor([-0.37, -0.55, 0.87]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                },
            },
            "robots": self.robot_init_state,
        }
        self.robot_dof_default_pos = {}
        self.robot_dof_default_pos_cpu = {}
        for robot in self.robots:
            self.robot_dof_default_pos[robot.name] = torch.tensor(
                list(self.init_states["robots"][robot.name]["dof_pos"].values()),
                dtype=torch.float32,
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
            joint and finger positions, velocity, and force information. The detail 422-dimensional
            observational space as shown in below:

            Description
            right robot proceptual observation
            right robot fingertip forces
            right robot actions
            left  robot proceptual observation
            left  robot fingertip forces
            left  robot actions
            object pose
            object linear velocity
            object angle velocity
            goal pose
            goal rot - object rot
            visual observation, currently RGB image (3 x 256 x 256)
        """
        if device is None:
            device = self.device
        num_envs = envstates.robots[self.robots[0].name].root_state.shape[0]
        if self.num_envs is None:
            self.num_envs = num_envs
        if self.goal_pos is None:
            self.goal_pos = (
                torch.tensor(self.init_goal_pos, dtype=torch.float32, device=self.device)
                .view(1, -1)
                .repeat(num_envs, 1)
            )
        if self.goal_rot is None:
            self.goal_rot = (
                torch.tensor(self.init_goal_rot, dtype=torch.float32, device=self.device)
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
            state_obs[:, t : t + 13] = envstates.objects[self.current_object_type].root_state
            state_obs[:, t + 10 : t + 13] *= self.vel_obs_scale  # object angvel
            t += 13
            state_obs[:, t : t + 7] = torch.cat(
                [self.goal_pos, self.goal_rot], dim=1
            )  # goal position and rotation (num_envs, 7)
            t += 7
            state_obs[:, t : t + 4] = math.quat_mul(
                envstates.objects[self.current_object_type].root_state[:, 3:7], math.quat_inv(self.goal_rot)
            )  # goal rotation - object rotation
            t += 4
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
        (reward, reset_buf, reset_goal_buf, success_buf) = compute_task_reward(
            reset_buf=reset_buf,
            reset_goal_buf=reset_goal_buf,
            episode_length_buf=episode_length_buf,
            success_buf=success_buf,
            max_episode_length=self.episode_length,
            object_pos=envstates.objects[self.current_object_type].root_state[:, :3],
            object_rot=envstates.objects[self.current_object_type].root_state[:, 3:7],
            target_pos=self.goal_pos,
            target_rot=self.goal_rot,
            right_hand_pos=self.robots[0].palm_state[:, :3],
            left_hand_pos=self.robots[1].palm_state[:, :3],
            dist_reward_scale=self.dist_reward_scale,
            action_penalty_scale=self.action_penalty_scale,
            actions=actions,
            success_tolerance=self.success_tolerance,
            reach_goal_bonus=self.reach_goal_bonus,
            throw_bonus=self.throw_bonus,
            fall_penalty=self.fall_penalty,
            leave_penalty=self.leave_penalty,
            is_testing=self.is_testing,
        )
        return reward, reset_buf, reset_goal_buf, success_buf

    def goal_reset_fn(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reset the goal position and rotation for the environment.

        Args:
            env_ids (torch.Tensor): The reset goal buffer of all environments at this time, shape (num_envs_to_reset,).
        """
        rand_floats = math.torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)
        x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((len(env_ids), 1))
        y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((len(env_ids), 1))

        new_rot = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1], x_unit_tensor, y_unit_tensor)

        self.goal_pos[env_ids] = self.init_goal_pos.clone()
        self.goal_rot[env_ids] = new_rot

        return

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
            if getattr(self, "robot_dof_default_pos_cpu", None) is None:
                self.robot_dof_default_pos_cpu = {}
                for robot in self.robots:
                    self.robot_dof_default_pos_cpu[robot.name] = torch.zeros(
                        robot.num_joints, dtype=torch.float32, device="cpu"
                    )
                    for idx, name in enumerate(robot.dof_names):
                        self.robot_dof_default_pos_cpu[robot.name][idx] = self.init_states["robots"][robot.name][
                            "dof_pos"
                        ][name]
            reset_state = deepcopy(init_states)
            num_dofs = self.robots[0].num_joints + self.robots[1].num_joints
            x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device="cpu").repeat((len(env_ids), 1))
            y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device="cpu").repeat((len(env_ids), 1))

            # generate random values
            rand_floats = math.torch_rand_float(-1.0, 1.0, (len(env_ids), num_dofs + 5), device="cpu")

            new_object_rot = randomize_rotation(rand_floats[:, 3], rand_floats[:, 4], x_unit_tensor, y_unit_tensor)
            start_idx = 5
            for i, env_id in enumerate(env_ids):
                for obj_name in reset_state[env_id]["objects"].keys():
                    reset_state[env_id]["objects"][obj_name]["pos"][:3] += (
                        self.reset_position_noise * rand_floats[i, :3]
                    )
                    reset_state[env_id]["objects"][obj_name]["rot"] = new_object_rot[i]

                # reset hand
                for robot in self.robots:
                    robot_dof_default_pos = self.robot_dof_default_pos_cpu[robot.name]
                    delta_max = robot.joint_limits_upper.cpu() - robot_dof_default_pos
                    delta_min = robot.joint_limits_lower.cpu() - robot_dof_default_pos
                    rand_delta = (
                        delta_min + (delta_max - delta_min) * rand_floats[i, start_idx : start_idx + robot.num_joints]
                    )
                    dof_pos = robot_dof_default_pos + self.reset_dof_pos_noise * rand_delta
                    reset_state[env_id]["robots"][robot.name]["dof_pos"] = {
                        name: dof_pos[idx].item() for idx, name in enumerate(robot.dof_names)
                    }
                    start_idx += robot.num_joints

            return reset_state
        elif isinstance(init_states, TensorState):
            reset_state = deepcopy(init_states)  # in sorted order
            num_dofs = self.robots[0].num_joints + self.robots[1].num_joints
            x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((len(env_ids), 1))
            y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((len(env_ids), 1))

            # generate random values
            rand_floats = math.torch_rand_float(-1.0, 1.0, (len(env_ids), num_dofs + 5), device=self.device)

            new_object_rot = randomize_rotation(rand_floats[:, 3], rand_floats[:, 4], x_unit_tensor, y_unit_tensor)
            for obj_id, obj in enumerate(self.objects):
                root_state = reset_state.objects[obj.name].root_state
                root_state[env_ids, :3] += self.reset_position_noise * rand_floats[:, :3]
                root_state[env_ids, 3:7] = new_object_rot
                obj_state = ObjectState(
                    root_state=root_state,
                )
                if isinstance(obj, ArticulationObjCfg):
                    joint_pos = reset_state.objects[obj.name].joint_pos
                    obj_state.joint_pos = joint_pos
                reset_state.objects[obj.name] = obj_state

            start_idx = 5
            for robot_id, robot in enumerate(self.robots):
                robot_dof_default_pos = reset_state.robots[robot.name].joint_pos[env_ids]
                delta_max = robot.joint_limits_upper - robot_dof_default_pos
                delta_min = robot.joint_limits_lower - robot_dof_default_pos
                rand_delta = (
                    delta_min + (delta_max - delta_min) * rand_floats[:, start_idx : start_idx + robot.num_joints]
                )
                dof_pos = robot_dof_default_pos + self.reset_dof_pos_noise * rand_delta
                joint_pos = reset_state.robots[robot.name].joint_pos
                joint_pos[env_ids.unsqueeze(1), robot.hand_dof_idx.unsqueeze(0)] = dof_pos[:, robot.hand_dof_idx]
                robot_state = RobotState(
                    root_state=reset_state.robots[robot.name].root_state,
                    joint_pos=joint_pos,
                )
                reset_state.robots[robot.name] = robot_state
                start_idx += robot.num_joints

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
    object_pos,
    object_rot,
    target_pos,
    target_rot,
    right_hand_pos,
    left_hand_pos,
    dist_reward_scale: float,
    action_penalty_scale: float,
    actions,
    success_tolerance: float,
    reach_goal_bonus: float,
    throw_bonus: float,
    fall_penalty: float,
    leave_penalty: float,
    is_testing: bool = False,
):
    """Compute the reward of all environment.

    Args:
        reset_buf (tensor): The reset buffer of all environments at this time, shape (num_envs,)

        reset_goal_buf (tensor): The reset goal buffer of all environments at this time, shape (num_envs,)

        episode_length_buf (tensor): The porgress buffer of all environments at this time, shape (num_envs,)

        success_buf (tensor): The success buffer of all environments at this time, shape (num_envs,)

        max_episode_length (float): The max episode length in this environment

        object_pos (tensor): The position of the object

        object_rot (tensor): The rotation of the object

        target_pos (tensor): The position of the target

        target_rot (tensor): The rotate of the target

        right_hand_pos (tensor): The position of the right hand, shape (num_envs, 3)

        left_hand_pos (tensor): The position of the left hand, shape (num_envs, 3)

        dist_reward_scale (float): The scale of the distance reward

        action_penalty_scale (float): The scale of the action penalty

        actions (tensor): The action buffer of all environments at this time

        success_tolerance (float): The tolerance of the success determined

        reach_goal_bonus (float): The reward given when the object reaches the goal

        throw_bonus (float): The reward given when the object is thrown

        fall_penalty (float): The reward given when the object is fell

        leave_penalty (float): The reward given when hand leaves its base position

        is_testing (bool): Whether the environment is in testing mode, default False
    """
    # Distance from the hand to the object
    diff_xy = target_pos[:, :2] - object_pos[:, :2]
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
    reward_dist = goal_dist

    # Orientation alignment for the cube in hand and goal cube
    quat_diff = math.quat_mul(object_rot, math.quat_inv(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))

    dist_rew = reward_dist

    action_penalty = torch.sum(actions**2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = torch.exp(-0.2 * (dist_rew * dist_reward_scale)) - action_penalty * action_penalty_scale

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(
        torch.abs(goal_dist) <= 0.03,
        torch.ones_like(reset_goal_buf),
        reset_goal_buf,
    )
    success_buf = torch.where(
        success_buf == 0,
        torch.where(
            torch.abs(goal_dist) <= 0.03,
            torch.ones_like(success_buf),
            success_buf,
        ),
        success_buf,
    )

    # Reward for throwing the object
    thrown = (diff_xy[:, 1] >= -0.40) & (diff_xy[:, 1] <= -0.1) & (object_pos[:, 2] >= 0.7)
    reward = torch.where(thrown, reward + throw_bonus, reward)

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threashold
    reward = torch.where(object_pos[:, 2] <= 0.5, reward - fall_penalty, reward)

    # Check env termination conditions, including maximum success number
    resets = torch.where(object_pos[:, 2] <= 0.5, torch.ones_like(reset_buf), reset_buf)

    right_hand_base_dist = torch.norm(
        right_hand_pos - torch.tensor([-0.25, -0.55, 0.77], dtype=torch.float, device=right_hand_pos.device),
        p=2,
        dim=-1,
    )
    left_hand_base_dist = torch.norm(
        left_hand_pos - torch.tensor([-0.25, -1.15, 0.77], dtype=torch.float, device=left_hand_pos.device), p=2, dim=-1
    )

    reward = torch.where(right_hand_base_dist >= 0.1, reward - leave_penalty, reward)
    reward = torch.where(left_hand_base_dist >= 0.1, reward - leave_penalty, reward)

    if not is_testing:
        resets = torch.where(right_hand_base_dist >= 0.1, torch.ones_like(resets), resets)
        resets = torch.where(left_hand_base_dist >= 0.1, torch.ones_like(resets), resets)

    # Reset because of terminate or fall or success
    resets = torch.where(episode_length_buf >= max_episode_length, torch.ones_like(resets), resets)
    resets = torch.where(success_buf >= 1, torch.ones_like(resets), resets)

    return reward, resets, goal_resets, success_buf
