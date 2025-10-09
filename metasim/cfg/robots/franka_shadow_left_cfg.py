from __future__ import annotations

from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import torch

from metasim.utils import configclass, math
from metasim.utils.bidex_util import _solve_ik_single_env, jax_to_torch, torch_to_jax

from .base_dex_cfg import BaseDexCfg
from .base_robot_cfg import BaseActuatorCfg


@configclass
class FrankaShadowHandLeftCfg(BaseDexCfg):
    """Cfg for the franka with left shadow hand robot."""

    name: str = "franka_shadow_left"
    num_joints: int = 31
    num_arm_joints: int = 7
    fix_base_link: bool = True
    urdf_path: str = "roboverse_data/robots/franka_shadow_hand/urdf/franka_shadow_left.urdf"
    mjcf_path: str = "roboverse_data/robots/franka_shadow_hand/mjcf/franka_shadow_left.xml"
    usd_path: str = "roboverse_data/robots/franka_shadow_hand/usd/franka_shadow_left.usd"
    project_root: Path = Path(__file__).resolve().parents[3]
    hand_urdf_path: Path = (
        project_root / "roboverse_data" / "robots" / "franka_shadow_hand" / "urdf" / "shadow_left.urdf"
    )
    isaacgym_read_mjcf: bool = False
    enabled_gravity: bool = False
    enabled_self_collisions: bool = False
    use_vhacd: bool = True
    dof_drive_mode: Literal["none", "position", "effort"] = "position"
    angular_damping: float = None
    linear_damping: float = None
    tendon_limit_stiffness: float = 30
    tendon_damping: float = 0.1
    friction = None  # Use default friction from MJCF
    robot_controller: Literal["ik", "dof_pos", "dof_effort"] = "ik"
    fingertips = ["ffdistal", "mfdistal", "rfdistal", "lfdistal", "thdistal"]
    fingertips_offset = [0.0, 0.005, 0.02]
    wrist = "forearm"
    palm = "palm"
    root_link = "panda_link0"
    shadow_hand_wrist_stiffness: float = 5
    shadow_hand_wrist_damping: float = 0.5
    shadow_hand_finger_stiffness: float = 1.0
    shadow_hand_finger_damping: float = 0.1
    vel_obs_scale: float = 0.2  # Scale for velocity observations
    force_torque_obs_scale: float = 10.0  # Scale for force and torque observations

    actuators: dict[str, BaseActuatorCfg] = {
        "FFJ1": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "FFJ2": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "FFJ3": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "FFJ4": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "LFJ1": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "LFJ2": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "LFJ3": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "LFJ4": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "LFJ5": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "MFJ1": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "MFJ2": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "MFJ3": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "MFJ4": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "RFJ1": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "RFJ2": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "RFJ3": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "RFJ4": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "THJ1": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "THJ2": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "THJ3": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "THJ4": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "THJ5": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "WRJ1": BaseActuatorCfg(stiffness=shadow_hand_wrist_stiffness, damping=shadow_hand_wrist_damping),
        "WRJ2": BaseActuatorCfg(stiffness=shadow_hand_wrist_stiffness, damping=shadow_hand_wrist_damping),
        "panda_joint1": BaseActuatorCfg(stiffness=1e5, damping=1e4, velocity_limit=2.175),
        "panda_joint2": BaseActuatorCfg(stiffness=1e4, damping=1e3, velocity_limit=2.175),
        "panda_joint3": BaseActuatorCfg(stiffness=1e5, damping=5e3, velocity_limit=2.175),
        "panda_joint4": BaseActuatorCfg(stiffness=1e5, damping=1e4, velocity_limit=2.175),
        "panda_joint5": BaseActuatorCfg(stiffness=400, damping=50, velocity_limit=2.61),
        "panda_joint6": BaseActuatorCfg(stiffness=400, damping=50, velocity_limit=2.61),
        "panda_joint7": BaseActuatorCfg(stiffness=800, damping=50, velocity_limit=2.61),
    }

    joint_limits: dict[str, tuple[float, float]] = {
        "FFJ1": (0, 1.571),
        "FFJ2": (0, 1.571),
        "FFJ3": (0, 1.571),
        "FFJ4": (-0.349, 0.349),
        "LFJ1": (0, 1.571),
        "LFJ2": (0, 1.571),
        "LFJ3": (0, 1.571),
        "LFJ4": (-0.349, 0.349),
        "LFJ5": (0, 0.785),
        "MFJ1": (0, 1.571),
        "MFJ2": (0, 1.571),
        "MFJ3": (0, 1.571),
        "MFJ4": (-0.349, 0.349),
        "RFJ1": (0, 1.571),
        "RFJ2": (0, 1.571),
        "RFJ3": (0, 1.571),
        "RFJ4": (-0.349, 0.349),
        "THJ1": (0, 1.571),
        "THJ2": (-0.524, 0.524),
        "THJ3": (-0.209, 0.209),
        "THJ4": (0, 1.222),
        "THJ5": (-0.2, 1.047),
        "WRJ1": (-0.698, 0.489),
        "WRJ2": (-0.489, 0.14),
        "panda_joint1": (-2.8973, 2.8973),
        "panda_joint2": (-1.7628, 1.7628),
        "panda_joint3": (-2.8973, 2.8973),
        "panda_joint4": (-3.0718, -0.0698),
        "panda_joint5": (-2.8973, 2.8973),
        "panda_joint6": (-0.0175, 3.7525),
        "panda_joint7": (-2.8973, 2.8973),
    }

    arm_dof_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]
    # set False for visualization correction. Also see https://forums.developer.nvidia.com/t/how-to-flip-collision-meshes-in-isaac-gym/260779 for another example.
    isaacgym_flip_visual_attachments = False

    default_joint_positions: dict[str, float] = {
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
        "panda_joint2": -0.785398,
        "panda_joint3": 0.0,
        "panda_joint4": -2.356194,
        "panda_joint5": 0.0,
        "panda_joint6": 3.1415926,
        "panda_joint7": -2.356194,
    }

    def __post_init__(self):
        if self.isaacgym_read_mjcf:
            self.actuators["LFJ1"].fully_actuated = False
            self.actuators["RFJ1"].fully_actuated = False
            self.actuators["MFJ1"].fully_actuated = False
            self.actuators["FFJ1"].fully_actuated = False
        super().__post_init__()
        self.load_robot_for_ik()

    def scale_hand_action(self, actions: torch.Tensor) -> torch.Tensor:
        if self.robot_controller != "dof_pos":
            raise ValueError("robot_controller must be 'dof_pos' to use scale_hand_action")
        if actions.shape[1] != len(self.hand_acutuated_idx):
            raise ValueError(
                f"action shape {actions.shape} does not match hand dof {self.num_joints - self.num_arm_joints}"
            )
        hand_dof = math.unscale_transform(
            actions,
            self.joint_limits_lower[self.hand_dof_idx][self.hand_acutuated_idx],
            self.joint_limits_upper[self.hand_dof_idx][self.hand_acutuated_idx],
        )
        control_actions = torch.zeros((self.dof_pos.shape[0], len(self.hand_dof_idx)), device=hand_dof.device)
        control_actions[:, self.hand_acutuated_idx] = hand_dof
        return control_actions

    def control_arm_ik(self, dpose, num_envs: int, device: str):
        # Set controller parameters
        # IK params
        dpose = dpose.unsqueeze(-1)
        damping = 0.05
        jacobian_tensor = self.jacobian[:, self.jacobian_body_reindex, :, :][
            ..., self.jacobian_joint_reindex
        ]  # (num_envs, num_bodies, 6, num_dofs)
        # solve damped least squares
        if self.fix_base_link:
            ik_idx = self.wrist_index - 1 if self.root_link_index < self.wrist_index else self.wrist_index
            j_eef = jacobian_tensor[:, ik_idx, :, self.arm_dof_idx]
        else:
            j_eef = jacobian_tensor[:, self.wrist_index, :, self.arm_dof_idx]
        j_eef_T = torch.transpose(j_eef, 1, 2)
        lmbda = torch.eye(6, device=device) * (damping**2)
        u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, -1)
        u += self.dof_pos[:, self.arm_dof_idx]
        u = torch.clamp(u, self.joint_limits_lower[self.arm_dof_idx], self.joint_limits_upper[self.arm_dof_idx])
        return u

    def load_robot_for_ik(self):
        import pyroki as pk
        import yourdfpy

        robot_urdf_path = self.hand_urdf_path

        def filename_handler(fname: str) -> str:
            base_path = robot_urdf_path.parent
            return yourdfpy.filename_handler_magic(fname, dir=base_path)

        try:
            urdf = yourdfpy.URDF.load(robot_urdf_path, filename_handler=filename_handler)
        except FileNotFoundError as err:
            raise FileNotFoundError(f"File {robot_urdf_path} not found.") from err

        # Create robot.
        self.hand = pk.Robot.from_urdf(urdf)
        self.solver = jax.jit(jax.vmap(_solve_ik_single_env, in_axes=(None, 0, 0, 0, None), out_axes=0))
        self.hand_dof_names_ik = self.hand.joints.names[1:]  # skip the first joint which is world joint
        self.ik_reindex = [self.hand_dof_names.index(name) for name in self.hand_dof_names_ik]
        self.ik_reindex_inverse = [self.hand_dof_names_ik.index(name) for name in self.hand_dof_names]

    def control_hand_ik(self, target_pos, target_rot):
        if self.robot_controller != "ik":
            raise ValueError("robot_controller must be 'ik' to use control_hand_ik")
        init_q = torch_to_jax(self.dof_pos[:, self.hand_dof_idx][:, self.ik_reindex])
        target_wxyz = torch_to_jax(
            math.quat_mul(
                math.quat_from_euler_xyz(target_rot[..., 0], target_rot[..., 1], target_rot[..., 2]),
                self.ft_relative_rot,
            )
        )
        target_position = target_pos + self.ft_pos
        target_position = torch_to_jax(
            math.quat_apply(
                math.quat_inv(self.wrist_rot).unsqueeze(1).repeat(1, self.num_fingertips, 1),
                target_position - self.wrist_pos.unsqueeze(1),
            )
        )
        if not hasattr(self, "target_link_indices"):
            self.target_link_indices = jnp.array(
                [self.hand.links.names.index(name) for name in self.fingertips], dtype=jnp.int32
            )

        solution = self.solver(
            self.hand,
            init_q,
            target_wxyz,
            target_position,
            self.target_link_indices,
        )
        hand_dof = jax_to_torch(solution)[:, self.ik_reindex_inverse]
        hand_dof = torch.clamp(
            hand_dof, self.joint_limits_lower[self.hand_dof_idx], self.joint_limits_upper[self.hand_dof_idx]
        )
        action = torch.zeros((self.dof_pos.shape[0], len(self.hand_dof_idx)), device=hand_dof.device)
        action[:, self.hand_acutuated_idx] = hand_dof[:, self.hand_acutuated_idx]
        return action

    def reward(self, target_pos):
        """Reward based on the distance between the fingertips and the target position.

        Args:
            target_pos: (num_envs, 3) target position
        Returns:
            reward: (num_envs,) reward
        """
        dists = self.ft_pos - target_pos.unsqueeze(1)  # (num_envs, num_fingertips, 3)
        dists = torch.norm(dists, p=2, dim=-1)  # (num_envs, num_fingertips)
        mean_dists = torch.mean(dists, dim=-1)  # (num_envs,)
        sum_dists = torch.sum(dists, dim=-1)  # (num_envs,)
        reward = 1.2 - sum_dists
        return reward, mean_dists
