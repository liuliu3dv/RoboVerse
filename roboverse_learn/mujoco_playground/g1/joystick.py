# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Joystick task for Unitree G1."""

from typing import Any
import torch
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np


from metasim.cfg.scenario import ScenarioCfg
from metasim.utils.setup_util import get_sim_env_class
from metasim.constants import SimType

CONFIG = {
    "episode_length": 1000,
    "action_repeat": 1,
    "action_scale": 0.5,
    "history_len": 1,
    "restricted_joint_range": False,
    "soft_joint_pos_limit_factor": 0.95,

    "noise_config": {
        "level": 1.0,  # Set to 0.0 to disable noise.
        "scales": {
            "joint_pos": 0.03,
            "joint_vel": 1.5,
            "gravity": 0.05,
            "linvel": 0.1,
            "gyro": 0.2,
        },
    },

    "reward_config": {
        "scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.75,
            "lin_vel_z": 0.0,
            "ang_vel_xy": -0.15,
            "orientation": -2.0,
            "base_height": 0.0,
            "torques": 0.0,
            "action_rate": 0.0,
            "energy": 0.0,
            "dof_acc": 0.0,
            "feet_clearance": 0.0,
            "feet_air_time": 2.0,
            "feet_slip": -0.25,
            "feet_height": 0.0,
            "feet_phase": 1.0,
            "alive": 0.0,
            "stand_still": -1.0,
            "termination": -100.0,
            "collision": -0.1,
            "contact_force": -0.01,
            "joint_deviation_knee": -0.1,
            "joint_deviation_hip": -0.25,
            "dof_pos_limits": -1.0,
            "pose": -0.1,
        },
        "tracking_sigma": 0.25,
        "max_foot_height": 0.15,
        "base_height_target": 0.5,
        "max_contact_force": 500.0,
    },

    "push_config": {
        "enable": True,
        "interval_range": [5.0, 10.0],
        "magnitude_range": [0.1, 2.0],
    },

    "command_config": {
        "a": [1.0, 0.8, 1.0],  # Amplitudes
        "b": [0.9, 0.25, 0.5],  # Probability of nonzero commands
    },

    "lin_vel_x": [-1.0, 1.0],
    "lin_vel_y": [-0.5, 0.5],
    "ang_vel_yaw": [-1.0, 1.0],
}


KNEE_BENT_STATE = {
    "robots": {
        "g1": {
            # [x, y, z | qw, qx, qy, qz | vx, vy, vz | wx, wy, wz]
            "root_state": torch.tensor(
                [[0.0, 0.0, 0.755,
                  1.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0]], dtype=torch.float32
            ),
            # joint order identical to simulator
            "joint_pos": torch.tensor(
                [[
                    # left leg
                    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
                    # right leg
                    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
                    # waist
                     0.0, 0.0, 0.073,
                    # left arm
                     0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
                    # right arm
                     0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0
                ]],
                dtype=torch.float32
            ),
        }
    },
}


class Joystick():
  """Track a joystick command."""

  def __init__(
      self,
      scenario: ScenarioCfg,
    #   task: str = "flat_terrain",
      device: str | torch.device | None = None,
  ):
    EnvironmentClass = get_sim_env_class(SimType(scenario.sim))
    self.env = EnvironmentClass(scenario)
    self.num_envs = scenario.num_envs
    self.robot = scenario.robots[0]
    self.robot_name = self.robot.name
    self.task = scenario.task
    self.device = device
    self._post_init()

  def _post_init(self) -> None:
    self.initial_state = KNEE_BENT_STATE     

    kb_joint_pos = KNEE_BENT_STATE["robots"][self.robot.name]["joint_pos"][0]
    self._default_pose = kb_joint_pos.to(device=self.device, dtype=self.dtype).clone()

    limits       = self.robot.joint_limits   
    self.joiners = torch.tensor(
        [limits[j][0] for j in self.joint_names],
        device=self.device, dtype=self.dtype,
    )
    self._uppers = torch.tensor(
        [limits[j][1] for j in self.joint_names],
        device=self.device, dtype=self.dtype,
    )

    c = (self._lowers + self._uppers) * 0.5
    r = self._uppers - self._lowers
    f = self._config.soft_joint_pos_limit_factor
    self._soft_lowers = c - 0.5 * r * f
    self._soft_uppers = c + 0.5 * r * f

    # fmt: off
    self._weights = torch.tensor([
        0.01, 1.0, 1.0, 0.01, 1.0, 1.0,
        0.01, 1.0, 1.0, 0.01, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ], device=self.device, dtype=self.dtype)

    self._cmd_a = torch.tensor(self._config.command_config["a"], device=self.device, dtype=self.dtype)
    self._cmd_b = torch.tensor(self._config.command_config["b"], device=self.device, dtype=self.dtype)

  def reset(self):
    # ---- basic tensors --------------------------------------------------
    dev, dt = self.device, self.dtype
    N       = self.num_envs
    kb      = KNEE_BENT_STATE["robots"][self.robot.name]

    # ---- dimensions (no mj_model) --------------------------------------
    nq      = kb["root_state"].shape[1]          # 13 (pos+quat+vel+ang_vel)
    nj      = kb["joint_pos"].shape[1]
    nv      = 6 + nj                             # 6 base dofs + joints

    qpos    = torch.zeros(N, nq, device=dev, dtype=dt)
    qvel    = torch.zeros(N, nv, device=dev, dtype=dt)

    # ---- template pose --------------------------------------------------
    root    = kb["root_state"][0]                # (13,)
    joints  = kb["joint_pos"][0]                 # (nj,)

    qpos[:, :3]   = root[:3]                     # base pos
    qpos[:, 3:7]  = root[3:7]                    # base quat
    qpos[:, 7:]   = joints.expand(N, nj)         # joint angles

    # ---- randomisation --------------------------------------------------
    qpos[:, :2]  += torch.empty(N, 2, device=dev, dtype=dt).uniform_(-0.5, 0.5)

    yaw           = torch.empty(N, 1, device=dev, dtype=dt).uniform_(-3.14, 3.14)
    quat_yaw      = torch.cat([torch.cos(yaw/2),
                            torch.zeros_like(yaw).repeat(1, 2),
                            torch.sin(yaw/2)], 1)
    qpos[:, 3:7]  = math.quat_mul(qpos[:, 3:7], quat_yaw)

    scale         = torch.empty(N, nj, device=dev, dtype=dt).uniform_(0.5, 1.5)
    qpos[:, 7:]  *= scale
    qvel[:, :6]   = torch.empty(N, 6, device=dev, dtype=dt).uniform_(-0.5, 0.5)

    # ---- write to TensorState ------------------------------------------
    ts = self.handler.get_state()
    rs = ts.robots[self.robot.name]

    rs.root_state[:, :3]   = qpos[:, :3]
    rs.root_state[:, 3:7]  = qpos[:, 3:7]
    rs.root_state[:, 7:10] = qvel[:, :3]
    rs.root_state[:,10:13] = qvel[:, 3:6]

    rs.joint_pos[:]        = qpos[:, 7:]
    rs.joint_vel[:]        = qvel[:, 6:6+nj]
    rs.joint_pos_target[:] = qpos[:, 7:]
    rs.joint_vel_target[:] = 0.0

    self.handler.set_state(ts)

    # ---- bookkeeping ----------------------------------------------------
    info = {"step": torch.zeros(N, dtype=torch.int64, device=dev)}
    self.info = info
    metrics = {}
    contact  = torch.zeros(N, len(self._feet_geom_id), device=dev, dtype=dt)

    obs   = self._get_obs(ts, info, contact)
    reward   = torch.zeros(N, device=dev, dtype=dt)
    done  = torch.zeros(N, device=dev, dtype=dt)
    return obs, reward, done, info, metrics

  def step(self, env_state, action):
    # -------------------------------------------------------------- fetch
    env_state = self.handler.get_state()             # TensorState (ignore arg)
    rs     = env_state.robots[self.robot_name]
    dev    = self.device
    dtype  = self.dtype

    # ------------------------------------------------------ push sampling
    push_theta     = torch.rand((), device=dev, dtype=dtype) * (2 * math.pi)
    m_lo, m_hi     = self._config.push_config.magnitude_range
    push_magnitude = torch.empty((), device=dev, dtype=dtype).uniform_(m_lo, m_hi)

    push = torch.stack([torch.cos(push_theta), torch.sin(push_theta)])
    push *= (
        torch.remainder(self.info["push_step"] + 1,
                        self.info["push_interval_steps"]) == 0
    )
    push *= self._config.push_config.enable

    # apply planar push to base linear velocity (vx, vy)
    rs.root_state[:, 7:9] = push * push_magnitude + rs.root_state[:, 7:9]

    # -------------------------------------------------------- motor cmd
    motor_targets            = self._default_pose + action * self._config.action_scale
    rs.joint_pos_target[:]   = motor_targets
    self.info["motor_targets"] = motor_targets

    # ----------------------------------------------------- write & step
    self.handler.set_state(env_state)
    self.handler.simulate       # advance simulation

    # -------------------------------------------------------- contacts
    contact = torch.stack([
        geoms_colliding(data, gid, self._floor_geom_id)
        for gid in self._feet_geom_id
    ])
    
    contact_filt          = contact | self.info["last_contact"]
    first_contact         = (self.info["feet_air_time"] > 0.0) * contact_filt
    self.info["feet_air_time"] += self.dt
    left_feet_pos = env_state.extras["left_foot_pos"]
    right_feet_pos = env_state.extras["right_foot_pos"]
    p_f   = torch.concatenate(
        [left_feet_pos, right_feet_pos], axis=-1
    )
    p_fz  = p_f[..., -1]
    self.info["swing_peak"] = torch.maximum(self.info["swing_peak"], p_fz)

    # ------------------------------------------------ obs / done / rwd
    obs  = self._get_obs(env_state, self.info, contact)
    done = self._get_termination(env_state)

    rewards = self._get_reward(
        env_state, action, self.info, self.metrics, done, first_contact, contact
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = sum(rewards.values()) * self.dt

    # --------------------------------------------------- bookkeeping
    self.info["push"]       = push
    self.info["step"]      += 1
    self.info["push_step"] += 1

    phase_tp1              = self.info["phase"] + self.info["phase_dt"]
    self.info["phase"]     = torch.fmod(phase_tp1 + math.pi, 2 * math.pi) - math.pi
    # NOTE(kevin): Enable this to make the policy stand still at 0 command.
    # self.info["phase"] = torch.where(
    #     torch.linalg.norm(self.info["command"]) > 0.01,
    #     self.info["phase"],
    #     torch.ones(2, device=dev, dtype=dtype) * math.pi,
    # )

    self.info["last_last_act"] = self.info["last_act"]
    self.info["last_act"]      = action

    # ------------------------------- command resample + maintenance
    if self.info["step"] > 500:
      self.info["command"] = self.sample_command()

    self.info["step"] = torch.where(
        done | (self.info["step"] > 500),
        torch.zeros_like(self.info["step"]),
        self.info["step"],
    )
    self.info["feet_air_time"] *= ~contact
    self.info["last_contact"]   = contact
    self.info["swing_peak"]    *= ~contact

    for k, v in rewards.items():
      self.metrics[f"reward/{k}"] = v
    self.metrics["swing_peak"] = torch.mean(self.info["swing_peak"])

    done = done.to(reward.dtype)

    # refresh state handle & return
    return obs, reward, done, self.info, self.metrics
  
  def _get_termination(self, data):
    fall_termination = self.get_gravity(data, "torso")[-1] < 0.0
    contact_termination = collision.geoms_colliding(
        data, self._right_foot_geom_id, self._left_foot_geom_id
    )
    contact_termination |= collision.geoms_colliding(
        data, self._left_foot_geom_id, self._right_shin_geom_id
    )
    contact_termination |= collision.geoms_colliding(
        data, self._right_foot_geom_id, self._left_shin_geom_id
    )
    return (
        fall_termination
        | contact_termination
        | torch.isnan(data.qpos).any()
        | torch.isnan(data.qvel).any()
    )


  def _get_obs(
      self, env_state, info: dict[str, Any], contact: jax.Array
  ) -> mjx_env.Observation:
    # gyro = self.get_gyro(data, "pelvis")
    gyro = env_state.extras["gyro_pelvis"]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )
    gravity = env_state.extras["pelvis_rot"] @ torch.tensor([0, 0, -1])
    # gravity = data.site_xmat[self._pelvis_imu_site_id].T @ jp.array([0, 0, -1])
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gravity = (
        gravity
        + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )

    joint_angles = env_state[7:]
        # joint_angles = data.qpos[7:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )

    joint_vel = env_state[]

    # joint_vel = data.qvel[6:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = (
        joint_vel
        + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_vel
    )

    cos = jp.cos(info["phase"])
    sin = jp.sin(info["phase"])
    phase = jp.concatenate([cos, sin])

    linvel = env_state.extras["local_linvel_pelvis"]
    # linvel = self.get_local_linvel(data, "pelvis")
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_linvel = (SensorData
        linvel
        + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.linvel
    )

    state = jp.hstack([
        noisy_linvel,  # 3
        noisy_gyro,  # 3
        noisy_gravity,  # 3
        info["command"],  # 3
        noisy_joint_angles - self._default_pose,  # 29
        noisy_joint_vel,  # 29
        info["last_act"],  # 29
        phase,
    ])
    accelerometer = env_state.extras["accelerometer_pelvis"]
    global_angvel = env_state.extras["global_angvel_pelvis"]
    # accelerometer = self.get_accelerometer(data, "pelvis")
    # global_angvel = self.get_global_angvel(data, "pelvis")
    left_feet_vel = env_state.extras["left_foot_global_linvel"]
    right_feet_vel = env_state.extras["right_foot_global_linvel"]
    feet_vel= torch.concatenate(
        [left_feet_vel, right_feet_vel], axis=-1
    )
    # feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
    root_height = env_state["robots"][self.robot_name]["pos"][2]

    privileged_state = torch.hstack([
        state,
        gyro,  # 3
        accelerometer,  # 3
        gravity,  # 3
        linvel,  # 3
        global_angvel,  # 3
        joint_angles - self._default_pose,
        joint_vel,
        root_height,  # 1
        env_state["robots"][self.robot_name][""],  # 29
        # data.actuator_force,  # 29
        contact,  # 2
        feet_vel,  # 4*3
        info["feet_air_time"],  # 2
    ])

    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  def _get_reward(
      self,
      env_state,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
      first_contact: jax.Array,
      contact: jax.Array,
  ) -> dict[str, jax.Array]:
    del metrics  # Unused.
    return {
        # Tracking rewards.
        "tracking_lin_vel": self._reward_tracking_lin_vel(
            info["command"], self.get_local_linvel(data, "pelvis")
        ),
        "tracking_ang_vel": self._reward_tracking_ang_vel(
            info["command"], self.get_gyro(data, "pelvis")
        ),
        # Base-related rewards.
        "lin_vel_z": self._cost_lin_vel_z(
            env_state.extras["global_linvel_pelvis"],
            env_state.extras["global_linvel_torso"],
        ),
        "ang_vel_xy": self._cost_ang_vel_xy(
            env_state.extras["global_angvel_torso"],
        ),
        "orientation": self._cost_orientation(env_state.extras["upvector_torso"]),
        "base_height": self._cost_base_height(data.qpos[2]),
        # Energy related rewards.
        "torques": self._cost_torques(data.actuator_force),
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
        "dof_acc": self._cost_dof_acc(data.qacc[6:]),
        # Feet related rewards.
        "feet_slip": self._cost_feet_slip(data, contact, info),
        "feet_clearance": self._cost_feet_clearance(data, info),
        "feet_height": self._cost_feet_height(
            info["swing_peak"], first_contact, info
        ),
        "feet_air_time": self._reward_feet_air_time(
            info["feet_air_time"], first_contact, info["command"]
        ),
        "feet_phase": self._reward_feet_phase(
            data,
            info["phase"],
            self._config.reward_config.max_foot_height,
            info["command"],
        ),
        # Other rewards.
        "alive": self._reward_alive(),
        "termination": self._cost_termination(done),
        "stand_still": self._cost_stand_still(info["command"], data.qpos[7:]),
        "collision": self._cost_collision(data),
        "contact_force": self._cost_contact_force(data),
        # Pose related rewards.
        "joint_deviation_hip": self._cost_joint_deviation_hip(
            data.qpos[7:], info["command"]
        ),
        "joint_deviation_knee": self._cost_joint_deviation_knee(data.qpos[7:]),
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
        "pose": self._cost_pose(data.qpos[7:]),
    }

  def _cost_contact_force(self, data: mjx.Data) -> jax.Array:
    l_contact_force = mjx_env.get_sensor_data(
        self.mj_model, data, "left_foot_force"
    )
    r_contact_force = mjx_env.get_sensor_data(
        self.mj_model, data, "right_foot_force"
    )
    cost = jp.clip(
        jp.abs(l_contact_force[2])
        - self._config.reward_config.max_contact_force,
        min=0.0,
    )
    cost += jp.clip(
        jp.abs(r_contact_force[2])
        - self._config.reward_config.max_contact_force,
        min=0.0,
    )
    return cost

  def _cost_collision(self, data: mjx.Data) -> jax.Array:
    c = collision.geoms_colliding(
        data, self._left_hand_geom_id, self._left_thigh_geom_id
    )
    c |= collision.geoms_colliding(
        data, self._right_hand_geom_id, self._right_thigh_geom_id
    )
    return jp.any(c)

  # Tracking rewards.

  def _cost_joint_deviation_hip(
      self, qpos: jax.Array, cmd: jax.Array
  ) -> jax.Array:
    error = qpos[self._hip_indices] - self._default_pose[self._hip_indices]
    # Allow roll deviation when lateral velocity is high.
    weight = jp.where(
        cmd[1] > 0.1,
        jp.array([0.0, 1.0, 0.0, 1.0]),
        jp.array([1.0, 1.0, 1.0, 1.0]),
    )
    cost = jp.sum(jp.abs(error) * weight)
    return cost

  def _cost_joint_deviation_knee(self, qpos: jax.Array) -> jax.Array:
    error = qpos[self._knee_indices] - self._default_pose[self._knee_indices]
    return jp.sum(jp.abs(error))

  def _cost_pose(self, qpos: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qpos - self._default_pose))

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)

  def _reward_tracking_lin_vel(
      self,
      commands: jax.Array,
      local_vel: jax.Array,
  ) -> jax.Array:
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    return jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)

  def _reward_tracking_ang_vel(
      self,
      commands: jax.Array,
      ang_vel: jax.Array,
  ) -> jax.Array:
    ang_vel_error = jp.square(commands[2] - ang_vel[2])
    return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)

  # Base-related rewards.

  def _cost_lin_vel_z(
      self,
      global_linvel_torso: jax.Array,
      global_linvel_pelvis: jax.Array,
  ) -> jax.Array:
    torso_cost = jp.square(global_linvel_torso[2])
    pelvis_cost = jp.square(global_linvel_pelvis[2])
    return torso_cost + pelvis_cost

  def _cost_ang_vel_xy(self, global_angvel_torso: jax.Array) -> jax.Array:
    return jp.sum(jp.square(global_angvel_torso[:2]))

  def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
    return jp.sum(jp.square(torso_zaxis - jp.array([0.073, 0.0, 1.0])))

  def _cost_base_height(self, base_height: jax.Array) -> jax.Array:
    return jp.square(
        base_height - self._config.reward_config.base_height_target
    )

  # Energy related rewards.

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sum(jp.abs(torques))

  def _cost_energy(
      self, qvel: jax.Array, qfrc_actuator: jax.Array
  ) -> jax.Array:
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    del last_last_act  # Unused.
    return jp.sum(jp.square(act - last_act))

  def _cost_dof_acc(self, qacc: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qacc))

  # Other rewards.

  def _cost_stand_still(
      self, commands: jax.Array, qpos: jax.Array
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands)
    cost = jp.sum(jp.abs(qpos - self._default_pose))
    cost *= cmd_norm < 0.01
    return cost

  def _cost_termination(self, done: jax.Array) -> jax.Array:
    return done

  def _reward_alive(self) -> jax.Array:
    return jp.array(1.0)

  # Feet related rewards.

  def _cost_feet_slip(
      self, data: mjx.Data, contact: jax.Array, info: dict[str, Any]
  ) -> jax.Array:
    del info  # Unused.
    body_vel = env_state.extras["global_linvel_pelvis"]
    # body_vel = self.get_global_linvel(data, "pelvis")[:2]
    reward = jp.sum(jp.linalg.norm(body_vel, axis=-1) * contact)
    return reward

  def _cost_feet_clearance(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> jax.Array:
    del info  # Unused.
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))

    left_feet_pos = env_state.extras["left_foot_pos"]
    right_feet_pos = env_state.extras["right_foot_pos"]
    foot_pos= jp.concatenate(
        [left_feet_pos, right_feet_pos], axis=-1
    )
    # foot_pos = data.site_xpos[self._feet_site_id]

    foot_z = foot_pos[..., -1]
    delta = jp.abs(foot_z - self._config.reward_config.max_foot_height)
    return jp.sum(delta * vel_norm)

  def _cost_feet_height(
      self,
      swing_peak: jax.Array,
      first_contact: jax.Array,
      info: dict[str, Any],
  ) -> jax.Array:
    del info  # Unused.
    error = swing_peak / self._config.reward_config.max_foot_height - 1.0
    return jp.sum(jp.square(error) * first_contact)

  def _reward_feet_air_time(
      self,
      air_time: jax.Array,
      first_contact: jax.Array,
      commands: jax.Array,
      threshold_min: float = 0.2,
      threshold_max: float = 0.5,
  ) -> jax.Array:
    del commands  # Unused.
    air_time = (air_time - threshold_min) * first_contact
    air_time = jp.clip(air_time, max=threshold_max - threshold_min)
    reward = jp.sum(air_time)
    return reward

  def _reward_feet_phase(
      self,
      env_state,
      env_extra,
      phase: jax.Array,
      foot_height: jax.Array,
      command: jax.Array,
  ) -> jax.Array:
    # Reward for tracking the desired foot height.
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    rz = gait.get_rz(phase, swing_height=foot_height)
    error = jp.sum(jp.square(foot_z - rz))
    reward = jp.exp(-error / 0.01)
    body_linvel = env_extra[key][:2]
    body_angvel = env_extra[key][2]
    # body_linvel = self.get_global_linvel(data, "pelvis")[:2]
    # body_angvel = self.get_global_angvel(data, "pelvis")[2]
    linvel_mask = jp.logical_or(
        jp.linalg.norm(body_linvel) > 0.1,
        jp.abs(body_angvel) > 0.1,
    )
    mask = jp.logical_or(linvel_mask, jp.linalg.norm(command) > 0.01)
    reward *= mask
    return reward

  def sample_command(self, rng: jax.Array) -> jax.Array:
    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

    lin_vel_x = jax.random.uniform(
        rng1, minval=self._config.lin_vel_x[0], maxval=self._config.lin_vel_x[1]
    )
    lin_vel_y = jax.random.uniform(
        rng2, minval=self._config.lin_vel_y[0], maxval=self._config.lin_vel_y[1]
    )
    ang_vel_yaw = jax.random.uniform(
        rng3,
        minval=self._config.ang_vel_yaw[0],
        maxval=self._config.ang_vel_yaw[1],
    )

    # With 10% chance, set everything to zero.
    return jp.where(
        jax.random.bernoulli(rng4, p=0.1),
        jp.zeros(3),
        jp.hstack([lin_vel_x, lin_vel_y, ang_vel_yaw]),
    )

