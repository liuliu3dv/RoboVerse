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
"""Base classes for G1."""

from typing import Any, Dict, Optional, Union

from etils import epath
import jax
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import torch
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.g1 import g1_constants as consts
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.query_type import SensorData, SitePos
from metasim.constants import SimType
from metasim.utils.setup_util import get_sim_env_class
def get_assets() -> Dict[str, bytes]:
  assets = {}
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls", "*.xml")
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls" / "assets")
  path = mjx_env.MENAGERIE_PATH / "unitree_g1"
  mjx_env.update_assets(assets, path, "*.xml")
  mjx_env.update_assets(assets, path / "assets")
  return assets


def extra_spec(self):
    """All extra observations required for the crawl task."""
    return {
        # site positions (if still needed)
        # "head_pos"         : SitePos(["head"]),

        # torso / pelvis IMU sensors
        "upvector_torso"        : SensorData("upvector_torso"),
        "local_linvel_torso"    : SensorData("local_linvel_torso"),
        "accelerometer_torso"   : SensorData("accelerometer_torso"),
        "gyro_torso"            : SensorData("gyro_torso"),
        "forwardvector_torso"   : SensorData("forwardvector_torso"),
        "orientation_torso"     : SensorData("orientation_torso"),
        "global_linvel_torso"   : SensorData("global_linvel_torso"),
        "global_angvel_torso"   : SensorData("global_angvel_torso"),

        "upvector_pelvis"       : SensorData("upvector_pelvis"),
        "local_linvel_pelvis"   : SensorData("local_linvel_pelvis"),
        "accelerometer_pelvis"  : SensorData("accelerometer_pelvis"),
        "gyro_pelvis"           : SensorData("gyro_pelvis"),
        "forwardvector_pelvis"  : SensorData("forwardvector_pelvis"),
        "orientation_pelvis"    : SensorData("orientation_pelvis"),
        "global_linvel_pelvis"  : SensorData("global_linvel_pelvis"),
        "global_angvel_pelvis"  : SensorData("global_angvel_pelvis"),

        # foot sensors
        "left_foot_global_linvel" : SensorData("left_foot_global_linvel"),
        "right_foot_global_linvel": SensorData("right_foot_global_linvel"),
        "left_foot_upvector"      : SensorData("left_foot_upvector"),
        "right_foot_upvector"     : SensorData("right_foot_upvector"),
        "left_foot_force"         : SensorData("left_foot_force"),
        "right_foot_force"        : SensorData("right_foot_force"),

        "left_foot_pos" : SitePos(["left_foot"]),
        "right_foot_pos": SitePos(["right_foot"]),
        "left_palm_pos" : SitePos(["left_palm"]),
        "right_palm_pos": SitePos(["right_palm"]),

        "pelvis_rot" : SiteXMat("imu_in_pelvis"),
    }

class G1Env(mjx_env.MjxEnv):
  """Base class for G1 environments."""

  def __init__(
    self,
    scenario: ScenarioCfg,
    device: str | torch.device | None = None,
  ) -> None:
    super().__init__(config, config_overrides)

    EnvironmentClass = get_sim_env_class(SimType(scenario.sim))
    self.env = EnvironmentClass(scenario)

    self.num_envs = scenario.num_envs
    self.robot = scenario.robots[0]
    self.task = scenario.task

    # self._model_assets = get_assets()
    # self._mj_model = mujoco.MjModel.from_xml_string(
    #     epath.Path(xml_path).read_text(), assets=self._model_assets
    # )
    # self._mj_model.opt.timestep = self.sim_dt

    # if self._config.restricted_joint_range:
    #   self._mj_model.jnt_range[1:] = consts.RESTRICTED_JOINT_RANGE
    #   self._mj_model.actuator_ctrlrange[:] = consts.RESTRICTED_JOINT_RANGE

    # self._mj_model.vis.global_.offwidth = 3840
    # self._mj_model.vis.global_.offheight = 2160

    # self._mjx_model = mjx.put_model(self._mj_model)
    # self._xml_path = xml_path

  # Sensor readings.


  def get_sensor_data(
      model: mujoco.MjModel, data: mjx.Data, sensor_name: str
  ) -> jax.Array:
    """Gets sensor data given sensor name."""
    sensor_id = model.sensor(sensor_name).id
    sensor_adr = model.sensor_adr[sensor_id]
    sensor_dim = model.sensor_dim[sensor_id]
    return data.sensordata[sensor_adr : sensor_adr + sensor_dim]



  def get_gravity(self, data: mjx.Data, frame: str) -> jax.Array:
    """Return the gravity vector in the world frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{consts.GRAVITY_SENSOR}_{frame}"
    )

  def get_global_linvel(self, data: mjx.Data, frame: str) -> jax.Array:
    """Return the linear velocity of the robot in the world frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{consts.GLOBAL_LINVEL_SENSOR}_{frame}"
    )

  def get_global_angvel(self, data: mjx.Data, frame: str) -> jax.Array:
    """Return the angular velocity of the robot in the world frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{consts.GLOBAL_ANGVEL_SENSOR}_{frame}"
    )

  def get_local_linvel(self, data: mjx.Data, frame: str) -> jax.Array:
    """Return the linear velocity of the robot in the local frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{consts.LOCAL_LINVEL_SENSOR}_{frame}"
    )

  def get_accelerometer(self, data: mjx.Data, frame: str) -> jax.Array:
    """Return the accelerometer readings in the local frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{consts.ACCELEROMETER_SENSOR}_{frame}"
    )

  def get_gyro(self, data: mjx.Data, frame: str) -> jax.Array:
    """Return the gyroscope readings in the local frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{consts.GYRO_SENSOR}_{frame}"
    )

  # Accessors.

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self._mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
