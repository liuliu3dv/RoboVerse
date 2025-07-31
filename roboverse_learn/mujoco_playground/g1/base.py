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

from metasim.cfg.query_type import SensorData, SitePos, SiteXMat


def extra_spec(self):
    """All extra observations required for the crawl task."""
    return {
        # torso / pelvis IMU sensors
        "upvector_torso": SensorData("upvector_torso"),
        "local_linvel_torso": SensorData("local_linvel_torso"),
        "accelerometer_torso": SensorData("accelerometer_torso"),
        "gyro_torso": SensorData("gyro_torso"),
        "forwardvector_torso": SensorData("forwardvector_torso"),
        "orientation_torso": SensorData("orientation_torso"),
        "global_linvel_torso": SensorData("global_linvel_torso"),
        "global_angvel_torso": SensorData("global_angvel_torso"),
        "upvector_pelvis": SensorData("upvector_pelvis"),
        "local_linvel_pelvis": SensorData("local_linvel_pelvis"),
        "accelerometer_pelvis": SensorData("accelerometer_pelvis"),
        "gyro_pelvis": SensorData("gyro_pelvis"),
        "forwardvector_pelvis": SensorData("forwardvector_pelvis"),
        "orientation_pelvis": SensorData("orientation_pelvis"),
        "global_linvel_pelvis": SensorData("global_linvel_pelvis"),
        "global_angvel_pelvis": SensorData("global_angvel_pelvis"),
        # foot sensors
        "left_foot_global_linvel": SensorData("left_foot_global_linvel"),
        "right_foot_global_linvel": SensorData("right_foot_global_linvel"),
        "left_foot_upvector": SensorData("left_foot_upvector"),
        "right_foot_upvector": SensorData("right_foot_upvector"),
        "left_foot_force": SensorData("left_foot_force"),
        "right_foot_force": SensorData("right_foot_force"),
        "left_foot_pos": SitePos(["left_foot"]),
        "right_foot_pos": SitePos(["right_foot"]),
        "left_palm_pos": SitePos(["left_palm"]),
        "right_palm_pos": SitePos(["right_palm"]),
        "pelvis_rot": SiteXMat("imu_in_pelvis"),
    }
