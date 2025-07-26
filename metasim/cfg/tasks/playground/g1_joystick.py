"""Base classes for G1."""

from metasim.cfg.query_type import SensorData, SitePos, SiteXMat
from metasim.cfg.tasks.base_task_cfg import SimParamCfg
from metasim.constants import TaskType
from metasim.utils import configclass


@configclass
class G1JoyStickCfg:
    """Base class for humanoid tasks."""

    decimation: int = 10
    task_type = TaskType.LOCOMOTION
    sim_params = SimParamCfg(
        dt=0.002,
    )

    def extra_spec(self):
        """All extra observations required for the crawl task."""
        return {
            # site positions (if still needed)
            # "head_pos"         : SitePos(["head"]),
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
