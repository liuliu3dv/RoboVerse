"""FootballShoot config in SkillBench in Skillblender"""

from __future__ import annotations

from typing import Callable

import torch

from metasim.cfg.objects import PrimitiveCubeCfg, PrimitiveSphereCfg
from metasim.cfg.simulator_params import SimParamCfg
from metasim.cfg.tasks.base_task_cfg import BaseRLTaskCfg
from metasim.cfg.tasks.skillblender.base_humanoid_cfg import BaseHumanoidCfg, BaseHumanoidCfgPPO
from metasim.constants import PhysicStateType
from metasim.sim import BaseSimHandler
from metasim.types import EnvState
from metasim.utils import configclass


# define new reward function
def reward_torso_pos(env_states: EnvState, robot_name: str, cfg: BaseRLTaskCfg):
    torso_pos = env_states.robots[robot_name].body_state[:, cfg.torso_indices, :3].squeeze(1)  # [envs, 3]
    torso_ori_ball_pos_diff = env_states.robots[robot_name].extra["ori_ball_pos"] - torso_pos
    torso_ori_ball_pos_diff = torso_ori_ball_pos_diff[:, :2]  # only xy
    torso_ori_ball_pos_error = torch.mean(torch.abs(torso_ori_ball_pos_diff), dim=1)
    return torch.exp(-4 * torso_ori_ball_pos_error), torso_ori_ball_pos_error


def reward_ball_pos(env_states: EnvState, robot_name: str, cfg: BaseRLTaskCfg):
    ball_goal_diff = (
        env_states.objects[cfg.objects[0].name].root_state[:, :3] - env_states.robots[robot_name].extra["goal_pos"]
    )
    ball_goal_error = torch.mean(torch.abs(ball_goal_diff), dim=1)
    return torch.exp(-1 * ball_goal_error), ball_goal_error


@configclass
class TaskBallCfgPPO(BaseHumanoidCfgPPO):
    @configclass
    class Policy(BaseHumanoidCfgPPO.Policy):
        """Network config class for PPO."""

        # HRL
        num_dofs = 19
        frame_stack = 1
        command_dim = 6
        # Expert skills
        skill_dict = {
            "h1_wrist_walking": {
                "experiment_name": "h1_wrist_walking",
                "load_run": "2025_0101_093233",
                "checkpoint": -1,
                "low_high": (-1, 1),
            },
            "h1_wrist_stepping": {
                "experiment_name": "h1_wrist_stepping",
                "load_run": "20250321_094203",
                "checkpoint": -1,
                "low_high": (-1, 1),
            },
        }

    @configclass
    class Runner(BaseHumanoidCfgPPO.Runner):
        policy_class_name = "ActorCriticHierarchical"
        max_iterations = 15001
        save_interval = 500
        experiment_name = "task_ball"
        run_name = ""

    policy = Policy()
    runner = Runner()


@configclass
class TaskBallCfg(BaseHumanoidCfg):
    """Cfg class for Skillbench:FootballShoot."""

    task_name = "task_ball"
    env_spacing = 10.0
    decimation = 10
    sim_params = SimParamCfg(
        dt=0.001,
        contact_offset=0.01,
        solver_type=1,
        substeps=1,
        num_position_iterations=4,
        max_depenetration_velocity=1.0,
        rest_offset=0.0,
        num_velocity_iterations=0,
        bounce_threshold_velocity=0.1,
        replace_cylinder_with_capsule=False,
        friction_offset_threshold=0.04,
        default_buffer_size_multiplier=5,
        num_threads=10,
    )

    ppo_cfg = TaskBallCfgPPO()
    command_ranges = BaseHumanoidCfg.CommandRanges(
        lin_vel_x=[-0, 0], lin_vel_y=[-0, 0], ang_vel_yaw=[-0, 0], heading=[-0, 0]
    )

    @configclass
    class TaskBallChecker(BaseHumanoidCfg.BaseLeggedRobotChecker):
        def check(self, handler: BaseSimHandler):
            reset_buf = super().check(handler)
            # if the ball hits the goal, reset the env
            envstates = handler.get_states()
            ball_pos = envstates.objects[handler.scenario.objects[0].name].root_state[:, :3]
            # HACK This is a hack to get the goal position for checker
            goal_pos = handler.task.goal_pos
            ball_goal_diff = ball_pos - goal_pos  # [envs, 3]
            ball_goal_dist = torch.norm(ball_goal_diff, dim=1)
            reset_buf |= ball_goal_dist < handler.task.command_ranges.threshold
            return reset_buf

    checker = TaskBallChecker()
    commands = BaseHumanoidCfg.CommandsConfig(num_commands=4, resampling_time=8.0)

    num_actions = 19
    frame_stack = 1
    c_frame_stack = 3
    command_dim = 6
    num_single_obs = 3 * num_actions + 6 + command_dim
    num_observations = int(frame_stack * num_single_obs)
    single_num_privileged_obs = 3 * num_actions + 33
    num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)

    reward_functions: list[Callable] = [reward_torso_pos, reward_ball_pos]
    reward_weights: dict[str, float] = {
        "torso_pos": 1,
        "ball_pos": 5,
    }

    objects = [
        PrimitiveSphereCfg(
            name="sphere",
            radius=0.2,
            color=[0.0, 0.5, 1.0],  # TODO domain randomization
            physics=PhysicStateType.RIGIDBODY,
            mass=0.4,  # TODO domain randomization
        ),
        PrimitiveCubeCfg(
            name="doll1",
            size=[0.05, 4.0, 2.0],
            color=[1.0, 1.0, 1.0],
            fix_base_link=True,
            enabled_gravity=True,
        ),
        PrimitiveCubeCfg(
            name="doll2",
            size=[1.0, 0.05, 2.0],
            color=[1.0, 1.0, 1.0],
            fix_base_link=True,
            enabled_gravity=True,
        ),
        PrimitiveCubeCfg(
            name="doll3",
            size=[1.0, 0.05, 2.0],
            color=[1.0, 1.0, 1.0],
            fix_base_link=True,
            enabled_gravity=True,
        ),
    ]

    init_states = [
        {
            "objects": {
                "sphere": {
                    "pos": torch.tensor([1.7, 0.0, 0.2]),  # TODO domain randomization as original repo
                    "rot": torch.tensor([
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                    ]),  # TODO domain randomize as original repo as original repo
                },
                "doll1": {
                    "pos": torch.tensor([5.0, 0.0, 1.0]),
                    "rot": torch.tensor([
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                    ]),
                },
                "doll2": {
                    "pos": torch.tensor([4.5, 2.0, 1.0]),
                    "rot": torch.tensor([
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                    ]),
                },
                "doll3": {
                    "pos": torch.tensor([4.5, -2.0, 1.0]),
                    "rot": torch.tensor([
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                    ]),
                },
            },
            "robots": {
                "h1_wrist": {
                    "pos": torch.tensor([0.0, 0.0, 1.0]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    "dof_pos": {
                        "left_hip_yaw": 0.0,
                        "left_hip_roll": 0.0,
                        "left_hip_pitch": -0.4,
                        "left_knee": 0.8,
                        "left_ankle": -0.4,
                        "right_hip_yaw": 0.0,
                        "right_hip_roll": 0.0,
                        "right_hip_pitch": -0.4,
                        "right_knee": 0.8,
                        "right_ankle": -0.4,
                        "torso": 0.0,
                        "left_shoulder_pitch": 0.0,
                        "left_shoulder_roll": 0.0,
                        "left_shoulder_yaw": 0.0,
                        "left_elbow": 0.0,
                        "right_shoulder_pitch": 0.0,
                        "right_shoulder_roll": 0.0,
                        "right_shoulder_yaw": 0.0,
                        "right_elbow": 0.0,
                    },
                },
            },
        }
    ]

    def __post_init__(self):
        super().__post_init__()
        # goal, related to asset size
        self.command_ranges.goal_x = [5.0, 5.0]
        self.command_ranges.goal_y = [-2.0, 2.0]
        self.command_ranges.goal_z = [0, 0.5]
        self.command_ranges.threshold = 0.5
        self.ball_range_x = [0.5, 1.0]
        self.ball_range_y = [-0.3, 0.3]
        self.ball_range_mass = [0.3, 0.5]
