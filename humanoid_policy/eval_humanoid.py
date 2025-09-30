from __future__ import annotations

import datetime
import os
import time
from dataclasses import dataclass
from typing import Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import imageio.v2 as iio
import numpy as np
import rootutils
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

from metasim.cfg.randomization import RandomizationCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors.cameras import PinholeCameraCfg
from metasim.constants import SimType
from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import get_robot, get_sim_env_class, get_task
from algorithms import PolicyRunner, get_runner

import torch
import numpy as np
import pyroki as pk
from pyroki import Robot
from yourdfpy import URDF
# from tqdm.rich import tqdm_rich as tqdm
from tqdm import tqdm
from metasim.cfg.robots.base_robot_cfg import BaseRobotCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import SimType
from metasim.sim import BaseSimHandler, EnvWrapper
from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import get_robot, get_sim_env_class, get_task
from metasim.utils.state import state_tensor_to_nested
from metasim.utils.tensor_util import tensor_to_cpu
import third_party.pyroki.examples.pyroki_snippets as pks
import pickle
import json

@dataclass
class Args:
    random: RandomizationCfg
    """Domain randomization options"""
    task: str
    """Task name"""
    robot: str = "franka"
    """Robot name"""
    num_envs: int = 1
    """Number of parallel environments, find a proper number for best performance on your machine"""
    sim: Literal["isaaclab", "mujoco", "isaacgym"] = "isaaclab"
    """Simulator backend"""
    max_demo: int | None = None
    """Maximum number of demos to collect, None for all demos"""
    headless: bool = False
    """Run in headless mode"""
    table: bool = True
    """Try to add a table"""
    task_id_range_low: int = 0
    """Low end of the task id range"""
    task_id_range_high: int = 1000
    """High end of the task id range"""
    checkpoint_path: str = ""
    """Path to the checkpoint"""
    algo: str = "diffusion_policy"
    """Algorithm to use"""
    subset: str = "pickcube_l0"
    """Subset your ckpt trained on"""
    action_set_steps: int = 1
    """Number of steps to take for each action set"""
    save_video_freq: int = 1
    """Frequency of saving videos"""
    max_step: int = 250
    """Maximum number of steps to collect"""
    gpu_id: int = 0
    """GPU ID to use"""

    def __post_init__(self):
        if self.random.table and not self.table:
            log.warning("Cannot enable table randomization without a table, disabling table randomization")
            self.random.table = False
        log.info(f"Args: {self}")


args = tyro.cli(Args)


def get_pk_robot(urdf) -> Robot:
    return pk.Robot.from_urdf(urdf)

def main():
    num_envs: int = args.num_envs
    log.info(f"Using GPU device: {args.gpu_id}")
    scene = "tapwater_scene_131"
    task = get_task(args.task)()
    task.episode_length = args.action_set_steps * args.max_step
    # task.decimation=3
    # task.contact_offset=0.01
    # task.rest_offset = 0.008
    # task.substeps=2
    # task.num_velocity_iterations = 4
    # task.max_depenetration_velocity = 5.0

    robot_franka = get_robot("franka")
    robot_g1 = get_robot("g1_hand")
    src_robot_urdf = URDF.load("./roboverse_data/robots/franka_with_gripper_extension/urdf/franka_with_gripper_extensions.urdf")
    src_robot = get_pk_robot(src_robot_urdf)
    tgt_robot_urdf = URDF.load(robot_g1.urdf_path)
    tgt_robot = get_pk_robot(tgt_robot_urdf)
    # camera = PinholeCameraCfg(data_types=["rgb", "depth"],pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0), width=1024, height=1024)
    camera = PinholeCameraCfg(data_types=["rgb", "depth"],pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))
    scenario = ScenarioCfg(
        # TODO retarget task
        task=args.task,
        robots=[robot_g1],
        scene=scene,
        cameras=[camera],
        random=args.random,
        try_add_table=args.table,
        sim=args.sim,
        headless=args.headless,
        num_envs=args.num_envs,
        humanoid=True
    )
    # camera = PinholeCameraCfg(
    #     data_types=["rgb", "depth"],
    #     pos=(1.5, -1.5, 1.5),
    #     look_at=(0.0, 0.0, 0.0),
    # )

    # # ✅ 统一 SceneCfg，并直接指定物理精度
    # scenario = ScenarioCfg(
    #     task=task,                   # 用 TaskCfg 而不是字符串
    #     robots=[robot_g1],
    #     scene="tapwater_scene_131",
    #     cameras=[camera],
    #     random=args.random,
    #     try_add_table=args.table,
    #     sim=args.sim,
    #     headless=args.headless,
    #     num_envs=args.num_envs,
    #     humanoid=True,
    #     sim_params=dict(
    #         substeps=2,
    #         num_position_iterations=12,
    #         num_velocity_iterations=4,
    #         contact_offset=0.005,
    #         rest_offset=0.001,
    #         bounce_threshold_velocity=0.2,
    #         solver_type=1,
    #         max_depenetration_velocity=1.0,
    #     )
    # )


    tic = time.time()
    env_class = get_sim_env_class(SimType(scenario.sim))
    env = env_class(scenario)
    toc = time.time()
    log.trace(f"Time to launch: {toc - tic:.2f}s")

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ckpt_name = args.checkpoint_path.split("/")[-1] + "_" + time_str
    ckpt_name = f"{args.task}/{args.algo}/{args.robot}/{ckpt_name}"
    runnerCls = get_runner(args.algo)
    policyRunner: PolicyRunner = runnerCls(
        scenario=scenario,
        num_envs=num_envs,
        checkpoint_path=args.checkpoint_path,
        device=f"cuda:{args.gpu_id}",
        task_name=args.task,
        subset=args.subset,
    )
    action_set_steps = 2 if policyRunner.policy_cfg.action_config.action_type == "ee" else 1
    ## Data
    tic = time.time()
    assert os.path.exists(task.traj_filepath), f"Trajectory file: {task.traj_filepath} does not exist."
    init_states, all_actions, all_states = get_traj(task, robot_franka, env.handler)
    num_demos = len(init_states)

    base_path = "/home/RoboVerse_Humanoid/humanoid_policy/roboverse_demo/demo_isaaclab/CloseBox-Level0/robot-g1_hand/"
    demo_list = os.listdir(base_path)
    for i, demo_dir in enumerate(demo_list):
        demo_path = base_path + demo_dir + "/"
        meta_data_path = demo_path + "metadata.pkl"
        meta_json_path = demo_path + "metadata.json"
        # Load pickle file
        # with open(meta_data_path, "rb") as f:  # "rb" means read in binary mode
        #     data = pickle.load(f)

        with open(meta_json_path, "r", encoding="utf-8") as f:  # "r" means read text
            data = json.load(f)
        # print(type(data))
        # print(data)
        init_dof_pos = {}
        for j in range(len(tgt_robot.joints.actuated_names)):
            init_dof_pos[tgt_robot.joints.actuated_names[j]] = data['joint_qpos'][i][j]
        init_states[i]["robots"]["g1"] = {
                    "pos": torch.tensor([-0.263, -0.0053, -0.]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0]),
                    "dof_pos": init_dof_pos
                }


    toc = time.time()
    log.trace(f"Time to load data: {toc - tic:.2f}s")

    total_success = 0
    total_completed = 0
    if args.max_demo is None:
        max_demos = args.task_id_range_high - args.task_id_range_low
    else:
        max_demos = args.max_demo
    max_demos = min(max_demos, num_demos)

    # init_states[0]["robots"]["g1"] = {
    #     "pos": torch.tensor([-0.263, -0.0053, -0.0]),
    #     "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
    # }

    for demo_start_idx in range(args.task_id_range_low, args.task_id_range_low + max_demos, num_envs):
        demo_end_idx = min(demo_start_idx + num_envs, num_demos)

        ## Reset before first step
        tic = time.time()
        obs, extras = env.reset(states=init_states[demo_start_idx:demo_end_idx])
        policyRunner.reset()
        toc = time.time()
        log.trace(f"Time to reset: {toc - tic:.2f}s")

        step = 0
        MaxStep = args.max_step
        SuccessOnce = [False] * num_envs
        TimeOut = [False] * num_envs
        images_list = []
        print(policyRunner.policy_cfg)
        while step < MaxStep:
            log.debug(f"Step {step}")
            new_obs = {
                "rgb": obs.cameras["camera0"].rgb,
                # "joint_qpos": obs.robots[args.robot].joint_pos,
                "joint_qpos": obs.robots["g1"].joint_pos,
            }

            images_list.append(np.array(new_obs["rgb"].cpu()))
            action = policyRunner.get_action(new_obs)

            for round_i in range(action_set_steps):
                # set left = 0
                # for joint, joint_value in action[0]['g1']['dof_pos_target'].items():
                #     if 'left' in joint:
                #         action[0]['g1']['dof_pos_target'][joint] = 0.0
                obs, reward, success, time_out, extras = env.step(action)

            # eval
            SuccessOnce = [SuccessOnce[i] or success[i] for i in range(num_envs)]
            TimeOut = [TimeOut[i] or time_out[i] for i in range(num_envs)]
            step += 1
            if all(SuccessOnce):
                break

        SuccessEnd = success.tolist()
        total_success += SuccessOnce.count(True)
        total_completed += len(SuccessOnce)
        os.makedirs(f"tmp/{ckpt_name}", exist_ok=True)
        for i, demo_idx in enumerate(range(demo_start_idx, demo_end_idx)):
            demo_idx_str = str(demo_idx).zfill(4)
            if i % args.save_video_freq == 0:
                iio.mimwrite(f"tmp/{ckpt_name}/{demo_idx}.mp4", [images[i] for images in images_list])
            with open(f"tmp/{ckpt_name}/{demo_idx_str}.txt", "w") as f:
                f.write(f"Demo Index: {demo_idx}\n")
                f.write(f"Num Envs: {num_envs}\n")
                f.write(f"SuccessOnce: {SuccessOnce[i]}\n")
                f.write(f"SuccessEnd: {SuccessEnd[i]}\n")
                f.write(f"TimeOut: {TimeOut[i]}\n")
                f.write(f"Cumulative Average Success Rate: {total_success / total_completed}\n")
        log.info("Demo Indices: ", range(demo_start_idx, demo_end_idx))
        log.info("Num Envs: ", num_envs)
        log.info(f"SuccessOnce: {SuccessOnce}")
        log.info(f"SuccessEnd: {SuccessEnd}")
        log.info(f"TimeOut: {TimeOut}")
    log.info(f"FINAL RESULTS: {total_success / total_completed}")
    with open(f"tmp/{ckpt_name}/final_stats.txt", "w") as f:
        f.write(f"Total Success: {total_success}\n")
        f.write(f"Total Completed: {total_completed}\n")
        f.write(f"Average Success Rate: {total_success / total_completed}\n")
    env.close()


if __name__ == "__main__":
    main()
