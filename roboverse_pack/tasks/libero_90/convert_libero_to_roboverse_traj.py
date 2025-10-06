#!/usr/bin/env python3
"""
Convert LIBERO HDF5 demonstration data to RoboVerse trajectory format
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def convert_libero_to_roboverse_traj(hdf5_path, output_dir, task_name, robot_name="franka", num_demos=None):
    """
    Convert LIBERO HDF5 demonstration data to RoboVerse trajectory format

    Args:
        hdf5_path: Path to the LIBERO HDF5 file
        output_dir: Directory where trajectory files will be saved
        task_name: Name of the task
        robot_name: Name of the robot (default: "franka")
        num_demos: Maximum number of demos to convert (None for all)
    """

    print(f"Converting {hdf5_path} to RoboVerse trajectory format")
    print(f"Output directory: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize trajectory data
    init_states = []
    all_actions = []
    all_states = []

    with h5py.File(hdf5_path, "r") as f:
        data_group = f["data"]
        demo_keys = [key for key in data_group.keys() if key.startswith("demo_")]

        if num_demos is not None:
            demo_keys = demo_keys[:num_demos]

        print(f"Found {len(demo_keys)} demos to convert")

        for demo_idx, demo_key in enumerate(tqdm(demo_keys, desc="Converting demos")):
            demo_data = data_group[demo_key]

            # Extract data from LIBERO format
            actions = demo_data["actions"][:]  # Shape: (T, 7)
            states = demo_data["states"][:]  # Shape: (T, 51) - flattened mujoco states
            joint_states = demo_data["obs"]["joint_states"][:]  # Shape: (T, 7)
            gripper_states = demo_data["obs"]["gripper_states"][:]  # Shape: (T, 2)

            # Convert to RoboVerse format (v2 format)
            # LIBERO states (51-dim): flattened mujoco states containing robot and object states
            #
            # Actual structure (verified from Kitchen Scene 1 data):
            # qpos 部分 (索引 0-25, 共26维):
            #   idx[0]         - 时间戳 (timestamp, seconds, +0.05 per step @ 20Hz)
            #   idx[1:8]       - 机器人7个关节 (panda_joint1-7, 略异于obs/joint_states)
            #   idx[8:10]      - 夹爪2个关节 (panda_finger_joint1-2, 接近obs/gripper_states)
            #   idx[9]         - 未使用 (padding or reserved)
            #   idx[10:13]     - 黑碗位置 (akita_black_bowl x,y,z)
            #   idx[13:17]     - 黑碗四元数 (akita_black_bowl w,x,y,z)
            #   idx[17:20]     - 盘子位置 (plate x,y,z)
            #   idx[20:24]     - 盘子四元数 (plate w,x,y,z)
            #   idx[24:26]     - 未使用/填充 (padding)
            #
            # 注意:
            #   - 抽屉位置不在qpos中，可能在场景固定参数或另外的结构中
            #   - 建议使用obs/joint_states和obs/gripper_states获取精确的机器人状态
            #
            # qvel (索引 26-50, 共25维):
            #   [26-32] - 机器人7个关节速度 (panda_joint1-7 velocities)
            #   [33-34] - 夹爪2个关节速度 (panda_finger_joint1-2 velocities)
            #   [35]    - 抽屉滑动速度 (bottom drawer joint velocity)
            #   [36-38] - 黑碗线速度 (akita_black_bowl linear velocity x,y,z)
            #   [39-41] - 黑碗角速度 (akita_black_bowl angular velocity x,y,z)
            #   [42-44] - 盘子线速度 (plate linear velocity x,y,z)
            #   [45-47] - 盘子角速度 (plate angular velocity x,y,z)
            #   [48-50] - 未使用/填充 (padding)

            # Extract drawer position (main task object)
            drawer_pos = 0.0  # Drawer is fixed, only joint position matters
            drawer_joint_pos = states[0, 9]  # Bottom drawer slide position

            # Extract bowl state
            bowl_pos = states[0, 10:13].tolist()  # [x, y, z]
            bowl_quat = states[0, 13:17].tolist()  # [w, x, y, z]

            # Extract plate state
            plate_pos = states[0, 17:20].tolist()  # [x, y, z]
            plate_quat = states[0, 20:24].tolist()  # [w, x, y, z]

            print(states[0])

            # Robot base is fixed (mounted on table), position is implicit
            robot_pos = [-0.66, 0.0, 0.912]  # Base position (fixed)
            robot_rot = [1.0, 0.0, 0.0, 0.0]  # Base rotation (identity)

            # Build joint positions dictionary from joint_states
            robot_dof_pos = {
                "panda_joint1": float(joint_states[0, 0]),
                "panda_joint2": float(joint_states[0, 1]),
                "panda_joint3": float(joint_states[0, 2]),
                "panda_joint4": float(joint_states[0, 3]),
                "panda_joint5": float(joint_states[0, 4]),
                "panda_joint6": float(joint_states[0, 5]),
                "panda_joint7": float(joint_states[0, 6]),
                "panda_finger_joint1": float(gripper_states[0, 0]),
                "panda_finger_joint2": float(gripper_states[0, 1]),
            }

            # Initial state - v2 format expects pos, rot, dof_pos for robots and objects
            init_state = {
                robot_name: {
                    "pos": robot_pos,
                    "rot": robot_rot,
                    "dof_pos": robot_dof_pos,
                },
                "wooden_cabinet": {  # Cabinet with drawers
                    "pos": [0.00102559, -0.29300899, 0.905],  # Fixed position on table (from BDDL)
                    "rot": [0, 0.000000e00, 0.000000e00, 1.000000e00],
                    "dof_pos": {
                        "bottom_level": float(drawer_joint_pos),  # Bottom drawer position
                    },
                },
                "akita_black_bowl": {
                    "pos": [-0.00568576, 0.01078732, 0.90],
                    "rot": [7.07106785e-01, -2.20655841e-05, 1.74663862e-06, 7.07106777e-01],
                },
                "plate": {
                    "pos": [0.02411016, 0.23273392, 0.90244668],
                    "rot": [7.07106785e-01, -2.20655841e-05, 1.74663862e-06, 7.07106777e-01],
                },
            }

            # Actions (v2 format) - each timestep
            episode_actions = []
            episode_states = []

            for t in range(len(actions)):
                # Action for this timestep - v2 format expects dof_pos_target with joint names
                action = {
                    "dof_pos_target": {
                        "panda_joint1": float(joint_states[t, 0]),
                        "panda_joint2": float(joint_states[t, 1]),
                        "panda_joint3": float(joint_states[t, 2]),
                        "panda_joint4": float(joint_states[t, 3]),
                        "panda_joint5": float(joint_states[t, 4]),
                        "panda_joint6": float(joint_states[t, 5]),
                        "panda_joint7": float(joint_states[t, 6]),
                        "panda_finger_joint1": float(gripper_states[t, 0]),
                        "panda_finger_joint2": float(gripper_states[t, 1]),
                    }
                }
                episode_actions.append(action)

                # Extract states for this timestep
                drawer_joint_pos_t = float(states[t, 9])
                bowl_pos_t = states[t, 10:13].tolist()
                bowl_quat_t = states[t, 13:17].tolist()
                plate_pos_t = states[t, 17:20].tolist()
                plate_quat_t = states[t, 20:24].tolist()

                robot_dof_pos_t = {
                    "panda_joint1": float(joint_states[t, 0]),
                    "panda_joint2": float(joint_states[t, 1]),
                    "panda_joint3": float(joint_states[t, 2]),
                    "panda_joint4": float(joint_states[t, 3]),
                    "panda_joint5": float(joint_states[t, 4]),
                    "panda_joint6": float(joint_states[t, 5]),
                    "panda_joint7": float(joint_states[t, 6]),
                    "panda_finger_joint1": float(gripper_states[t, 0]),
                    "panda_finger_joint2": float(gripper_states[t, 1]),
                }

                # State for this timestep - v2 format expects pos, rot, dof_pos for robots and objects
                state = {
                    robot_name: {
                        "pos": robot_pos,
                        "rot": robot_rot,
                        "dof_pos": robot_dof_pos_t,
                    },
                    "wooden_cabinet_1": {
                        "pos": [0.0, -0.3, 0.0],
                        "rot": [0.0, 0.0, 1.0, 0.0],
                        "dof_pos": {
                            "drawer_bottom": drawer_joint_pos_t,
                        },
                    },
                    "akita_black_bowl_1": {
                        "pos": bowl_pos_t,
                        "rot": bowl_quat_t,
                    },
                    "plate_1": {
                        "pos": plate_pos_t,
                        "rot": plate_quat_t,
                    },
                }
                episode_states.append(state)

            # Add to trajectory data
            init_states.append(init_state)
            all_actions.append(episode_actions)
            all_states.append(episode_states)

    # Save trajectory data in v2 format
    # v2 format expects data organized by robot name
    # Each demo contains init_state with all objects, actions, and states
    traj_data = {robot_name: []}

    # Convert to v2 format for each episode
    for i in range(len(init_states)):
        # Episode data with complete initial state and per-timestep states
        episode_data = {
            "init_state": init_states[i],  # Contains all objects
            "actions": all_actions[i],
            "states": all_states[i],  # Each state contains all objects
        }
        traj_data[robot_name].append(episode_data)

    # Add metadata
    traj_data["metadata"] = {
        "task_name": task_name,
        "robot_name": robot_name,
        "num_episodes": len(demo_keys),
        "source": "libero",
        "original_file": os.path.basename(hdf5_path),
    }

    # Save as pickle file (RoboVerse format)
    traj_file = os.path.join(output_dir, f"{task_name}_traj_v2.pkl")
    with open(traj_file, "wb") as f:
        pickle.dump(traj_data, f)

    # Also save metadata as JSON for inspection
    metadata_file = os.path.join(output_dir, f"{task_name}_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(traj_data["metadata"], f, indent=2)

    print(f"Conversion completed!")
    print(f"Total episodes: {len(demo_keys)}")
    print(f"Trajectory file: {traj_file}")
    print(f"Metadata file: {metadata_file}")

    # Print some statistics
    total_steps = sum(len(actions) for actions in all_actions)
    avg_steps = total_steps / len(demo_keys) if len(demo_keys) > 0 else 0
    print(f"Total timesteps: {total_steps}")
    print(f"Average steps per episode: {avg_steps:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Convert LIBERO HDF5 to RoboVerse trajectory format")
    parser.add_argument(
        "--hdf5_path",
        type=str,
        default="roboverse_pack/tasks/libero_90/KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_demo.hdf5",
        help="Path to the LIBERO HDF5 file",
    )
    parser.add_argument(
        "--output_dir", type=str, default="libero_traj_output", help="Directory where trajectory files will be saved"
    )
    parser.add_argument("--task_name", type=str, default="libero_90_kitchen_scene1", help="Name of the task")
    parser.add_argument("--robot_name", type=str, default="franka", help="Name of the robot")
    parser.add_argument("--num_demos", type=int, default=3, help="Maximum number of demos to convert (None for all)")

    args = parser.parse_args()

    convert_libero_to_roboverse_traj(args.hdf5_path, args.output_dir, args.task_name, args.robot_name, args.num_demos)


if __name__ == "__main__":
    main()
