
# Developer Guide: Migrating New Tasks
## Preparing Assets
You need to convert the assets into usd files and put them under `./data_isaaclab/assets/<benchmark_name>/<task_name>`.

Isaac Lab provides easy-to-use tools to convert urdf, mjcf, and mesh files into usd files, see [here](https://isaac-sim.github.io/IsaacLab/main/source/how-to/import_new_asset.html).

Specifically, if you are migrating from old version, please refer to `scripts/convert_usd.py` to ensure the new assets are compatible with the Isaac Lab standard.

## Implementing Tasks
Typically, you will need to define 4 methods in your task class:
- `add_object`: Load the assets to the scene. For physical states, there could be 3 cases:
    - No gravity, No collision (`XFormPrimView` in Isaac Sim)
        ```python
        spawn=sim_utils.UsdFileCfg(
            usd_path=...,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
        ),
        ```
        Refer to 'base' object in 'plug_charger' task.
    - No gravity, Has collision (`GeometryPrimView` in Isaac Sim)
        ```python
        spawn=sim_utils.UsdFileCfg(
            usd_path=...,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        ```
    - Has gravity, Has collision (`RigidPrimView` in Isaac Sim)
        ```python
        spawn=sim_utils.UsdFileCfg(
            usd_path=...,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        ```
        Refer to 'charger' object in 'plug_charger' task.
- `init_buffer`: Initialize the buffers to store task-specific data.
- `_get_dones`: Check whether the task is successful. It is recommended to use structured success checkers under `src/checkers`.
- `_reset_idx`: Reset the buffers, initialize the object pose.

And set three parameters in env cfg:
- `decimation`: Task decimation rate.
- `episode_length_s`: Episode length in seconds.
- `demo_file_path`: Path to the demonstration file.

## Migrating Demonstrations
Prepare a .pkl file containing demonstration data. Use the following format:

```python
{
    'source': 'ManiSkill2-rigid_body-PickSingleYCB-v0',
    'max_episode_len': 147,
    'demos': {
        "franka": [
            {
                'name': 'banana_traj_0',
                'description': 'Banana nana bana ba banana.', # optional
                'env_setup': {
                    "init_q": [...],          # [q_len], 9 for Franka
                    "init_robot_pos": [...],  # [3]
                    "init_robot_quat": [...], # [4]
                    "init_{obj_name}_pos": [...],    # [3]
                    "init_{obj_name}_quat": [...],   # [4]
                    "init_{joint_name}_q": [...],  # [1], for articulations
                    ...
                },
                'robot_traj': {
                    "q": [[...], ...],        # [demo_len x q_len]
                    "ee_act": [[...], ...],   # [demo_len x 1] 0.0~1.0 0 for close and 1 for open
                }
            },
            ...
        ],
    }
}
```

Explanation:
- `max_episode_len`: The maximum length of the demo in this file. This is used to pad the trajectories so they can be stacked into a tensor for efficiency.
- `robot_traj`:
    - `q`: Robot joint positions
    - `ee_act`: End-effector actions
- In each `env_setup`:
  - `init_q`: initial robot joint positions
  - `init_pos`: initial robot position
  - `init_quat`: initial robot rotation (quaterion)
  - The other `init_*` are used to initialize each environment, and what you save here could be retrived in the `_reset_idx` method of the task env. This usually includes the intial object position, rotation, scaling, joint positions, etc.
