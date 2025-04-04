# Diffusion Policy

## Installation

```bash
cd roboverse_learn/algorithms/diffusion_policy
pip install -e .
cd ../../../

pip install pandas wandb
```

## Training Procedure

The main script for training is `train.sh`, which automates both data preparation and training.

### Usage of `train.sh`

```shell
bash roboverse_learn/algorithms/diffusion_policy/train.sh <task_name> <robot> <expert_data_num> <level> <seed> <gpu_id> <DEBUG> <num_epochs> <obs_space> <act_space> [<delta_ee>]
```

| Argument          | Description                                                 |
|-------------------|-------------------------------------------------------------|
| `task_name`       | Name of the task                                            |
| `robot`           | Robot type used for training                                |
| `expert_data_num` | Number of expert demonstrations that were collected         |
| `level`           | Randomization level of the demonstrations                   |
| `seed`            | Random seed for reproducibility                             |
| `gpu_id`          | ID of the GPU to use                                        |
| `DEBUG`           | Debug mode toggle (`True` or `False`)                       |
| `num_epochs`      | Number of training epochs                                   |
| `obs_space`       | Observation space (`joint_pos` or `ee`)                     |
| `act_space`       | Action space (`joint_pos` or `ee`)                          |
| `delta_ee`        | Optional: Delta control (`0` absolute, `1` delta; default 0)|

**Example:**
```shell
bash roboverse_learn/algorithms/diffusion_policy/train.sh CloseBox franka 100 0 42 0 False 200 ee ee 0
```

This script runs in two parts:

1. **Data Preparation**: _data2zarr_dp.py_ Converts the metadata into Zarr format for efficient dataloading. Automatically parses arguments and points to the correct `metadata_dir` (the location of data collected by the `collect_demo` script) to convert demonstration data into Zarr format.
2. **Training**: _diffusion_policy/train.py_ Uses the generated Zarr data, which gets stored in the `data_policy/` directory, to train the diffusion policy model.

We chose to combine these two parts for consistency of action-space and observation-space data processing, but these two parts can be ran independently if desired.

### Understanding data2zarr_dp.py

The `data2zarr_dp.py` script converts demonstration data into Zarr format for efficient data loading. While `train.sh` handles this automatically, you may want to run this step separately for custom data preprocessing.

```bash
python roboverse_learn/algorithms/diffusion_policy/data2zarr_dp.py [arguments]
```

**Key Arguments:**
| Argument | Description |
|----------|-------------|
| `--task_name` | Name of the task (e.g., StackCube_franka) |
| `--expert_data_num` | Number of episodes to process |
| `--metadata_dir` | Path to the demonstration metadata |
| `--downsample_ratio` | Downsample ratio for demonstration data |
| `--custom_name` | Custom name for the output Zarr file |
| `--observation_space` | Observation space to use (`joint_pos` or `ee`) |
| `--action_space` | Action space to use (`joint_pos` or `ee`) |
| `--delta_ee` | (optional) Delta control mode for end effector (0: absolute, 1: delta) |
| `--joint_pos_padding` | (optional) If > 0, pad joint positions to this length when using `joint_pos` observation/action space |

The processed data is saved to `data_policy/[task_name]_[expert_data_num]_[custom_name].zarr` and is ready for training. This script also saves a metadata.json which contains some of the above parameters so that the downstream policy training can see how the data is processed.

**Important Parameter Overrides:**
- `horizon`, `n_obs_steps`, and `n_action_steps` are set directly in `train.sh` and override the YAML configurations.
- All other parameters (e.g., batch size, number of epochs) can be manually adjusted in the YAML file: `roboverse_learn/algorithms/diffusion_policy/diffusion_policy/config/robot_dp.yaml`
- If you alter observation and action spaces, verify the corresponding shapes in: `roboverse_learn/algorithms/diffusion_policy/diffusion_policy/config/task/default_task.yaml`

### Switching between Joint Position and End Effector Control

- **Joint Position Control**: Set both `obs_space` and `act_space` to `joint_pos`.
- **End Effector Control**: Set both `obs_space` and `act_space` to `ee`. You may use `delta_ee=1` for delta mode or `delta_ee=0` for absolute positioning.

Adjust relevant configuration parameters in:
- `roboverse_learn/algorithms/diffusion_policy/diffusion_policy/config/robot_dp.yaml`

## Evaluation

To deploy and evaluate the trained policy:

```bash
python roboverse_learn/eval.py --task CloseBox --algo diffusion_policy --num_envs <up to ~50 envs works on RTX> --checkpoint_path <absolute_checkpoint_path>
```

Ensure that `<absolute_checkpoint_path>` points to your trained model checkpoint.
