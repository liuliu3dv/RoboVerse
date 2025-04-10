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


### Option1: 2 Step, Pre-processing and Training

#### Step1: Data Preparation
**Data Preparation**: _data2zarr_dp.py_ Converts the metadata into Zarr format for efficient dataloading. Automatically parses arguments and points to the correct `metadata_dir` (the location of data collected by the `collect_demo` script) to convert demonstration data into Zarr format.

Command:
```shell
python roboverse_learn/algorithms/data2zarr_dp.py \
--task_name <task_name> \
--expert_data_num <expert_data_num> \
--metadata_dir <metadata_dir> \
--custom_name <custom_name> \
--action_space <action_space> \
--observation_space <observation_space>
```
| Argument | Description |
|----------|-------------|
| `task_name` | Name of the task (e.g., CloseBox_Franka_Level0_16) |
| `expert_data_num` | Number of expert demonstrations to process |
| `metadata_dir` | Path to the directory containing demonstration metadata |
| `custom_name` | Custom string to append to the output filename |
| `action_space` | Type of action space to use (options: 'joint_pos' or 'ee') |
| `observation_space` | Type of observation space to use (options: 'joint_pos' or 'ee') |

**Example:**
```shell
python roboverse_learn/algorithms/data2zarr_dp.py \
--task_name CloseBox_Franka_Level0_16 \
--expert_data_num 100 \
--metadata_dir roboverse_demo/demo_isaaclab/CloseBox-Level0/robot-franka \
--custom_name customized_additional_string \
--action_space joint_pos \
--observation_space joint_pos
```

#### Step2: Training
**Training**: _diffusion_policy/train.py_ Uses the generated Zarr data, which gets stored in the `data_policy/` directory, to train the diffusion policy model.

Command:
```shell
python roboverse_learn/algorithms/diffusion_policy/train.py \
--config-name=robot_dp.yaml \
task.name=<task_name>_<robot>_level<level>_<expert_data_num>_<name> \
task.dataset.zarr_path=<zarr_path> \
training.debug=<debug_mode> \
training.seed=<seed> \
exp_name=<experiment_name> \
logging.mode=<wandb_mode> \
horizon=<horizon> \
n_obs_steps=<n_obs_steps> \
n_action_steps=<n_action_steps> \
training.num_epochs=<num_epochs> \
policy_runner.obs.obs_type=<obs_type> \
policy_runner.action.action_type=<action_type> \
policy_runner.action.delta=<delta> \
training.device=<device>
```
| Argument | Description |
|----------|-------------|
| `task_name` | Name of the task |
| `robot` | Type of robot being used |
| `level` | Difficulty level of the task |
| `expert_data_num` | Number of expert demonstrations |
| `name` | Custom name for the experiment |
| `zarr_path` | Path to the zarr dataset created in Step 1 |
| `debug_mode` | Enable/disable debug mode (True/False) |
| `seed` | Random seed for reproducibility |
| `experiment_name` | Name for the experiment run |
| `wandb_mode` | Weights & Biases logging mode |
| `horizon` | Time horizon for the policy |
| `n_obs_steps` | Number of observation steps |
| `n_action_steps` | Number of action steps |
| `num_epochs` | Number of training epochs |
| `obs_type` | Observation type (joint_pos or ee) |
| `action_type` | Action type (joint_pos or ee) |
| `delta` | Delta control mode (0 for absolute, 1 for delta) |
| `device` | GPU device to use (e.g., "cuda:0") |

**Example:**
```shell
python roboverse_learn/algorithms/diffusion_policy/train.py \
--config-name=robot_dp.yaml \
task.name=${task_name}_${robot}_level${level}_${expert_data_num}_${name} \        task.dataset.zarr_path="data_policy/CloseBox_Franka_Level0_16_100_test.zarr" \
training.debug=False \
training.seed=0 \
exp_name=CloseBox_Franka_Level0_16_100_test \
logging.mode=${wandb_mode} \
horizon=16 \
n_obs_steps=8 \
n_action_steps=8 \
training.num_epochs=1000 \
policy_runner.obs.obs_type=joint_pos \
policy_runner.action.action_type=joint_pos \
policy_runner.action.delta=0 \
training.device="cuda:7"
```



### Option2: Run with Single Command: train.sh

We further wrap the data preparation and training into a single command: `train.sh`.

```shell
bash roboverse_learn/algorithms/diffusion_policy/train.sh <metadata_dir> <task_name> <robot> <expert_data_num> <level> <seed> <gpu_id> <DEBUG> <num_epochs> <obs_space> <act_space> [<delta_ee>]
```

| Argument          | Description                                                 |
|-------------------|-------------------------------------------------------------|
| `metadata_dir`    | Path to the directory containing the demonstration metadata |
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
bash roboverse_learn/algorithms/diffusion_policy/train.sh roboverse_demo/demo_isaaclab/CloseBox-Level0/robot-franka CloseBox franka 100 0 42 0 False 500 ee ee 0
```

This script runs in two parts:

1. **Data Preparation**: _data2zarr_dp.py_ Converts the metadata into Zarr format for efficient dataloading. Automatically parses arguments and points to the correct `metadata_dir` (the location of data collected by the `collect_demo` script) to convert demonstration data into Zarr format.
2. **Training**: _diffusion_policy/train.py_ Uses the generated Zarr data, which gets stored in the `data_policy/` directory, to train the diffusion policy model.

We chose to combine these two parts for consistency of action-space and observation-space data processing, but these two parts can be ran independently if desired.

#### Understanding data2zarr_dp.py

The `data2zarr_dp.py` script converts demonstration data into Zarr format for efficient data loading. While `train.sh` handles this automatically, you may want to run this step separately for custom data preprocessing.

```bash
python roboverse_learn/algorithms/data2zarr_dp.py [arguments]
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

#### Switching between Joint Position and End Effector Control

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
