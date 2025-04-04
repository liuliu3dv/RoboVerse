# ACT
Act (Action Chunking with Transformers) implements a transformer-based VAE policy, which generates chunks of ~100 actions at each step. These are averaged using temporal ensembling to generate a single action. This algorithm was introduced by the [Aloha](https://arxiv.org/abs/2304.13705) paper, and uses the same implementation.
## Installation

```bash
cd roboverse_learn/algorithms/act/detr
pip install -e .
cd ../../../

pip install pandas wandb
```

## Training Procedure

The main script for training is `train_act.sh`, which automates both data preparation and training. This is very similar to the diffusion policy training.

### Usage of `train_act.sh`

```shell
bash roboverse_learn/algorithms/act/train_act.sh <task_name> <robot> <expert_data_num> <level> <seed> <gpu_id> <num_epochs> <obs_space> <act_space> [<delta_ee>]
```

| Argument          | Description                                                 |
|-------------------|-------------------------------------------------------------|
| `task_name`       | Name of the task                                            |
| `robot`           | Robot type used for training                                |
| `expert_data_num` | Number of expert demonstrations that were collected         |
| `level`           | Randomization level of the demonstrations                   |
| `seed`            | Random seed for reproducibility                             |
| `gpu_id`          | ID of the GPU to use                                        |
| `num_epochs`      | Number of training epochs                                   |
| `obs_space`       | Observation space (`joint_pos` or `ee`)                     |
| `act_space`       | Action space (`joint_pos` or `ee`)                          |
| `delta_ee`        | Optional: Delta control (`0` absolute, `1` delta; default 0)|

**Example:**
```shell
bash roboverse_learn/algorithms/act/train_act.sh CloseBox franka 100 0 42 0 3000 joint_pos joint_pos 0
```

This script runs in two parts:

1. **Data Preparation**: _data2zarr_dp.py_ Converts the metadata into Zarr format for efficient dataloading. Automatically parses arguments and points to the correct `metadata_dir` (the location of data collected by the `collect_demo` script) to convert demonstration data into Zarr format. The diffusion policy page has more details regarding this script.
2. **Training**: _roboverse_learn/algorithms/act/train.py_ Uses the generated Zarr data, which gets stored in the `data_policy/` directory, to train the ACT model.


**Important Parameter Overrides:**
- Key hyperparameters including `kl_weight`, `chunk_size`, `hidden_dim`, `batch_size`, `state_dim`, `dim_feedforward` are set directly in `train_act.sh`.
- Learning rate is set to `1e-5` by default.
- Notably chunk size is the most important parameter, which is defaulted to 100 actions per step

### Switching between Joint Position and End Effector Control

- **Joint Position Control**: Set both `obs_space` and `act_space` to `joint_pos`.
- **End Effector Control**: Set both `obs_space` and `act_space` to `ee`. You may use `delta_ee=1` for delta mode or `delta_ee=0` for absolute positioning.
- Note the original ACT paper uses an action joint space of 14, but we modify the code to allow a parameterized action dimensionalty `state_dim` to be passed into the training python script, which we default to 9 for Franka joint space or end effector space.

## Evaluation

To deploy and evaluate the trained policy:

```bash
python roboverse_learn/eval.py --task CloseBox --algo ACT --num_envs <up to ~50 envs works on RTX> --checkpoint_path <absolute_checkpoint_path>
```

Ensure that `<absolute_checkpoint_path>` points to your trained model checkpoint, which should get saved to `info/outputs`
