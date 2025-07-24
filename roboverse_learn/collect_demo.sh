## run nvidia-smi to check available GPUs
export CUDA_VISIBLE_DEVICES=0

## Parameters
task_name_set=CloseBox
random_level=0          # 0: No randomization 1: Randomize visual material 2: Randomize camera pose 3: Randomize object reflection and lighting
num_envs=1              # Number of parallel environments
demo_start_idx=0        # Index of the first demo to collect
max_demo_idx=1000       # Maximum index of demos to collect

obs_space=joint_pos
act_space=joint_pos
delta_ee=0              # Delta control
extra="obs:${obs_space}_act:${act_space}"
if [ "${delta_ee}" = 1 ]; then
  extra="${extra}_delta"
fi

## Collecting demonstration data run_all,
python metasim/scripts/collect_demo.py --task=${task_name_set} --num_envs=${num_envs} --run_all --headless --random.level=${random_level} --demo_start_idx=${demo_start_idx} --max_demo_idx=${max_demo_idx}


## Convert demonstration data
python roboverse_learn/algorithms/data2zarr_dp.py \
--task_name ${task_name_set}FrankaL${random_level}_${extra} \
--expert_data_num 100 \
--metadata_dir roboverse_demo/demo_isaaclab/${task_name_set}-Level${random_level}/robot-franka \
--action_space ${act_space} \
--observation_space ${obs_space}

# expert_data_num    Number of expert demonstrations to process
# metadata_dir       Path to the directory containing demonstration metadata saved by collect_demo
# action_space       Type of action space to use
# observation_space  Type of observation space to use
