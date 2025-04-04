# Examples:
# bash data2zarr_dp.sh StackCube_franka 100 ~/RoboVerse/data_isaaclab/demo/StackCube/robot-franka

# The format of 'task_name' must be {task}_{robot} e.g.StackCube_franka
# 'expert_data_num' means number of training data
# 'metadata_dir' means the location of metadata



task_name=${1}
expert_data_num=${2}
metadata_dir=${3}

python data2zarr_dp.py ${task_name} ${expert_data_num} ${metadata_dir}
