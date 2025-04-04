# Examples:
# bash roboverse_learn/algorithms/act/train_act.sh CloseBox franka 100 0 42 0 3000 joint_pos joint_pos 0

# 'expert_data_num' means number of training data. e.g.100
# 'level' means the level of the task. e.g.0
# 'seed' means random seed, select any number you like, e.g.42
# 'gpu_id' means single gpu id, e.g.0
# 'DEBUG' means whether to run in debug mode. e.g. False

task_name=${1}
robot=${2}
expert_data_num=${3}
level=${4}
seed=${5}
gpu_id=${6}

num_epochs=${7}
obs_space=${8} # joint_pos or ee
act_space=${9} # joint_pos or ee
delta_ee=${10:-0} # 0 or 1 (only matters if act_space is ee, 0 means absolute 1 means delta control )

alg_name=ACT


name="obs:${obs_space}_act:${act_space}"
if [ "${delta_ee}" = 1 ]; then
  name="${name}_delta"
fi

python roboverse_learn/algorithms/data2zarr_dp.py --task_name ${task_name}_${robot}_level${level} --expert_data_num ${expert_data_num} --metadata_dir ~/RoboVerse/roboverse_demo/demo_isaaclab/${task_name}-Level${level}/robot-${robot} --custom_name ${name} --action_space ${act_space} --observation_space ${obs_space} --delta_ee ${delta_ee}


CUDA_VISIBLE_DEVICES=${gpu_id}
python -m roboverse_learn.algorithms.act.train \
--task_name ${task_name}_${robot}_level${level}_${expert_data_num}_${name} \
--num_episodes ${expert_data_num} \
--dataset_dir data_policy/${task_name}_${robot}_level${level}_${expert_data_num}_${name}.zarr \
--policy_class ${alg_name} --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs ${num_epochs}  --lr 1e-5 --state_dim 9 \
--seed ${seed}
