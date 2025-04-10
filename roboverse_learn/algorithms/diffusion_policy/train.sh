# Examples:
# bash roboverse_learn/algorithms/diffusion_policy/train.sh CloseBox franka 100 1 42 0 False 200 ee ee 0

# 'expert_data_num' means number of training data. e.g.100
# 'level' means the level of the task. e.g.0
# 'seed' means random seed, select any number you like, e.g.42
# 'gpu_id' means single gpu id, e.g.0
# 'DEBUG' means whether to run in debug mode. e.g. False

metadata_dir=${1}
task_name=${2}
robot=${3}
expert_data_num=${4}
level=${5}
seed=${6}
gpu_id=${7}

DEBUG=${8} # True or False
num_epochs=${9}
obs_space=${10} # joint_pos or ee
act_space=${11} # joint_pos or ee
delta_ee=${12:-0} # 0 or 1 (only matters if act_space is ee, 0 means absolute 1 means delta control )

save_ckpt=True

alg_name=robot_dp


horizon=8
n_obs_steps=3
n_action_steps=4


name="obs:${obs_space}_act:${act_space}"
if [ "${delta_ee}" = 1 ]; then
  name="${name}_delta"
fi

python roboverse_learn/algorithms/data2zarr_dp.py \
--task_name ${task_name}_${robot}_level${level} \
--expert_data_num ${expert_data_num} \
--metadata_dir ${metadata_dir} \
--custom_name ${name} \
--action_space ${act_space} \
--observation_space ${obs_space} \
--delta_ee ${delta_ee}




config_name=${alg_name}
addition_info=max250
exp_name=${task_name}_${robot}_${alg_name}-${addition_info}
run_dir="info/outputs/${exp_name}_${horizon}_${n_obs_steps}_${n_action_steps}_${addition_info}_seed${seed}"




config_name=${alg_name}
addition_info=max250
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="info/outputs/${exp_name}_${horizon}_${n_obs_steps}_${n_action_steps}_${addition_info}_seed${seed}"

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"



export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
python roboverse_learn/algorithms/diffusion_policy/train.py --config-name=${config_name}.yaml \
                task.name=${task_name}_${robot}_level${level}_${expert_data_num}_${name} \
                task.dataset.zarr_path="data_policy/${task_name}_${robot}_level${level}_${expert_data_num}_${name}.zarr" \
                training.debug=$DEBUG \
                training.seed=${seed} \
                exp_name=${exp_name} \
                logging.mode=${wandb_mode} \
                horizon=${horizon} \
                n_obs_steps=${n_obs_steps} \
                n_action_steps=${n_action_steps} \
                training.num_epochs=${num_epochs} \
                policy_runner.obs.obs_type=${obs_space} \
                policy_runner.action.action_type=${act_space} \
                policy_runner.action.delta=${delta_ee} \
                training.device="cuda:${gpu_id}"
                # checkpoint.save_ckpt=${save_ckpt}
                # shape_meta.action.shape=8, \

# # example:
# python roboverse_learn/algorithms/data2zarr_dp.py \
# --task_name CloseBox_Franka_Level0_16 \
# --expert_data_num 100 \
# --metadata_dir roboverse_demo/demo_isaaclab/CloseBox-Level0-env16/robot-franka \
# --custom_name test \
# --action_space joint_pos \
# --observation_space joint_pos

# python roboverse_learn/algorithms/diffusion_policy/train.py --config-name=robot_dp.yaml \
#                 task.name=${task_name}_${robot}_level${level}_${expert_data_num}_${name} \
#                 task.dataset.zarr_path="data_policy/CloseBox_Franka_Level0_16_100_test.zarr" \
#                 training.debug=False \
#                 training.seed=0 \
#                 exp_name=CloseBox_Franka_Level0_16_100_test \
#                 logging.mode=${wandb_mode} \
#                 horizon=16 \
#                 n_obs_steps=8 \
#                 n_action_steps=8 \
#                 training.num_epochs=1000 \
#                 policy_runner.obs.obs_type=joint_pos \
#                 policy_runner.action.action_type=joint_pos \
#                 policy_runner.action.delta=0 \
#                 training.device="cuda:7"
