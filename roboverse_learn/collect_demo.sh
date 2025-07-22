export CUDA_VISIBLE_DEVICES=2
task_name=CloseBox
random_level=0
num_envs=1
demo_start_idx=0
max_demo_idx=1000

python metasim/scripts/collect_demo.py --task=${task_name} --num_envs=${num_envs} --run_all --headless --random.level=${random_level} --demo_start_idx=${demo_start_idx} --max_demo_idx=${max_demo_idx}

python roboverse_learn/algorithms/data2zarr_dp.py \
--task_name roboverse_demo/demo_isaaclab/${task_name}-Level${random_level}/robot-franka \
--expert_data_num 100 \
--metadata_dir ${task_name}FrankaL${random_level} \
--action_space joint_pos \
--observation_space joint_pos
