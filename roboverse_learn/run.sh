task_name=CloseBox
level=0
config_name=dp_runner
num_epochs=100               # Number of training epochs
port=50010
seed=42
gpu=0
obs_space=joint_pos
act_space=joint_pos
delta_ee=0
eval_num_envs=1
eval_max_step=500
expert_data_num=100


extra="obs:${obs_space}_act:${act_space}"
if [ "${delta_ee}" = 1 ]; then
  extra="${extra}_delta"
fi

python roboverse_learn/main.py --config-name=${config_name}.yaml \
task_name="${task_name}_${extra}" \
dataset_config.zarr_path="data_policy/${task_name}FrankaL${level}_${extra}_${expert_data_num}.zarr" \
train_config.training_params.seed=${seed} \
train_config.training_params.num_epochs=${num_epochs} \
train_config.training_params.device=${gpu} \
eval_config.policy_runner.obs.obs_type=${obs_space} \
eval_config.policy_runner.action.action_type=${act_space} \
eval_config.policy_runner.action.delta=${delta_ee} \
eval_config.eval_args.task=${task_name} \
eval_config.eval_args.max_step=${eval_max_step} \
eval_config.eval_args.num_envs=${eval_num_envs} \
eval_config.eval_args.random.level=${level} \

## Seperate training and evaluation
# 1. open runner/dp_runner.py
# 2. only training: set `def run(self, train=True, eval=True, ckpt_path=None):` to
#                       `def run(self, train=True, eval=False, ckpt_path="None"):`
# 4. only evaluation: set `def run(self, train=True, eval=True, ckpt_path=None):` to
#                         `def run(self, train=False, eval=True, ckpt_path="/path/to/your/checkpoint.ckpt"):`
