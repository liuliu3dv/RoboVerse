import os
import shutil

base_path = "/home/RoboVerse_Humanoid/roboverse_data/trajs/calvin/"

# task_list = [
#     "basketball_in_hoop",
#     # "beat_the_buzz",  # has bug
#     "block_pyramid",
#     "change_clock",
#     "close_fridge",
#     "empty_dishwasher",
#     "insert_onto_square_peg",
#     "lamp_on",
#     "light_bulb_in",
#     "meat_on_grill",
#     "open_box",
#     # "reach_and_drag" # bug
#     # "take_cup_out_from_cabinet"  # AttributeError: 'RigidObject' object has no attribute '_data'. Did you mean: 'data'?
#     "play_jenga"
# ]
task_list = os.listdir(base_path)
task_list.sort()

for task in task_list:
    task_path = base_path + task + "/"
    old_traj_path = task_path + "/v2/franka_v2.pkl.gz"
    new_traj_path = task_path + "/v2/g1_v2.pkl.gz"
    shutil.copyfile(old_traj_path, new_traj_path)
