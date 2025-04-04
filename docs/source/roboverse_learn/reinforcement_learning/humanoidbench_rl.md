# Humanoidbench RL

We provide a basic RL training example for Humanoidbench tasks.

RL framework: `stable-baselines3`

RL algorithm: `PPO`

Simulator: `MuJoCo` and `IsaacGym`

## Installation

```bash
pip install stable-baselines3
pip install wandb
pip install wandb[media]
pip install tensorboard
```

Wandb login, enter your wandb account token.

```bash
wandb login
```

## Training

```bash
python roboverse_learn/humanoidbench_rl/train_sb3.py --sim mujoco --num_envs 1 --robot=h1 --task humanoidbench:Stand --wandb_entity <your_wandb_entity_name> --use_wandb
```

```bash
python roboverse_learn/humanoidbench_rl/train_sb3.py --sim isaacgym --num_envs 2 --robot=h1 --task humanoidbench:Stand --wandb_entity <your_wandb_entity_name> --use_wandb
```

```bash
python roboverse_learn/humanoidbench_rl/train_sb3.py --sim isaaclab --num_envs 2 --robot=h1 --task humanoidbench:Stand --wandb_entity <your_wandb_entity_name> --use_wandb
```

## Warning

- Isaacgym has problem now, see issue:
  - https://github.com/RoboVerseOrg/RoboVerse/issues/61, corresponding to the code: metasim/cfg/robots/h1_cfg.py
    - Switch the `fix_base_link` to `True` to ignore this bug.
  - https://github.com/RoboVerseOrg/RoboVerse/issues/57, corresponding to the code: metasim/utils/humanoid_robot_util.py
    - Switch the `head_body` to `mid360_link` to ignore this bug.
- Isaaclab has problem now:
  - H1 usd doesn't have head body, see code: metasim/utils/humanoid_robot_util.py
    - Switch the `head_body` to `mid360_link` to ignore this bug.

Current implementation of Isaacgym and Isaaclab has problem, after dealing with these 2 bugs or ignoring them, it's running well.

## Task list

- [ ]  Balance
    - Not Implemented because of collision detection issues.
- [x]  Crawl
- [x]  Cube
- [x]  Door
- [ ]  Highbar
    - Not implemented due to the need to connect H1 and Highbar.
- [ ]  Hurdle
    - Not Implemented because of collision detection issues.
- [ ]   Maze
    - Not Implemented because of collision detection issues.
- [x]  Package
- [ ]  Pole
    - Not Implemented because of collision detection issues.
- [x]  Powerlift
- [x]  Push
- [x]  Run
- [x]  Sit
- [x]  Slide
- [ ]  Spoon
    - Not Implemented because of sensor detection issues.
- [x]  Stair
- [x]  Stand
