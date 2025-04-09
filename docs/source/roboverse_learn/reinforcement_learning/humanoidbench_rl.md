# Humanoidbench RL

We provide a basic RL training example for Humanoidbench tasks.

RL framework: `stable-baselines3`

RL algorithm: `PPO`

Simulator: `MuJoCo` and `IsaacGym` and `IsaacLab`

## Installation

```bash
pip install stable-baselines3
pip install wandb
pip install tensorboard
```

Wandb login, enter your wandb account token.

```bash
wandb login
```

## Training

> NOTE: You can add `--use_wandb --wandb_entity <your_wandb_entity_name>` to use wandb to log the training process.

- MuJoCo:

    ```bash
    python roboverse_learn/humanoidbench_rl/train_sb3.py --sim mujoco --num_envs 1 --robot=h1 --task humanoidbench:Stand
    ```

- IsaacGym:

    ```bash
    python roboverse_learn/humanoidbench_rl/train_sb3.py --sim isaacgym --num_envs 2 --robot=h1 --task humanoidbench:Stand
    ```

- IsaacLab:

    > IsaacLab is not supported currently due to incompatibilities between its infrastructure and the MuJoCo and IsaacGym frameworks.
    > We are continuing to work on this issue and will update the documentation when it is resolved.

    ```bash
    python roboverse_learn/humanoidbench_rl/train_sb3.py --sim isaaclab --num_envs 2 --robot=h1 --task humanoidbench:Stand
    ```

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
