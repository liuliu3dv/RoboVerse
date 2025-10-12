# Calvin Task Migration Guide

## Calvin Dataset

https://github.com/mees/calvin

This repo contains the raw dataset of calvin.

In dataset/README.md, you can see how the dataset is structured, and how to download the dataset.

## Calvin Environment

https://github.com/mees/calvin_env/tree/main

This repo contains the environment setup for calvin.

## What's Finished Now

[x] Loading data from the Calvin dataset
[x] Static environment setup for calvin_scene_A
[x] Joint-state based control of the robot

## Missing Blocks

[ ] Implement the functions of light and buttom in the Calvin Env.
[ ] Split the large dataset into small clips and save as trajectories.
[ ] Write the cfg file for all scenes (A, B, C, D)

## TODO

1. Implement the functions of light and buttom in the Calvin Env.

The detailed functions of light and buttom can be checked from the original repo of calvin env: calvin_env/scene/objects

2. Split the large dataset into small clips and save as trajectories.

Currently, the dataset consists of long sequences of robot motion. As error accumulates over time, the later part of the sequence may not be very accurate. The result is that the robot may miss the object when it tries to pick it up. To solve this problem, we can split the long sequence into small clips, each of which contains one or two robot-object interactions. This way, we can reduce the error accumulation and improve the quality of the trajectories.

3. Write the cfg file for all scenes (A, B, C, D)

Currently, we only wrote the cfg file for calvin_scene_A. We need to write the cfg file for all scenes (A, B, C, D) so that we can use them in the training and evaluation.