# Tasks Overview

RoboVerse provides a diverse set of tasks for robots to learn and evaluate their capabilities. We mainly migrate the tasks from existing benchmarks and simulation environments. We further migrate trajectories, collect demonstrations, roll out existing policies or heuristics, and train policies on them. Please follow the license of the original tasks and datasets when using them.

## Task Organization

We organize the tasks with our standard metasim configuration, as shown in `metasim/cfg/tasks`.

## Task List

We provide the following benchmarks or task categories:

- "RLBench": RLBench tasks.
- "Garment": Garment manipulation tasks.
- "Humanoid": Humanoid manipulation tasks.
- "PointGoal": Point goal navigation tasks.
- "PointGoalRGB": Point goal navigation tasks with RGB observation.
- "PointGoalRGBD": Point goal navigation tasks with RGB-D observation.
