# Scenes in RoboVerse

RoboVerse supports a wide variety of **USD**-formatted assets that contribute to building immersive worlds. These assets are categorized for easy identification and usage. While not all assets are utilized in our demonstration tasks, they are available for you to incorporate into your own custom tasks. Below is an overview of the different types of assets available:

- [Scenes in RoboVerse](#scenes-in-roboverse)
  - [Ground Materials](#ground-materials)
  - [Scenes](#scenes)

## Ground Materials
Ground Materials are mainly adapted from [ARNOLD](https://arnold-benchmark.github.io/). These materials are ideal for texturing environments and can be applied easily in your tasks. Here’s an example of how you can use them:

```python
omni_python src/scripts/demo.py task=PickCube controller=ReplayController robot=Franka num_envs=4 demo_collection=true ground.material_mdl_path=data/source_data/arnold/materials/Wood/Bamboo.mdl

# other options
# data/source_data/arnold/materials/Wood/Walnut_Planks.mdl
# data/source_data/arnold/materials/Wood/Walnut.mdl
# data/source_data/arnold/materials/Wood/Birch_Planks.mdl
# data/source_data/arnold/materials/Wood/Cherry.mdl
# data/source_data/arnold/materials/Wood/Parquet_Floor.md
# data/source_data/arnold/materials/Wood/Mahogany.mdl
# data/source_data/arnold/materials/Wood/Cork.mdl
# data/source_data/arnold/materials/Wood/Plywood.mdl
# data/source_data/arnold/materials/Wood/Cherry_Planks.mdl
# data/source_data/arnold/materials/Wood/Timber_Cladding.mdl
# data/source_data/arnold/materials/Wood/Bamboo_Planks.mdl
# data/source_data/arnold/materials/Wood/Ash_Planks.mdl
# data/source_data/arnold/materials/Wood/Oak.mdl
# data/source_data/arnold/materials/Wood/Ash.mdl
# data/source_data/arnold/materials/Wood/Timber.mdl
# data/source_data/arnold/materials/Wood/Mahogany_Planks.mdl
# data/source_data/arnold/materials/Wood/Oak_Planks.mdl
# data/source_data/arnold/materials/Wood/Birch.mdl
# data/source_data/arnold/materials/Wood/Bamboo.mdl
# data/source_data/arnold/materials/Wood/Beadboard.mdl
```

## Scenes

RoboVerse also includes various predefined scenes that can be easily loaded into your simulation.

We are now actively collecting scenes from multiple sources and updating the existing scenes to match our newest configuration system. The length of the list and number of scenes will continuously increase. You are also welcomed to contribute your own scene to RoboVerse.

| Source | Categories & Amounts                      |
| ------ | ----------------------------------------- |
| [ARNOLD](https://arnold-benchmark.github.io/) | House (23) |
| [CALVIN](http://calvin.cs.uni-freiburg.de/) | CALVIN Table (4: Calvin Table A, B, C, D) |
| [DMControl](https://deepmind.google/discover/blog/dm-control-software-and-tasks-for-continuous-control/) | Locomotion (1) |
| Fetch | Manipulation (1) |
| [GAPartManip](https://arxiv.org/abs/2411.18276) | Manipulation (2) |
| [GAPartNet](https://pku-epic.github.io/GAPartNet/) | Manipulation (5) |
| GPT(GPT Generated Tasks) | Manipulation (1) |
| [GraspNet](https://graspnet.net/) | Grasping (1) |
| [HumanoidBench](https://humanoid-bench.github.io/) | Humanoid (19) |
| [IsaacgymEnvs](https://github.com/isaac-sim/IsaacGymEnvs) | Locomotion (1) Manipulation (1) |
| [LIBERO](https://libero-project.github.io/main.html) | Manipulation (10) |
| [Maniskill](https://www.maniskill.ai/) | Manipulation (7) |
| [MetaWorld](https://meta-world.github.io/) | Manipulation (6) |
| [Open6Dor](https://pku-epic.github.io/Open6DOR/) | Manipulation (68) |
| [RLAfford](https://sites.google.com/view/rlafford/) | Manipulation (1) |
| [RLBench](https://github.com/stepjam/RLBench) | Manipulation (68) |
| [RoboSuite](https://robosuite.ai/) | Manipulation (7) |
| [SimplerEnv](https://simpler-env.github.io/) | Manipulation (1) |
| [UH1](Learning from Massive Human Videos for Universal Humanoid Pose Control) | Humanoid (1) |
