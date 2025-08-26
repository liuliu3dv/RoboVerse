"""Domain Randomization Example for MetaSim."""

from __future__ import annotations

import rootutils

rootutils.setup_root(__file__, pythonpath=True)

import os
from typing import Literal

import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

from metasim.constants import PhysicStateType, SimType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils import configclass
from metasim.utils.obs_utils import ObsSaver
from metasim.utils.setup_util import get_sim_handler_class
from roboverse_pack.randomization import (
    FrictionRandomCfg,
    FrictionRandomizer,
    MassRandomCfg,
    MassRandomizer,
    MaterialPresets,
    MaterialRandomizer,
)

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


def run_domain_randomization(args):
    """Demonstrate domain randomization with specified simulator."""
    log.info(f"=== {args.simulator.upper()} Domain Randomization Demo ===")

    # Create scenario and update simulator
    scenario = ScenarioCfg(
        robots=["franka"],
        num_envs=args.num_envs,  # Multiple environments for parallel testing
        simulator=args.simulator,  # Will be overridden
        headless=args.headless,  # Will be overridden
    )

    # Add cameras for video recording
    scenario.cameras = [PinholeCameraCfg(width=1024, height=1024, pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))]

    # Add objects (same as 0_static_scene.py)
    scenario.objects = [
        PrimitiveCubeCfg(
            name="cube",
            size=(0.1, 0.1, 0.1),
            color=[1.0, 0.0, 0.0],
            physics=PhysicStateType.RIGIDBODY,
        ),
        PrimitiveSphereCfg(
            name="sphere",
            radius=0.1,
            color=[0.0, 0.0, 1.0],
            physics=PhysicStateType.RIGIDBODY,
        ),
        ArticulationObjCfg(
            name="box_base",
            fix_base_link=True,
            usd_path="metasim/example/example_assets/box_base/usd/box_base.usd",
            urdf_path="metasim/example/example_assets/box_base/urdf/box_base_unique.urdf",
            mjcf_path="metasim/example/example_assets/box_base/mjcf/box_base_unique.mjcf",
        ),
    ]

    # Get handler class based on simulator
    env_class = get_sim_handler_class(SimType(args.simulator))
    env = env_class(scenario)

    env.launch()
    init_states = [
        {
            "objects": {
                "cube": {
                    "pos": torch.tensor([0.3, -0.2, 0.05]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                },
                "sphere": {
                    "pos": torch.tensor([0.4, -0.6, 0.05]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                },
                "box_base": {
                    "pos": torch.tensor([0.5, 0.2, 0.1]),
                    "rot": torch.tensor([0.0, 0.7071, 0.0, 0.7071]),
                    "dof_pos": {"box_joint": 0.0},
                },
            },
            "robots": {
                "franka": {
                    "pos": torch.tensor([0.0, 0.0, 0.0]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    "dof_pos": {
                        "panda_joint1": 0.0,
                        "panda_joint2": -0.785398,
                        "panda_joint3": 0.0,
                        "panda_joint4": -2.356194,
                        "panda_joint5": 0.0,
                        "panda_joint6": 1.570796,
                        "panda_joint7": 0.785398,
                        "panda_finger_joint1": 0.04,
                        "panda_finger_joint2": 0.04,
                    },
                },
            },
        }
    ] * scenario.num_envs

    env.set_states(init_states)

    # Initialize video recording
    os.makedirs("get_started/output", exist_ok=True)
    obs_saver = ObsSaver(video_path=f"get_started/output/12_domain_randomization_{args.simulator}.mp4")
    obs = env.get_states(mode="dict")
    obs_saver.add(obs)

    # initialize randomizers
    cube_mass_config = MassRandomCfg(
        obj_name="cube",
        range=(0.3, 0.7),
        operation="abs",
        distribution="uniform",
    )
    cube_mass_randomizer = MassRandomizer(cube_mass_config)
    cube_mass_randomizer.bind_handler(env)

    franka_friction_config = FrictionRandomCfg(
        obj_name="franka",
        range=(0.5, 1.5),
        operation="add",
        distribution="gaussian",
    )
    franka_friction_randomizer = FrictionRandomizer(franka_friction_config)
    franka_friction_randomizer.bind_handler(env)

    franka_mass_config = MassRandomCfg(
        obj_name="franka",
        range=(0.2, 0.4),
        operation="abs",
        distribution="log_uniform",
    )
    franka_mass_randomizer = MassRandomizer(franka_mass_config)
    franka_mass_randomizer.bind_handler(env)

    # Initialize material randomizers with different strategies

    # Cube: Wood with MDL textures (combined mode - physics + visual)
    cube_material_randomizer = MaterialRandomizer(
        MaterialPresets.wood_object("cube", use_mdl=True, randomization_mode="combined")
    )
    cube_material_randomizer.bind_handler(env)

    # Sphere: Rubber with high bounce (combined mode - physics + visual)
    sphere_material_randomizer = MaterialRandomizer(
        MaterialPresets.rubber_object("sphere", randomization_mode="combined")
    )
    sphere_material_randomizer.bind_handler(env)

    # Box: Metal with MDL textures (combined mode - physics + visual)
    box_material_randomizer = MaterialRandomizer(
        MaterialPresets.wood_object("box_base", use_mdl=True, randomization_mode="combined")
    )
    box_material_randomizer.bind_handler(env)

    # Get cube mass using randomizer
    cube_mass = cube_mass_randomizer.get_body_mass("cube")
    robot_friction = franka_friction_randomizer.get_body_friction("franka")
    robot_mass = franka_mass_randomizer.get_body_mass("franka")

    # Get initial material properties for comparison
    initial_cube_physical = cube_material_randomizer.get_physical_properties()
    initial_sphere_physical = sphere_material_randomizer.get_physical_properties()

    # Store initial values for comparison
    initial_values = {
        "cube_mass": cube_mass.clone(),
        "franka_mass": robot_mass.clone(),
        "franka_friction": robot_friction.clone(),
        "cube_physical": initial_cube_physical,
        "sphere_physical": initial_sphere_physical,
    }

    # run randomization for cube mass
    cube_mass_randomizer()
    randomized_cube_mass = cube_mass_randomizer.get_body_mass("cube")
    log.info("================================================")
    log.info("randomizing cube mass by uniform distribution")
    log.info(f"  Before: {initial_values['cube_mass'].cpu().numpy().round(3)} kg")
    log.info(f"  After:  {randomized_cube_mass.cpu().numpy().round(3)} kg")

    # run randomization for franka friction
    franka_friction_randomizer()
    randomized_robot_friction = franka_friction_randomizer.get_body_friction("franka")
    log.info("================================================")
    log.info("randomizing franka friction by gaussian distribution")
    log.info(f"  Before: {initial_values['franka_friction'][0].cpu().numpy().round(3)}")
    log.info(f"  After:  {randomized_robot_friction[0].cpu().numpy().round(3)}")

    # run randomization for franka mass
    franka_mass_randomizer()
    randomized_sphere_mass = franka_mass_randomizer.get_body_mass("franka")
    log.info("================================================")
    log.info("randomizing franka mass by log_uniform distribution")
    log.info(f"  Before: {initial_values['franka_mass'].cpu().numpy().round(3)} kg")
    log.info(f"  After:  {randomized_sphere_mass.cpu().numpy().round(3)} kg")

    # run material randomization
    log.info("================================================")
    log.info("randomizing cube material (Wood: combined mode)")
    cube_material_randomizer()
    randomized_cube_physical = cube_material_randomizer.get_physical_properties()
    log.info("  Applied: Wood MDL texture + Physics properties")
    if "friction" in initial_values["cube_physical"] and "friction" in randomized_cube_physical:
        log.info(f"  Cube friction before: {initial_values['cube_physical']['friction'][0].round(3)}")
        log.info(f"  Cube friction after:  {randomized_cube_physical['friction'][0].round(3)}")

    log.info("================================================")
    log.info("randomizing sphere material (Rubber: combined mode)")
    sphere_material_randomizer()
    randomized_sphere_physical = sphere_material_randomizer.get_physical_properties()
    log.info("  Applied: Rubber PBR + Physics (high bounce)")
    if "friction" in initial_values["sphere_physical"] and "friction" in randomized_sphere_physical:
        log.info(f"  Sphere friction before: {initial_values['sphere_physical']['friction'][0].round(3)}")
        log.info(f"  Sphere friction after:  {randomized_sphere_physical['friction'][0].round(3)}")
    if "restitution" in initial_values["sphere_physical"] and "restitution" in randomized_sphere_physical:
        log.info(f"  Sphere restitution before: {initial_values['sphere_physical']['restitution'][0].round(3)}")
        log.info(f"  Sphere restitution after:  {randomized_sphere_physical['restitution'][0].round(3)}")

    log.info("================================================")
    log.info("randomizing box_base material (Metal: combined mode)")
    try:
        box_material_randomizer()
        log.info("  Applied: Metal MDL texture + Physics properties")
    except Exception as e:
        log.warning(f"  Metal material randomization failed: {e}")
        log.info("  This is expected if MDL files are not available")

    # Run simulation for a few steps with video recording
    log.info("================================================")
    log.info("Running simulation with randomized materials...")

    for step in range(100):
        log.debug(f"Simulation step {step}")
        env.simulate()
        obs = env.get_states(mode="dict")
        obs_saver.add(obs)

        # Apply randomization every 30 steps to show material changes (less frequent to reduce flicker)
        if step % 30 == 0 and step > 0:
            log.info(f"  Step {step}: Re-randomizing materials...")
            cube_material_randomizer()
            sphere_material_randomizer()
            box_material_randomizer()

    # Save video and close
    log.info("================================================")
    log.info("Saving video and closing simulation...")
    obs_saver.save()
    env.close()


def main():
    @configclass
    class Args:
        """Arguments for the static scene."""

        robot: str = "franka"

        ## Handlers
        simulator: Literal["isaacsim"] = "isaacsim"

        ## Others
        num_envs: int = 1
        headless: bool = False

        def __post_init__(self):
            """Post-initialization configuration."""
            log.info(f"Args: {self}")

    args = tyro.cli(Args)
    """Main function to run the domain randomization demo."""
    log.info("Starting Domain Randomization Demo")
    log.info("This demo showcases:")
    log.info("  - Mass randomization (cube and franka)")
    log.info("  - Friction randomization (franka)")
    log.info("  - Advanced material randomization with combined mode:")
    log.info("    * Cube: Wood (MDL + physics)")
    log.info("    * Sphere: Rubber (PBR + physics, high bounce)")
    log.info("    * Box: Metal (MDL + physics)")
    log.info("  - Flexible and extensible material configuration system")
    # Run IsaacSim demo
    run_domain_randomization(args)
    log.info("\nRandomization demo completed! Check the logs above for detailed results.")
    log.info(f"Video saved to: get_started/output/12_domain_randomization_{args.simulator}.mp4")


if __name__ == "__main__":
    main()
