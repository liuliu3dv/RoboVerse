"""Domain randomization example with flexible baseline management.

This script demonstrates how to use domain randomization with flexible baseline management.
Key features:
- Each randomization is independent and based on a chosen baseline
- Can randomize based on original baseline or updated baseline
- Shows different randomization intensities (conservative, default, aggressive)
- Generates videos to demonstrate randomization diversity

Usage:
    # Basic usage with adaptive baseline (default)
    python 12_domain_randomization.py --sim isaacsim --randomization_mode default

    # Use original baseline (always randomize from initial scene)
    python 12_domain_randomization.py --sim isaacsim --randomization_mode default --use_original_baseline

    # Aggressive randomization with more scenarios
    python 12_domain_randomization.py --sim isaacsim --randomization_mode aggressive --num_scenarios 8

    # Update baseline every 2 scenarios (adaptive mode)
    python 12_domain_randomization.py --sim isaacsim --update_baseline_every 2
"""

from __future__ import annotations

from typing import Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import rootutils
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

import numpy as np

from get_started.utils import ObsSaver
from metasim.constants import PhysicStateType, SimType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.lights import DistantLightCfg, SphereLightCfg
from metasim.scenario.objects import PrimitiveCubeCfg, PrimitiveSphereCfg
from metasim.scenario.render import RenderCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils import configclass
from metasim.utils.setup_util import get_sim_handler_class

# Import domain randomization and utils
from roboverse_pack.randomization import DomainRandomizer


@configclass
class Args:
    """Arguments for domain randomization demo."""

    robot: str = "franka"
    """Robot to use in the simulation"""

    sim: Literal["isaacsim", "isaaclab"] = "isaacsim"
    """Simulator backend"""

    num_envs: int = 1
    """Number of parallel environments"""

    headless: bool = False
    """Run in headless mode"""

    randomization_mode: Literal["off", "conservative", "default", "aggressive"] = "default"
    """Domain randomization intensity"""

    num_scenarios: int = 5
    """Number of different randomized scenarios to generate"""

    frames_per_scenario: int = 50
    """Number of frames to record for each scenario"""

    use_original_baseline: bool = False
    """If True, always randomize from original baseline; if False, use adaptive baseline"""

    update_baseline_every: int = 3
    """Update baseline every N scenarios (only if use_original_baseline=False)"""

    save_videos: bool = True
    """Whether to save videos showing randomization effects"""

    def __post_init__(self):
        log.info(f"Args: {self}")


def create_base_scenario(args: Args) -> ScenarioCfg:
    """Create the base scenario configuration."""

    scenario = ScenarioCfg(
        robots=[args.robot],
        headless=args.headless,
        num_envs=args.num_envs,
        simulator=args.sim,
        render=RenderCfg(mode="pathtracing") if args.sim == "isaacsim" else RenderCfg(),
    )

    # Add observation camera
    scenario.cameras = [
        PinholeCameraCfg(
            name="obs_camera",
            width=512,
            height=512,
            pos=(2.0, -1.5, 1.5),
            look_at=(0.0, 0.0, 0.5),
            focal_length=35.0,
            horizontal_aperture=20.955,
            data_types=["rgb", "depth"],
        )
    ]

    # Add basic lighting
    scenario.lights = [
        DistantLightCfg(name="main_distant_light", intensity=1500.0, color=(1.0, 0.95, 0.9), polar=30.0, azimuth=45.0),
        SphereLightCfg(
            name="main_sphere_light", intensity=800.0, color=(0.9, 0.9, 1.0), radius=0.3, pos=(1.0, 1.0, 2.0)
        ),
    ]

    # Add base objects (these form the base scene)
    scenario.objects = [
        # Table surface
        PrimitiveCubeCfg(
            name="table_surface",
            size=[1.0, 1.0, 0.02],
            color=[0.8, 0.6, 0.4],
            mass=10.0,
            default_position=(0.5, 0.5, 0.41),
            physics=PhysicStateType.RIGIDBODY,
            fix_base_link=True,
        ),
        # Some basic objects to manipulate
        PrimitiveCubeCfg(
            name="red_cube",
            size=[0.1, 0.1, 0.1],
            color=[1.0, 0.0, 0.0],
            mass=0.5,
            default_position=(0.3, -0.2, 0.5),
            physics=PhysicStateType.RIGIDBODY,
        ),
        PrimitiveCubeCfg(
            name="green_cube",
            size=[0.08, 0.08, 0.08],
            color=[0.0, 1.0, 0.0],
            mass=0.3,
            default_position=(0.5, -0.1, 0.5),
            physics=PhysicStateType.RIGIDBODY,
        ),
        PrimitiveSphereCfg(
            name="blue_sphere",
            radius=0.06,
            color=[0.0, 0.0, 1.0],
            mass=0.4,
            default_position=(0.4, -0.3, 0.5),
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]

    log.info(
        f"Created base scenario with {len(scenario.objects)} objects, "
        f"{len(scenario.lights)} lights, {len(scenario.cameras)} cameras"
    )

    return scenario


def create_domain_randomizer(args: Args) -> DomainRandomizer:
    """Create domain randomizer based on the specified mode."""

    if args.randomization_mode == "off":
        domain_randomizer = DomainRandomizer(seed=42)
        domain_randomizer.disable_all()
        return domain_randomizer

    elif args.randomization_mode == "conservative":
        return DomainRandomizer(
            lighting_config={
                "intensity_range": (800.0, 2000.0),
                "color_variation": 0.1,
                "preset_probability": 0.2,  # 20% complete preset replacement
                "modify_probability": 0.7,  # 70% modify existing lights
                "add_only_probability": 0.1,  # 10% add new lights only
                "max_additional_lights": 1,
            },
            camera_config={
                "distance_range": (1.8, 2.5),
                "elevation_range": (20.0, 60.0),
                "azimuth_range": (-60.0, 60.0),
                "look_at_noise": 0.05,
                "use_preset_probability": 0.7,  # 70% use YAML presets
                "randomize_intrinsics": False,  # Conservative: don't change intrinsics
                "randomize_focus": False,  # Conservative: don't change focus
            },
            material_config={
                "randomize_objects": True,
                "object_change_probability": 0.6,  # Conservative: 60% chance to change object materials
                "randomize_environment": True,
                "environment_change_probability": 0.4,  # Conservative: 40% chance to change environment
                "randomize_robots": False,
                "apply_physics_materials": False,  # Conservative: no physics materials
                "split": "train",
            },
            object_config={
                "modify_existing_probability": 0.8,
                "add_yaml_objects_probability": 0.2,
                "add_parametric_objects_probability": 0.0,
                "position_noise": 0.05,
                "rotation_noise": 0.1,
                "scale_variation": 0.05,
                "mass_variation": 0.1,
                "color_variation": 0.1,
                "max_additional_objects": 1,
            },
            seed=42,
        )

    elif args.randomization_mode == "default":
        return DomainRandomizer(
            lighting_config={
                "intensity_range": (500.0, 2500.0),
                "color_variation": 0.2,
                "preset_probability": 0.3,  # 30% complete preset replacement
                "modify_probability": 0.4,  # 40% modify existing lights
                "add_only_probability": 0.3,  # 30% add new lights only
                "max_additional_lights": 2,
            },
            camera_config={
                "distance_range": (1.5, 3.0),
                "elevation_range": (15.0, 75.0),
                "azimuth_range": (-90.0, 90.0),
                "look_at_noise": 0.1,
                "use_preset_probability": 0.5,  # 50% use YAML presets
                "randomize_intrinsics": True,  # Default: randomize intrinsics
                "randomize_focus": False,  # Default: don't change focus
            },
            material_config={
                "randomize_objects": True,
                "object_change_probability": 0.8,  # Default: 80% chance to change object materials
                "randomize_environment": True,
                "environment_change_probability": 0.6,  # Default: 60% chance to change environment
                "randomize_robots": False,
                "apply_physics_materials": True,  # Default: include physics materials
                "physics_change_probability": 0.3,  # Default: 30% chance for physics materials
                "split": "train",
            },
            object_config={
                "modify_existing_probability": 0.5,
                "add_yaml_objects_probability": 0.3,
                "add_parametric_objects_probability": 0.2,
                "position_noise": 0.1,
                "rotation_noise": 0.2,
                "scale_variation": 0.1,
                "mass_variation": 0.2,
                "color_variation": 0.15,
                "max_additional_objects": 2,
            },
            seed=42,
        )

    elif args.randomization_mode == "aggressive":
        return DomainRandomizer(
            lighting_config={
                "intensity_range": (300.0, 4000.0),
                "color_variation": 0.3,
                "preset_probability": 0.4,  # 40% complete preset replacement
                "modify_probability": 0.3,  # 30% modify existing lights
                "add_only_probability": 0.3,  # 30% add new lights only
                "max_additional_lights": 4,
            },
            camera_config={
                "distance_range": (1.0, 4.0),
                "elevation_range": (10.0, 90.0),
                "azimuth_range": (-180.0, 180.0),
                "look_at_noise": 0.2,
                "use_preset_probability": 0.3,  # 30% use YAML presets
                "randomize_intrinsics": True,  # Aggressive: randomize intrinsics
                "randomize_focus": True,  # Aggressive: also randomize focus
            },
            material_config={
                "randomize_objects": True,
                "object_change_probability": 0.9,  # Aggressive: 90% chance to change object materials
                "randomize_environment": True,
                "environment_change_probability": 0.8,  # Aggressive: 80% chance to change environment
                "randomize_robots": True,  # Aggressive: also randomize robot materials
                "robot_change_probability": 0.5,  # Aggressive: 50% chance for robot materials
                "apply_physics_materials": True,  # Aggressive: include physics materials
                "physics_change_probability": 0.5,  # Aggressive: 50% chance for physics materials
                "split": "train",
            },
            object_config={
                "modify_existing_probability": 0.3,
                "add_yaml_objects_probability": 0.4,
                "add_parametric_objects_probability": 0.3,
                "position_noise": 0.15,
                "rotation_noise": 0.4,
                "scale_variation": 0.2,
                "mass_variation": 0.3,
                "color_variation": 0.25,
                "max_additional_objects": 3,
                "use_intelligent_placement": True,
            },
            seed=42,
        )

    else:
        raise ValueError(f"Unknown randomization mode: {args.randomization_mode}")


def get_robot_action(robot_name: str, step: int) -> dict:
    """Generate a simple robot action for demonstration."""
    if robot_name == "franka":
        # Simple sinusoidal motion for demonstration
        t = step * 0.1
        return {
            "dof_pos_target": {
                "panda_joint1": 0.3 * np.sin(t),
                "panda_joint2": -0.785398 + 0.2 * np.cos(t),
                "panda_joint3": 0.0,
                "panda_joint4": -2.356194 + 0.3 * np.sin(t * 0.5),
                "panda_joint5": 0.0,
                "panda_joint6": 1.570796,
                "panda_joint7": 0.785398 + 0.4 * np.cos(t * 0.3),
                "panda_finger_joint1": 0.04,
                "panda_finger_joint2": 0.04,
            }
        }
    else:
        return {}


def main():
    """Main function demonstrating flexible domain randomization."""

    args = tyro.cli(Args)

    log.info(f"Starting domain randomization demo with {args.sim}")
    log.info(f"Randomization mode: {args.randomization_mode}")
    log.info(f"Will generate {args.num_scenarios} scenarios")
    log.info(f"Baseline strategy: {'original' if args.use_original_baseline else 'adaptive'}")

    # Create base scenario and domain randomizer
    base_scenario = create_base_scenario(args)
    domain_randomizer = create_domain_randomizer(args)

    # Print randomization status
    status = domain_randomizer.get_status()
    log.info(f"Enabled components: {status['enabled_components']}")

    # Capture baseline from base scenario
    domain_randomizer.capture_baseline_from_scenario(base_scenario)

    # Create and launch environment once with base scenario
    log.info(f"Initializing {args.sim} environment...")
    env_class = get_sim_handler_class(SimType(args.sim))
    env = env_class(base_scenario)
    env.launch()
    log.info("Environment launched!")

    try:
        # Generate multiple randomized scenarios
        for scenario_idx in range(args.num_scenarios):
            log.info(f"\nScenario {scenario_idx + 1}/{args.num_scenarios}")

            baseline_type = "original" if args.use_original_baseline else "adaptive"
            log.info(f"Applying randomization (baseline: {baseline_type})...")

            # Apply domain randomization (this will automatically use dynamic scene updates)
            domain_randomizer.randomize_on_reset(env, use_original_baseline=args.use_original_baseline)

            # Environment is already launched with randomized objects, just reset robot state
            init_states = [
                {
                    "objects": {
                        obj.name: {
                            "pos": torch.tensor(obj.default_position),
                            "rot": torch.tensor(obj.default_orientation),
                        }
                        for obj in env.objects
                    },
                    "robots": {
                        args.robot: {
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
            ]

            # Apply states
            env.set_states(init_states)

            # Force camera updates for Isaac Sim
            for _ in range(3):
                if hasattr(env, "_update_camera_pose"):
                    env._update_camera_pose()
                env.simulate()

            # Create video recorder
            if args.save_videos:
                output_dir = f"get_started/output/domain_randomization_{args.randomization_mode}"
                video_path = f"{output_dir}/scenario_{scenario_idx + 1:02d}_{baseline_type}.mp4"
                obs_saver = ObsSaver(video_path=video_path)

            # Record continuous motion
            for frame in range(args.frames_per_scenario):
                # Generate robot actions
                robot_action = get_robot_action(args.robot, frame)
                actions = [{args.robot: robot_action}]

                # Step simulation
                env.set_dof_targets(actions)
                env.simulate()

                # Capture frame
                if args.save_videos:
                    obs = env.get_states()
                    obs_saver.add(obs)

            # Save video
            if args.save_videos:
                obs_saver.save()
                log.info(f"Saved scenario_{scenario_idx + 1:02d}_{baseline_type}.mp4")

            # Note: Baseline update is now handled automatically in randomize_on_reset()
            # No manual baseline update needed here for adaptive mode

            log.info(f"Scenario {scenario_idx + 1} completed")

        log.info("\nDomain randomization demo completed!")
        log.info("Statistics:")
        log.info(f"   - Total scenarios: {args.num_scenarios}")
        log.info(f"   - Frames per scenario: {args.frames_per_scenario}")
        log.info(f"   - Randomization mode: {args.randomization_mode}")
        log.info(f"   - Baseline strategy: {baseline_type}")
        log.info(f"   - Enabled components: {status['enabled_components']}")

        if args.save_videos:
            output_dir = f"get_started/output/domain_randomization_{args.randomization_mode}"
            log.info(f"   - Videos saved to: {output_dir}")
            log.info("\nEach video shows a different randomized environment!")
            log.info("Compare videos to see the diversity achieved through randomization!")

    finally:
        # Clean up - close environment only once at the end
        log.info("Cleaning up...")
        env.close()


if __name__ == "__main__":
    main()
