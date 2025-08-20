"""Material randomization using YAML configuration files.

Follows the same simple pattern as other randomizers.
"""

from __future__ import annotations

import random
from typing import Any

from loguru import logger as log

from .base import BaseRandomizer
from .config_loader import ConfigLoader


class MaterialRandomizer(BaseRandomizer):
    """Material randomizer that assigns random materials to objects across different simulators."""

    def __init__(
        self,
        config_dir: str | None = None,
        split: str = "train",
        # Object material settings
        randomize_objects: bool = True,
        object_change_probability: float = 0.8,
        # Environment material settings
        randomize_environment: bool = True,
        environment_change_probability: float = 0.6,
        # Robot material settings
        randomize_robots: bool = False,
        robot_change_probability: float = 0.5,
        # Physics material settings
        apply_physics_materials: bool = False,
        physics_change_probability: float = 0.3,
        # General settings
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)

        self.config_loader = ConfigLoader(config_dir)
        self.split = split

        # Material randomization settings
        self.randomize_objects = randomize_objects
        self.object_change_probability = object_change_probability
        self.randomize_environment = randomize_environment
        self.environment_change_probability = environment_change_probability
        self.randomize_robots = randomize_robots
        self.robot_change_probability = robot_change_probability
        self.apply_physics_materials = apply_physics_materials
        self.physics_change_probability = physics_change_probability

        log.info(
            f"MaterialRandomizer: objects={randomize_objects}, environment={randomize_environment}, "
            f"robots={randomize_robots}, physics={apply_physics_materials}"
        )

    def randomize(self, scenario_cfg: Any, env_ids: list[int] | None = None, **kwargs) -> None:
        """Apply material randomization to the scenario configuration."""
        if not self.enabled:
            return

        log.debug("Randomizing materials from YAML configuration")

        # Prepare material assignments for different object types
        material_assignments = {
            "object_materials": [],
            "environment_materials": [],
            "robot_materials": [],
            "physics_materials": [],
        }

        # Generate material assignments based on randomization settings
        if self.randomize_objects:
            material_assignments["object_materials"] = self._generate_object_materials(scenario_cfg)

        if self.randomize_environment:
            material_assignments["environment_materials"] = self._generate_environment_materials(scenario_cfg)

        if self.randomize_robots:
            material_assignments["robot_materials"] = self._generate_robot_materials(scenario_cfg)

        if self.apply_physics_materials:
            material_assignments["physics_materials"] = self._generate_physics_materials(scenario_cfg)

        # Store material assignments in scenario config for later application
        scenario_cfg.material_assignments = material_assignments

        log.debug(
            f"Material randomization complete: {len(material_assignments['object_materials'])} object materials, "
            f"{len(material_assignments['environment_materials'])} environment materials, "
            f"{len(material_assignments['robot_materials'])} robot materials, "
            f"{len(material_assignments['physics_materials'])} physics materials"
        )

    def _generate_object_materials(self, scenario_cfg: Any) -> list[dict]:
        """Generate material assignments for objects."""
        materials = []

        # Get all objects from scenario_cfg
        objects = getattr(scenario_cfg, "objects", [])

        for obj in objects:
            # Skip if this is an environment object (has environment-related name)
            if self._is_environment_object(obj):
                continue

            # Skip if this is a robot object
            if self._is_robot_object(obj):
                continue

            # Decide whether to change this object's material
            if random.random() > self.object_change_probability:
                continue

            # Determine object type for material selection
            obj_type = self._get_object_type(obj)

            # Get material from YAML config
            material_config = self.config_loader.get_random_object_material(for_object_type=obj_type)

            if material_config:
                materials.append({"object": obj, "material_config": material_config, "category": "object"})

        return materials

    def _generate_environment_materials(self, scenario_cfg: Any) -> list[dict]:
        """Generate material assignments for environment objects."""
        materials = []

        # Get all objects and filter for environment objects
        objects = getattr(scenario_cfg, "objects", [])

        for obj in objects:
            env_type = self._classify_environment_object(obj)
            log.debug(f"Object '{getattr(obj, 'name', 'unnamed')}' classified as environment type: {env_type}")

            if env_type:
                prob_roll = random.random()
                log.debug(
                    f"Environment object '{getattr(obj, 'name', 'unnamed')}' ({env_type}): probability roll {prob_roll:.3f} vs threshold {self.environment_change_probability}"
                )

                if prob_roll < self.environment_change_probability:
                    material_config = self.config_loader.get_random_environment_material(env_type, self.split)
                    log.debug(f"Got environment material config for {env_type}: {material_config is not None}")

                    if material_config:
                        materials.append({
                            "object": obj,
                            "material_config": material_config,
                            "category": "environment",
                            "env_type": env_type,
                        })
                        log.debug(f"Added environment material for '{getattr(obj, 'name', 'unnamed')}'")
                    else:
                        log.warning(f"No environment material found for type '{env_type}' in split '{self.split}'")

        return materials

    def _generate_robot_materials(self, scenario_cfg: Any) -> list[dict]:
        """Generate material assignments for robots."""
        materials = []

        # Get robots from scenario_cfg
        robots = getattr(scenario_cfg, "robots", [])

        # Also check objects for robot-like objects
        objects = getattr(scenario_cfg, "objects", [])
        robot_objects = [obj for obj in objects if self._is_robot_object(obj)]

        all_robots = robots + robot_objects

        for robot in all_robots:
            if random.random() < self.robot_change_probability:
                # Robots should use appropriate materials (metals, plastics)
                material_config = self.config_loader.get_random_object_material(for_object_type="robots")

                if material_config:
                    materials.append({"object": robot, "material_config": material_config, "category": "robot"})

        return materials

    def _generate_physics_materials(self, scenario_cfg: Any) -> list[dict]:
        """Generate physics material assignments for objects."""
        materials = []

        # Get all objects that can have physics materials
        objects = getattr(scenario_cfg, "objects", [])
        robots = getattr(scenario_cfg, "robots", [])
        all_objects = objects + robots

        for obj in all_objects:
            if random.random() < self.physics_change_probability:
                # Choose physics material based on object type
                obj_type = self._get_object_type(obj)
                friction_type = self._get_friction_type_for_object(obj_type)

                physics_config = self.config_loader.get_random_physics_material(friction_type)

                if physics_config:
                    materials.append({"object": obj, "material_config": physics_config, "category": "physics"})

        return materials

    def _classify_environment_object(self, obj: Any) -> str | None:
        """Classify an object as environment type based on name or properties."""
        if not hasattr(obj, "name") or not obj.name:
            return None

        obj_name = str(obj.name).lower()

        # Table objects (check first, before ground, since table_surface contains both "table" and "surface")
        if any(keyword in obj_name for keyword in ["table", "desk", "counter", "workbench"]):
            return "tables"

        # Ground/floor objects
        if any(keyword in obj_name for keyword in ["ground", "floor", "plane", "terrain", "surface"]):
            return "ground"

        # Wall objects
        if any(keyword in obj_name for keyword in ["wall", "barrier", "partition", "panel"]):
            return "walls"

        return None

    def _is_environment_object(self, obj: Any) -> bool:
        """Check if an object is an environment object."""
        return self._classify_environment_object(obj) is not None

    def _is_robot_object(self, obj: Any) -> bool:
        """Check if an object is a robot."""
        if hasattr(obj, "name") and obj.name:
            obj_name = str(obj.name).lower()
            if any(
                keyword in obj_name for keyword in ["robot", "arm", "manipulator", "agent", "franka", "ur5", "panda"]
            ):
                return True

        # Check class name
        class_name = obj.__class__.__name__.lower()
        if any(keyword in class_name for keyword in ["robot", "articulation", "manipulator", "agent"]):
            return True

        return False

    def _get_object_type(self, obj: Any) -> str:
        """Determine the type of object for material selection."""
        class_name = obj.__class__.__name__.lower()

        if "cube" in class_name:
            return "cubes"
        elif "sphere" in class_name:
            return "spheres"
        elif "cylinder" in class_name:
            return "cylinders"
        elif any(keyword in class_name for keyword in ["robot", "articulation", "manipulator"]):
            return "robots"
        else:
            return "objects"  # Generic category

    def _get_friction_type_for_object(self, obj_type: str) -> str:
        """Get appropriate friction type for object."""
        friction_mapping = {
            "cubes": "medium_friction",
            "spheres": "low_friction",
            "cylinders": "medium_friction",
            "robots": "medium_friction",
            "objects": "medium_friction",
        }
        return friction_mapping.get(obj_type, "medium_friction")
