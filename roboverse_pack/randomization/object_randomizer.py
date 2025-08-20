"""Object randomization using YAML configuration files."""

from typing import Any, List, Optional, Tuple

import numpy as np
from loguru import logger as log

from .base import BaseRandomizer
from .config_loader import ConfigLoader


class ObjectRandomizer(BaseRandomizer):
    """Enhanced object randomizer with multiple strategies and parametric generation."""

    def __init__(
        self,
        config_dir: Optional[str] = None,
        split: str = "train",
        modify_existing_probability: float = 0.6,
        add_yaml_objects_probability: float = 0.3,
        add_parametric_objects_probability: float = 0.1,
        max_additional_objects: int = 2,
        position_noise: float = 0.1,
        rotation_noise: float = 0.2,
        scale_variation: float = 0.1,
        mass_variation: float = 0.2,
        color_variation: float = 0.15,
        use_intelligent_placement: bool = True,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)

        self.config_loader = ConfigLoader(config_dir)
        self.split = split

        # Normalize probabilities to sum to 1.0
        total = modify_existing_probability + add_yaml_objects_probability + add_parametric_objects_probability
        self.modify_existing_probability = modify_existing_probability / total
        self.add_yaml_probability = add_yaml_objects_probability / total
        self.add_parametric_probability = add_parametric_objects_probability / total

        self.max_additional_objects = max_additional_objects
        self.position_noise = position_noise
        self.rotation_noise = rotation_noise
        self.scale_variation = scale_variation
        self.mass_variation = mass_variation
        self.color_variation = color_variation
        self.use_intelligent_placement = use_intelligent_placement

        # Note: Unique IDs are now generated using timestamp + random

        log.info(
            f"ObjectRandomizer: modify={self.modify_existing_probability:.1%}, "
            f"yaml_add={self.add_yaml_probability:.1%}, parametric_add={self.add_parametric_probability:.1%}"
        )

    def randomize(self, scenario_cfg: Any, env_ids: Optional[List[int]] = None, **kwargs) -> None:
        if not self.enabled:
            return

        log.debug("Randomizing objects using enhanced strategies")

        # Choose strategy based on normalized probabilities
        rand_val = np.random.random()

        if rand_val < self.modify_existing_probability:
            # Strategy A: Modify existing objects only
            log.debug("Using Strategy A: Modify existing objects")
            self._modify_existing_objects(scenario_cfg)

        elif rand_val < self.modify_existing_probability + self.add_yaml_probability:
            # Strategy B: Add objects from YAML config
            log.debug("Using Strategy B: Add YAML objects")
            self._add_yaml_objects(scenario_cfg)

        else:
            # Strategy C: Add parametric objects
            log.debug("Using Strategy C: Add parametric objects")
            self._add_parametric_objects(scenario_cfg)

    def _modify_existing_objects(self, scenario_cfg: Any) -> None:
        """Modify positions and properties of existing objects."""
        for obj in scenario_cfg.objects:
            # Apply random modifications to all existing objects
            # (In adaptive mode, previously added objects become part of the baseline)
            self._randomize_object_properties(obj)

        log.debug("Modified existing objects")

    def _randomize_object_properties(self, obj: Any) -> None:
        """Randomize properties of a single object with enhanced attribute support."""
        # 1. Add noise to position
        current_pos = getattr(obj, "default_position", (0, 0, 0))
        noise = np.random.uniform(-self.position_noise, self.position_noise, 3).astype(np.float32)
        new_pos_array = np.array(current_pos, dtype=np.float32) + noise
        # Keep objects above ground
        new_pos_array[2] = max(new_pos_array[2], 0.01)
        new_pos = tuple(float(coord) for coord in new_pos_array)
        obj.default_position = new_pos

        # 2. Add noise to orientation (rotation around z-axis)
        angle_noise = float(np.random.uniform(-self.rotation_noise, self.rotation_noise))
        cos_half = float(np.cos(angle_noise / 2))
        sin_half = float(np.sin(angle_noise / 2))
        new_quat = (cos_half, 0.0, 0.0, sin_half)  # w, x, y, z
        obj.default_orientation = new_quat

        # 3. Randomize color if available
        if hasattr(obj, "color"):
            base_color = obj.color
            color_noise = np.random.uniform(-self.color_variation, self.color_variation, 3)
            new_color = [float(np.clip(c + noise, 0.0, 1.0)) for c, noise in zip(base_color, color_noise)]
            obj.color = new_color

        # 4. Randomize mass if available
        if hasattr(obj, "mass"):
            base_mass = obj.mass
            mass_multiplier = np.random.uniform(1 - self.mass_variation, 1 + self.mass_variation)
            obj.mass = float(base_mass * mass_multiplier)
            obj.mass = max(obj.mass, 0.01)  # Minimum mass

        # 5. Randomize size properties based on object type
        if hasattr(obj, "size"):  # PrimitiveCubeCfg
            base_size = obj.size
            scale_factors = np.random.uniform(1 - self.scale_variation, 1 + self.scale_variation, 3)
            new_size = [float(s * f) for s, f in zip(base_size, scale_factors)]
            # Ensure minimum size
            new_size = [max(s, 0.01) for s in new_size]
            obj.size = new_size

        elif hasattr(obj, "radius"):  # PrimitiveSphereCfg or PrimitiveCylinderCfg
            base_radius = obj.radius
            radius_multiplier = np.random.uniform(1 - self.scale_variation, 1 + self.scale_variation)
            obj.radius = float(base_radius * radius_multiplier)
            obj.radius = max(obj.radius, 0.005)  # Minimum radius

            # For cylinders, also randomize height
            if hasattr(obj, "height"):  # PrimitiveCylinderCfg
                base_height = obj.height
                height_multiplier = np.random.uniform(1 - self.scale_variation, 1 + self.scale_variation)
                obj.height = float(base_height * height_multiplier)
                obj.height = max(obj.height, 0.01)  # Minimum height

        # 6. Randomize scale for file-based objects
        elif hasattr(obj, "scale"):
            if isinstance(obj.scale, (list, tuple)):
                base_scale = obj.scale
                scale_factors = np.random.uniform(1 - self.scale_variation, 1 + self.scale_variation, 3)
                new_scale = tuple(float(s * f) for s, f in zip(base_scale, scale_factors))
            else:
                base_scale = obj.scale
                scale_factor = np.random.uniform(1 - self.scale_variation, 1 + self.scale_variation)
                new_scale = float(base_scale * scale_factor)
            obj.scale = new_scale

    def _add_yaml_objects(self, scenario_cfg: Any) -> None:
        """Add random additional objects from YAML configuration."""
        objects = getattr(scenario_cfg, "objects", [])
        num_to_add = np.random.randint(1, self.max_additional_objects + 1)

        # Get spawn area from config
        spawn_area = self.config_loader.get_spawn_area()
        if spawn_area:
            spawn_bounds = (spawn_area["x_range"], spawn_area["y_range"], spawn_area["z_range"])
        else:
            spawn_bounds = ((0.2, 0.8), (-0.4, 0.4), (0.5, 0.8))  # fallback

        added_count = 0
        for i in range(num_to_add):
            # Get random object from YAML config
            new_obj = self.config_loader.get_random_object(self.split)

            if new_obj is None:
                continue

            # Generate unique ID for naming (simple timestamp + random)
            import random
            import time

            object_id = int(time.time() * 1000) % 100000 + random.randint(1, 999)

            new_obj.name = f"added_obj_{object_id}"

            # Set random position
            pos = (
                float(np.random.uniform(*spawn_bounds[0])),
                float(np.random.uniform(*spawn_bounds[1])),
                float(np.random.uniform(*spawn_bounds[2])),
            )
            new_obj.default_position = pos
            new_obj.default_orientation = self._random_quaternion()

            # Check for collisions with existing objects
            if self._check_object_collision(pos, objects):
                log.debug(f"Skipping object {new_obj.name} due to collision")
                continue

            objects.append(new_obj)
            added_count += 1
            log.debug(f"Added new object {new_obj.name} at position {pos}")

        log.debug(f"Added {added_count} new objects from YAML config")

    def _check_object_collision(self, pos: Tuple[float, float, float], existing_objects: List[Any]) -> bool:
        """Check if a position collides with existing objects."""
        min_distance = 0.08  # 8cm minimum distance

        for obj in existing_objects:
            obj_pos = getattr(obj, "default_position", (0, 0, 0))
            distance = np.linalg.norm(np.array(pos[:2]) - np.array(obj_pos[:2]))
            if distance < min_distance:
                return True

        return False

    def _random_quaternion(self) -> Tuple[float, float, float, float]:
        """Generate a random quaternion for rotation."""
        angle = float(np.random.uniform(0, 2 * np.pi))
        cos_half = float(np.cos(angle / 2))
        sin_half = float(np.sin(angle / 2))
        return (cos_half, 0.0, 0.0, sin_half)  # w, x, y, z

    def _add_parametric_objects(self, scenario_cfg: Any) -> None:
        """Add parametrically generated objects using randomization_ranges."""
        objects = getattr(scenario_cfg, "objects", [])
        num_to_add = np.random.randint(1, self.max_additional_objects + 1)

        # Try to get randomization ranges from config
        ranges = self.config_loader.objects_config.get("randomization_settings", {})
        if not ranges:
            log.warning("No randomization_settings found in objects.yml, falling back to YAML objects")
            self._add_yaml_objects(scenario_cfg)
            return

        # Get spawn area (prefer extended area for parametric objects)
        spawn_area = self.config_loader.get_spawn_area("extended")
        if not spawn_area:
            spawn_area = self.config_loader.get_spawn_area()  # fallback

        if spawn_area:
            spawn_bounds = (spawn_area["x_range"], spawn_area["y_range"], spawn_area["z_range"])
        else:
            spawn_bounds = ((0.1, 0.9), (-0.5, 0.5), (0.5, 0.8))  # fallback

        added_count = 0
        max_attempts = ranges.get("max_placement_attempts", 20)

        for i in range(num_to_add):
            # Generate parametric object
            new_obj = self._generate_parametric_object(ranges)

            if new_obj is None:
                continue

            # Generate unique ID for naming (simple timestamp + random)
            import random
            import time

            object_id = int(time.time() * 1000) % 100000 + random.randint(1, 999)

            new_obj.name = f"param_obj_{object_id}"

            # Try to find a valid position
            placed = False
            for attempt in range(max_attempts):
                pos = (
                    float(np.random.uniform(*spawn_bounds[0])),
                    float(np.random.uniform(*spawn_bounds[1])),
                    float(np.random.uniform(*spawn_bounds[2])),
                )

                # Intelligent collision checking
                if self.use_intelligent_placement:
                    collision_radius = self._estimate_object_radius(new_obj)
                    min_distance = ranges.get("min_distance_between_objects", 0.08)
                    if not self._check_intelligent_collision(pos, objects, collision_radius, min_distance):
                        new_obj.default_position = pos
                        new_obj.default_orientation = self._random_quaternion()
                        placed = True
                        break
                else:
                    # Simple collision checking
                    if not self._check_object_collision(pos, objects):
                        new_obj.default_position = pos
                        new_obj.default_orientation = self._random_quaternion()
                        placed = True
                        break

            if placed:
                objects.append(new_obj)
                added_count += 1
                log.debug(f"Added parametric object {new_obj.name} at {new_obj.default_position}")
            else:
                log.debug(f"Failed to place parametric object after {max_attempts} attempts")

        log.debug(f"Added {added_count} parametric objects")

    def _generate_parametric_object(self, ranges: dict) -> Any:
        """Generate an object using parametric ranges."""
        # Choose object type randomly (include file-based objects if available)
        object_types = ["cube", "sphere", "cylinder"]

        # Add file-based types if asset paths are provided in ranges
        if ranges.get("rigid_asset_paths"):
            object_types.append("rigid")
        if ranges.get("articulation_asset_paths"):
            object_types.append("articulation")

        obj_type = np.random.choice(object_types)

        try:
            if obj_type == "cube":
                return self._generate_parametric_cube(ranges)
            elif obj_type == "sphere":
                return self._generate_parametric_sphere(ranges)
            elif obj_type == "cylinder":
                return self._generate_parametric_cylinder(ranges)
            elif obj_type == "rigid":
                return self._generate_parametric_rigid(ranges)
            elif obj_type == "articulation":
                return self._generate_parametric_articulation(ranges)
        except Exception as e:
            log.warning(f"Failed to generate parametric {obj_type}: {e}")
            return None

        return None

    def _generate_parametric_cube(self, ranges: dict) -> Any:
        """Generate a parametric cube object."""
        from metasim.constants import PhysicStateType
        from metasim.scenario.objects import PrimitiveCubeCfg

        # Size ranges
        size_base = 0.08  # Base size 8cm
        size_variation = ranges.get("scale_variation", 0.3)
        size_range = (size_base * (1 - size_variation), size_base * (1 + size_variation))

        # Generate random size (can be rectangular)
        size = [
            float(np.random.uniform(*size_range)),
            float(np.random.uniform(*size_range)),
            float(np.random.uniform(*size_range)),
        ]

        # Mass range based on volume
        volume = size[0] * size[1] * size[2]
        density_range = (200, 800)  # kg/m³
        mass_range = (volume * density_range[0], volume * density_range[1])
        mass = float(np.random.uniform(*mass_range))

        # Random color
        color = self._generate_random_color()

        return PrimitiveCubeCfg(
            name="param_cube",  # Will be overridden
            size=size,
            mass=mass,
            color=color,
            physics=PhysicStateType.RIGIDBODY,
            collision_enabled=True,
            enabled_gravity=True,
            fix_base_link=False,
        )

    def _generate_parametric_sphere(self, ranges: dict) -> Any:
        """Generate a parametric sphere object."""
        from metasim.constants import PhysicStateType
        from metasim.scenario.objects import PrimitiveSphereCfg

        # Radius range
        radius_base = 0.04  # Base radius 4cm
        radius_variation = ranges.get("scale_variation", 0.3)
        radius_range = (radius_base * (1 - radius_variation), radius_base * (1 + radius_variation))
        radius = float(np.random.uniform(*radius_range))

        # Mass range based on volume
        volume = (4 / 3) * np.pi * radius**3
        density_range = (200, 800)  # kg/m³
        mass_range = (volume * density_range[0], volume * density_range[1])
        mass = float(np.random.uniform(*mass_range))

        # Random color
        color = self._generate_random_color()

        return PrimitiveSphereCfg(
            name="param_sphere",  # Will be overridden
            radius=radius,
            mass=mass,
            color=color,
            physics=PhysicStateType.RIGIDBODY,
            collision_enabled=True,
            enabled_gravity=True,
            fix_base_link=False,
        )

    def _generate_parametric_cylinder(self, ranges: dict) -> Any:
        """Generate a parametric cylinder object."""
        from metasim.constants import PhysicStateType
        from metasim.scenario.objects import PrimitiveCylinderCfg

        # Radius and height ranges
        radius_base = 0.03  # Base radius 3cm
        height_base = 0.08  # Base height 8cm
        scale_variation = ranges.get("scale_variation", 0.3)

        radius_range = (radius_base * (1 - scale_variation), radius_base * (1 + scale_variation))
        height_range = (height_base * (1 - scale_variation), height_base * (1 + scale_variation))

        radius = float(np.random.uniform(*radius_range))
        height = float(np.random.uniform(*height_range))

        # Mass range based on volume
        volume = np.pi * radius**2 * height
        density_range = (200, 800)  # kg/m³
        mass_range = (volume * density_range[0], volume * density_range[1])
        mass = float(np.random.uniform(*mass_range))

        # Random color
        color = self._generate_random_color()

        return PrimitiveCylinderCfg(
            name="param_cylinder",  # Will be overridden
            radius=radius,
            height=height,
            mass=mass,
            color=color,
            physics=PhysicStateType.RIGIDBODY,
            collision_enabled=True,
            enabled_gravity=True,
            fix_base_link=False,
        )

    def _generate_random_color(self) -> list:
        """Generate a random RGB color."""
        # Color palette ranges
        color_palettes = [
            # Warm colors
            [(0.8, 1.0), (0.2, 0.6), (0.0, 0.3)],  # Red-orange range
            [(0.6, 1.0), (0.6, 1.0), (0.0, 0.4)],  # Yellow range
            # Cool colors
            [(0.0, 0.4), (0.4, 0.8), (0.7, 1.0)],  # Blue range
            [(0.0, 0.5), (0.6, 1.0), (0.3, 0.7)],  # Green range
            # Neutral colors
            [(0.4, 0.8), (0.4, 0.8), (0.4, 0.8)],  # Gray range
            [(0.5, 0.9), (0.3, 0.7), (0.6, 1.0)],  # Purple range
        ]

        # Choose random palette
        palette = np.random.choice(len(color_palettes))
        color_ranges = color_palettes[palette]

        color = [
            float(np.random.uniform(*color_ranges[0])),
            float(np.random.uniform(*color_ranges[1])),
            float(np.random.uniform(*color_ranges[2])),
        ]

        return color

    def _estimate_object_radius(self, obj: Any) -> float:
        """Estimate the effective radius of an object for collision checking."""
        if hasattr(obj, "radius"):
            return obj.radius
        elif hasattr(obj, "size"):
            # For cubes, use half the maximum dimension
            return max(obj.size) / 2
        else:
            # Default fallback
            return 0.05

    def _check_intelligent_collision(
        self, pos: Tuple[float, float, float], existing_objects: List[Any], object_radius: float, min_distance: float
    ) -> bool:
        """Enhanced collision checking considering object sizes."""
        for obj in existing_objects:
            obj_pos = getattr(obj, "default_position", (0, 0, 0))
            obj_radius = self._estimate_object_radius(obj)

            # Calculate 3D distance
            distance = np.linalg.norm(np.array(pos) - np.array(obj_pos))

            # Required distance is sum of radii plus minimum separation
            required_distance = object_radius + obj_radius + min_distance

            if distance < required_distance:
                return True  # Collision detected

        return False  # No collision

    def _generate_parametric_rigid(self, ranges: dict) -> Any:
        """Generate a parametric rigid object from asset paths."""
        from metasim.constants import PhysicStateType
        from metasim.scenario.objects import RigidObjCfg

        # Get available asset paths
        asset_paths = ranges.get("rigid_asset_paths", [])
        if not asset_paths:
            return None

        # Choose random asset
        asset_config = np.random.choice(asset_paths)

        # Random scale
        scale_variation = ranges.get("scale_variation", 0.2)
        scale_base = 1.0
        scale = float(np.random.uniform(scale_base * (1 - scale_variation), scale_base * (1 + scale_variation)))

        return RigidObjCfg(
            name="param_rigid",  # Will be overridden
            usd_path=asset_config.get("usd_path"),
            mesh_path=asset_config.get("mesh_path"),
            urdf_path=asset_config.get("urdf_path"),
            mjcf_path=asset_config.get("mjcf_path"),
            scale=scale,
            physics=PhysicStateType.RIGIDBODY,
            collision_enabled=True,
            enabled_gravity=True,
            fix_base_link=False,
        )

    def _generate_parametric_articulation(self, ranges: dict) -> Any:
        """Generate a parametric articulation object from asset paths."""
        from metasim.scenario.objects import ArticulationObjCfg

        # Get available asset paths
        asset_paths = ranges.get("articulation_asset_paths", [])
        if not asset_paths:
            return None

        # Choose random asset
        asset_config = np.random.choice(asset_paths)

        # Random scale
        scale_variation = ranges.get("scale_variation", 0.1)  # Smaller variation for articulations
        scale_base = 1.0
        scale = float(np.random.uniform(scale_base * (1 - scale_variation), scale_base * (1 + scale_variation)))

        return ArticulationObjCfg(
            name="param_articulation",  # Will be overridden
            usd_path=asset_config.get("usd_path"),
            mesh_path=asset_config.get("mesh_path"),
            urdf_path=asset_config.get("urdf_path"),
            mjcf_path=asset_config.get("mjcf_path"),
            mjx_mjcf_path=asset_config.get("mjx_mjcf_path"),
            scale=scale,
            enabled_gravity=asset_config.get("enabled_gravity", True),
            fix_base_link=asset_config.get("fix_base_link", False),
        )
