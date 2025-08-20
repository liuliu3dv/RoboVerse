"""Lighting randomization using YAML configuration files."""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger as log

from .base import BaseRandomizer
from .config_loader import ConfigLoader


class LightingRandomizer(BaseRandomizer):
    """Lighting randomizer with three strategies: preset replacement, modification, or addition."""

    def __init__(
        self,
        config_dir: str | None = None,
        split: str = "train",
        preset_probability: float = 0.3,
        modify_probability: float = 0.4,
        add_only_probability: float = 0.3,
        max_additional_lights: int = 2,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)

        self.config_loader = ConfigLoader(config_dir)
        self.split = split

        # Normalize probabilities to sum to 1.0
        total = preset_probability + modify_probability + add_only_probability
        self.preset_probability = preset_probability / total
        self.modify_probability = modify_probability / total
        self.add_only_probability = add_only_probability / total

        self.max_additional_lights = max_additional_lights

        log.info(
            f"LightingRandomizer strategies: preset={self.preset_probability:.1%}, modify={self.modify_probability:.1%}, add_only={self.add_only_probability:.1%}"
        )

    def randomize(self, scenario_cfg: Any, env_ids: list[int] | None = None, **kwargs) -> None:
        """Apply lighting randomization to the scenario."""
        if not self.enabled:
            return

        log.debug("Randomizing lighting from YAML configuration")

        # Choose strategy based on normalized probabilities
        rand_val = np.random.random()

        if rand_val < self.preset_probability:
            # Strategy A: Complete preset replacement
            log.debug("Using Strategy A: Complete preset replacement")
            self._apply_lighting_preset(scenario_cfg)

        elif rand_val < self.preset_probability + self.modify_probability:
            # Strategy B: Modify existing lights only
            log.debug("Using Strategy B: Modify existing lights")
            self._modify_existing_lights(scenario_cfg)

        else:
            # Strategy C: Add new lights only (keep existing unchanged)
            log.debug("Using Strategy C: Add new lights only")
            self._add_random_lights(scenario_cfg)

    def _modify_existing_lights(self, scenario_cfg: Any) -> None:
        """Modify properties of existing lights."""
        lights = getattr(scenario_cfg, "lights", [])

        for light_cfg in lights:
            # Apply random modifications to all existing lights
            # (In adaptive mode, previously added lights become part of the baseline)
            self._randomize_light_properties(light_cfg)

        log.debug(f"Modified {len(lights)} existing lights")

    def _randomize_light_properties(self, light_cfg: Any) -> None:
        """Randomize properties of a single light."""
        # Randomize intensity (±50% variation)
        if hasattr(light_cfg, "intensity"):
            base_intensity = light_cfg.intensity
            variation = np.random.uniform(0.5, 1.5)
            light_cfg.intensity = float(base_intensity * variation)

        # Randomize color slightly
        if hasattr(light_cfg, "color"):
            base_color = light_cfg.color
            new_color = []
            for channel in base_color:
                noise = np.random.uniform(-0.1, 0.1)
                new_channel = np.clip(channel + noise, 0.0, 1.0)
                new_color.append(float(new_channel))
            light_cfg.color = tuple(new_color)

        # Randomize exposure (±2 stops)
        if hasattr(light_cfg, "exposure"):
            if not hasattr(light_cfg, "exposure") or light_cfg.exposure is None:
                light_cfg.exposure = 0.0
            light_cfg.exposure += float(np.random.uniform(-2.0, 2.0))
            light_cfg.exposure = np.clip(light_cfg.exposure, -10.0, 10.0)

        # Randomize color temperature if enabled
        if hasattr(light_cfg, "enable_color_temperature") and hasattr(light_cfg, "color_temperature"):
            if np.random.random() < 0.3:  # 30% chance to enable color temperature
                light_cfg.enable_color_temperature = True
                # Randomize temperature in realistic range
                light_cfg.color_temperature = float(np.random.uniform(2700, 6500))

        # For directional lights, slightly randomize direction
        if hasattr(light_cfg, "polar") and hasattr(light_cfg, "azimuth"):
            light_cfg.polar += float(np.random.uniform(-15, 15))
            light_cfg.azimuth += float(np.random.uniform(-30, 30))
            light_cfg.polar = np.clip(light_cfg.polar, 0, 90)
            light_cfg.azimuth = light_cfg.azimuth % 360

        # For distant lights, randomize angular size
        if hasattr(light_cfg, "angle"):
            if not hasattr(light_cfg, "angle") or light_cfg.angle is None:
                light_cfg.angle = 0.53  # Default sun angle
            light_cfg.angle *= float(np.random.uniform(0.5, 3.0))
            light_cfg.angle = np.clip(light_cfg.angle, 0.1, 10.0)

        # Randomize position for positional lights
        if hasattr(light_cfg, "pos"):
            base_pos = np.array(light_cfg.pos)
            # Add noise to position (±0.5m in each axis)
            noise = np.random.uniform(-0.5, 0.5, size=3)
            new_pos = base_pos + noise
            # Keep lights above ground
            new_pos[2] = max(new_pos[2], 0.5)
            light_cfg.pos = tuple(new_pos.astype(float))

        # Randomize size properties
        if hasattr(light_cfg, "radius"):
            base_radius = light_cfg.radius
            variation = np.random.uniform(0.7, 1.3)
            light_cfg.radius = float(base_radius * variation)
            light_cfg.radius = max(light_cfg.radius, 0.05)  # Minimum radius

        if hasattr(light_cfg, "length"):  # For cylinder lights
            base_length = light_cfg.length
            variation = np.random.uniform(0.7, 1.3)
            light_cfg.length = float(base_length * variation)
            light_cfg.length = max(light_cfg.length, 0.1)  # Minimum length

        # Randomize special properties
        if hasattr(light_cfg, "normalize"):
            # Randomly enable/disable normalization
            light_cfg.normalize = np.random.choice([True, False])

        if hasattr(light_cfg, "treat_as_point"):
            # 20% chance to treat sphere as point light
            light_cfg.treat_as_point = np.random.random() < 0.2

        if hasattr(light_cfg, "treat_as_line"):
            # 20% chance to treat cylinder as line light
            light_cfg.treat_as_line = np.random.random() < 0.2

    def _add_random_lights(self, scenario_cfg: Any) -> None:
        """Add random additional lights using both YAML config and parametric generation."""
        lights = getattr(scenario_cfg, "lights", [])
        num_to_add = np.random.randint(1, self.max_additional_lights + 1)

        for i in range(num_to_add):
            # Choose between YAML config (70%) and parametric generation (30%)
            if np.random.random() < 0.7:
                # Method A: Get additional light from YAML config
                new_light = self.config_loader.get_random_additional_light()
            else:
                # Method B: Generate parametric light using randomization_ranges
                new_light = self._generate_parametric_light()

            if new_light:
                # Generate unique ID for naming (simple timestamp + random)
                import random
                import time

                light_id = int(time.time() * 1000) % 100000 + random.randint(1, 999) + i

                # Always assign a unique name
                if not hasattr(new_light, "name") or not new_light.name:
                    new_light.name = f"added_light_{light_id}"
                else:
                    # Add unique ID to existing name to avoid conflicts
                    original_name = new_light.name
                    new_light.name = f"{original_name}_{light_id}"

                lights.append(new_light)
                log.debug(f"Added new light: {new_light.__class__.__name__} with intensity {new_light.intensity}")

        log.debug(f"Added {num_to_add} new lights to scene")

    def _generate_parametric_light(self) -> Any | None:
        """Generate a light using parametric ranges from lights.yml."""
        ranges = self.config_loader.lights_config.get("randomization_ranges", {})
        if not ranges:
            log.warning("No randomization_ranges found in lights.yml, falling back to YAML config")
            return self.config_loader.get_random_additional_light()

        # Choose random light type
        light_types = ["distant", "sphere", "cylinder", "disk", "dome"]
        light_type = np.random.choice(light_types)

        # Generate a unique ID for this light
        # Note: We can't access sim_handler here, so use a timestamp-based fallback
        import time

        light_id = int(time.time() * 1000) % 100000

        try:
            if light_type == "distant":
                return self._generate_parametric_distant_light(ranges, light_id)
            elif light_type == "sphere":
                return self._generate_parametric_sphere_light(ranges, light_id)
            elif light_type == "cylinder":
                return self._generate_parametric_cylinder_light(ranges, light_id)
            elif light_type == "disk":
                return self._generate_parametric_disk_light(ranges, light_id)
            elif light_type == "dome":
                return self._generate_parametric_dome_light(ranges, light_id)
        except Exception as e:
            log.warning(f"Failed to generate parametric {light_type} light: {e}, falling back to YAML")
            return self.config_loader.get_random_additional_light()

        return None

    def _apply_lighting_preset(self, scenario_cfg: Any) -> None:
        """Apply a complete lighting preset, replacing all existing lights."""
        # Get a random lighting preset
        preset = self.config_loader.get_random_lighting_preset(self.split)

        if not preset:
            log.warning(f"No lighting presets available for split '{self.split}', falling back to modification")
            self._modify_existing_lights(scenario_cfg)
            return

        log.debug(f"Applying lighting preset: {preset.get('name', 'unnamed')}")

        # Create new lights from preset
        new_lights = self.config_loader.create_lights_from_preset(preset)

        if not new_lights:
            log.warning("Failed to create lights from preset, falling back to modification")
            self._modify_existing_lights(scenario_cfg)
            return

        # Assign unique names to avoid conflicts
        for i, light in enumerate(new_lights):
            # Generate unique ID for naming (simple timestamp + random)
            import random
            import time

            light_id = int(time.time() * 1000) % 100000 + random.randint(1, 999) + i

            # Only assign name if not already named or name is empty
            if not hasattr(light, "name") or not light.name:
                light.name = f"preset_light_{light_id}"

        # Replace all lights with preset lights
        scenario_cfg.lights = new_lights

        log.info(f"Applied lighting preset '{preset.get('name', 'unnamed')}' with {len(new_lights)} lights")

        # Optionally add some random variation to preset lights
        if np.random.random() < 0.5:  # 50% chance to add small variations
            for light in new_lights:
                # Apply subtle randomization to preset lights
                self._apply_subtle_randomization(light)

    def _apply_subtle_randomization(self, light_cfg: Any) -> None:
        """Apply subtle randomization to preset lights to add variety."""
        # Smaller intensity variation for presets (±20% instead of ±50%)
        if hasattr(light_cfg, "intensity"):
            base_intensity = light_cfg.intensity
            variation = np.random.uniform(0.8, 1.2)
            light_cfg.intensity = float(base_intensity * variation)

        # Smaller color variation for presets
        if hasattr(light_cfg, "color"):
            base_color = light_cfg.color
            new_color = []
            for channel in base_color:
                noise = np.random.uniform(-0.05, 0.05)  # Smaller variation
                new_channel = np.clip(channel + noise, 0.0, 1.0)
                new_color.append(float(new_channel))
            light_cfg.color = tuple(new_color)

        # Small exposure variation
        if hasattr(light_cfg, "exposure"):
            light_cfg.exposure += float(np.random.uniform(-0.5, 0.5))  # Smaller range
            light_cfg.exposure = np.clip(light_cfg.exposure, -5.0, 5.0)

        # Small position variation for positional lights
        if hasattr(light_cfg, "pos"):
            base_pos = np.array(light_cfg.pos)
            noise = np.random.uniform(-0.2, 0.2, size=3)  # Smaller variation
            new_pos = base_pos + noise
            new_pos[2] = max(new_pos[2], 0.5)  # Keep above ground
            light_cfg.pos = tuple(new_pos.astype(float))

    def _generate_parametric_distant_light(self, ranges: dict, light_id: int) -> Any:
        """Generate a distant light using parametric ranges."""
        from metasim.scenario.lights import DistantLightCfg

        intensity_range = ranges.get("intensity_ranges", {}).get("distant", [1000.0, 3000.0])
        intensity = float(np.random.uniform(*intensity_range))

        color = self._generate_random_color(ranges)

        polar_range = ranges.get("polar_range", [10.0, 80.0])
        azimuth_range = ranges.get("azimuth_range", [-180.0, 180.0])
        polar = float(np.random.uniform(*polar_range))
        azimuth = float(np.random.uniform(*azimuth_range))

        angle_range = ranges.get("angle_range", [0.1, 5.0])
        angle = float(np.random.uniform(*angle_range))

        light_cfg = DistantLightCfg(
            name=f"parametric_distant_light_{light_id}",
            intensity=intensity,
            color=color,
            polar=polar,
            azimuth=azimuth,
            angle=angle,
            exposure=self._generate_random_exposure(ranges),
            normalize=self._get_random_bool_property(ranges, "normalize_probability"),
            enable_color_temperature=self._get_random_bool_property(ranges, "color_temperature_probability"),
            color_temperature=self._generate_random_color_temperature(ranges),
        )

        return light_cfg

    def _generate_parametric_sphere_light(self, ranges: dict, light_id: int) -> Any:
        """Generate a sphere light using parametric ranges."""
        from metasim.scenario.lights import SphereLightCfg

        intensity_range = ranges.get("intensity_ranges", {}).get("sphere", [200.0, 1000.0])
        intensity = float(np.random.uniform(*intensity_range))

        color = self._generate_random_color(ranges)
        pos = self._generate_random_position(ranges)

        radius_range = ranges.get("radius_range", [0.1, 1.0])
        radius = float(np.random.uniform(*radius_range))

        light_cfg = SphereLightCfg(
            name=f"parametric_sphere_light_{light_id}",
            intensity=intensity,
            color=color,
            pos=pos,
            radius=radius,
            exposure=self._generate_random_exposure(ranges),
            normalize=self._get_random_bool_property(ranges, "normalize_probability"),
            treat_as_point=self._get_random_bool_property(ranges, "treat_as_point_probability"),
            enable_color_temperature=self._get_random_bool_property(ranges, "color_temperature_probability"),
            color_temperature=self._generate_random_color_temperature(ranges),
        )

        return light_cfg

    def _generate_parametric_cylinder_light(self, ranges: dict, light_id: int) -> Any:
        """Generate a cylinder light using parametric ranges."""
        from metasim.scenario.lights import CylinderLightCfg

        intensity_range = ranges.get("intensity_ranges", {}).get("cylinder", [300.0, 800.0])
        intensity = float(np.random.uniform(*intensity_range))

        color = self._generate_random_color(ranges)
        pos = self._generate_random_position(ranges)

        radius_range = ranges.get("radius_range", [0.1, 1.0])
        radius = float(np.random.uniform(*radius_range))

        length_range = ranges.get("length_range", [0.5, 2.0])
        length = float(np.random.uniform(*length_range))

        # Random rotation quaternion
        rot = self._generate_random_rotation()

        light_cfg = CylinderLightCfg(
            name=f"parametric_cylinder_light_{light_id}",
            intensity=intensity,
            color=color,
            pos=pos,
            rot=rot,
            radius=radius,
            length=length,
            exposure=self._generate_random_exposure(ranges),
            normalize=self._get_random_bool_property(ranges, "normalize_probability"),
            treat_as_line=self._get_random_bool_property(ranges, "treat_as_line_probability"),
            enable_color_temperature=self._get_random_bool_property(ranges, "color_temperature_probability"),
            color_temperature=self._generate_random_color_temperature(ranges),
        )

        return light_cfg

    def _generate_parametric_disk_light(self, ranges: dict, light_id: int) -> Any:
        """Generate a disk light using parametric ranges."""
        from metasim.scenario.lights import DiskLightCfg

        intensity_range = ranges.get("intensity_ranges", {}).get("disk", [400.0, 1200.0])
        intensity = float(np.random.uniform(*intensity_range))

        color = self._generate_random_color(ranges)
        pos = self._generate_random_position(ranges)

        radius_range = ranges.get("radius_range", [0.1, 1.0])
        radius = float(np.random.uniform(*radius_range))

        # Random rotation quaternion
        rot = self._generate_random_rotation()

        light_cfg = DiskLightCfg(
            name=f"parametric_disk_light_{light_id}",
            intensity=intensity,
            color=color,
            pos=pos,
            rot=rot,
            radius=radius,
            exposure=self._generate_random_exposure(ranges),
            normalize=self._get_random_bool_property(ranges, "normalize_probability"),
            enable_color_temperature=self._get_random_bool_property(ranges, "color_temperature_probability"),
            color_temperature=self._generate_random_color_temperature(ranges),
        )

        return light_cfg

    def _generate_parametric_dome_light(self, ranges: dict, light_id: int) -> Any:
        """Generate a dome light using parametric ranges."""
        from metasim.scenario.lights import DomeLightCfg

        intensity_range = ranges.get("intensity_ranges", {}).get("dome", [100.0, 500.0])
        intensity = float(np.random.uniform(*intensity_range))

        color = self._generate_random_color(ranges)

        light_cfg = DomeLightCfg(
            name=f"parametric_dome_light_{light_id}",
            intensity=intensity,
            color=color,
            texture_file=None,  # Pure color dome
            exposure=self._generate_random_exposure(ranges),
            normalize=self._get_random_bool_property(ranges, "normalize_probability"),
            visible_in_primary_ray=self._get_random_bool_property(ranges, "visible_in_primary_ray_probability"),
            enable_color_temperature=self._get_random_bool_property(ranges, "color_temperature_probability"),
            color_temperature=self._generate_random_color_temperature(ranges),
        )

        return light_cfg

    def _generate_random_color(self, ranges: dict) -> tuple:
        """Generate a random color using the color ranges."""
        color_variation = ranges.get("color_variation", 0.2)

        # Choose color temperature type
        color_type = np.random.choice(["warm", "cool", "neutral"])

        if color_type == "warm":
            color_range = ranges.get("warm_color_range", [0.9, 1.0, 0.7, 1.0, 0.6, 0.9])
        elif color_type == "cool":
            color_range = ranges.get("cool_color_range", [0.7, 0.9, 0.8, 1.0, 0.9, 1.0])
        else:  # neutral
            color_range = ranges.get("neutral_color_range", [0.85, 1.0, 0.85, 1.0, 0.85, 1.0])

        r = float(np.random.uniform(color_range[0], color_range[1]))
        g = float(np.random.uniform(color_range[2], color_range[3]))
        b = float(np.random.uniform(color_range[4], color_range[5]))

        return (r, g, b)

    def _generate_random_position(self, ranges: dict) -> tuple:
        """Generate a random position using the position ranges."""
        pos_ranges = ranges.get("position_ranges", {})
        x_range = pos_ranges.get("x", [-2.0, 3.0])
        y_range = pos_ranges.get("y", [-2.0, 2.0])
        z_range = pos_ranges.get("z", [0.5, 3.0])

        x = float(np.random.uniform(*x_range))
        y = float(np.random.uniform(*y_range))
        z = float(np.random.uniform(*z_range))

        return (x, y, z)

    def _generate_random_rotation(self) -> tuple:
        """Generate a random rotation quaternion."""
        # Generate random rotation using quaternion
        # Simple random rotation around Z axis for most lights
        angle = np.random.uniform(0, 2 * np.pi)
        w = np.cos(angle / 2)
        x = 0.0
        y = 0.0
        z = np.sin(angle / 2)

        return (float(w), float(x), float(y), float(z))

    def _generate_random_exposure(self, ranges: dict) -> float:
        """Generate random exposure value."""
        exposure_range = ranges.get("exposure_range", [-2.0, 2.0])
        return float(np.random.uniform(*exposure_range))

    def _generate_random_color_temperature(self, ranges: dict) -> float:
        """Generate random color temperature."""
        temp_range = ranges.get("color_temperature_range", [2700.0, 6500.0])
        return float(np.random.uniform(*temp_range))

    def _get_random_bool_property(self, ranges: dict, property_name: str) -> bool:
        """Get a random boolean property based on probability."""
        probability = ranges.get(property_name, 0.5)
        return np.random.random() < probability
