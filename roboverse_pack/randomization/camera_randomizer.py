"""Camera randomization with both YAML-based and parametric positioning."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from loguru import logger as log

from .base import BaseRandomizer
from .config_loader import ConfigLoader


class CameraRandomizer(BaseRandomizer):
    """Enhanced camera randomizer with both preset and parametric positioning."""

    def __init__(
        self,
        config_dir: str | None = None,
        split: str = "train",
        distance_range: tuple[float, float] = (1.5, 3.0),
        elevation_range: tuple[float, float] = (15.0, 75.0),
        azimuth_range: tuple[float, float] = (-90.0, 90.0),
        look_at_noise: float = 0.1,
        use_preset_probability: float = 0.5,
        randomize_intrinsics: bool = True,
        randomize_focus: bool = False,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)

        self.config_loader = ConfigLoader(config_dir)
        self.split = split

        # Parametric positioning parameters
        self.distance_range = distance_range
        self.elevation_range = elevation_range
        self.azimuth_range = azimuth_range
        self.look_at_noise = look_at_noise
        self.use_preset_probability = use_preset_probability
        self.randomize_intrinsics = randomize_intrinsics
        self.randomize_focus = randomize_focus

        log.info(
            f"CameraRandomizer: preset_prob={use_preset_probability:.1%}, "
            f"distance={distance_range}, elevation={elevation_range}°, azimuth={azimuth_range}°"
        )

    def randomize(self, scenario_cfg: Any, env_ids: list[int] | None = None, **kwargs) -> None:
        """Apply camera randomization to the scenario."""
        if not self.enabled:
            return

        log.debug("Randomizing camera configurations")

        # Randomize camera poses
        for camera_cfg in scenario_cfg.cameras:
            if not hasattr(camera_cfg, "mount_to") or camera_cfg.mount_to is None:
                if np.random.random() < self.use_preset_probability:
                    # Strategy A: Use preset from YAML
                    log.debug(f"📷 Using preset positioning for {camera_cfg.name}")
                    self._apply_preset_positioning(camera_cfg)
                else:
                    # Strategy B: Use parametric positioning
                    log.debug(f"📷 Using parametric positioning for {camera_cfg.name}")
                    self._apply_parametric_positioning(camera_cfg)

                # Optionally randomize intrinsics
                if self.randomize_intrinsics:
                    self._randomize_intrinsics(camera_cfg)

                # Optionally randomize focus
                if self.randomize_focus:
                    self._randomize_focus_properties(camera_cfg)

                log.debug(f"📷 Camera {camera_cfg.name}: pos={camera_cfg.pos}, look_at={camera_cfg.look_at}")

    def _apply_preset_positioning(self, camera_cfg: Any) -> None:
        """Apply preset positioning from YAML configuration."""
        # Get random position from split-specific configurations
        position_cfg = self.config_loader.get_random_camera_position(self.split)
        if position_cfg:
            camera_cfg.pos = tuple(np.float32(x) for x in position_cfg["position"])
            camera_cfg.look_at = tuple(np.float32(x) for x in position_cfg["look_at"])

            # Add small noise from YAML config
            ranges = self.config_loader.cameras_config.get("randomization_ranges", {})
            look_at_noise = ranges.get("look_at_noise", {})

            if look_at_noise:
                noise_x = float(np.random.uniform(*look_at_noise.get("x", [0, 0])))
                noise_y = float(np.random.uniform(*look_at_noise.get("y", [0, 0])))
                noise_z = float(np.random.uniform(*look_at_noise.get("z", [0, 0])))

                new_look_at = (
                    np.float32(camera_cfg.look_at[0] + noise_x),
                    np.float32(camera_cfg.look_at[1] + noise_y),
                    np.float32(camera_cfg.look_at[2] + noise_z),
                )
                camera_cfg.look_at = new_look_at

    def _apply_parametric_positioning(self, camera_cfg: Any) -> None:
        """Apply parametric positioning using spherical coordinates."""
        # Default look-at point (can be customized)
        look_at_base = np.array([0.5, 0.0, 0.5])

        # Add small noise to look-at point
        look_at_noise_vec = np.random.uniform(-self.look_at_noise, self.look_at_noise, 3)
        look_at_point = look_at_base + look_at_noise_vec

        # Generate camera position using spherical coordinates
        distance = np.random.uniform(*self.distance_range)
        elevation_deg = np.random.uniform(*self.elevation_range)
        azimuth_deg = np.random.uniform(*self.azimuth_range)

        # Convert to radians
        elevation_rad = math.radians(elevation_deg)
        azimuth_rad = math.radians(azimuth_deg)

        # Spherical to Cartesian coordinates
        # Elevation is measured from horizontal plane (0° = horizontal, 90° = straight up)
        x = distance * math.cos(elevation_rad) * math.cos(azimuth_rad)
        y = distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
        z = distance * math.sin(elevation_rad)

        # Position relative to look-at point
        camera_pos = look_at_point + np.array([x, y, z])

        # Ensure camera is above ground
        camera_pos[2] = max(camera_pos[2], 0.3)

        # Update camera configuration with explicit float32 conversion
        camera_cfg.pos = tuple(camera_pos.astype(np.float32))
        camera_cfg.look_at = tuple(look_at_point.astype(np.float32))

        log.debug(
            f"Generated parametric pose: distance={distance:.1f}m, "
            f"elevation={elevation_deg:.1f}°, azimuth={azimuth_deg:.1f}°"
        )

    def _randomize_intrinsics(self, camera_cfg: Any) -> None:
        """Randomize camera intrinsic parameters."""
        if hasattr(camera_cfg, "focal_length") and hasattr(camera_cfg, "horizontal_aperture"):
            # Option 1: Use preset from YAML
            if np.random.random() < 0.5:
                intrinsics_cfg = self.config_loader.get_random_camera_intrinsics()
                if intrinsics_cfg:
                    camera_cfg.focal_length = float(intrinsics_cfg["focal_length"])
                    camera_cfg.horizontal_aperture = float(intrinsics_cfg["horizontal_aperture"])
            else:
                # Option 2: Parametric variation of current values
                focal_variation = np.random.uniform(0.8, 1.2)
                aperture_variation = np.random.uniform(0.9, 1.1)

                camera_cfg.focal_length *= focal_variation
                camera_cfg.horizontal_aperture *= aperture_variation

                # Clamp to reasonable ranges
                camera_cfg.focal_length = np.clip(camera_cfg.focal_length, 15.0, 100.0)
                camera_cfg.horizontal_aperture = np.clip(camera_cfg.horizontal_aperture, 15.0, 40.0)

    def _randomize_focus_properties(self, camera_cfg: Any) -> None:
        """Randomize camera focus and clipping properties."""
        if hasattr(camera_cfg, "focus_distance"):
            # Randomize focus distance (simulate depth of field effects)
            focus_variation = np.random.uniform(0.5, 2.0)
            camera_cfg.focus_distance *= focus_variation
            camera_cfg.focus_distance = np.clip(camera_cfg.focus_distance, 50.0, 1000.0)

        if hasattr(camera_cfg, "clipping_range"):
            # Slightly randomize clipping planes
            near, far = camera_cfg.clipping_range
            near_variation = np.random.uniform(0.8, 1.2)
            far_variation = np.random.uniform(0.8, 1.2)

            new_near = np.clip(near * near_variation, 0.01, 0.1)
            new_far = np.clip(far * far_variation, 1000.0, 100000.0)

            camera_cfg.clipping_range = (float(new_near), float(new_far))
