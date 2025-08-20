"""Domain randomizer that orchestrates all randomization components."""

import copy
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger as log

from .camera_randomizer import CameraRandomizer
from .lighting_randomizer import LightingRandomizer
from .material_randomizer import MaterialRandomizer
from .object_randomizer import ObjectRandomizer


class DomainRandomizer:
    """Domain randomization controller with YAML-based configuration."""

    def __init__(
        self,
        config_dir: Optional[str] = None,
        split: str = "train",
        lighting_config: Optional[Dict] = None,
        camera_config: Optional[Dict] = None,
        material_config: Optional[Dict] = None,
        object_config: Optional[Dict] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

        # Initialize components with YAML support
        lighting_cfg = lighting_config or {}
        lighting_cfg.update({"config_dir": config_dir, "split": split})
        self.lighting_randomizer = LightingRandomizer(seed=seed, **lighting_cfg)

        camera_cfg = camera_config or {}
        camera_cfg.update({"config_dir": config_dir, "split": split})
        self.camera_randomizer = CameraRandomizer(seed=seed, **camera_cfg)

        object_cfg = object_config or {}
        object_cfg.update({"config_dir": config_dir, "split": split})
        self.object_randomizer = ObjectRandomizer(seed=seed, **object_cfg)

        # Material randomizer runs AFTER object randomizer to handle all objects
        material_cfg = material_config or {}
        material_cfg.update({"config_dir": config_dir, "split": split})
        self.material_randomizer = MaterialRandomizer(seed=seed, **material_cfg)

        # Baseline management
        self.original_baseline_scenario_cfg = None
        self.current_baseline_scenario_cfg = None
        self.original_baseline_captured = False

        # Note: Unique IDs are now generated using timestamp + random in each randomizer

        log.info(f"DomainRandomizer initialized with split '{split}'")

    def capture_baseline_from_scenario(self, scenario_cfg: Any) -> None:
        """Captures the baseline from a scenario configuration."""
        if not self.original_baseline_captured:
            self.original_baseline_scenario_cfg = copy.deepcopy(scenario_cfg)
            self.current_baseline_scenario_cfg = copy.deepcopy(scenario_cfg)
            self.original_baseline_captured = True
            log.info("Captured original baseline scenario configuration.")
        else:
            log.debug("Original baseline already captured.")

    def capture_baseline(self, sim_handler: Any) -> None:
        """Captures the initial state of the scenario as the original baseline."""
        self.capture_baseline_from_scenario(sim_handler.scenario_cfg)

    def update_baseline(self, sim_handler: Any) -> None:
        """Updates the current baseline to the current state of the scenario."""
        if self.original_baseline_captured:
            self.current_baseline_scenario_cfg = copy.deepcopy(sim_handler.scenario_cfg)
            log.info("Updated current baseline scenario configuration.")
        else:
            log.warning("Cannot update baseline: original baseline not captured yet.")

    def create_randomized_scenario(self, use_original_baseline: bool = False) -> Any:
        """Create a new randomized scenario configuration.

        This method:
        1. Starts with a baseline scenario configuration (original or current)
        2. Applies all randomizers directly to the scenario config
        3. Returns the modified scenario configuration

        The randomizers only edit the scenario configuration data - they don't touch the actual scene.
        The scene update happens later via sim_handler.update_scene_from_scenario().

        Args:
            use_original_baseline: Whether to use original baseline or current baseline

        Returns:
            A new scenario configuration with randomization applied
        """
        if not self.original_baseline_captured:
            raise RuntimeError("Must capture baseline before creating randomized scenarios")

        # Start with the chosen baseline
        if use_original_baseline:
            scenario_cfg = copy.deepcopy(self.original_baseline_scenario_cfg)
            log.debug("Using original baseline for randomization.")
        else:
            scenario_cfg = copy.deepcopy(self.current_baseline_scenario_cfg)
            log.debug("Using current baseline for randomization.")

        # Apply randomization in order: lighting -> camera -> objects -> materials
        # Each randomizer modifies the scenario_cfg directly
        self.lighting_randomizer.randomize(scenario_cfg)
        self.camera_randomizer.randomize(scenario_cfg)
        self.object_randomizer.randomize(scenario_cfg)

        # Materials LAST - after all objects are finalized
        self.material_randomizer.randomize(scenario_cfg)

        log.debug("Created randomized scenario configuration")
        return scenario_cfg

    def randomize_on_reset(self, sim_handler: Any, env_ids: Optional[List[int]] = None, **kwargs) -> None:
        """Apply randomization when environments are reset.

        This is the main entry point for domain randomization. It handles two modes:

        1. **Original baseline mode** (use_original_baseline=True):
           - Always randomizes from the original captured baseline scenario
           - Provides consistent randomization starting point
           - Good for comparing randomization effects

        2. **Adaptive baseline mode** (use_original_baseline=False):
           - Each randomization becomes the new baseline for the next one
           - Creates cumulative/evolving randomization effects
           - Good for exploring diverse scenarios

        Args:
            sim_handler: The simulation handler to apply randomization to
            env_ids: Environment IDs to randomize (optional)
            **kwargs: Additional arguments, including:
                - use_original_baseline (bool): Whether to use original or adaptive baseline

        Note:
            The method automatically updates the baseline for adaptive mode, so callers
            don't need to manually manage baseline updates.
        """
        # Extract use_original_baseline from kwargs to avoid duplicate argument
        use_original_baseline = kwargs.pop("use_original_baseline", False)

        # Ensure baseline is captured before randomization
        if not self.original_baseline_captured:
            self.capture_baseline(sim_handler)

        # Create a randomized scenario and update the scene dynamically
        randomized_scenario = self.create_randomized_scenario(use_original_baseline=use_original_baseline)

        # Update the scene with the new randomized scenario
        sim_handler.update_scene_from_scenario(randomized_scenario)

        # IMPORTANT: For adaptive mode, update the current baseline to the new randomized scenario
        # This ensures the next randomization builds upon this one rather than the original
        if not use_original_baseline:
            self.current_baseline_scenario_cfg = copy.deepcopy(randomized_scenario)
            log.debug("Updated current baseline to new randomized scenario (adaptive mode)")

    def get_status(self) -> Dict:
        """Get the current status of all randomization components."""
        enabled_components = []
        if self.lighting_randomizer.enabled:
            enabled_components.append("lighting")
        if self.camera_randomizer.enabled:
            enabled_components.append("camera")
        if self.object_randomizer.enabled:
            enabled_components.append("object")
        if self.material_randomizer.enabled:
            enabled_components.append("material")

        return {"enabled_components": enabled_components, "baseline_captured": self.original_baseline_captured}

    def enable_all(self) -> None:
        """Enable all randomization components."""
        self.lighting_randomizer.enabled = True
        self.camera_randomizer.enabled = True
        self.object_randomizer.enabled = True
        self.material_randomizer.enabled = True
        log.info("Enabled all randomization components")

    def disable_all(self) -> None:
        """Disable all randomization components."""
        self.lighting_randomizer.enabled = False
        self.camera_randomizer.enabled = False
        self.object_randomizer.enabled = False
        self.material_randomizer.enabled = False
        log.info("Disabled all randomization components")

    @classmethod
    def create_default(cls, seed: Optional[int] = None, **kwargs):
        """Create a domain randomizer with default settings."""
        return cls(
            lighting_config={
                "enable_additional_lights": True,
                "max_additional_lights": 2,
                "modify_existing_probability": 0.8,
            },
            camera_config={},
            object_config={
                "enable_additional_objects": True,
                "max_additional_objects": 2,
                "modify_existing_probability": 0.9,
            },
            material_config={"randomize_objects": True, "randomize_robots": False, "object_change_probability": 0.8},
            seed=seed,
            **kwargs,
        )

    @classmethod
    def create_conservative(cls, seed: Optional[int] = None, **kwargs):
        """Create a conservative domain randomizer."""
        return cls(
            lighting_config={"enable_additional_lights": False, "modify_existing_probability": 0.6},
            camera_config={},
            object_config={
                "enable_additional_objects": False,
                "modify_existing_probability": 0.7,
                "position_noise": 0.05,
                "rotation_noise": 0.1,
            },
            material_config={"randomize_objects": True, "randomize_robots": False, "object_change_probability": 0.6},
            seed=seed,
            **kwargs,
        )

    @classmethod
    def create_aggressive(cls, seed: Optional[int] = None, **kwargs):
        """Create an aggressive domain randomizer."""
        return cls(
            lighting_config={
                "enable_additional_lights": True,
                "max_additional_lights": 4,
                "modify_existing_probability": 1.0,
            },
            camera_config={},
            object_config={
                "enable_additional_objects": True,
                "max_additional_objects": 3,
                "modify_existing_probability": 1.0,
                "position_noise": 0.15,
                "rotation_noise": 0.4,
            },
            material_config={"randomize_objects": True, "randomize_robots": True, "object_change_probability": 1.0},
            seed=seed,
            **kwargs,
        )
