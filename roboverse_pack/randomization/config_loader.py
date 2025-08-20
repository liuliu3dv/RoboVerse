"""Configuration loader for domain randomization.
Loads pre-defined configurations from YAML files for random selection.
"""

import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from loguru import logger as log

from metasim.constants import PhysicStateType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.lights import CylinderLightCfg, DiskLightCfg, DistantLightCfg, DomeLightCfg, SphereLightCfg
from metasim.scenario.objects import PrimitiveCubeCfg, PrimitiveCylinderCfg, PrimitiveSphereCfg

try:
    from metasim.scenario.materials import (
        EnvironmentMaterialCfg,
        GlassMaterialCfg,
        MDLMaterialCfg,
        PBRMaterialCfg,
        PhysicsMaterialCfg,
        PreviewSurfaceCfg,
    )

    MATERIALS_AVAILABLE = True
except ImportError:
    MATERIALS_AVAILABLE = False
    log.warning("Materials module not available")


class ConfigLoader:
    """Loads configurations from YAML files for domain randomization."""

    def __init__(self, config_dir: Optional[str] = None):
        if config_dir is None:
            # Default to the cfg directory relative to this file
            self.config_dir = os.path.join(os.path.dirname(__file__), "cfg")
        else:
            self.config_dir = config_dir

        # Load all configuration files
        self.cameras_config = self._load_yaml("cameras.yml")
        self.lights_config = self._load_yaml("lights.yml")
        self.objects_config = self._load_yaml("objects.yml")
        self.materials_config = self._load_yaml("materials.yml")

        log.info(f"ConfigLoader initialized from {self.config_dir}")

    def _load_yaml(self, filename: str) -> Dict:
        """Load a YAML configuration file."""
        filepath = os.path.join(self.config_dir, filename)
        try:
            with open(filepath) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            log.warning(f"Configuration file {filename} not found")
            return {}
        except yaml.YAMLError as e:
            log.error(f"Error loading {filename}: {e}")
            return {}

    # Camera configuration methods
    def get_random_camera_position(self, split: str = "train") -> Optional[Dict]:
        """Get a random camera position from the configuration."""
        positions = self.cameras_config.get("camera_positions", {}).get(split, [])
        if not positions:
            log.warning(f"No camera positions found for split '{split}'")
            return None

        return random.choice(positions)

    def get_random_camera_intrinsics(self) -> Optional[Dict]:
        """Get random camera intrinsics from the configuration."""
        intrinsics = self.cameras_config.get("camera_intrinsics", [])
        if not intrinsics:
            return None

        return random.choice(intrinsics)

    def apply_camera_randomization(self, camera_cfg: PinholeCameraCfg) -> None:
        """Apply randomization to a camera configuration."""
        # Get random position
        position_cfg = self.get_random_camera_position()
        if position_cfg:
            camera_cfg.pos = tuple(float(x) for x in position_cfg["position"])
            camera_cfg.look_at = tuple(float(x) for x in position_cfg["look_at"])

            # Add small noise
            ranges = self.cameras_config.get("randomization_ranges", {})
            look_at_noise = ranges.get("look_at_noise", {})

            if look_at_noise:
                noise_x = float(np.random.uniform(*look_at_noise.get("x", [0, 0])))
                noise_y = float(np.random.uniform(*look_at_noise.get("y", [0, 0])))
                noise_z = float(np.random.uniform(*look_at_noise.get("z", [0, 0])))

                new_look_at = (
                    camera_cfg.look_at[0] + noise_x,
                    camera_cfg.look_at[1] + noise_y,
                    camera_cfg.look_at[2] + noise_z,
                )
                camera_cfg.look_at = new_look_at

        # Get random intrinsics
        intrinsics_cfg = self.get_random_camera_intrinsics()
        if intrinsics_cfg:
            camera_cfg.focal_length = float(intrinsics_cfg["focal_length"])
            camera_cfg.horizontal_aperture = float(intrinsics_cfg["horizontal_aperture"])

            # Add small variation
            ranges = self.cameras_config.get("randomization_ranges", {})
            focal_variation = ranges.get("focal_length_variation", 0)
            aperture_variation = ranges.get("aperture_variation", 0)

            if focal_variation > 0:
                variation = np.random.uniform(-focal_variation, focal_variation)
                camera_cfg.focal_length *= 1 + variation

            if aperture_variation > 0:
                variation = np.random.uniform(-aperture_variation, aperture_variation)
                camera_cfg.horizontal_aperture *= 1 + variation

    # Lighting configuration methods
    def get_random_lighting_preset(self, split: str = "train") -> Optional[Dict]:
        """Get a random lighting preset from the configuration."""
        presets = self.lights_config.get("lighting_presets", {}).get(split, [])
        if not presets:
            log.warning(f"No lighting presets found for split '{split}'")
            return None

        return random.choice(presets)

    def create_lights_from_preset(self, preset: Dict) -> List[Any]:
        """Create light objects from a preset configuration."""
        lights = []

        for idx, light_cfg in enumerate(preset.get("lights", [])):
            light_type = light_cfg["type"]
            # Generate a unique name for each light
            light_name = f"{preset.get('name', 'preset')}_{light_type}_{idx}"

            if light_type == "distant":
                light = DistantLightCfg(
                    name=light_name,
                    intensity=float(light_cfg["intensity"]),
                    color=tuple(float(x) for x in light_cfg["color"]),
                    polar=float(light_cfg["polar"]),
                    azimuth=float(light_cfg["azimuth"]),
                    # Support new attributes
                    angle=float(light_cfg.get("angle", 0.53)),
                    exposure=float(light_cfg.get("exposure", 0.0)),
                    normalize=light_cfg.get("normalize", False),
                    enable_color_temperature=light_cfg.get("enable_color_temperature", False),
                    color_temperature=float(light_cfg.get("color_temperature", 6500.0)),
                )

            elif light_type == "sphere":
                light = SphereLightCfg(
                    name=light_name,
                    intensity=float(light_cfg["intensity"]),
                    color=tuple(float(x) for x in light_cfg["color"]),
                    radius=float(light_cfg["radius"]),
                    pos=tuple(float(x) for x in light_cfg["position"]),
                    # Support new attributes
                    normalize=light_cfg.get("normalize", False),
                    treat_as_point=light_cfg.get("treat_as_point", False),
                    exposure=float(light_cfg.get("exposure", 0.0)),
                    enable_color_temperature=light_cfg.get("enable_color_temperature", False),
                    color_temperature=float(light_cfg.get("color_temperature", 6500.0)),
                )

            elif light_type == "cylinder":
                light = CylinderLightCfg(
                    name=light_name,
                    intensity=float(light_cfg["intensity"]),
                    color=tuple(float(x) for x in light_cfg["color"]),
                    radius=float(light_cfg["radius"]),
                    length=float(light_cfg["length"]),
                    pos=tuple(float(x) for x in light_cfg["position"]),
                    rot=tuple(float(x) for x in light_cfg["rotation"]),
                    # Support new attributes
                    treat_as_line=light_cfg.get("treat_as_line", False),
                    exposure=float(light_cfg.get("exposure", 0.0)),
                    normalize=light_cfg.get("normalize", False),
                    enable_color_temperature=light_cfg.get("enable_color_temperature", False),
                    color_temperature=float(light_cfg.get("color_temperature", 6500.0)),
                )

            elif light_type == "disk":
                light = DiskLightCfg(
                    name=light_name,
                    intensity=float(light_cfg["intensity"]),
                    color=tuple(float(x) for x in light_cfg["color"]),
                    radius=float(light_cfg["radius"]),
                    pos=tuple(float(x) for x in light_cfg["position"]),
                    rot=tuple(float(x) for x in light_cfg.get("rotation", [1.0, 0.0, 0.0, 0.0])),
                    # Support new attributes
                    normalize=light_cfg.get("normalize", False),
                    exposure=float(light_cfg.get("exposure", 0.0)),
                    enable_color_temperature=light_cfg.get("enable_color_temperature", False),
                    color_temperature=float(light_cfg.get("color_temperature", 6500.0)),
                )

            elif light_type == "dome":
                light = DomeLightCfg(
                    name=light_name,
                    intensity=float(light_cfg["intensity"]),
                    color=tuple(float(x) for x in light_cfg["color"]),
                    texture_file=light_cfg.get("texture_file"),
                    # Support new attributes
                    texture_format=light_cfg.get("texture_format", "automatic"),
                    visible_in_primary_ray=light_cfg.get("visible_in_primary_ray", True),
                    exposure=float(light_cfg.get("exposure", 0.0)),
                    normalize=light_cfg.get("normalize", False),
                    enable_color_temperature=light_cfg.get("enable_color_temperature", False),
                    color_temperature=float(light_cfg.get("color_temperature", 6500.0)),
                )

            else:
                log.warning(f"Unknown light type: {light_type}")
                continue

            lights.append(light)

        return lights

    def get_random_additional_light(self, light_type: str = None) -> Optional[Any]:
        """Get a random additional light for adding to the scene."""
        additional = self.lights_config.get("additional_lights", {})

        if light_type is None:
            # Choose random light type
            available_types = list(additional.keys())
            if not available_types:
                return None
            light_type = random.choice(available_types)

        light_configs = additional.get(light_type, [])
        if not light_configs:
            return None

        config = random.choice(light_configs)

        if light_type == "sphere_lights":
            # Create sphere light with enhanced properties
            light_cfg = SphereLightCfg(
                name=f"additional_{light_type}",
                intensity=float(np.random.uniform(*config["intensity_range"])),
                color=tuple(random.choice(config["color_options"])),
                radius=float(np.random.uniform(*config["radius_range"])),
                pos=tuple(random.choice(config["positions"])),
                normalize=random.choice(config.get("normalize", [True])),
            )

            # Add optional properties
            if "exposure_range" in config:
                light_cfg.exposure = float(np.random.uniform(*config["exposure_range"]))
            if "treat_as_point" in config:
                light_cfg.treat_as_point = random.choice(config["treat_as_point"])

            return light_cfg

        elif light_type == "distant_lights":
            # Create distant light with enhanced properties
            light_cfg = DistantLightCfg(
                name=f"additional_{light_type}",
                intensity=float(np.random.uniform(*config["intensity_range"])),
                color=tuple(random.choice(config["color_options"])),
                polar=float(np.random.uniform(*config["polar_range"])),
                azimuth=float(np.random.uniform(*config["azimuth_range"])),
            )

            # Add optional properties
            if "exposure_range" in config:
                light_cfg.exposure = float(np.random.uniform(*config["exposure_range"]))
            if "angle_range" in config:
                light_cfg.angle = float(np.random.uniform(*config["angle_range"]))
            if "color_temperature_range" in config and "enable_color_temperature_probability" in config:
                if random.random() < config["enable_color_temperature_probability"]:
                    light_cfg.enable_color_temperature = True
                    light_cfg.color_temperature = float(np.random.uniform(*config["color_temperature_range"]))

            return light_cfg

        elif light_type == "disk_lights":
            # Create disk light with enhanced properties
            light_cfg = DiskLightCfg(
                name=f"additional_{light_type}",
                intensity=float(np.random.uniform(*config["intensity_range"])),
                color=tuple(random.choice(config["color_options"])),
                radius=float(np.random.uniform(*config["radius_range"])),
                pos=tuple(random.choice(config["positions"])),
                normalize=random.choice(config.get("normalize", [True])),
            )

            # Add optional properties
            if "exposure_range" in config:
                light_cfg.exposure = float(np.random.uniform(*config["exposure_range"]))
            if "rotations" in config:
                light_cfg.rot = tuple(random.choice(config["rotations"]))

            return light_cfg

        elif light_type == "cylinder_lights":
            # Create cylinder light with enhanced properties
            light_cfg = CylinderLightCfg(
                name=f"additional_{light_type}",
                intensity=float(np.random.uniform(*config["intensity_range"])),
                color=tuple(random.choice(config["color_options"])),
                radius=float(np.random.uniform(*config["radius_range"])),
                length=float(np.random.uniform(*config["length_range"])),
                pos=tuple(random.choice(config["positions"])),
            )

            # Add optional properties
            if "exposure_range" in config:
                light_cfg.exposure = float(np.random.uniform(*config["exposure_range"]))
            if "normalize" in config:
                light_cfg.normalize = random.choice(config["normalize"])
            if "treat_as_line" in config:
                light_cfg.treat_as_line = random.choice(config["treat_as_line"])
            if "rotations" in config:
                light_cfg.rot = tuple(random.choice(config["rotations"]))

            return light_cfg

        elif light_type == "dome_lights":
            # Create dome light with enhanced properties
            light_cfg = DomeLightCfg(
                name=f"additional_{light_type}",
                intensity=float(np.random.uniform(*config["intensity_range"])),
                color=tuple(random.choice(config["color_options"])),
                texture_file=None,  # Pure color dome
            )

            # Add optional properties
            if "exposure_range" in config:
                light_cfg.exposure = float(np.random.uniform(*config["exposure_range"]))
            if "visible_in_primary_ray" in config:
                light_cfg.visible_in_primary_ray = random.choice(config["visible_in_primary_ray"])

            return light_cfg

        return None

    # Object configuration methods
    def get_random_object(self, split: str = "train", object_type: str = None) -> Optional[Any]:
        """Get a random object from the configuration."""
        objects = self.objects_config.get("objects", {}).get(split, {})

        if object_type is None:
            # Choose random object type
            available_types = list(objects.keys())
            if not available_types:
                return None
            object_type = random.choice(available_types)

        object_configs = objects.get(object_type, [])
        if not object_configs:
            return None

        config = random.choice(object_configs)
        return self._create_object_from_config(config)

    def _create_object_from_config(self, config: Dict) -> Optional[Any]:
        """Create an object from a configuration dictionary."""
        obj_type = config["type"]

        # Parse physics state
        physics_str = config.get("physics", "RIGIDBODY")
        physics = getattr(PhysicStateType, physics_str, PhysicStateType.RIGIDBODY)

        # Common properties
        collision_enabled = config.get("collision_enabled", True)
        enabled_gravity = config.get("enabled_gravity", True)
        fix_base_link = config.get("fix_base_link", False)

        # Select random color
        color_options = config.get("color_options", [[0.5, 0.5, 0.5]])
        color = random.choice(color_options)

        if obj_type == "cube":
            size = config["size"]
            mass_range = config.get("mass_range", [0.1, 0.5])
            mass = float(np.random.uniform(*mass_range))

            return PrimitiveCubeCfg(
                name=config["name"],
                size=size,
                color=color,
                mass=mass,
                physics=physics,
                collision_enabled=collision_enabled,
                enabled_gravity=enabled_gravity,
                fix_base_link=fix_base_link,
            )

        elif obj_type == "sphere":
            radius_range = config.get("radius_range", [0.05, 0.05])
            radius = float(np.random.uniform(*radius_range))
            mass_range = config.get("mass_range", [0.1, 0.5])
            mass = float(np.random.uniform(*mass_range))

            return PrimitiveSphereCfg(
                name=config["name"],
                radius=radius,
                color=color,
                mass=mass,
                physics=physics,
                collision_enabled=collision_enabled,
                enabled_gravity=enabled_gravity,
                fix_base_link=fix_base_link,
            )

        elif obj_type == "cylinder":
            radius_range = config.get("radius_range", [0.03, 0.05])
            radius = float(np.random.uniform(*radius_range))
            height_range = config.get("height_range", [0.06, 0.1])
            height = float(np.random.uniform(*height_range))
            mass_range = config.get("mass_range", [0.1, 0.5])
            mass = float(np.random.uniform(*mass_range))

            return PrimitiveCylinderCfg(
                name=config["name"],
                radius=radius,
                height=height,
                color=color,
                mass=mass,
                physics=physics,
                collision_enabled=collision_enabled,
                enabled_gravity=enabled_gravity,
                fix_base_link=fix_base_link,
            )

        elif obj_type == "rigid":
            # File-based rigid object
            from metasim.scenario.objects import RigidObjCfg

            # Random scale
            scale_range = config.get("scale_range", [1.0, 1.0])
            scale = float(np.random.uniform(*scale_range))

            return RigidObjCfg(
                name=config["name"],
                usd_path=config.get("usd_path"),
                mesh_path=config.get("mesh_path"),
                urdf_path=config.get("urdf_path"),
                mjcf_path=config.get("mjcf_path"),
                scale=scale,
                physics=physics,
                collision_enabled=collision_enabled,
                enabled_gravity=enabled_gravity,
                fix_base_link=fix_base_link,
            )

        elif obj_type == "articulation":
            # File-based articulated object
            from metasim.scenario.objects import ArticulationObjCfg

            # Random scale
            scale_range = config.get("scale_range", [1.0, 1.0])
            scale = float(np.random.uniform(*scale_range))

            return ArticulationObjCfg(
                name=config["name"],
                usd_path=config.get("usd_path"),
                mesh_path=config.get("mesh_path"),
                urdf_path=config.get("urdf_path"),
                mjcf_path=config.get("mjcf_path"),
                mjx_mjcf_path=config.get("mjx_mjcf_path"),
                scale=scale,
                enabled_gravity=enabled_gravity,
                fix_base_link=fix_base_link,
            )

        return None

    def get_spawn_area(self, area_name: str = None) -> Optional[Dict]:
        """Get a spawn area configuration."""
        spawn_areas = self.objects_config.get("spawn_areas", {})

        if area_name is None:
            # Choose random spawn area
            available_areas = list(spawn_areas.keys())
            if not available_areas:
                return None
            area_name = random.choice(available_areas)

        return spawn_areas.get(area_name)

    # Material configuration methods
    def get_random_object_material(self, for_object_type: str = None, material_type: str = None) -> Optional[Dict]:
        """Get a random material for objects from the configuration."""
        if not MATERIALS_AVAILABLE:
            return None

        object_materials = self.materials_config.get("object_materials", {})

        if material_type is None:
            # Choose material type based on object preferences or weights
            if for_object_type:
                preferences = self.materials_config.get("assignment_rules", {}).get("object_preferences", {})
                obj_prefs = preferences.get(for_object_type, {})
                preferred_types = obj_prefs.get("preferred_types", list(object_materials.keys()))
                avoid_types = obj_prefs.get("avoid_types", [])
                weights = obj_prefs.get("weights", {})

                valid_types = [t for t in preferred_types if t not in avoid_types and t in object_materials]
                if not valid_types:
                    valid_types = list(object_materials.keys())

                # Use object-specific weights if available
                if weights:
                    type_weights = [weights.get(t, 0.1) for t in valid_types]
                    if sum(type_weights) > 0:
                        material_type = np.random.choice(valid_types, p=np.array(type_weights) / sum(type_weights))
                    else:
                        material_type = random.choice(valid_types)
                else:
                    material_type = random.choice(valid_types)
            else:
                # Use global weights from configuration
                weights = self.materials_config.get("randomization_settings", {}).get("global_type_weights", {})
                valid_types = list(object_materials.keys())
                type_weights = [weights.get(t, 0.1) for t in valid_types]

                if sum(type_weights) > 0:
                    material_type = np.random.choice(valid_types, p=np.array(type_weights) / sum(type_weights))
                else:
                    material_type = random.choice(valid_types)

        materials = object_materials.get(material_type, [])
        if not materials:
            return None

        return random.choice(materials)

    def get_random_environment_material(self, env_type: str = "ground", split: str = "train") -> Optional[Dict]:
        """Get a random environment material (ground, walls, tables) from the configuration."""
        if not MATERIALS_AVAILABLE:
            return None

        env_materials = self.materials_config.get("environment_materials", {}).get(env_type, {})
        preferences = (
            self.materials_config.get("assignment_rules", {}).get("environment_preferences", {}).get(env_type, {})
        )
        use_mdl_prob = preferences.get("use_mdl_probability", 0.8)

        # Decide whether to use MDL or fallback PBR
        if random.random() < use_mdl_prob and "mdl_library" in env_materials:
            # Use MDL materials
            mdl_materials = env_materials["mdl_library"].get(split, [])
            if mdl_materials:
                return random.choice(mdl_materials)

        # Use fallback PBR materials
        fallback_materials = env_materials.get("pbr_fallback", [])
        if fallback_materials:
            return random.choice(fallback_materials)

        # Fallback to object materials if no environment materials
        fallback_types = preferences.get("fallback_types", ["basic_colors"])
        fallback_type = random.choice(fallback_types)
        return self.get_random_object_material(material_type=fallback_type)

    def get_random_physics_material(self, friction_type: str = None) -> Optional[Dict]:
        """Get a random physics material from the configuration."""
        if not MATERIALS_AVAILABLE:
            return None

        physics_materials = self.materials_config.get("physics_materials", {})

        if friction_type is None:
            # Choose random friction type
            available_types = list(physics_materials.keys())
            if not available_types:
                return None
            friction_type = random.choice(available_types)

        materials = physics_materials.get(friction_type, [])
        if not materials:
            return None

        return random.choice(materials)

    def get_random_material(self, material_type: str = None, for_object_type: str = None) -> Optional[Dict]:
        """Legacy method for backward compatibility."""
        return self.get_random_object_material(for_object_type=for_object_type, material_type=material_type)

    def create_material_from_config(self, config: Dict) -> Optional[Any]:
        """Create a material object from configuration."""
        if not MATERIALS_AVAILABLE:
            return None

        material_type = config.get("type", "pbr")
        settings = self.materials_config.get("randomization_settings", {})

        if material_type == "pbr":
            return self._create_pbr_material(config, settings)
        elif material_type == "mdl":
            return self._create_mdl_material(config, settings)
        elif material_type == "physics":
            return self._create_physics_material(config, settings)
        elif material_type == "preview_surface":
            return self._create_preview_surface_material(config, settings)
        else:
            log.warning(f"Unknown material type: {material_type}")
            return None

    def _create_pbr_material(self, config: Dict, settings: Dict) -> PBRMaterialCfg:
        """Create a PBR material with randomization."""
        color_var = settings.get("color_variation", 0.15)
        metallic_var = settings.get("metallic_variation", 0.1)
        roughness_var = settings.get("roughness_variation", 0.15)
        specular_var = settings.get("specular_variation", 0.1)
        opacity_var = settings.get("opacity_variation", 0.05)
        emission_var = settings.get("emission_intensity_variation", 0.3)

        # Randomize diffuse color
        base_color = config.get("diffuse_color", [0.8, 0.8, 0.8])
        diffuse_color = []
        for c in base_color:
            noise = np.random.uniform(-color_var, color_var)
            diffuse_color.append(float(np.clip(c + noise, 0.0, 1.0)))

        # Randomize other properties
        metallic = config.get("metallic", 0.0)
        metallic += np.random.uniform(-metallic_var, metallic_var)
        metallic = float(np.clip(metallic, 0.0, 1.0))

        roughness = config.get("roughness", 0.5)
        roughness += np.random.uniform(-roughness_var, roughness_var)
        roughness = float(np.clip(roughness, 0.0, 1.0))

        specular = config.get("specular", 0.5)
        specular += np.random.uniform(-specular_var, specular_var)
        specular = float(np.clip(specular, 0.0, 1.0))

        opacity = config.get("opacity", 1.0)
        opacity += np.random.uniform(-opacity_var, opacity_var)
        opacity = float(np.clip(opacity, 0.0, 1.0))

        emission_intensity = config.get("emission_intensity", 0.0)
        if emission_intensity > 0:
            emission_intensity += np.random.uniform(-emission_var, emission_var)
            emission_intensity = float(np.clip(emission_intensity, 0.0, 2.0))

        return PBRMaterialCfg(
            name=config["name"],
            diffuse_color=tuple(diffuse_color),
            emissive_color=tuple(config.get("emissive_color", [0.0, 0.0, 0.0])),
            metallic=metallic,
            roughness=roughness,
            specular=specular,
            opacity=opacity,
            clearcoat=float(config.get("clearcoat", 0.0)),
            clearcoat_roughness=float(config.get("clearcoat_roughness", 0.1)),
            ior=float(config.get("ior", 1.5)),
            transmission=float(config.get("transmission", 0.0)),
            emission_intensity=emission_intensity,
            anisotropy=float(config.get("anisotropy", 0.0)),
            subsurface=float(config.get("subsurface", 0.0)),
            subsurface_color=tuple(config.get("subsurface_color", [1.0, 1.0, 1.0])),
        )

    def _create_mdl_material(self, config: Dict, settings: Dict) -> MDLMaterialCfg:
        """Create an MDL material with randomization."""
        albedo_var = settings.get("albedo_brightness_variation", 0.2)
        texture_scale_var = settings.get("texture_scale_variation", 0.3)

        # Randomize albedo brightness
        albedo_brightness = config.get("albedo_brightness")
        if albedo_brightness is not None:
            if isinstance(albedo_brightness, list) and len(albedo_brightness) == 2:
                # Range specified in config
                albedo_brightness = float(np.random.uniform(*albedo_brightness))
            else:
                # Single value with variation
                variation = np.random.uniform(-albedo_var, albedo_var)
                albedo_brightness = float(np.clip(albedo_brightness + variation, 0.1, 3.0))

        # Randomize texture scale
        texture_scale = config.get("texture_scale")
        if texture_scale is not None:
            if isinstance(texture_scale, list) and len(texture_scale) == 2 and isinstance(texture_scale[0], list):
                # Range specified in config [[min_u, min_v], [max_u, max_v]]
                min_scale, max_scale = texture_scale
                texture_scale = (
                    float(np.random.uniform(min_scale[0], max_scale[0])),
                    float(np.random.uniform(min_scale[1], max_scale[1])),
                )
            elif isinstance(texture_scale, list) and len(texture_scale) == 2:
                # Fixed scale with variation
                scale_u = texture_scale[0] * (1 + np.random.uniform(-texture_scale_var, texture_scale_var))
                scale_v = texture_scale[1] * (1 + np.random.uniform(-texture_scale_var, texture_scale_var))
                texture_scale = (float(np.clip(scale_u, 0.1, 5.0)), float(np.clip(scale_v, 0.1, 5.0)))

        return MDLMaterialCfg(
            name=config["name"],
            mdl_path=config["mdl_path"],
            project_uvw=config.get("project_uvw"),
            albedo_brightness=albedo_brightness,
            texture_scale=texture_scale,
        )

    def _create_physics_material(self, config: Dict, settings: Dict) -> PhysicsMaterialCfg:
        """Create a physics material (no randomization for physics properties)."""
        return PhysicsMaterialCfg(
            name=config["name"],
            static_friction=float(config.get("static_friction", 0.5)),
            dynamic_friction=float(config.get("dynamic_friction", 0.5)),
            rolling_friction=float(config.get("rolling_friction", 0.0)),
            spinning_friction=float(config.get("spinning_friction", 0.0)),
            restitution=float(config.get("restitution", 0.0)),
            contact_offset=float(config.get("contact_offset", 0.02)),
            rest_offset=float(config.get("rest_offset", 0.0)),
            friction_combine_mode=config.get("friction_combine_mode", "average"),
            restitution_combine_mode=config.get("restitution_combine_mode", "average"),
            density=config.get("density"),
        )

    def _create_preview_surface_material(self, config: Dict, settings: Dict) -> PreviewSurfaceCfg:
        """Create a preview surface material with randomization."""
        color_var = settings.get("color_variation", 0.15)
        metallic_var = settings.get("metallic_variation", 0.1)
        roughness_var = settings.get("roughness_variation", 0.15)
        opacity_var = settings.get("opacity_variation", 0.05)

        # Randomize diffuse color
        base_color = config.get("diffuse_color", [0.18, 0.18, 0.18])
        diffuse_color = []
        for c in base_color:
            noise = np.random.uniform(-color_var, color_var)
            diffuse_color.append(float(np.clip(c + noise, 0.0, 1.0)))

        # Randomize properties
        metallic = config.get("metallic", 0.0)
        metallic += np.random.uniform(-metallic_var, metallic_var)
        metallic = float(np.clip(metallic, 0.0, 1.0))

        roughness = config.get("roughness", 0.5)
        roughness += np.random.uniform(-roughness_var, roughness_var)
        roughness = float(np.clip(roughness, 0.0, 1.0))

        opacity = config.get("opacity", 1.0)
        opacity += np.random.uniform(-opacity_var, opacity_var)
        opacity = float(np.clip(opacity, 0.0, 1.0))

        return PreviewSurfaceCfg(
            name=config["name"],
            diffuse_color=tuple(diffuse_color),
            emissive_color=tuple(config.get("emissive_color", [0.0, 0.0, 0.0])),
            roughness=roughness,
            metallic=metallic,
            opacity=opacity,
        )
