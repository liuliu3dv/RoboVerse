"""Configuration classes for materials used in the simulation."""

from __future__ import annotations

from typing import Literal

from metasim.utils import configclass


@configclass
class BaseMaterialCfg:
    """Base configuration for a material."""

    name: str = "default_material"
    """Name of the material"""
    material_type: Literal["pbr", "mdl", "physics", "preview_surface"] = "pbr"
    """Type of material to create"""
    enabled: bool = True
    """Whether this material is enabled for randomization"""


@configclass
class MDLMaterialCfg(BaseMaterialCfg):
    """Material configuration using external MDL files (for Isaac Sim)."""

    material_type: Literal["mdl"] = "mdl"
    mdl_path: str | None = None
    """Path to the MDL material file"""
    project_uvw: bool | None = None
    """Whether to project the UVW coordinates of the material"""
    albedo_brightness: float | None = None
    """Multiplier for the diffuse color of the material"""
    texture_scale: tuple[float, float] | None = None
    """The scale of the texture"""


@configclass
class PBRMaterialCfg(BaseMaterialCfg):
    """Physically-Based Rendering (PBR) material configuration."""

    material_type: Literal["pbr"] = "pbr"

    # Base color properties
    diffuse_color: tuple[float, float, float] = (0.8, 0.8, 0.8)
    """The RGB diffusion color. This is the base color of the surface"""
    emissive_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """The RGB emission component of the surface"""

    # Core PBR properties
    metallic: float = 0.0
    """Metallic property of the material (0.0 = dielectric, 1.0 = metallic)"""
    roughness: float = 0.5
    """Roughness of the material surface (0.0 = mirror, 1.0 = completely rough)"""
    specular: float = 0.5
    """Specular reflection intensity"""
    opacity: float = 1.0
    """The opacity of the surface (0.0 = transparent, 1.0 = opaque)"""
    clearcoat: float = 0.0
    """Clearcoat layer intensity (for car paint, etc.)"""
    clearcoat_roughness: float = 0.1
    """Clearcoat roughness"""

    # Optical properties
    ior: float = 1.5
    """Index of refraction for dielectric materials"""
    transmission: float = 0.0
    """Transmission for glass-like materials (0.0 = opaque, 1.0 = transparent)"""

    # Emission properties
    emission_intensity: float = 0.0
    """Emission intensity for emissive materials"""

    # Additional visual properties
    anisotropy: float = 0.0
    """Anisotropy for brushed metal effects (-1.0 to 1.0)"""
    subsurface: float = 0.0
    """Subsurface scattering amount"""
    subsurface_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Subsurface scattering color"""


@configclass
class TextureMaterialCfg(PBRMaterialCfg):
    """Material configuration with comprehensive texture support."""

    # Base texture maps
    diffuse_texture: str | None = None
    """Path to diffuse/albedo texture file (e.g., .png, .jpg, .exr)"""
    normal_texture: str | None = None
    """Path to normal map texture file"""
    roughness_texture: str | None = None
    """Path to roughness texture file"""
    metallic_texture: str | None = None
    """Path to metallic texture file"""

    # Additional texture maps
    specular_texture: str | None = None
    """Path to specular texture file"""
    emission_texture: str | None = None
    """Path to emission texture file"""
    opacity_texture: str | None = None
    """Path to opacity/alpha texture file"""
    height_texture: str | None = None
    """Path to height/displacement texture file"""
    ambient_occlusion_texture: str | None = None
    """Path to ambient occlusion texture file"""

    # Texture properties
    texture_scale: tuple[float, float] = (1.0, 1.0)
    """UV scaling for textures (U, V)"""
    texture_offset: tuple[float, float] = (0.0, 0.0)
    """UV offset for textures (U, V)"""
    texture_rotation: float = 0.0
    """Texture rotation in radians"""

    # Texture filtering
    texture_wrap_u: str = "repeat"
    """Texture wrap mode for U coordinate: 'repeat', 'clamp', 'mirror'"""
    texture_wrap_v: str = "repeat"
    """Texture wrap mode for V coordinate: 'repeat', 'clamp', 'mirror'"""


@configclass
class PhysicsMaterialCfg(BaseMaterialCfg):
    """Physics material configuration for contact properties."""

    material_type: Literal["physics"] = "physics"

    # Friction properties
    static_friction: float = 0.5
    """Static friction coefficient"""
    dynamic_friction: float = 0.5
    """Dynamic friction coefficient"""
    rolling_friction: float = 0.0
    """Rolling friction coefficient (for wheels, balls, etc.)"""
    spinning_friction: float = 0.0
    """Spinning friction coefficient"""

    # Restitution properties
    restitution: float = 0.0
    """Restitution/bounciness coefficient (0.0 = no bounce, 1.0 = perfect bounce)"""

    # Contact properties
    contact_offset: float = 0.02
    """Contact offset for collision detection"""
    rest_offset: float = 0.0
    """Rest offset for stable contact"""

    # Combine modes
    friction_combine_mode: str = "average"
    """How to combine friction values: 'average', 'min', 'max', 'multiply'"""
    restitution_combine_mode: str = "average"
    """How to combine restitution values: 'average', 'min', 'max', 'multiply'"""

    # Material density (for mass calculation)
    density: float | None = None
    """Material density in kg/m³ (if specified, overrides object mass)"""


@configclass
class ProceduralMaterialCfg(PBRMaterialCfg):
    """Material configuration with procedural generation support."""

    # Procedural noise parameters
    use_procedural_noise: bool = False
    """Whether to use procedural noise for surface variation"""
    noise_scale: float = 1.0
    """Scale of procedural noise"""
    noise_amplitude: float = 0.1
    """Amplitude of procedural noise effect"""
    noise_octaves: int = 4
    """Number of noise octaves for detail"""

    # Procedural patterns
    use_procedural_pattern: bool = False
    """Whether to use procedural patterns (stripes, checkers, etc.)"""
    pattern_type: str = "checker"
    """Type of pattern: 'checker', 'stripe', 'brick', 'hexagon'"""
    pattern_scale: tuple[float, float] = (10.0, 10.0)
    """Scale of the procedural pattern"""
    pattern_color1: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """First color of the pattern"""
    pattern_color2: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Second color of the pattern"""


@configclass
class PreviewSurfaceCfg(BaseMaterialCfg):
    """Preview surface material configuration for simple PBR rendering."""

    material_type: Literal["preview_surface"] = "preview_surface"

    diffuse_color: tuple[float, float, float] = (0.18, 0.18, 0.18)
    """The RGB diffusion color. This is the base color of the surface"""
    emissive_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """The RGB emission component of the surface"""
    roughness: float = 0.5
    """The roughness for specular lobe (0 = smooth, 1 = rough)"""
    metallic: float = 0.0
    """The metallic component (0 = dielectric, 1 = metal)"""
    opacity: float = 1.0
    """The opacity of the surface (0 = transparent, 1 = opaque)"""


@configclass
class GlassMaterialCfg(MDLMaterialCfg):
    """Glass material configuration using MDL."""

    mdl_path: str = "OmniGlass.mdl"
    """Path to the glass MDL material"""
    glass_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """The RGB color or tint of the glass"""
    frosting_roughness: float = 0.0
    """The amount of reflectivity of the surface (0 = clear, 1 = frosted)"""
    thin_walled: bool = False
    """Whether to perform thin-walled refraction"""
    glass_ior: float = 1.491
    """The index of refraction for glass"""


@configclass
class EnvironmentMaterialCfg(BaseMaterialCfg):
    """Special material configuration for environment objects (ground, walls, sky)."""

    apply_to_type: Literal["ground", "wall", "sky", "table"] = "ground"
    """Type of environment object this material applies to"""
    use_mdl_library: bool = True
    """Whether to use MDL materials from library paths"""
    fallback_material: PBRMaterialCfg | None = None
    """Fallback PBR material if MDL fails"""


@configclass
class CompleteMaterialCfg(TextureMaterialCfg, PhysicsMaterialCfg, ProceduralMaterialCfg):
    """Complete material configuration with all features."""

    pass


# Common material presets
DEFAULT_MATERIAL = PBRMaterialCfg(name="default", diffuse_color=(0.8, 0.8, 0.8), metallic=0.0, roughness=0.5)

METAL_MATERIAL = PBRMaterialCfg(name="metal", diffuse_color=(0.7, 0.7, 0.8), metallic=1.0, roughness=0.1)

PLASTIC_MATERIAL = PBRMaterialCfg(name="plastic", diffuse_color=(0.9, 0.9, 0.9), metallic=0.0, roughness=0.3)

GLASS_MATERIAL = GlassMaterialCfg(name="glass", glass_color=(0.95, 0.95, 0.95), thin_walled=False, glass_ior=1.5)
