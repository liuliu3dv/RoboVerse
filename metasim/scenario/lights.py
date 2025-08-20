"""Configuration classes for lights used in the simulation."""

from __future__ import annotations

import math

import torch
from loguru import logger as log

from metasim.utils import configclass
from metasim.utils.math import quat_from_euler_xyz


@configclass
class BaseLightCfg:
    """Base configuration for a light."""

    name: str = ""
    """Name of the light for tracking and management"""
    intensity: float = 500.0
    """Intensity of the light"""
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Color of the light"""
    exposure: float = 0.0
    """Scales the power of the light exponentially as a power of 2. Default is 0.0."""
    normalize: bool = False
    """Normalizes power by the surface area of the light. Default is False."""
    enable_color_temperature: bool = False
    """Enables color temperature. Default is False."""
    color_temperature: float = 6500.0
    """Color temperature (in Kelvin) representing the white point. Valid range is [1000, 10000]. Default is 6500K."""
    is_global: bool = False
    """Whether the light is a global light that is not copied to each environment"""


@configclass
class DistantLightCfg(BaseLightCfg):
    """Configuration for a distant light. The default direction is (0, 0, -1), pointing towards Z- direction."""

    polar: float = 0.0
    """Polar angle of the light (in degrees). Default is 0, which means the light is pointing towards Z- direction."""
    azimuth: float = 0.0
    """Azimuth angle of the light (in degrees). Default is 0."""
    angle: float = 0.53
    """Angular size of the light (in degrees). Default is 0.53 degrees (approximate sun angle)."""
    is_global: bool = True
    """Whether the light is a global light that is not copied to each environment. For distant light, it must be global."""

    @property
    def quat(self) -> tuple[float, float, float, float]:
        """Quaternion of the light direction. (1, 0, 0, 0)a means the light is pointing towards Z- direction."""
        roll = torch.tensor(self.polar / 180.0 * math.pi)
        pitch = torch.tensor(0.0)
        yaw = torch.tensor(self.azimuth / 180.0 * math.pi)
        return tuple(quat_from_euler_xyz(roll, pitch, yaw).squeeze(0).tolist())

    def __post_init__(self):
        """Post-initialization hook to check if the light is global."""
        if not self.is_global:
            log.warning("Distant light must be global, overriding the value.")
            self.is_global = True


@configclass
class CylinderLightCfg(BaseLightCfg):
    """Configuration for a cylinder light."""

    length: float = 1.0
    """Length of the cylinder (in m). Default is 1.0m."""
    radius: float = 0.5
    """Radius of the cylinder (in m). Default is 0.5m."""
    treat_as_line: bool = False
    """Treats the cylinder as a line source, i.e. a zero-radius cylinder. Default is False."""
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position of the cylinder (in m). Default is (0.0, 0.0, 0.0)."""
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Orientation of the cylinder. Default is (1.0, 0.0, 0.0, 0.0)."""


@configclass
class DomeLightCfg(BaseLightCfg):
    """Configuration for a dome light. Provides uniform lighting from all directions, simulating sky lighting."""

    texture_file: str | None = None
    """Path to HDR texture file for environment lighting. If None, uses uniform color."""
    texture_format: str = "automatic"
    """The parametrization format of the color map file. Default is 'automatic'."""
    visible_in_primary_ray: bool = True
    """Whether the dome light is visible in the primary ray. Default is True."""
    is_global: bool = True
    """Whether the light is a global light that is not copied to each environment. For dome light, it should be global."""

    def __post_init__(self):
        """Post-initialization hook to check if the light is global."""
        if not self.is_global:
            log.warning("Dome light should be global, overriding the value.")
            self.is_global = True


@configclass
class SphereLightCfg(BaseLightCfg):
    """Configuration for a sphere light. Emits light from a spherical area."""

    radius: float = 0.5
    """Radius of the sphere light (in m). Default is 0.5m."""
    treat_as_point: bool = False
    """Treats the sphere as a point source, i.e. a zero-radius sphere. Default is False."""
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position of the sphere light (in m). Default is (0.0, 0.0, 0.0)."""


@configclass
class DiskLightCfg(BaseLightCfg):
    """Configuration for a disk light. Emits light from a circular disk area."""

    radius: float = 1.0
    """Radius of the disk (in m). Default is 1.0m."""
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position of the disk light (in m). Default is (0.0, 0.0, 0.0)."""
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Orientation of the disk. Default is (1.0, 0.0, 0.0, 0.0) (pointing down)."""
    normalize: bool = True
    """Whether to normalize the light intensity based on the disk area."""
