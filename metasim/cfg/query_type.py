from dataclasses import dataclass


@dataclass
class SitePos:
    """Return worldframe (x,y,z) position of a given site."""

    site: str


@dataclass
class ContactForce:
    """Return 6D contact force/torque measured by a named sensor."""

    sensor_name: str


@dataclass
class SensorData:
    """Return a named sensor."""

    name: str


@dataclass
class SiteXMat:
    """World frame 3*3 rotation matrix of one site (flattened to 9)."""

    name: str


@dataclass
class BodyMass:
    """Scalar subtree mass of a body."""

    name: str
