from __future__ import annotations

from metasim.utils.configclass import configclass

from .base_scene_cfg import SceneCfg


@configclass
class Kujiale0003Cfg(SceneCfg):
    """Config class for kujiale 0003 scene"""

    name = "kujiale_0003"
    usd_path = "roboverse_data/scenes/manycore/kujiale_0003/usd/kujiale_0003_flatten.usd"
    default_position = (0.0, 0.0, 0.0)
    quat = (1.0, 0.0, 0.0, 0.0)
    scale = (0.01, 0.01, 0.01)
