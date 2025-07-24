from __future__ import annotations

from metasim.utils.configclass import configclass

from .base_scene_cfg import SceneCfg


@configclass
class Kujiale0021Cfg(SceneCfg):
    """Config class for kujiale scene"""

    name: str = "kujiale_0020"
    usd_path: str = "roboverse_data/scenes/manycore/kujiale_0021/usd/kujiale_0021_flatten.usd"
    positions: list[tuple[float, float, float]] = [
        (-9.59, 2.05, -0.85),  # ? <- (959, -205, 85)
        # (0.62354, -1.77833, -0.73064),  # (0.62, -1.78, -0.74) <- (-62, 73, -178)
    ]  # XXX: only positions are randomized for now
    default_position: tuple[float, float, float] = (-10.16, 2.25, -0.85)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    scale: tuple[float, float, float] = (0.01, 0.01, 0.01)
