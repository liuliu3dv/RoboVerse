from __future__ import annotations

from metasim.utils.configclass import configclass

from .base_scene_cfg import SceneCfg


@configclass
class Kujiale0020Cfg(SceneCfg):
    """Config class for kujiale scene"""

    name: str = "kujiale_0020"
    usd_path: str = "roboverse_data/scenes/manycore/kujiale_0020/usd/kujiale_0020_pour_test.usd"
    positions: list[tuple[float, float, float]] = [
        (2.60, 3.17, -0.32),  # ? <- (-260, -317, 32)
        # (0.62354, -1.77833, -0.73064),  # (0.62, -1.78, -0.74) <- (-62, 73, -178)
    ]  # XXX: only positions are randomized for now
    default_position: tuple[float, float, float] = (2.60, 3.17, -0.32)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    scale: tuple[float, float, float] = (0.01, 0.01, 0.01)
