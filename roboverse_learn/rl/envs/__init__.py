"""Environment wrappers with lazy imports to avoid simulator conflicts"""

from __future__ import annotations

import importlib
from typing import Any

# Dictionary to store wrapper classes once imported
_wrapper_classes: dict[str, Any] = {}


def _import_wrapper(name: str) -> Any:
    """Dynamically import a wrapper class when requested"""
    if name in _wrapper_classes:
        return _wrapper_classes[name]

    if name == "IsaacGymWrapper":
        module = importlib.import_module("roboverse_learn.rl.envs.isaacgym_wrapper")
        cls = getattr(module, name)
    elif name == "MujocoWrapper":
        module = importlib.import_module("roboverse_learn.rl.envs.mujoco_wrapper")
        cls = getattr(module, name)
    elif name == "IsaacLabWrapper":
        module = importlib.import_module("roboverse_learn.rl.envs.isaaclab_wrapper")
        cls = getattr(module, name)
    else:
        raise ImportError(f"Unknown wrapper: {name}")

    _wrapper_classes[name] = cls
    return cls


# Define properties for lazy-loading the wrapper classes
def __getattr__(name: str) -> Any:
    """Lazy load wrapper classes only when requested"""
    if name in ["IsaacGymWrapper", "MujocoWrapper", "IsaacLabWrapper"]:
        return _import_wrapper(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Define what's available for import from this module
__all__ = ["IsaacGymWrapper", "IsaacLabWrapper", "MujocoWrapper"]
