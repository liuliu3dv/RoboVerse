"""Base class for domain randomization components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from loguru import logger as log


class BaseRandomizer(ABC):
    """Base class for all domain randomization components."""

    def __init__(self, seed: int | None = None, enabled: bool = True, **kwargs):
        """Initialize the base randomizer.

        Args:
            seed: Random seed for reproducibility
            enabled: Whether randomization is enabled
            **kwargs: Additional configuration parameters
        """
        self.enabled = enabled
        self.config = kwargs

        # Set random seeds
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        log.info(f"Initialized {self.__class__.__name__} (enabled={self.enabled})")

    @abstractmethod
    def randomize(self, scenario_cfg: Any, env_ids: list | None = None, **kwargs) -> None:
        """Apply randomization to the scenario configuration.

        Args:
            scenario_cfg: The scenario configuration containing scene elements to randomize
            env_ids: List of environment IDs to randomize. If None, randomize all environments.
            **kwargs: Additional parameters for randomization
        """
        pass

    def enable(self) -> None:
        """Enable randomization."""
        self.enabled = True
        log.info(f"Enabled {self.__class__.__name__}")

    def disable(self) -> None:
        """Disable randomization."""
        self.enabled = False
        log.info(f"Disabled {self.__class__.__name__}")

    def update_config(self, **kwargs) -> None:
        """Update configuration parameters."""
        self.config.update(kwargs)
        log.debug(f"Updated {self.__class__.__name__} config: {kwargs}")
