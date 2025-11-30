# deepassimilate/__init__.py
"""
deepassimilate

Two-step library for:
1) Unconditional diffusion model training (using diffusers + PyTorch).
2) Generative data assimilation on top of the trained model.
"""

from .training.uncond_trainer import UncondTrainConfig, train_unconditional
from .assimilation.pipeline import run_data_assimilation
from .assimilation.da_posterior import DAConfig
from .assimilation.observation_ops import ObservationOperator, IdentityObservationOperator

# Import utility functions
from .utils import get_device, get_device_info, print_device_info

# Import dataset utilities
from .datasets import WeatherDataset, GriddedDataset

# Version
__version__ = "0.1.0"


__all__ = [
    "__version__",
    "UncondTrainConfig",
    "train_unconditional",
    "run_data_assimilation",
    "DAConfig",
    "ObservationOperator",
    "IdentityObservationOperator",
    "get_device",
    "get_device_info",
    "print_device_info",
    "WeatherDataset",
    "GriddedDataset",
]
