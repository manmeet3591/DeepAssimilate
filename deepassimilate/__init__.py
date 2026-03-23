# deepassimilate/__init__.py
"""
deepassimilate

Three-step library for diffusion-based generative data assimilation:
  1) Architecture search (NAS) to find the best diffusion model
  2) Unconditional diffusion model training (using diffusers + PyTorch)
  3) Score-based data assimilation with sparse observations
"""

# Step 1: NAS
from .nas import search_architecture, NASConfig, NASResult

# Step 2: Training
from .training.uncond_trainer import UncondTrainConfig, train_unconditional

# Step 3: Data Assimilation
from .assimilation.pipeline import run_data_assimilation
from .assimilation.score import SDAConfig, score_based_assimilation
from .assimilation.observation_ops import (
    ObservationOperator,
    IdentityObservationOperator,
    MaskedObservationOperator,
    make_random_mask,
    make_station_mask,
)

# Backward compat
from .assimilation.da_posterior import DAConfig

# Utilities
from .utils import get_device, get_device_info, print_device_info

# Datasets
from .datasets import WeatherDataset, GriddedDataset

# Schedulers
from .schedulers import build_scheduler, build_distilled_scheduler
from .schedulers.factory import list_schedulers

# Models
from .models import build_unet_2d, build_model_from_config, count_parameters, list_presets

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # Step 1: NAS
    "search_architecture",
    "NASConfig",
    "NASResult",
    # Step 2: Training
    "UncondTrainConfig",
    "train_unconditional",
    # Step 3: DA
    "run_data_assimilation",
    "SDAConfig",
    "score_based_assimilation",
    "ObservationOperator",
    "IdentityObservationOperator",
    "MaskedObservationOperator",
    "make_random_mask",
    "make_station_mask",
    "DAConfig",
    # Utils
    "get_device",
    "get_device_info",
    "print_device_info",
    # Datasets
    "WeatherDataset",
    "GriddedDataset",
    # Schedulers
    "build_scheduler",
    "build_distilled_scheduler",
    "list_schedulers",
    # Models
    "build_unet_2d",
    "build_model_from_config",
    "count_parameters",
    "list_presets",
]
