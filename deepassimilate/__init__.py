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

__all__ = [
    "UncondTrainConfig",
    "train_unconditional",
    "run_data_assimilation",
    "DAConfig",
    "ObservationOperator",
    "IdentityObservationOperator",
]
