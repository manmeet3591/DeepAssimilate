# deepassimilate/assimilation/__init__.py
from .observation_ops import (
    ObservationOperator,
    IdentityObservationOperator,
    MaskedObservationOperator,
    LinearObservationOperator,
    make_random_mask,
    make_station_mask,
)
from .score import SDAConfig, score_based_assimilation, get_mu_sigma_from_scheduler
from .pipeline import run_data_assimilation

# Backward compat
from .da_posterior import DAConfig

__all__ = [
    "ObservationOperator",
    "IdentityObservationOperator",
    "MaskedObservationOperator",
    "LinearObservationOperator",
    "make_random_mask",
    "make_station_mask",
    "SDAConfig",
    "score_based_assimilation",
    "get_mu_sigma_from_scheduler",
    "run_data_assimilation",
    "DAConfig",
]
