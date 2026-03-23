# deepassimilate/assimilation/observation_ops.py
from abc import ABC, abstractmethod
from typing import Optional

import torch
import numpy as np


class ObservationOperator(ABC):
    """Maps a full state x to observation space H(x).

    Implement forward() for your observation mapping (e.g., masking grid points,
    projecting to a subset, etc.).
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class IdentityObservationOperator(ObservationOperator):
    """H(x) = x. Observations are the full state."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class MaskedObservationOperator(ObservationOperator):
    """Observation operator for sparse gridded observations.

    Given a boolean mask, returns only the observed grid points.
    This is the most common case: station observations on a grid where
    most grid points are unobserved (NaN).

    Args:
        mask: Boolean tensor [C, H, W] or [H, W] where True = observed.
    """

    def __init__(self, mask: torch.Tensor):
        self.mask = mask.bool()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.mask.to(x.device)
        if mask.ndim < x.ndim:
            mask = mask.unsqueeze(0).expand_as(x)
        return x * mask.float()


class LinearObservationOperator(ObservationOperator):
    """Linear observation operator H(x) = Ax.

    For cases where observations are linear combinations of state variables
    (e.g., spatial interpolation, channel mixing).

    Args:
        matrix: Observation matrix [obs_dim, state_dim].
    """

    def __init__(self, matrix: torch.Tensor):
        self.matrix = matrix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        matrix = self.matrix.to(x.device)
        shape = x.shape
        x_flat = x.reshape(shape[0], -1)
        out = x_flat @ matrix.T
        return out.reshape(shape)


def make_random_mask(shape, obs_fraction, seed=None):
    """Create a random observation mask.

    Args:
        shape: Tuple for mask shape, e.g. (1, 32, 32) or (32, 32).
        obs_fraction: Fraction of points to observe (0.0 to 1.0).
        seed: Optional random seed.

    Returns:
        Boolean tensor where True = observed.
    """
    if seed is not None:
        torch.manual_seed(seed)
    return torch.rand(shape) < obs_fraction


def make_station_mask(lat, lon, station_lats, station_lons, tolerance=1.0):
    """Create observation mask from station locations.

    Args:
        lat: 1D array of grid latitudes.
        lon: 1D array of grid longitudes.
        station_lats: 1D array of station latitudes.
        station_lons: 1D array of station longitudes.
        tolerance: Max distance (in grid units) to snap station to grid.

    Returns:
        Boolean tensor [len(lat), len(lon)] where True = station present.
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    mask = np.zeros((len(lat), len(lon)), dtype=bool)

    for slat, slon in zip(station_lats, station_lons):
        lat_idx = np.argmin(np.abs(lat - slat))
        lon_idx = np.argmin(np.abs(lon - slon))
        if (np.abs(lat[lat_idx] - slat) <= tolerance
                and np.abs(lon[lon_idx] - slon) <= tolerance):
            mask[lat_idx, lon_idx] = True

    return torch.from_numpy(mask)
