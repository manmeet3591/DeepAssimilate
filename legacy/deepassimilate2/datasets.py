# deepassimilate/datasets.py
"""
Dataset utilities for weather/ocean data and other scientific data formats.
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr


class WeatherDataset(Dataset):
    """
    PyTorch Dataset for weather reanalysis data.
    
    Handles xarray DataArrays or numpy arrays with automatic normalization.
    Designed for time-series gridded data (e.g., temperature, precipitation, winds).
    
    Args:
        data: numpy array of shape [time, lat, lon] or xarray DataArray
        normalize: tuple of (min, max) for normalization, or None to compute from data.
                   If None, normalization is computed from the data itself.
        channel_dim: Whether to add a channel dimension (default: True for UNet compatibility)
    
    Examples:
        >>> import xarray as xr
        >>> import deepassimilate as da
        >>> 
        >>> # Load xarray data
        >>> ds = xr.open_dataset("temperature.nc")
        >>> 
        >>> # Create dataset with auto-normalization
        >>> dataset = da.WeatherDataset(ds.temperature)
        >>> 
        >>> # Or with pre-computed normalization
        >>> dataset = da.WeatherDataset(ds.temperature, normalize=(250.0, 310.0))
    """
    
    def __init__(
        self,
        data: Union[np.ndarray, xr.DataArray],
        normalize: Optional[Tuple[float, float]] = None,
        channel_dim: bool = True,
    ):
        # Convert xarray to numpy if needed
        if isinstance(data, xr.DataArray):
            data = data.values
        
        self.data = data.astype(np.float32)
        
        # Compute or use provided normalization
        if normalize is None:
            # Compute normalization from data
            self.min = float(self.data.min())
            self.max = float(self.data.max())
        else:
            self.min, self.max = normalize
        
        # Normalize to [0, 1]
        self.data_norm = (self.data - self.min) / (self.max - self.min + 1e-8)
        
        # Store normalization parameters for denormalization
        self.normalize_params = (self.min, self.max)
        self.channel_dim = channel_dim
        
        # Print info
        print(f"Dataset shape: {self.data_norm.shape}")
        print(f"Value range: [{self.data_norm.min():.3f}, {self.data_norm.max():.3f}] (normalized)")
        print(f"Original range: [{self.min:.2f}, {self.max:.2f}] (original units)")
    
    def __len__(self) -> int:
        """Return number of timesteps in dataset."""
        return len(self.data_norm)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a single timestep.
        
        Returns:
            torch.Tensor: Shape [1, H, W] if channel_dim=True, or [H, W] if False
        """
        # Get single timestep: shape [lat, lon] -> [1, lat, lon] (add channel dim)
        x = torch.from_numpy(self.data_norm[idx])
        
        if self.channel_dim:
            x = x.unsqueeze(0)  # Add channel dimension for UNet compatibility
        
        return x.float()
    
    def denormalize(self, normalized_data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Convert normalized data back to original scale.
        
        Args:
            normalized_data: Data in normalized [0, 1] range
        
        Returns:
            Data in original units
        """
        if isinstance(normalized_data, torch.Tensor):
            return normalized_data * (self.max - self.min) + self.min
        else:
            return normalized_data * (self.max - self.min) + self.min
    
    @property
    def shape(self) -> tuple:
        """Return the shape of the normalized data."""
        return self.data_norm.shape
    
    @property
    def spatial_shape(self) -> tuple:
        """Return the spatial shape (lat, lon) of a single timestep."""
        if len(self.data_norm.shape) == 3:
            return self.data_norm.shape[1:]  # (lat, lon)
        elif len(self.data_norm.shape) == 2:
            return self.data_norm.shape  # (lat, lon)
        else:
            raise ValueError(f"Unexpected data shape: {self.data_norm.shape}")


class GriddedDataset(Dataset):
    """
    More general dataset for gridded data with multiple channels.
    
    Args:
        data: numpy array of shape [time, channels, lat, lon] or [time, lat, lon]
        normalize: dict mapping channel indices to (min, max) tuples, or None
        channel_names: Optional list of channel names for multi-channel data
    """
    
    def __init__(
        self,
        data: Union[np.ndarray, xr.DataArray],
        normalize: Optional[dict] = None,
        channel_names: Optional[list] = None,
    ):
        if isinstance(data, xr.DataArray):
            data = data.values
        
        self.data = data.astype(np.float32)
        
        # Handle normalization per channel if multi-channel
        if len(self.data.shape) == 4:  # [time, channels, lat, lon]
            self.is_multi_channel = True
            self.num_channels = self.data.shape[1]
        elif len(self.data.shape) == 3:  # [time, lat, lon]
            self.is_multi_channel = False
            self.num_channels = 1
            # Add channel dimension
            self.data = self.data[:, np.newaxis, :, :]
        else:
            raise ValueError(f"Expected 3D or 4D data, got shape {self.data.shape}")
        
        self.channel_names = channel_names or [f"channel_{i}" for i in range(self.num_channels)]
        self.normalize = normalize or {}
        
        # Normalize each channel
        self.data_norm = np.zeros_like(self.data)
        self.normalize_params = {}
        
        for i in range(self.num_channels):
            if i in self.normalize:
                min_val, max_val = self.normalize[i]
            else:
                min_val = float(self.data[:, i, :, :].min())
                max_val = float(self.data[:, i, :, :].max())
            
            self.normalize_params[i] = (min_val, max_val)
            self.data_norm[:, i, :, :] = (self.data[:, i, :, :] - min_val) / (max_val - min_val + 1e-8)
        
        print(f"Dataset shape: {self.data_norm.shape}")
        print(f"Number of channels: {self.num_channels}")
        if self.channel_names:
            print(f"Channels: {', '.join(self.channel_names)}")
    
    def __len__(self) -> int:
        return len(self.data_norm)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a single timestep with shape [C, H, W]."""
        return torch.from_numpy(self.data_norm[idx]).float()
    
    def denormalize(self, normalized_data: Union[np.ndarray, torch.Tensor], channel: int = 0) -> Union[np.ndarray, torch.Tensor]:
        """Convert normalized data back to original scale for a specific channel."""
        min_val, max_val = self.normalize_params[channel]
        if isinstance(normalized_data, torch.Tensor):
            return normalized_data * (max_val - min_val) + min_val
        else:
            return normalized_data * (max_val - min_val) + min_val

