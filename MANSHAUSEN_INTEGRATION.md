# Manshausen Data Assimilation Integration

This document describes the integration of the Manshausen et al. diffusion-based data assimilation approach into the DeepAssimilate library.

## Overview

The Manshausen approach implements zero-shot data assimilation using diffusion models. It:
1. Trains a diffusion model (UNet2DModel) on gridded data
2. Integrates sparse observations during inference using a likelihood-based correction step

## What Was Added

### 1. New Module: `deepassimilate/manshausen_da.py`

This module contains the core implementation:

- **Diffusion Schedule Functions:**
  - `alpha(t, eta)`: Computes alpha for the diffusion schedule
  - `mu(t, eta)`: Computes mu for the diffusion schedule  
  - `sigma(t, eta)`: Computes sigma for the diffusion schedule

- **Observation Correction:**
  - `corrected_noise(x, obs, t, std, gamma, eta)`: Applies likelihood-based correction during sampling to integrate observations

- **Training:**
  - `train_diffusion_model(...)`: Trains a UNet2DModel on gridded data
  - `DiffusionDataset`: Dataset class for diffusion model training

- **Inference:**
  - `sample_with_observations(...)`: Generates samples with observation correction

### 2. Updated `deepassimilate/deepassimilate.py`

Added a new `ManshausenDA` class that provides a clean API:

```python
from deepassimilate import ManshausenDA

# Initialize
da = ManshausenDA(device='cuda')

# Train on gridded data
da.train(
    data=train_data,  # [N, H, W] numpy array
    val_data=val_data,  # Optional
    sample_size=64,
    n_epochs=10
)

# Assimilate sparse observations
assimilated = da.assimilate(
    observations=sparse_obs,  # [H, W] with NaNs for missing data
    num_inference_steps=50
)
```

### 3. Updated Requirements

Added to `requirements.txt`:
- `xarray`: For handling netCDF data (optional but useful)
- `tqdm`: For progress bars during sampling

### 4. Example Usage

Created `examples/manshausen_example.py` with three examples:
1. Basic numpy array usage
2. Xarray/netCDF data handling
3. Testing different observation densities

## Key Features

1. **Zero-shot Data Assimilation**: No retraining needed when new observations arrive
2. **Sparse Observations**: Handles missing data (NaNs) naturally
3. **Flexible Input**: Works with numpy arrays or xarray datasets
4. **Configurable**: Adjustable observation error (std), regularization (gamma), and inference steps

## Usage from Notebook

The implementation matches your notebook approach:

1. **Training** (from notebook cells 14-26):
   - Normalize data to [0, 1]
   - Train UNet2DModel with HeunDiscreteScheduler
   - Use MSE loss on predicted noise

2. **Inference with Observations** (from notebook cells 37-58):
   - Start from noise
   - Denoise using the trained model
   - Apply `corrected_noise` at each timestep to integrate observations
   - Denormalize output

## Parameters

### Training Parameters
- `sample_size`: Spatial dimensions (H=W, e.g., 64)
- `num_train_timesteps`: Number of diffusion steps (default: 1000)
- `n_epochs`: Training epochs
- `batch_size`: Batch size for training
- `learning_rate`: Optimizer learning rate (default: 1e-4)

### Inference Parameters
- `num_inference_steps`: Denoising steps (default: 50, fewer = faster)
- `std`: Observation error standard deviation (default: 0.5)
- `gamma`: Regularization parameter (default: 0.01)
- `eta`: Numerical stability constant (default: 1e-3)

## Differences from Notebook

1. **Clean API**: Wrapped in classes and functions for easier use
2. **Normalization Handling**: Automatically handles [0,1] ↔ [-1,1] mapping
3. **Error Handling**: Better validation and error messages
4. **Flexibility**: Works with different data shapes and formats

## Next Steps

To use this in your workflow:

1. **Load your data** (numpy or xarray):
   ```python
   import xarray as xr
   ds = xr.open_mfdataset('air.2m*.nc')
   data = ds.air.values  # [time, lat, lon]
   ```

2. **Train the model**:
   ```python
   from deepassimilate import ManshausenDA
   da = ManshausenDA()
   da.train(data[:train_end], val_data=data[val_start:val_end])
   ```

3. **Assimilate observations**:
   ```python
   # Create sparse observations (with NaNs for missing data)
   observations = true_field.copy()
   observations[~observation_mask] = np.nan
   
   # Assimilate
   result = da.assimilate(observations)
   ```

## Files Modified/Created

- ✅ `deepassimilate/manshausen_da.py` (new)
- ✅ `deepassimilate/deepassimilate.py` (updated - added ManshausenDA class)
- ✅ `deepassimilate/__init__.py` (updated - exports new functions)
- ✅ `requirements.txt` (updated - added xarray, tqdm)
- ✅ `examples/manshausen_example.py` (new)

## Testing

You can test the implementation by running:
```bash
python examples/manshausen_example.py
```

Or use it in your own scripts:
```python
from deepassimilate import ManshausenDA
# ... your code ...
```

