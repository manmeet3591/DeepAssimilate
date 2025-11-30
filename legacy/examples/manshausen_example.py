"""
Example usage of Manshausen et al. Diffusion-based Data Assimilation

This example demonstrates how to use the ManshausenDA class to:
1. Train a diffusion model on gridded data
2. Assimilate sparse observations into the gridded field

Based on the approach from:
Manshausen et al. (2024) - Score-based data assimilation using diffusion models.
"""

import numpy as np
import xarray as xr
from deepassimilate import ManshausenDA

# Example 1: Using numpy arrays directly
def example_numpy():
    """Example using numpy arrays (similar to notebook approach)"""
    
    # Generate synthetic gridded data [N, H, W]
    # In practice, this would come from your model output or reanalysis
    np.random.seed(42)
    n_samples = 100
    height, width = 64, 64
    data = np.random.randn(n_samples, height, width) * 10 + 280  # Temperature-like data
    
    # Split into train and validation
    train_data = data[:80]
    val_data = data[80:90]
    test_data = data[90:]
    
    # Initialize Manshausen DA
    da = ManshausenDA(device='cuda')
    
    # Train the diffusion model
    print("Training diffusion model...")
    da.train(
        data=train_data,
        val_data=val_data,
        sample_size=64,
        n_epochs=10,
        batch_size=20,
        save_path='best_manshausen_model.pth'
    )
    
    # Create sparse observations (similar to notebook)
    # In practice, this would be your actual station observations
    true_field = test_data[0]  # [H, W]
    obs_prob = 0.01  # 1% of pixels have observations
    observations = true_field.copy()
    mask = np.random.rand(*true_field.shape) > (1 - obs_prob)
    observations[~mask] = np.nan  # Set missing observations to NaN
    
    # Assimilate observations
    print("Assimilating observations...")
    assimilated = da.assimilate(
        observations=observations,
        num_inference_steps=50,
        std=0.5,
        gamma=0.01
    )
    
    print(f"True field shape: {true_field.shape}")
    print(f"Assimilated shape: {assimilated.shape}")
    print(f"Observations available: {np.sum(~np.isnan(observations))}")
    
    return true_field, observations, assimilated


# Example 2: Using xarray (for netCDF data)
def example_xarray():
    """Example using xarray datasets (for netCDF files)"""
    
    # This is a template - replace with your actual data loading
    # Example: loading from netCDF files
    # ds = xr.open_mfdataset('air.2m*.nc')
    # data = ds.air.values  # [time, lat, lon] or [time, lat, lon]
    
    # For demonstration, create synthetic xarray-like data
    times = np.arange(100)
    lats = np.linspace(30, 40, 64)
    lons = np.linspace(-80, -70, 64)
    
    np.random.seed(42)
    data = np.random.randn(len(times), len(lats), len(lons)) * 10 + 280
    
    # Create xarray dataset
    ds = xr.Dataset(
        {'air': (['time', 'lat', 'lon'], data)},
        coords={'time': times, 'lat': lats, 'lon': lons}
    )
    
    # Extract training data
    train_data = ds.air.isel(time=slice(0, 80)).values  # [80, 64, 64]
    val_data = ds.air.isel(time=slice(80, 90)).values   # [10, 64, 64]
    test_data = ds.air.isel(time=slice(90, 100)).values # [10, 64, 64]
    
    # Initialize and train
    da = ManshausenDA(device='cuda')
    
    print("Training diffusion model on xarray data...")
    da.train(
        data=train_data,
        val_data=val_data,
        sample_size=64,
        n_epochs=10,
        batch_size=20
    )
    
    # Create observations for one time step
    true_field = test_data[0]  # [64, 64]
    obs_prob = 0.05  # 5% observations
    observations = true_field.copy()
    mask = np.random.rand(*true_field.shape) > (1 - obs_prob)
    observations[~mask] = np.nan
    
    # Assimilate
    print("Assimilating observations...")
    assimilated = da.assimilate(
        observations=observations,
        num_inference_steps=50
    )
    
    # Create output dataset
    result_ds = xr.Dataset({
        'true': (['lat', 'lon'], true_field),
        'observations': (['lat', 'lon'], observations),
        'assimilated': (['lat', 'lon'], assimilated[0, 0, :, :])
    }, coords={'lat': lats, 'lon': lons})
    
    return result_ds


# Example 3: Testing different observation densities
def example_observation_density():
    """Test assimilation with different observation densities"""
    
    # Generate data
    np.random.seed(42)
    data = np.random.randn(100, 64, 64) * 10 + 280
    train_data = data[:80]
    val_data = data[80:90]
    test_data = data[90:]
    
    # Train model
    da = ManshausenDA(device='cuda')
    print("Training model...")
    da.train(
        data=train_data,
        val_data=val_data,
        sample_size=64,
        n_epochs=5,  # Fewer epochs for quick demo
        batch_size=20
    )
    
    # Test different observation densities
    true_field = test_data[0]
    obs_densities = [0.01, 0.05, 0.30, 0.50]
    
    results = {}
    for obs_prob in obs_densities:
        print(f"\nTesting with {obs_prob*100:.0f}% observations...")
        observations = true_field.copy()
        mask = np.random.rand(*true_field.shape) > (1 - obs_prob)
        observations[~mask] = np.nan
        
        assimilated = da.assimilate(
            observations=observations,
            num_inference_steps=50
        )
        
        results[obs_prob] = {
            'observations': observations,
            'assimilated': assimilated[0, 0, :, :]
        }
    
    return true_field, results


if __name__ == "__main__":
    print("=" * 60)
    print("Manshausen DA Example")
    print("=" * 60)
    
    # Run numpy example
    print("\n1. Numpy array example:")
    true_field, observations, assimilated = example_numpy()
    
    # Run xarray example (commented out to avoid errors if xarray not available)
    # print("\n2. Xarray example:")
    # result_ds = example_xarray()
    
    # Run observation density test
    print("\n3. Testing different observation densities:")
    true_field, results = example_observation_density()
    
    print("\nExamples completed!")

