import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.optim as optim
from copy import deepcopy
from sklearn.metrics import r2_score

# Function to generate random noise images
def generate_random_image(seed, img_size=(256, 256)):
    np.random.seed(seed)
    random_image = np.random.rand(*img_size)
    return random_image

# Bicubic interpolation for downscaling
def bicubic_downscale(image, scale_factor):
    height, width = image.shape
    new_size = (int(width // scale_factor), int(height // scale_factor))
    downscaled_image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
    return downscaled_image

# Bicubic upscaling to original size
def bicubic_upscale(image, original_size):
    upscaled_image = cv2.resize(image, original_size, interpolation=cv2.INTER_CUBIC)
    return upscaled_image

# Function to create "stations" image with 90% data missing
def create_stations_image(original_image, gap_ratio=0.9):
    mask = np.random.rand(*original_image.shape) < gap_ratio
    stations_image = original_image.copy()
    stations_image[mask] = np.nan  # Mask out 90% of the data
    return stations_image, mask  # Also return the mask for loss calculation

def nearest_neighbor_resize(image, target_size):
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)
    return resized_image

# Function to generate a random image with multiple channels (e.g., 3 for RGB)
def generate_random_image(seed, img_size, num_channels):
    np.random.seed(seed)
    # Generate random image with shape (C, H, W), where C is the number of channels
    return np.random.rand(num_channels, img_size[0], img_size[1])

# Function to downscale an image (multi-channel supported)
def bicubic_downscale(image, scale_factor):
    # Resize each channel using bicubic interpolation
    channels = [cv2.resize(image[c], (image.shape[2] // scale_factor, image.shape[1] // scale_factor), interpolation=cv2.INTER_CUBIC) for c in range(image.shape[0])]
    return np.stack(channels, axis=0)

# Function to upscale an image (multi-channel supported)
def bicubic_upscale(image, target_size):
    # Resize each channel using bicubic interpolation
    channels = [cv2.resize(image[c], target_size, interpolation=cv2.INTER_CUBIC) for c in range(image.shape[0])]
    return np.stack(channels, axis=0)

# Function to create the stations image with missing data (set missing values to NaN)
def create_stations_image(image, gap_ratio):
    mask = np.random.rand(*image.shape) > gap_ratio
    stations_image = np.where(mask, image, np.nan)  # Set values to NaN where mask is False
    return stations_image, mask

def nearest_neighbor_resize_with_nan(image, target_size):
    # Create a mask for NaNs
    nan_mask = np.isnan(image)

    # Replace NaNs with a placeholder value (e.g., 0) before resizing
    image_filled = np.where(nan_mask, 0, image)

    # Perform nearest neighbor resizing
    channels = [cv2.resize(image_filled[c], target_size, interpolation=cv2.INTER_NEAREST) for c in range(image_filled.shape[0])]

    # Resize the mask separately
    resized_nan_mask = [cv2.resize(nan_mask[c].astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST) for c in range(nan_mask.shape[0])]

    # Stack channels and mask
    resized_image = np.stack(channels, axis=0)
    resized_nan_mask = np.stack(resized_nan_mask, axis=0).astype(bool)

    # Restore NaN values in the resized image
    resized_image_with_nan = np.where(resized_nan_mask, np.nan, resized_image)

    return resized_image_with_nan
def torch_nanmax(tensor):
    # Replace NaN values with a very large negative number (-inf)
    tensor_no_nan = torch.where(torch.isnan(tensor), torch.tensor(float('-inf'), device=tensor.device), tensor)

    # Apply the max function
    return torch.max(tensor_no_nan)
def masked_mse_loss(output, target):
    # Create a mask for non-NaN values in both the target and output tensors
    mask = ~torch.isnan(target) & ~torch.isnan(output)

    # Apply the mask to both the output and target
    masked_output = output[mask]
    masked_target = target[mask]
    # print('sum of target = ', torch.nansum(target))
    # Diagnostic: Print how many valid (non-NaN) elements remain
    # print(f'Valid elements for loss calculation: {masked_target.numel()}')
    # print(torch.nansum(target))

    # Check if the mask has selected any valid (non-NaN) elements
    if masked_output.numel() == 0:  # No valid elements to compute loss
        return torch.tensor(0.0, device=output.device)  # Return a zero loss if there are no valid elements

    # Compute MSE loss only on valid (non-NaN) elements
    return nn.functional.mse_loss(masked_output, masked_target)

def calculate_r2(model, test_dataloader, device):
    model.eval()
    all_sr = []
    all_hr = []
    with torch.no_grad():
        for batch in test_dataloader:
            lr, hr, _ = batch  # Only use low-resolution (lr) and high-resolution target (hr)
            lr, hr = lr.to(device), hr.to(device)

            sr = model(lr)  # Predicted super-resolved image

            # Collect outputs and targets for R² calculation
            all_sr.append(sr.cpu().numpy())
            all_hr.append(hr.cpu().numpy())

    # Convert to NumPy arrays
    all_sr = np.concatenate(all_sr, axis=0).reshape(-1)
    all_hr = np.concatenate(all_hr, axis=0).reshape(-1)

    # Compute R²
    return r2_score(all_hr, all_sr)

