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

# Dataset definition remains the same
class ncDataset(Dataset):
    def __init__(self, data, targets, stations):
        # Assuming data, targets, and stations are already PyTorch tensors
        self.data = data
        self.targets = targets
        self.stations = stations

    def __getitem__(self, index):
        x = self.data[index].unsqueeze(0)  # No need for torch.from_numpy()
        y = self.targets[index].unsqueeze(0)  # Already a tensor
        z = self.stations[index].unsqueeze(0)  # Already a tensor
        return x, y, z

    def __len__(self):
        return len(self.data)


