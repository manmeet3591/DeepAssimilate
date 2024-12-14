# -*- coding: utf-8 -*-
"""
Example usage of DeepAssimilate package
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from copy import deepcopy
from deepassimilate.models import SRCNN, masked_mse_loss
from deepassimilate.dataset import ncDataset
from deepassimilate.utils import (
    generate_random_image,
    bicubic_downscale,
    bicubic_upscale,
    nearest_neighbor_resize_with_nan,
)

# Generate random input image
seed = 42
img_size = (256, 256)
num_channels = 1
original_image = generate_random_image(seed, img_size)

# Input Image: 4x downscaled and 2x upscaled
input_image_4x = bicubic_downscale(original_image, 4)
input_image_4x_upscaled_2x = bicubic_upscale(
    input_image_4x, (int(original_image.shape[0] / 2), int(original_image.shape[1] / 2))
)

# Target Image: 2x downscaled
target_image_2x = bicubic_downscale(original_image, 2)

# Create "stations" image (ground truth with 90% missing data)
stations_image, mask = nearest_neighbor_resize_with_nan(
    original_image, (target_image_2x.shape[1], target_image_2x.shape[0])
)

# Convert images to PyTorch tensors
input_image_4x_upscaled_2x_tensor = torch.tensor(input_image_4x_upscaled_2x, dtype=torch.float32)
target_image_2x_tensor = torch.tensor(target_image_2x, dtype=torch.float32)
stations_image_resized_tensor = torch.tensor(stations_image, dtype=torch.float32)

# Dataset Preparation
x_train_patches = input_image_4x_upscaled_2x_tensor
y_train_patches = target_image_2x_tensor
z_train_patches = stations_image_resized_tensor

train_dataset = ncDataset(x_train_patches, y_train_patches, z_train_patches)
train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)

# Initialize model, criterion, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SRCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
print_interval = 2

for epoch in range(1, num_epochs + 1):
    model.train()
    train_loss = 0.0
    for batch in train_dataloader:
        lr, hr, station = batch
        lr, hr, station = lr.to(device), hr.to(device), station.to(device)
        optimizer.zero_grad()

