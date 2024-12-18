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


from utils import *

# Generate random input image with multiple channels
seed = 42
img_size = (256, 256)
num_channels = 100  # For RGB or more channels if needed
original_image = generate_random_image(seed, img_size, num_channels)

# Input Image: 4x downscaled and 2x upscaled
input_image_4x = bicubic_downscale(original_image, 4)
input_image_4x_upscaled_2x = bicubic_upscale(input_image_4x, (original_image.shape[2] // 2, original_image.shape[1] // 2))

# Target Image: 2x downscaled
target_image_2x = bicubic_downscale(original_image, 2)

print("Input image (4x upscaled to 2x) shape:", input_image_4x_upscaled_2x.shape)
print("Target image (2x downscaled) shape:", target_image_2x.shape)



# Create "stations" image (ground truth with 90% missing data)
stations_image, mask = create_stations_image(original_image, gap_ratio=0.99)

# Nearest neighbor interpolation for resizing stations_image

# Apply the new resizing function
stations_image_resized = nearest_neighbor_resize_with_nan(stations_image, (target_image_2x.shape[1], target_image_2x.shape[2]))



train_assimilate(input_image_4x_upscaled_2x, target_image_2x, stations_image_resized)


all_predictions = assimilate(input_image_4x_upscaled_2x)


# Print the shapes
# print("Shape of Training Input (Low-Resolution):", all_inputs.shape)
print("Shape of Predicted Target (Super-Resolved):", all_predictions.shape)
# print("Shape of Ground Truth Target (High-Resolution):", all_targets.shape)
