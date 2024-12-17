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


# Until this point all the data is in numpy
def train_assimilate(input_image_4x_upscaled_2x, target_image_2x, stations_image_resized):
    import torch
    """# Diffusion"""
    
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torch.optim import AdamW
    from diffusers import DDPMScheduler, UNet2DModel
    from copy import deepcopy
    
    class ncDataset(Dataset):
        def __init__(self, data, targets, targets_z):
            self.data = data
            self.targets = targets
            self.targets_z = targets_z
    
        def __getitem__(self, index):
            # Convert to tensor if the data is not already a tensor
            x = self.data[index]
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x).unsqueeze(0)
            else:
                x = x.unsqueeze(0)
    
            y = self.targets[index]
            if not isinstance(y, torch.Tensor):
                y = torch.from_numpy(y).unsqueeze(0)
            else:
                y = y.unsqueeze(0)
    
            z = self.targets_z[index]
            if not isinstance(z, torch.Tensor):
                z = torch.from_numpy(z).unsqueeze(0)
            else:
                z = z.unsqueeze(0)
    
            return x, y, z
    
        def __len__(self):
            return len(self.data)
    print("Resized stations image shape:", stations_image_resized.shape)
    
    # Convert the images to PyTorch tensors (multi-channel)
    input_image_4x_upscaled_2x_tensor = torch.tensor(input_image_4x_upscaled_2x, dtype=torch.float32)
    target_image_2x_tensor = torch.tensor(target_image_2x, dtype=torch.float32)
    stations_image_resized_tensor = torch.tensor(stations_image_resized, dtype=torch.float32)
    
    # Verify the shapes of the tensors
    print(f"Input tensor shape: {input_image_4x_upscaled_2x_tensor.shape}")
    print(f"Target tensor shape: {target_image_2x_tensor.shape}")
    print(f"Stations tensor shape: {stations_image_resized_tensor.shape}")
    
    print(np.isnan(stations_image).sum())
    
    print(np.isnan(stations_image_resized).sum())
    
    print(stations_image.shape)
    print(stations_image_resized.shape)
    
    print(torch.isnan(stations_image_resized_tensor).sum())
    
    
    
    
    print(torch.nansum(stations_image_resized_tensor[0,:,:]))
    
    
    print(stations_image_resized_tensor.dtype)
    
    test_tensor = torch.tensor([[float('nan'), 1.0], [0.5, float('nan')]])
    print(test_tensor)
    print(torch.isnan(test_tensor))
    
    
    ########################
    import torch
    
    def split_data(x_patches, y_patches, z_patches):
        # Get total length of the data
        total_samples = x_patches.shape[0]
    
        # Calculate split indices
        train_end = int(0.6 * total_samples)  # 60% for training
        val_end = int(0.8 * total_samples)    # 20% for validation, 80% cumulative
    
        # Training data
        x_train_patches = x_patches[:train_end]
        y_train_patches = y_patches[:train_end]
        z_train_patches = z_patches[:train_end]
    
        # Validation data
        x_val_patches = x_patches[train_end:val_end]
        y_val_patches = y_patches[train_end:val_end]
        z_val_patches = z_patches[train_end:val_end]
    
        # Test data
        x_test_patches = x_patches[val_end:]
        y_test_patches = y_patches[val_end:]
        z_test_patches = z_patches[val_end:]
    
        return (x_train_patches, y_train_patches, z_train_patches,
                x_val_patches, y_val_patches, z_val_patches,
                x_test_patches, y_test_patches, z_test_patches)
    
    # Normalize the input tensors
    x_train_patches = input_image_4x_upscaled_2x_tensor
    y_train_patches = target_image_2x_tensor
    z_train_patches = stations_image_resized_tensor
    
    x_train_max = x_train_patches.max()
    y_train_max = y_train_patches.max()
    z_train_max = torch_nanmax(z_train_patches)
    
    x_train_patches /= x_train_max
    y_train_patches /= y_train_max
    z_train_patches /= z_train_max
    
    # Split the data into train, val, and test
    (x_train_patches, y_train_patches, z_train_patches,
     x_val_patches, y_val_patches, z_val_patches,
     x_test_patches, y_test_patches, z_test_patches) = split_data(
        x_train_patches, y_train_patches, z_train_patches
    )
    
    # Verify the shapes
    print(f"Train shapes: {x_train_patches.shape}, {y_train_patches.shape}, {z_train_patches.shape}")
    print(f"Val shapes: {x_val_patches.shape}, {y_val_patches.shape}, {z_val_patches.shape}")
    print(f"Test shapes: {x_test_patches.shape}, {y_test_patches.shape}, {z_test_patches.shape}")
    
    ########################
    
    
    # x_train_patches = input_image_4x_upscaled_2x_tensor
    # y_train_patches = target_image_2x_tensor
    # z_train_patches = stations_image_resized_tensor
    
    # x_train_max = x_train_patches.max()
    # y_train_max = y_train_patches.max()
    # z_train_max = torch_nanmax(z_train_patches)
    # x_train_patches /= x_train_max
    # y_train_patches /= y_train_max
    # z_train_patches /= z_train_max
    
    # x_val_patches = x_train_patches[60:80]
    # y_val_patches = y_train_patches[60:80]
    # z_val_patches = z_train_patches[60:80]
    
    # x_test_patches = x_train_patches[80:]
    # y_test_patches = y_train_patches[80:]
    # z_test_patches = z_train_patches[80:]
    
    # x_train_patches = x_train_patches[:60]
    # y_train_patches = y_train_patches[:60]
    # z_train_patches = z_train_patches[:60]
    
    # torch.nansum(stations_image_resized_tensor)
    
    # Dataset Preparation
    train_dataset = ncDataset(x_train_patches, y_train_patches, z_train_patches)
    val_dataset = ncDataset(x_val_patches, y_val_patches, z_val_patches)
    test_dataset = ncDataset(x_test_patches, y_test_patches, z_test_patches)
    train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=20, shuffle=True)
    
    from torch.utils.tensorboard import SummaryWriter
    # Test dataset DataLoader
    test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=False)
    
    
    
    
    
    
    
    def train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device):
        model.train()
        train_loss = 0.0
        train_loss_z = 0.0  # For additional loss
    
        for batch in train_dataloader:
            lr, hr, zr = batch  # Low resolution (lr), high resolution (hr), and z
            lr, hr, zr = lr.to(device), hr.to(device), zr.to(device)
    
            # Sample noise to add to the high-resolution images
            noise = torch.randn_like(hr)
    
            # Sample a random timestep for each image
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (lr.shape[0],)).long().to(device)
    
            # Get the noisy high-resolution images
            noisy_images = scheduler.add_noise(hr, noise, timesteps)
    
            # Concatenate the low-resolution and noisy high-resolution images
            x_t = torch.cat((noisy_images, lr), dim=1)
    
            # Get the model's prediction (predicted noise)
            noise_pred = model(x_t, timesteps).sample
    
            # Predict the high-resolution image (denoised output)
            pred_hr = []
            for i in range(lr.shape[0]):  # Iterate over the batch
                scheduler_output = scheduler.step(
                    model_output=noise_pred[i:i+1],  # Predicted noise for the current sample
                    timestep=timesteps[i],           # Timestep for the current sample
                    sample=noisy_images[i:i+1]       # Noisy image for the current sample
                )
                pred_hr.append(scheduler_output.prev_sample)
    
            # Stack predictions to form the batch
            pred_hr = torch.cat(pred_hr, dim=0)
    
            # Compute the noise prediction loss
            loss = criterion(noise_pred, noise)
    
            # Additional loss between predicted hr and z (zr)
            loss_z = masked_mse_loss(pred_hr, zr)
    
            # Combine the losses
            total_loss = loss + loss_z
            train_loss += loss.item()
            train_loss_z += loss_z.item()
    
            # Backpropagation and optimization
            optimizer.zero_grad()
            #total_loss.backward()
            loss_z.backward()
            optimizer.step()
    
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
        train_loss /= len(train_dataloader)
        train_loss_z /= len(train_dataloader)
    
        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_loss_z = 0.0  # For additional loss
    
        with torch.no_grad():
            for batch in val_dataloader:
                lr, hr, zr = batch
                lr, hr, zr = lr.to(device), hr.to(device), zr.to(device)
    
                # Sample noise to add to the high-resolution images
                noise = torch.randn_like(hr)
    
                # Sample a random timestep for each image
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (lr.shape[0],)).long().to(device)
    
                # Get the noisy high-resolution images
                noisy_images = scheduler.add_noise(hr, noise, timesteps)
    
                # Concatenate the low-resolution and noisy high-resolution images
                x_t = torch.cat((noisy_images, lr), dim=1)
    
                # Get the model's prediction (predicted noise)
                noise_pred = model(x_t, timesteps).sample
    
                # Predict the high-resolution image (denoised output)
                pred_hr = []
                for i in range(lr.shape[0]):  # Iterate over the batch
                    scheduler_output = scheduler.step(
                        model_output=noise_pred[i:i+1],  # Predicted noise for the current sample
                        timestep=timesteps[i],           # Timestep for the current sample
                        sample=noisy_images[i:i+1]       # Noisy image for the current sample
                    )
                    pred_hr.append(scheduler_output.prev_sample)
    
                # Stack predictions to form the batch
                pred_hr = torch.cat(pred_hr, dim=0)
    
                # Compute the noise prediction loss
                loss = criterion(noise_pred, noise)
    
                # Additional loss between predicted hr and z (zr)
                loss_z = masked_mse_loss(pred_hr, zr)
    
                # Combine the losses
                val_loss += loss.item()
                val_loss_z += loss_z.item()
    
        val_loss /= len(val_dataloader)
        val_loss_z /= len(val_dataloader)
    
        return train_loss, val_loss, train_loss_z, val_loss_z
    
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model setup
    patch_size = 32
    model = UNet2DModel(
        sample_size=patch_size,  # Input image size
        in_channels=2,           # Input channels (low resolution + noisy high resolution)
        out_channels=1,          # Output channels (high resolution)
        layers_per_block=4,
        block_out_channels=(64, 128, 256, 512),
    ).to(device)
    
    # Scheduler setup
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,  # Number of diffusion steps
        beta_schedule="linear"
    )
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=1e-4)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Hyperparameters
    num_epochs = 1
    print_interval = 10
    patience = 50
    best_val_loss = float('inf')
    counter = 0
    best_model = None
    
    # Dataloaders (assuming x_train_patches, y_train_patches, z_train_patches, etc., are defined)
    train_dataset = ncDataset(x_train_patches, y_train_patches, z_train_patches)
    val_dataset = ncDataset(x_val_patches, y_val_patches, z_val_patches)
    
    train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=20, shuffle=True)
    
    # Path to save the trained model
    diffusion_model_save_path = "best_diffusion_model.pth"
    # Main Training Loop
    for epoch in range(1, num_epochs + 1):
        train_loss, val_loss, train_loss_z, val_loss_z = train(
            model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device
        )
    
        # Log the losses
        print(f"Epoch [{epoch}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Loss Z: {train_loss_z:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Loss Z: {val_loss_z:.4f}")
    
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = deepcopy(model)
            counter = 0
            # Save the best model
            torch.save(best_model.state_dict(), diffusion_model_save_path)
            print(f"Best model saved at epoch {epoch}")
        else:
            counter += 1
    
        if counter >= patience:
            print("Early stopping triggered.")
            break
    
    print("Training Complete.")
    torch.save(model.state_dict(), diffusion_model_save_path)
    print(f"Final model saved to {diffusion_model_save_path}")
    return True

train_assimilate(input_image_4x_upscaled_2x, target_image_2x, stations_image_resized)
# Save the final model after training
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from diffusers import DDPMScheduler, UNet2DModel
from copy import deepcopy
# Load the trained model
patch_size = 32
loaded_diffusion_model = UNet2DModel(
    sample_size=patch_size,
    in_channels=2,
    out_channels=1,
    layers_per_block=4,
    block_out_channels=(64, 128, 256, 512),
).to(device)

# Load weights into the model
loaded_diffusion_model.load_state_dict(torch.load(diffusion_model_save_path))
loaded_diffusion_model.eval()
print(f"Model loaded from {diffusion_model_save_path}")

# Set model to evaluation mode
model.eval()

# Collect all predictions
all_inputs = []
all_predictions = []
all_targets = []

# Scheduler for noise handling
scheduler.set_timesteps(num_inference_steps=1000)  # Ensure proper timestep range for inference

# Loop through the training data
with torch.no_grad():
    for lr, hr, _ in train_dataloader:
        lr = lr.to(device)  # Move input to device
        hr = hr.to(device)  # Move target to device

        # Initialize predictions for the batch
        sr_batch = []

        # Start from pure noise
        noisy_images = torch.randn_like(hr)

        for t in reversed(scheduler.timesteps):  # Loop through timesteps in reverse order
            # Concatenate low-resolution input with noisy images
            x_t = torch.cat((noisy_images, lr), dim=1)

            # Predict noise using the model
            noise_pred = model(x_t, t).sample

            # Perform the scheduler step
            noisy_images = scheduler.step(noise_pred, t, noisy_images).prev_sample

        # Append the final denoised images to the batch predictions
        sr_batch.append(noisy_images)

        # Collect the inputs, predictions, and targets
        all_inputs.append(lr.cpu().numpy())  # Convert back to CPU
        all_predictions.append(torch.cat(sr_batch, dim=0).cpu().numpy())  # Convert back to CPU
        all_targets.append(hr.cpu().numpy())  # Convert back to CPU

# Concatenate all inputs, predictions, and targets
all_inputs = np.concatenate(all_inputs, axis=0)
all_predictions = np.concatenate(all_predictions, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

# Print the shapes
print("Shape of Training Input (Low-Resolution):", all_inputs.shape)
print("Shape of Predicted Target (Super-Resolved):", all_predictions.shape)
print("Shape of Ground Truth Target (High-Resolution):", all_targets.shape)
