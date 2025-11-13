"""
Manshausen et al. Diffusion-based Data Assimilation

This module implements the zero-shot data assimilation approach from:
Manshausen et al. (2024) - Score-based data assimilation using diffusion models.

The approach trains a diffusion model on gridded data and integrates sparse
observations during inference using a correction step.
"""

import math
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, Tuple
from tqdm import tqdm
from diffusers import HeunDiscreteScheduler, UNet2DModel
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


# Diffusion schedule functions
def alpha(t: torch.Tensor, eta: float = 1e-3) -> torch.Tensor:
    """
    Compute alpha(t) for the diffusion schedule.
    
    Args:
        t: Timestep tensor (normalized to [0, 1])
        eta: Small constant for numerical stability
        
    Returns:
        Alpha value at timestep t
    """
    const = math.acos(math.sqrt(eta))
    return torch.cos(const * t) ** 2


def mu(t: torch.Tensor, eta: float = 1e-3) -> torch.Tensor:
    """
    Compute mu(t) for the diffusion schedule.
    
    Args:
        t: Timestep tensor (normalized to [0, 1])
        eta: Small constant for numerical stability
        
    Returns:
        Mu value at timestep t
    """
    return alpha(t, eta)


def sigma(t: torch.Tensor, eta: float = 1e-3) -> torch.Tensor:
    """
    Compute sigma(t) for the diffusion schedule.
    
    Args:
        t: Timestep tensor (normalized to [0, 1])
        eta: Small constant for numerical stability
        
    Returns:
        Sigma value at timestep t
    """
    return (1 - alpha(t, eta) ** 2 + eta ** 2).sqrt()


def corrected_noise(
    x: torch.Tensor,
    obs: torch.Tensor,
    t: torch.Tensor,
    std: float = 0.5,
    gamma: float = 0.01,
    eta: float = 1e-3
) -> torch.Tensor:
    """
    Apply observation correction step during diffusion sampling.
    
    This function implements the likelihood-based correction from Manshausen et al.
    that integrates sparse observations into the diffusion process.
    
    Args:
        x: Current sample tensor [B, C, H, W]
        obs: Observation tensor broadcastable to x (can contain NaNs for missing obs)
        t: Current timestep (tensor or scalar, normalized to [0, 1])
        std: Observation error standard deviation
        gamma: Regularization parameter
        eta: Small constant for numerical stability
        
    Returns:
        Corrected sample tensor
    """
    mu_ = mu(t, eta)
    sigma_ = sigma(t, eta)
    eps = 1e-8  # numerical floor for variance / counts

    # Ensure obs is on the same device/dtype and broadcastable
    obs = obs.to(x.device, dtype=x.dtype)
    if obs.ndim == 2:  # [H, W] -> [1,1,H,W]
        obs = obs.unsqueeze(0).unsqueeze(0)
    elif obs.ndim == 3:  # [B, H, W] -> [B,1,H,W]
        obs = obs.unsqueeze(1)

    # Build a mask of valid (non-NaN) observation pixels
    valid_mask = torch.isfinite(obs)
    # Expand mask to x's shape for channel-wise broadcasting
    while valid_mask.ndim < x.ndim:
        valid_mask = valid_mask.unsqueeze(1)
    valid_mask = valid_mask.expand_as(x)

    # Replace NaNs/Infs in obs with x (so err=0 on invalid spots after masking)
    safe_obs = torch.where(valid_mask, obs.expand_as(x), x)

    with torch.enable_grad():
        x = x.detach().requires_grad_(True)

        # err and variance; only valid entries will contribute via the mask
        err = x - safe_obs

        # variance term (broadcastable); clamp for stability
        var = (std ** 2) + gamma * (sigma_ / (mu_ + eps)) ** 2
        var = torch.as_tensor(var, device=x.device, dtype=x.dtype)
        var = torch.clamp(var, min=eps)

        # Compute masked negative quadratic term. Normalize by #valid to keep scale stable.
        valid_count = valid_mask.sum().to(x.dtype).clamp_min(1.0)
        quad = (err * err) / var  # broadcasted
        # Convert mask to same dtype for multiplication
        log_p = -(quad * valid_mask.to(x.dtype)).sum() / (2.0 * valid_count)

        # s = ∇_x log p(obs | x)
        (s,) = torch.autograd.grad(log_p, x, create_graph=False, retain_graph=False)

    # Small step in the direction of higher likelihood
    update = torch.as_tensor(sigma_, device=x.device, dtype=x.dtype) * s
    x_new = x - update

    # Keep x unchanged where obs was NaN
    x_new = torch.where(valid_mask, x_new, x)

    return x_new


class DiffusionDataset(Dataset):
    """Dataset for diffusion model training on gridded data."""
    
    def __init__(self, data: np.ndarray):
        """
        Args:
            data: Array of shape [N, H, W] or [N, C, H, W] containing gridded data
        """
        self.data = data
        
    def __getitem__(self, index):
        x = torch.from_numpy(self.data[index]).float()
        if x.ndim == 2:  # [H, W] -> [1, H, W]
            x = x.unsqueeze(0)
        return x

    def __len__(self):
        return len(self.data)


def train_diffusion_model(
    data: np.ndarray,
    val_data: Optional[np.ndarray] = None,
    sample_size: int = 64,
    in_channels: int = 1,
    out_channels: int = 1,
    num_train_timesteps: int = 1000,
    batch_size: int = 20,
    n_epochs: int = 10,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    device: str = 'cuda',
    save_path: str = 'best_manshausen_model.pth',
    early_stopping_patience: int = 5000,
    print_step: int = 1
) -> UNet2DModel:
    """
    Train a diffusion model on gridded data following Manshausen et al. approach.
    
    Args:
        data: Training data array of shape [N, H, W] or [N, C, H, W]
        val_data: Validation data array (optional)
        sample_size: Spatial size of the data (H and W should match this)
        in_channels: Number of input channels
        out_channels: Number of output channels
        num_train_timesteps: Number of diffusion timesteps
        batch_size: Batch size for training
        n_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        device: Device to train on ('cuda' or 'cpu')
        save_path: Path to save the best model
        early_stopping_patience: Number of steps to wait before early stopping
        print_step: Print loss every N epochs
        
    Returns:
        Trained UNet2DModel
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    # Normalize data to [0, 1]
    data_min, data_max = data.min(), data.max()
    data_norm = (data - data_min) / (data_max - data_min)
    
    if val_data is not None:
        val_norm = (val_data - data_min) / (data_max - data_min)
        val_dataset = DiffusionDataset(val_norm)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    else:
        val_dataloader = None
    
    train_dataset = DiffusionDataset(data_norm)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize scheduler
    noise_scheduler = HeunDiscreteScheduler(num_train_timesteps=num_train_timesteps)
    
    # Initialize model
    model = UNet2DModel(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=out_channels,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=(
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ),
    ).to(device)
    
    # Loss function and optimizer
    loss_fn = nn.MSELoss()
    opt = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5000, eta_min=1e-6)
    
    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    
    for epoch in range(n_epochs):
        losses = []
        model.train()
        for batch in train_dataloader:
            x = batch.to(device)
            # Map to (-1, 1) for diffusion
            x = x * 2.0 - 1.0
            
            noise = torch.randn_like(x)
            timesteps = torch.randint(0, num_train_timesteps - 1, (x.shape[0],)).long().to(device)
            noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
            
            pred = model(noisy_x, timesteps).sample
            loss = loss_fn(pred, noise)
            
            loss.backward()
            opt.step()
            opt.zero_grad()
            scheduler.step()
            
            losses.append(loss.item())
        
        avg_train_loss = sum(losses) / len(losses)
        
        # Validation
        if val_dataloader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_dataloader:
                    x = batch.to(device)
                    x = x * 2.0 - 1.0
                    noise = torch.randn_like(x)
                    timesteps = torch.randint(0, num_train_timesteps - 1, (x.shape[0],)).long().to(device)
                    noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
                    pred = model(noisy_x, timesteps).sample
                    loss = loss_fn(pred, noise)
                    val_losses.append(loss.item())
            
            avg_val_loss = sum(val_losses) / len(val_losses)
        else:
            avg_val_loss = avg_train_loss
        
        if epoch % print_step == 0:
            print(f"Epoch {epoch}: Avg Train Loss: {avg_train_loss:.6f}, Avg Val Loss: {avg_val_loss:.6f}")
        
        # Checkpoint: Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print("Saved best model!")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered! Training stopped.")
                break
    
    # Load best model
    model.load_state_dict(torch.load(save_path))
    print(f"Training complete. Best model saved to {save_path}")
    
    return model, data_min, data_max


def sample_with_observations(
    model: UNet2DModel,
    observations: np.ndarray,
    data_min: float,
    data_max: float,
    sample_size: int = 64,
    in_channels: int = 1,
    num_inference_steps: int = 50,
    std: float = 0.5,
    gamma: float = 0.01,
    eta: float = 1e-3,
    device: str = 'cuda',
    num_train_timesteps: int = 1000,
    batch_size: int = 1
) -> np.ndarray:
    """
    Generate samples using the trained diffusion model with observation correction.
    
    Args:
        model: Trained UNet2DModel
        observations: Observation array of shape [H, W] or [N, H, W] (can contain NaNs)
        data_min: Minimum value used for normalization during training
        data_max: Maximum value used for normalization during training
        sample_size: Spatial size of the data
        in_channels: Number of input channels
        num_inference_steps: Number of denoising steps
        std: Observation error standard deviation
        gamma: Regularization parameter for correction
        eta: Small constant for numerical stability
        device: Device to run inference on
        num_train_timesteps: Number of timesteps the model was trained with
        batch_size: Batch size for inference
        
    Returns:
        Denormalized predictions as numpy array
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    noise_scheduler = HeunDiscreteScheduler(num_train_timesteps=num_train_timesteps)
    
    # Normalize observations to [0, 1]
    obs_norm = (observations - data_min) / (data_max - data_min)
    
    # Handle different input shapes
    if obs_norm.ndim == 2:  # [H, W]
        obs_norm = obs_norm[np.newaxis, ...]  # [1, H, W]
    
    if obs_norm.ndim == 3 and obs_norm.shape[0] == 1:  # [1, H, W]
        obs_norm = obs_norm[np.newaxis, ...]  # [1, 1, H, W]
    
    # Convert to tensor
    obs_tensor = torch.from_numpy(obs_norm).float().to(device)
    if obs_tensor.ndim == 3:
        obs_tensor = obs_tensor.unsqueeze(1)  # Add channel dimension
    
    # Map observations from [0, 1] to [-1, 1] to match training
    obs_tensor = obs_tensor * 2.0 - 1.0
    
    pred_patches = []
    
    for i in range(obs_tensor.shape[0]):
        # Get single observation
        obs = obs_tensor[i:i+1]  # [1, C, H, W]
        
        # Start from noise
        sample = torch.randn_like(obs)
        
        # Set timesteps
        noise_scheduler.set_timesteps(num_inference_steps, device=sample.device)
        
        # Normalize timesteps to [0, 1] for alpha/mu/sigma functions
        timesteps_normalized = noise_scheduler.timesteps.float() / (num_train_timesteps - 1)
        
        for idx, t in enumerate(tqdm(noise_scheduler.timesteps, desc="Sampling steps")):
            with torch.no_grad():
                t_in = t.expand(sample.shape[0])
                residual = model(sample, t_in).sample
            
            out = noise_scheduler.step(residual, t, sample)
            sample = out.prev_sample
            
            # Apply observation correction
            t_norm = timesteps_normalized[idx]
            sample = corrected_noise(sample, obs, t_norm, std=std, gamma=gamma, eta=eta)
        
        # Map back from [-1, 1] to [0, 1]
        sample = (sample + 1.0) / 2.0
        pred_patches.append(sample.cpu().detach().numpy())
    
    # Denormalize from [0, 1] to original scale
    predicted = np.concatenate(pred_patches, axis=0)
    predicted = predicted * (data_max - data_min) + data_min
    
    return predicted

