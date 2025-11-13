from .models import SRCNN
from .utils import preprocess_data, masked_mse_loss, train_model
from .manshausen_da import train_diffusion_model, sample_with_observations
import torch
import numpy as np
from typing import Optional

class deepassimilate:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = SRCNN().to(device)

    def assimilate(self, gridded_data, station_data, epochs=1000, batch_size=20):
        """
        Assimilate station data into gridded data.
        
        Args:
            gridded_data (np.array): Gridded data input.
            station_data (np.array): Station data input.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            
        Returns:
            torch.Tensor: Assimilated data.
        """
        # Preprocess data
        x_train, y_train, z_train = preprocess_data(gridded_data, station_data)

        # Train the model
        train_loader, val_loader = train_model(
            self.model, x_train, y_train, z_train, self.device, epochs, batch_size
        )

        # Output assimilated data
        self.model.eval()
        with torch.no_grad():
            output = self.model(torch.tensor(gridded_data).to(self.device))
        return output.cpu()


class ManshausenDA:
    """
    Manshausen et al. Diffusion-based Data Assimilation
    
    Implements zero-shot data assimilation using diffusion models.
    Trains a diffusion model on gridded data and integrates sparse
    observations during inference.
    """
    
    def __init__(self, device='cuda'):
        """
        Initialize Manshausen DA.
        
        Args:
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device
        self.model = None
        self.data_min = None
        self.data_max = None
        self.sample_size = None
        
    def train(
        self,
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
        save_path: str = 'best_manshausen_model.pth',
        early_stopping_patience: int = 5000,
        print_step: int = 1
    ):
        """
        Train the diffusion model on gridded data.
        
        Args:
            data: Training data array of shape [N, H, W] or [N, C, H, W]
            val_data: Validation data array (optional)
            sample_size: Spatial size of the data
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_train_timesteps: Number of diffusion timesteps
            batch_size: Batch size for training
            n_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            save_path: Path to save the best model
            early_stopping_patience: Number of steps to wait before early stopping
            print_step: Print loss every N epochs
        """
        self.sample_size = sample_size
        self.model, self.data_min, self.data_max = train_diffusion_model(
            data=data,
            val_data=val_data,
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_train_timesteps=num_train_timesteps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=self.device,
            save_path=save_path,
            early_stopping_patience=early_stopping_patience,
            print_step=print_step
        )
        
    def assimilate(
        self,
        observations: np.ndarray,
        num_inference_steps: int = 50,
        std: float = 0.5,
        gamma: float = 0.01,
        eta: float = 1e-3,
        num_train_timesteps: int = 1000,
        batch_size: int = 1
    ) -> np.ndarray:
        """
        Assimilate sparse observations into gridded field.
        
        Args:
            observations: Observation array of shape [H, W] or [N, H, W] (can contain NaNs)
            num_inference_steps: Number of denoising steps
            std: Observation error standard deviation
            gamma: Regularization parameter for correction
            eta: Small constant for numerical stability
            num_train_timesteps: Number of timesteps the model was trained with
            batch_size: Batch size for inference
            
        Returns:
            Assimilated data as numpy array
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        return sample_with_observations(
            model=self.model,
            observations=observations,
            data_min=self.data_min,
            data_max=self.data_max,
            sample_size=self.sample_size,
            in_channels=1,
            num_inference_steps=num_inference_steps,
            std=std,
            gamma=gamma,
            eta=eta,
            device=self.device,
            num_train_timesteps=num_train_timesteps,
            batch_size=batch_size
        )
