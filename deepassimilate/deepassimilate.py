from .models import SRCNN
from .utils import preprocess_data, masked_mse_loss, train_model
import torch

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
