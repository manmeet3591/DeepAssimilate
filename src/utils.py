import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

class ncDataset(Dataset):
    def __init__(self, data, targets, stations):
        self.data = data
        self.targets = targets
        self.stations = stations

    def __getitem__(self, index):
        return self.data[index], self.targets[index], self.stations[index]

    def __len__(self):
        return len(self.data)

def preprocess_data(gridded_data, station_data):
    # Implement preprocessing logic here
    # Example: normalize, handle missing data, etc.
    return torch.tensor(gridded_data), torch.tensor(station_data)

def masked_mse_loss(output, target):
    mask = ~torch.isnan(target)
    return nn.functional.mse_loss(output[mask], target[mask])

def train_model(model, x_train, y_train, z_train, device, epochs, batch_size):
    # Create datasets
    dataset = ncDataset(x_train, y_train, z_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = masked_mse_loss

    # Training loop
    for epoch in range(epochs):
        for batch in dataloader:
            x, y, z = batch
            x, y, z = x.to(device), y.to(device), z.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
    return dataloader, dataloader
