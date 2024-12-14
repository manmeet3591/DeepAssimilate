import torch
from torch.utils.data import Dataset

class ncDataset(Dataset):
    def __init__(self, data, targets, stations):
        self.data = data
        self.targets = targets
        self.stations = stations

    def __getitem__(self, index):
        x = self.data[index].unsqueeze(0)
        y = self.targets[index].unsqueeze(0)
        z = self.stations[index].unsqueeze(0)
        return x, y, z

    def __len__(self):
        return len(self.data)
