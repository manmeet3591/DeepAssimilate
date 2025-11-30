import torch
import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

def masked_mse_loss(output, target):
    mask = ~torch.isnan(target)
    masked_output = output[mask]
    masked_target = target[mask]
    if masked_output.numel() == 0:
        return torch.tensor(0.0, device=output.device)
    return nn.functional.mse_loss(masked_output, masked_target)
