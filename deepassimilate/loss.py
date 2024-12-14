import torch
import torch.nn.functional as F

def masked_mse_loss(output, target):
    mask = ~torch.isnan(target)
    masked_output = output[mask]
    masked_target = target[mask]
    if masked_output.numel() == 0:
        return torch.tensor(0.0, device=output.device)
    return F.mse_loss(masked_output, masked_target)
