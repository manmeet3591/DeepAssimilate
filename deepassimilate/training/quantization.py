# deepassimilate/training/quantization.py
import torch
from torch.ao.quantization import quantize_dynamic


def maybe_quantize(model: torch.nn.Module) -> torch.nn.Module:
    """
    Very simple dynamic quantization stub. This is mostly effective
    for Linear-heavy models; for UNet you may want to extend this or
    integrate with more specialized libraries later.
    """
    print("[deepassimilate] Quantization: applying dynamic quantization to Linear layers.")
    q_model = quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    return q_model
