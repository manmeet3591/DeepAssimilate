# deepassimilate/utils.py
"""
Utility functions for deepassimilate.
"""

import torch
from typing import Optional, Union


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Get the appropriate torch device for computation.
    
    This function auto-detects the best available device (CUDA GPU if available,
    otherwise CPU) or returns a specified device.
    
    Args:
        device: Optional device specification. Can be:
            - None: Auto-detect (default - prefers CUDA if available)
            - str: Device string like 'cuda', 'cpu', 'cuda:0', 'mps' (Apple Silicon)
            - torch.device: Already constructed device object
    
    Returns:
        torch.device: The device to use for computation.
    
    Examples:
        >>> import deepassimilate as da
        >>> device = da.get_device()  # Auto-detect
        >>> device = da.get_device('cuda:0')  # Specific GPU
        >>> device = da.get_device('cpu')  # Force CPU
        >>> device = da.get_device('mps')  # Apple Silicon (if available)
    """
    if device is None:
        # Auto-detect: prefer CUDA, then MPS (Apple Silicon), then CPU
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    elif isinstance(device, torch.device):
        # Already a device object, return as-is
        return device
    
    return torch.device(device)


def get_device_info(device: Optional[Union[str, torch.device]] = None) -> dict:
    """
    Get information about the specified or auto-detected device.
    
    Args:
        device: Optional device specification. If None, auto-detects.
    
    Returns:
        dict: Dictionary with device information including:
            - device: The torch.device object
            - device_type: Type of device ('cuda', 'cpu', 'mps')
            - is_cuda: Boolean indicating if CUDA is available
            - is_mps: Boolean indicating if MPS (Apple Silicon) is available
            - cuda_device_count: Number of CUDA devices (if CUDA available)
            - cuda_device_name: Name of CUDA device (if CUDA available)
    """
    device_obj = get_device(device)
    
    info = {
        "device": device_obj,
        "device_type": device_obj.type,
        "is_cuda": torch.cuda.is_available(),
        "is_mps": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
    }
    
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        if device_obj.type == "cuda":
            info["cuda_device_name"] = torch.cuda.get_device_name(device_obj.index if device_obj.index is not None else 0)
    
    return info


def print_device_info(device: Optional[Union[str, torch.device]] = None):
    """
    Print information about the specified or auto-detected device.
    
    Args:
        device: Optional device specification. If None, auto-detects.
    
    Examples:
        >>> import deepassimilate as da
        >>> da.print_device_info()
        Device: cuda
        Device type: cuda
        CUDA available: True
        CUDA devices: 1
        CUDA device name: NVIDIA GeForce RTX 3090
    """
    info = get_device_info(device)
    
    print(f"Device: {info['device']}")
    print(f"Device type: {info['device_type']}")
    print(f"CUDA available: {info['is_cuda']}")
    
    if info['is_cuda']:
        print(f"CUDA devices: {info['cuda_device_count']}")
        if 'cuda_device_name' in info:
            print(f"CUDA device name: {info['cuda_device_name']}")
    
    if info['is_mps']:
        print(f"MPS (Apple Silicon) available: {info['is_mps']}")

