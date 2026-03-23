# deepassimilate/models/factory.py
"""Model factory supporting all diffusers architectures for unconditional generation."""

from typing import Optional, Dict, Any, Callable, Union
import torch.nn as nn


# Preset architecture configs for UNet2DModel
PRESETS = {
    "edm_unet_2d": {
        "block_out_channels": (128, 256, 256, 256),
        "down_block_types": ("DownBlock2D",) * 4,
        "up_block_types": ("UpBlock2D",) * 4,
        "layers_per_block": 2,
        "mid_block_scale_factor": 1.0,
        "time_embedding_type": "fourier",
    },
    "edm_unet_2d_attn": {
        "block_out_channels": (64, 128, 256, 512),
        "down_block_types": (
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        "up_block_types": (
            "UpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ),
        "layers_per_block": 2,
        "time_embedding_type": "fourier",
    },
    "basic_unet": {
        "block_out_channels": (64, 128, 128),
        "down_block_types": ("DownBlock2D",) * 3,
        "up_block_types": ("UpBlock2D",) * 3,
        "layers_per_block": 2,
    },
    "tiny_unet": {
        "block_out_channels": (32, 64),
        "down_block_types": ("DownBlock2D",) * 2,
        "up_block_types": ("UpBlock2D",) * 2,
        "layers_per_block": 1,
    },
}


def list_presets():
    """Return available preset architecture names."""
    return list(PRESETS.keys())


def build_unet_2d(
    architecture: str = "edm_unet_2d",
    img_size: int = 64,
    in_channels: int = 1,
    out_channels: int = 1,
    config_overrides: Optional[Dict[str, Any]] = None,
    custom_builder: Optional[Callable[..., nn.Module]] = None,
) -> nn.Module:
    """Build a UNet2DModel from diffusers with preset or custom config.

    Args:
        architecture: Preset name (see list_presets()) or "custom".
        img_size: Spatial size (square assumed).
        in_channels: Input channels.
        out_channels: Output channels (usually same as in_channels).
        config_overrides: Extra kwargs merged on top of preset defaults.
        custom_builder: For architecture="custom", callable(img_size, in_channels, out_channels) -> model.

    Returns:
        A diffusers model instance.
    """
    from diffusers import UNet2DModel

    config_overrides = dict(config_overrides or {})

    if architecture in PRESETS:
        merged = {**PRESETS[architecture], **config_overrides}
        return UNet2DModel(
            sample_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            **merged,
        )
    elif architecture == "custom":
        if custom_builder is None:
            raise ValueError("architecture='custom' requires a custom_builder callable.")
        return custom_builder(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
        )
    else:
        raise ValueError(
            f"Unknown architecture: '{architecture}'. "
            f"Available: {list_presets() + ['custom']}"
        )


def build_model_from_config(config: Dict[str, Any]) -> nn.Module:
    """Build a diffusers model directly from a config dict.

    This is used by the NAS module to instantiate arbitrary architectures.
    The config must include a 'model_class' key (e.g. 'UNet2DModel').

    Args:
        config: Dict with 'model_class' and all constructor kwargs.

    Returns:
        Instantiated model.
    """
    import diffusers

    config = dict(config)
    model_class_name = config.pop("model_class", "UNet2DModel")

    model_class = getattr(diffusers, model_class_name, None)
    if model_class is None:
        raise ValueError(
            f"Model class '{model_class_name}' not found in diffusers. "
            f"Available: UNet2DModel, UNet2DConditionModel, etc."
        )

    return model_class(**config)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
