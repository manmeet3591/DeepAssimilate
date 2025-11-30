# deepassimilate/models/factory.py
from typing import Optional, Dict, Any, Callable
from diffusers import UNet2DModel


def build_unet_2d(
    architecture: str,
    img_size: int,
    in_channels: int,
    out_channels: int,
    config_overrides: Optional[Dict[str, Any]] = None,
    custom_builder: Optional[Callable[..., UNet2DModel]] = None,
) -> UNet2DModel:
    """
    Build an unconditional UNet2DModel with some preset architectures.

    Args:
        architecture: "edm_unet_2d", "basic_unet", or "custom".
        img_size: image / field size (assumed square, or max(H, W) for non-square inputs).
        in_channels: number of channels.
        out_channels: usually same as in_channels (predict noise).
        config_overrides: extra kwargs passed into UNet2DModel.
        custom_builder: callable(img_size, in_channels, out_channels) -> UNet2DModel if architecture="custom".
    """
    cfg = dict(config_overrides or {})

    if architecture == "edm_unet_2d":
        # Simple EDM-inspired UNet config
        model = UNet2DModel(
            sample_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            block_out_channels=(128, 256, 256, 256),
            down_block_types=("DownBlock2D",) * 4,
            up_block_types=("UpBlock2D",) * 4,
            layers_per_block=2,
            mid_block_scale_factor=1.0,
            time_embedding_type="fourier",
            **cfg,
        )
    elif architecture == "basic_unet":
        # Small baseline UNet
        model = UNet2DModel(
            sample_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            block_out_channels=(64, 128, 128),
            down_block_types=("DownBlock2D",) * 3,
            up_block_types=("UpBlock2D",) * 3,
            layers_per_block=2,
            **cfg,
        )
    elif architecture == "custom":
        if custom_builder is None:
            raise ValueError("For architecture='custom', you must provide a custom_builder callable.")
        model = custom_builder(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    return model
