# deepassimilate/training/uncond_trainer.py
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
from torch.utils.data import DataLoader
from diffusers.optimization import get_cosine_schedule_with_warmup

from ..models import build_unet_2d
from ..schedulers import build_scheduler
from .quantization import maybe_quantize


@dataclass
class UncondTrainConfig:
    """
    Configuration for unconditional diffusion training.
    """

    # Model/scheduler choices
    architecture: str = "edm_unet_2d"  # "edm_unet_2d", "basic_unet", or "custom"
    scheduler: str = "heun_edm"        # "heun_edm", "ddpm", "ddim"
    img_size: int = 128
    channels: int = 1

    # Optimization
    batch_size: int = 64
    num_epochs: int = 50
    lr: float = 1e-4
    device: str = "cuda"

    # Training noise schedule
    num_train_timesteps: int = 1000

    # Inference 'distillation' is scheduler-only; we don't modify model
    distilled_num_steps: Optional[int] = None  # used later in DA or sampling

    # Quantization
    quantize: bool = False

    # Misc
    ckpt_dir: str = "checkpoints"
    extra_model_args: Optional[Dict[str, Any]] = None
    extra_sched_args: Optional[Dict[str, Any]] = None


def train_unconditional(
    dataset,
    cfg: UncondTrainConfig,
):
    """
    Train an unconditional diffusion model using diffusers-style UNet + scheduler.

    Args:
        dataset: a PyTorch Dataset yielding tensors of shape [C, H, W]
                 normalized consistently (e.g. [-1, 1] or [0, 1]).
        cfg: UncondTrainConfig.

    Returns:
        model: trained UNet2DModel.
        scheduler: training scheduler (with full num_train_timesteps).
        distilled_num_steps: same as cfg.distilled_num_steps for convenience.
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_unet_2d(
        architecture=cfg.architecture,
        img_size=cfg.img_size,
        in_channels=cfg.channels,
        out_channels=cfg.channels,
        config_overrides=cfg.extra_model_args or {},
    ).to(device)

    scheduler = build_scheduler(
        name=cfg.scheduler,
        num_train_timesteps=cfg.num_train_timesteps,
        scheduler_kwargs=cfg.extra_sched_args or {},
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    num_training_steps = cfg.num_epochs * max(1, len(loader))

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    model.train()
    global_step = 0

    for epoch in range(cfg.num_epochs):
        for batch in loader:
            # If dataset returns dict or tuple, user should customize this part.
            if isinstance(batch, dict):
                x = batch["input"]
            elif isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            x = x.to(device)

            # sample timesteps
            t = torch.randint(
                0,
                scheduler.num_train_timesteps,
                (x.shape[0],),
                device=device,
            ).long()

            noise = torch.randn_like(x)
            noisy = scheduler.add_noise(x, noise, t)

            pred = model(noisy, t).sample
            loss = torch.nn.functional.mse_loss(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()

            global_step += 1

        # TODO: logging / checkpointing here if you want.

    if cfg.quantize:
        model = maybe_quantize(model)

    return model, scheduler, cfg.distilled_num_steps
