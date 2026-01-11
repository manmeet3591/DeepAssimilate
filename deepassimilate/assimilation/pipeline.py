# deepassimilate/assimilation/pipeline.py
from typing import Optional
import torch

from ..schedulers import build_distilled_scheduler
from .observation_ops import ObservationOperator, IdentityObservationOperator
from .da_posterior import DAConfig, corrected_sample_step


def run_data_assimilation(
    model,
    scheduler,
    observations: torch.Tensor,
    obs_operator: Optional[ObservationOperator] = None,
    n_samples: int = 1,
    da_cfg: Optional[DAConfig] = None,
    distilled_num_steps: Optional[int] = None,
    device: Optional[str] = None,
):
    """
    High-level Manshausen-style generative data assimilation.

    Args:
        model: trained unconditional diffusion model (UNet2DModel).
        scheduler: base scheduler used in training.
        observations: [B, C, H, W] tensor of observations (normalized like training data).
        obs_operator: ObservationOperator mapping state -> observation. Defaults to identity.
        n_samples: number of posterior samples per observation.
        da_cfg: DAConfig with DA hyperparameters.
        distilled_num_steps: if not None, how many inference steps to use
                             (noise-scheduler-only 'distillation').
        device: torch device string.

    Returns:
        analysis_samples: [B * n_samples, C, H, W] tensor of posterior samples.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    da_cfg = da_cfg or DAConfig()
    obs_operator = obs_operator or IdentityObservationOperator()

    model = model.to(device)
    model.eval()

    observations = observations.to(device)
    B, C, H, W = observations.shape

    # Expand observations if we want multiple samples per obs
    if n_samples > 1:
        observations = observations.repeat_interleave(n_samples, dim=0)
    batch_size = observations.shape[0]

    # Scheduler-only "distillation"
    scheduler = build_distilled_scheduler(scheduler, distilled_num_steps)
    timesteps = scheduler.timesteps

    # Initialize samples from prior at highest noise (standard Gaussian)
    samples = torch.randn(
        (batch_size, C, H, W),
        device=device,
    )

    # Diffusion + DA loop
    for i, t in enumerate(timesteps):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # Prior diffusion step (no gradients needed here)
        with torch.no_grad():
            model_out = model(samples, t_batch).sample
            prior_step = scheduler.step(model_out, t_batch, samples)
            samples = prior_step.prev_sample

        # DA correction step every few diffusion steps
        # spacing set so we perform ~da_cfg.da_steps corrections over trajectory
        if da_cfg.da_steps > 0:
            interval = max(1, len(timesteps) // da_cfg.da_steps)
            if i % interval == 0:
                samples = corrected_sample_step(
                    sample=samples,
                    t=t_batch,
                    observation=observations,
                    obs_operator=obs_operator,
                    cfg=da_cfg,
                )

    return samples
