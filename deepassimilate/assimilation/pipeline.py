# deepassimilate/assimilation/pipeline.py
"""High-level API for score-based data assimilation."""

from typing import Callable, Optional

import torch
from torch import Tensor

from .score import SDAConfig, score_based_assimilation
from .observation_ops import ObservationOperator


def run_data_assimilation(
    model,
    scheduler,
    observations: Tensor,
    obs_mask: Optional[Tensor] = None,
    obs_operator: Optional[Callable[[Tensor], Tensor]] = None,
    n_samples: int = 1,
    obs_noise_std: float = 0.5,
    gamma: float = 1e-3,
    num_inference_steps: int = 50,
    guidance_scale: float = 1.0,
    corrections: int = 0,
    device: Optional[str] = None,
    seed: Optional[int] = None,
) -> Tensor:
    """One-liner data assimilation using score-based diffusion.

    Combines an unconditional diffusion prior with sparse observations
    to produce posterior samples (Manshausen et al. 2024).

    Args:
        model: Trained unconditional diffusion model (e.g. UNet2DModel).
        scheduler: Diffusers noise scheduler used during training.
        observations: [B, C, H, W] tensor. Use NaN for missing values.
        obs_mask: [B, C, H, W] boolean mask (True=observed). If None,
            inferred from non-NaN values in observations.
        obs_operator: Callable H(x) or ObservationOperator mapping state -> obs space.
        n_samples: Number of posterior samples per observation.
        obs_noise_std: Observation noise std (normalized data space).
        gamma: Likelihood regularization parameter.
        num_inference_steps: Denoising steps.
        guidance_scale: Multiplier on likelihood gradient.
        corrections: Langevin corrector steps (0=none).
        device: Device string (auto-detected if None).
        seed: Random seed for reproducibility.

    Returns:
        [B * n_samples, C, H, W] posterior (assimilated) samples.

    Example:
        >>> import deepassimilate as da
        >>> result = da.run_data_assimilation(
        ...     model=model,
        ...     scheduler=scheduler,
        ...     observations=obs_with_nans,
        ...     obs_noise_std=0.5,
        ...     gamma=1e-3,
        ... )
    """
    if obs_mask is None:
        obs_mask = ~torch.isnan(observations)

    # Unwrap ObservationOperator to callable
    if isinstance(obs_operator, ObservationOperator):
        obs_op_fn = obs_operator.forward
    else:
        obs_op_fn = obs_operator

    cfg = SDAConfig(
        obs_noise_std=obs_noise_std,
        gamma=gamma,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        corrections=corrections,
    )

    return score_based_assimilation(
        model=model,
        scheduler=scheduler,
        observations=observations,
        obs_mask=obs_mask,
        cfg=cfg,
        obs_operator=obs_op_fn,
        n_samples=n_samples,
        device=device,
        seed=seed,
    )
