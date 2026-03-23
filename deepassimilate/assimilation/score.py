"""
Score-based data assimilation following Manshausen et al. (2024).

Implements the VP-SDE framework for diffusion-based generative data assimilation,
compatible with Hugging Face diffusers schedulers.

References:
    Manshausen et al. (2024) "Generative data assimilation of sparse weather
    station observations at scale" (arXiv:2406.16947)

    NVIDIA PhysicsNeMo ReGen:
    github.com/NVIDIA/physicsnemo/blob/main/examples/weather/regen/sda/score.py
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class SDAConfig:
    """Configuration for score-based data assimilation.

    Args:
        obs_noise_std: Observation noise standard deviation (normalized data space).
        gamma: Regularization for likelihood variance. Controls trust in prior
            vs observations at high noise levels.
        num_inference_steps: Number of denoising steps.
        corrections: Langevin corrector steps per denoising step (0 = no correction).
        tau: Step size for Langevin corrector.
        guidance_scale: Multiplier on likelihood gradient (1.0 = standard).
    """

    obs_noise_std: float = 0.5
    gamma: float = 1e-3
    num_inference_steps: int = 50
    corrections: int = 0
    tau: float = 1.0
    guidance_scale: float = 1.0


def get_mu_sigma_from_scheduler(scheduler, device):
    """Extract mu(t) and sigma(t) arrays from a diffusers scheduler.

    EDM/Heun: x_t = x_0 + sigma * eps  =>  mu=1, sigma=sigmas
    DDPM/DDIM: x_t = sqrt(alpha_bar)*x_0 + sqrt(1-alpha_bar)*eps

    Returns:
        (mu_arr, sigma_arr, scheduler_type)
    """
    if hasattr(scheduler, "sigmas") and scheduler.sigmas is not None:
        sigma_arr = scheduler.sigmas.to(device)
        mu_arr = torch.ones_like(sigma_arr)
        return mu_arr, sigma_arr, "edm"
    elif hasattr(scheduler, "alphas_cumprod"):
        alphas_cumprod = scheduler.alphas_cumprod.to(device)
        mu_arr = torch.sqrt(alphas_cumprod)
        sigma_arr = torch.sqrt(1.0 - alphas_cumprod)
        return mu_arr, sigma_arr, "ddpm"
    else:
        raise ValueError(
            f"Cannot extract mu/sigma from scheduler type {type(scheduler).__name__}. "
            "Scheduler must have either 'sigmas' (EDM) or 'alphas_cumprod' (DDPM/DDIM)."
        )


def score_posterior_step(
    samples: Tensor,
    observations: Tensor,
    obs_mask: Tensor,
    sigma_t: float,
    mu_t: float,
    cfg: SDAConfig,
    obs_operator: Optional[Callable[[Tensor], Tensor]] = None,
) -> Tensor:
    """Single posterior correction step via score of log p(y|x).

    Computes grad_x log p(y|x) where p(y|x) = N(y; H(x), var*I)
    and var = obs_noise_std^2 + gamma * (sigma_t / mu_t)^2.

    Args:
        samples: Current denoised samples [B, C, H, W].
        observations: Clean observations (NaN-free) [B, C, H, W].
        obs_mask: Boolean mask, True=observed [B, C, H, W].
        sigma_t: Current noise level sigma(t).
        mu_t: Current signal scale mu(t).
        cfg: SDAConfig.
        obs_operator: Optional H(x). Identity if None.

    Returns:
        Corrected samples [B, C, H, W].
    """
    if obs_operator is None:
        obs_operator = lambda x: x

    with torch.enable_grad():
        x = samples.detach().requires_grad_(True)

        predicted = obs_operator(x)

        err = (predicted - observations) * obs_mask.float()
        var = cfg.obs_noise_std ** 2 + cfg.gamma * (sigma_t / (mu_t + 1e-8)) ** 2
        var = max(var if isinstance(var, float) else var.item(), 1e-12)

        log_p = -(err ** 2).sum() / (2.0 * var)
        (grad,) = torch.autograd.grad(log_p, x)

    return samples + cfg.guidance_scale * sigma_t * grad


def score_based_assimilation(
    model: nn.Module,
    scheduler,
    observations: Tensor,
    obs_mask: Tensor,
    cfg: Optional[SDAConfig] = None,
    obs_operator: Optional[Callable[[Tensor], Tensor]] = None,
    n_samples: int = 1,
    device: Optional[str] = None,
    seed: Optional[int] = None,
) -> Tensor:
    """Run score-based data assimilation (Manshausen et al. 2024).

    Three-step process at each denoising step:
      1. Unconditional prior denoising step (diffusion model)
      2. Likelihood correction via grad log p(y|x) (observations)
      3. Optional Langevin corrector steps

    Args:
        model: Trained unconditional diffusion model (e.g. UNet2DModel).
        scheduler: Diffusers noise scheduler (Heun, DDPM, DDIM, etc.).
        observations: [B, C, H, W] with NaN for missing values.
        obs_mask: [B, C, H, W] boolean, True = observed.
        cfg: SDAConfig. Uses defaults if None.
        obs_operator: H(x) mapping state -> obs space. Identity if None.
        n_samples: Posterior samples per observation.
        device: Auto-detected if None.
        seed: For reproducibility.

    Returns:
        [B * n_samples, C, H, W] posterior samples.
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    cfg = cfg or SDAConfig()

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    model = model.to(device)
    model.eval()

    observations = observations.to(device)
    obs_mask = obs_mask.to(device).bool()
    B, C, H, W = observations.shape

    if n_samples > 1:
        observations = observations.repeat_interleave(n_samples, dim=0)
        obs_mask = obs_mask.repeat_interleave(n_samples, dim=0)
    batch_size = observations.shape[0]

    obs_clean = torch.nan_to_num(observations, nan=0.0)

    scheduler.set_timesteps(cfg.num_inference_steps, device=device)
    mu_arr, sigma_arr, sched_type = get_mu_sigma_from_scheduler(scheduler, device)

    samples = torch.randn((batch_size, C, H, W), device=device)

    for i_t, t in enumerate(scheduler.timesteps):
        samples = torch.clamp(samples, min=-5.0, max=5.0)
        t_batch = t.expand(batch_size)

        # 1. Prior: unconditional denoising step
        with torch.no_grad():
            noise_pred = model(samples, t_batch).sample
            noise_pred = torch.nan_to_num(noise_pred, nan=0.0)
            samples = scheduler.step(noise_pred, t, samples).prev_sample

        # 2. Likelihood correction
        ii = min(max(i_t, 0), sigma_arr.shape[0] - 1)
        sig = sigma_arr[ii]
        mu = mu_arr[ii]

        samples = score_posterior_step(
            samples, obs_clean, obs_mask, sig, mu, cfg, obs_operator
        )

        # 3. Optional Langevin corrector
        for _ in range(cfg.corrections):
            z = torch.randn_like(samples)
            with torch.no_grad():
                eps = model(samples, t_batch).sample
            delta = cfg.tau / (eps.square().mean() + 1e-8)
            samples = samples - (delta * eps + torch.sqrt(2 * delta) * z) * sig

    return samples.detach()
