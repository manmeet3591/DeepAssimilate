# deepassimilate/assimilation/da_posterior.py
from dataclasses import dataclass
from typing import Callable
import torch

from .observation_ops import ObservationOperator


@dataclass
class DAConfig:
    """
    Configuration for generative data assimilation.

    da_steps:     how many times to apply DA correction along the diffusion trajectory.
    da_strength:  step size for the gradient ascent on log posterior.
    obs_noise_std: standard deviation of observation noise.
    gamma:        additional inflation term (tweak to match Manshausen equations).
    """

    da_steps: int = 50
    da_strength: float = 0.1
    obs_noise_std: float = 0.1
    gamma: float = 0.0


def default_mu(t: torch.Tensor) -> torch.Tensor:
    """
    Placeholder for EDM-style mu(t). Replace with your exact expression if needed.
    Currently returns 1 for all t.
    """
    return torch.ones_like(t, dtype=torch.float32)


def default_sigma(t: torch.Tensor) -> torch.Tensor:
    """
    Placeholder for EDM-style sigma(t). Replace with your exact expression.
    Currently returns 1 for all t.
    """
    return torch.ones_like(t, dtype=torch.float32)


def corrected_sample_step(
    sample: torch.Tensor,
    t: torch.Tensor,
    observation: torch.Tensor,
    obs_operator: ObservationOperator,
    cfg: DAConfig,
    mu_fn: Callable[[torch.Tensor], torch.Tensor] = default_mu,
    sigma_fn: Callable[[torch.Tensor], torch.Tensor] = default_sigma,
) -> torch.Tensor:
    """
    One DA correction step: x -> x' using gradient of log p(y | x).

    You can plug in your exact Manshausen/EDM formulas here by modifying mu_fn,
    sigma_fn, and the log likelihood expression.
    """
    # t is [B] long; convert to float
    t_f = t.float()
    mu_t = mu_fn(t_f).view(-1, 1, 1, 1)
    sigma_t = sigma_fn(t_f).view(-1, 1, 1, 1)

    std = cfg.obs_noise_std
    gamma = cfg.gamma

    sample = sample.detach().requires_grad_(True)

    obs_model = obs_operator(sample)
    var = std**2 + gamma * (sigma_t / (mu_t + 1e-6)) ** 2

    # Gaussian likelihood p(y | x) ~ N(H(x), var)
    err = obs_model - observation
    log_p = -(err**2 / (2.0 * var)).sum()

    (grad_x,) = torch.autograd.grad(log_p, sample, retain_graph=False)

    corrected = sample + cfg.da_strength * sigma_t * grad_x
    return corrected.detach()
