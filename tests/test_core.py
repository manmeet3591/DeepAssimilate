"""Core tests for deepassimilate library."""

import pytest
import torch
import numpy as np


def test_import():
    import deepassimilate as da
    assert hasattr(da, "__version__")
    assert da.__version__ == "0.2.0"


def test_list_presets():
    from deepassimilate.models import list_presets
    presets = list_presets()
    assert "edm_unet_2d" in presets
    assert "basic_unet" in presets
    assert "tiny_unet" in presets


def test_list_schedulers():
    from deepassimilate.schedulers.factory import list_schedulers
    scheds = list_schedulers()
    assert "ddpm" in scheds
    assert "ddim" in scheds
    assert "heun_edm" in scheds


def test_build_tiny_unet():
    from deepassimilate import build_unet_2d, count_parameters
    model = build_unet_2d(architecture="tiny_unet", img_size=32, in_channels=1, out_channels=1)
    assert count_parameters(model) > 0
    # Forward pass
    x = torch.randn(2, 1, 32, 32)
    t = torch.randint(0, 100, (2,))
    out = model(x, t).sample
    assert out.shape == (2, 1, 32, 32)


def test_build_scheduler_ddpm():
    from deepassimilate import build_scheduler
    sched = build_scheduler("ddpm", num_train_timesteps=100)
    assert hasattr(sched, "alphas_cumprod")


def test_build_scheduler_heun():
    from deepassimilate import build_scheduler
    sched = build_scheduler("heun_edm", num_train_timesteps=100)
    sched.set_timesteps(10)
    assert hasattr(sched, "sigmas")


def test_get_mu_sigma_ddpm():
    from deepassimilate import build_scheduler
    from deepassimilate.assimilation.score import get_mu_sigma_from_scheduler
    sched = build_scheduler("ddpm", num_train_timesteps=100)
    sched.set_timesteps(10)
    mu, sigma, stype = get_mu_sigma_from_scheduler(sched, "cpu")
    assert stype == "ddpm"
    assert mu.shape == sigma.shape


def test_get_mu_sigma_edm():
    from deepassimilate import build_scheduler
    from deepassimilate.assimilation.score import get_mu_sigma_from_scheduler
    sched = build_scheduler("heun_edm", num_train_timesteps=100)
    sched.set_timesteps(10)
    mu, sigma, stype = get_mu_sigma_from_scheduler(sched, "cpu")
    assert stype == "edm"
    assert (mu == 1.0).all()


def test_observation_ops():
    from deepassimilate import make_random_mask, MaskedObservationOperator
    mask = make_random_mask((1, 8, 8), 0.5, seed=42)
    assert mask.shape == (1, 8, 8)
    assert mask.dtype == torch.bool

    op = MaskedObservationOperator(mask)
    x = torch.ones(2, 1, 8, 8)
    y = op(x)
    assert y.shape == x.shape
    # Masked points should be 0
    assert (y[:, ~mask] == 0).all()


def test_weather_dataset():
    from deepassimilate import WeatherDataset
    data = np.random.randn(10, 16, 16).astype(np.float32)
    ds = WeatherDataset(data)
    assert len(ds) == 10
    sample = ds[0]
    assert sample.shape == (1, 16, 16)


def test_sda_config():
    from deepassimilate import SDAConfig
    cfg = SDAConfig(obs_noise_std=0.3, gamma=0.01)
    assert cfg.obs_noise_std == 0.3
    assert cfg.gamma == 0.01
    assert cfg.num_inference_steps == 50


def test_score_posterior_step():
    from deepassimilate.assimilation.score import score_posterior_step, SDAConfig
    cfg = SDAConfig(obs_noise_std=0.5, gamma=1e-3)
    samples = torch.randn(1, 1, 8, 8)
    obs = torch.randn(1, 1, 8, 8)
    mask = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    result = score_posterior_step(samples, obs, mask, sigma_t=1.0, mu_t=1.0, cfg=cfg)
    assert result.shape == samples.shape
    # With observations, the result should differ from input
    assert not torch.allclose(result, samples)


def test_nas_config():
    from deepassimilate import NASConfig
    cfg = NASConfig(time_budget_seconds=10, max_experiments=2)
    assert cfg.time_budget_seconds == 10


def test_build_model_from_config():
    from deepassimilate import build_model_from_config
    config = {
        "model_class": "UNet2DModel",
        "sample_size": 16,
        "in_channels": 1,
        "out_channels": 1,
        "block_out_channels": (32,),
        "down_block_types": ("DownBlock2D",),
        "up_block_types": ("UpBlock2D",),
        "layers_per_block": 1,
    }
    model = build_model_from_config(config)
    x = torch.randn(1, 1, 16, 16)
    t = torch.zeros(1, dtype=torch.long)
    out = model(x, t).sample
    assert out.shape == (1, 1, 16, 16)
