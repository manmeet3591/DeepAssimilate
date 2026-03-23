# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is DeepAssimilate

A Python library for diffusion-based generative data assimilation, targeting the climate/weather community. Built on PyTorch and Hugging Face Diffusers. Three-step workflow:

1. **Architecture search** (`da.search_architecture`) - autoresearch-style NAS with fixed time budget
2. **Unconditional diffusion training** (`da.train_unconditional`) - train best architecture
3. **Score-based data assimilation** (`da.run_data_assimilation`) - posterior sampling with sparse observations

Based on Manshausen et al. (2024, arXiv:2406.16947) and NVIDIA PhysicsNeMo ReGen.

## Build and Test

```bash
# Install
pip install -e .

# Run tests
python -m pytest tests/test_core.py -v

# Run a single test
python -m pytest tests/test_core.py::test_score_posterior_step -v
```

## Architecture

```
deepassimilate/
  __init__.py              # Top-level API (all 3 steps)
  nas/
    search.py              # Step 1: autoresearch-style NAS (search_architecture, NASConfig)
  training/
    uncond_trainer.py      # Step 2: unconditional diffusion training (train_unconditional)
    quantization.py        # Optional dynamic quantization
  assimilation/
    score.py               # Step 3 core: VPSDE score-based DA (score_based_assimilation, SDAConfig)
    pipeline.py            # Step 3 high-level API (run_data_assimilation - the 1-liner)
    observation_ops.py     # Observation operators (Masked, Linear, Identity, station masks)
    da_posterior.py         # Legacy DA (kept for backward compat)
  models/
    factory.py             # Model factory (build_unet_2d, build_model_from_config, PRESETS)
  schedulers/
    factory.py             # Scheduler factory (build_scheduler, all diffusers schedulers)
    distilled.py           # Scheduler-only distillation
  datasets.py              # WeatherDataset, GriddedDataset (xarray/numpy -> PyTorch)
  utils.py                 # Device detection utilities
```

## Key Design Decisions

- **Diffusers-native**: All models and schedulers come from HF diffusers. The library wraps them with presets and factories, but any diffusers class can be used directly.
- **EDM vs DDPM**: The score.py module auto-detects scheduler type via `get_mu_sigma_from_scheduler()` - checks for `sigmas` (EDM) or `alphas_cumprod` (DDPM/DDIM).
- **NAS**: Fixed wall-clock time budget per candidate (like Karpathy's autoresearch). Results logged to TSV. Best model checkpoint saved.
- **Observation operators**: `obs_mask` (boolean tensor) + optional `obs_operator` callable. NaN values in observations auto-generate mask if none provided.

## DA Math (score.py)

At each denoising step t:
```
x_t <- x_t + guidance_scale * sigma(t) * grad_x log p(y|x_t)
```
where `log p(y|x) = -||y - H(x)||^2 / (2 * var)` and `var = obs_noise_std^2 + gamma * (sigma_t / mu_t)^2`.

## Examples

- `examples/01_training_diffusion_priors_weather.ipynb` - Full training on NCEP reanalysis
- `examples/02_diffusion_da.ipynb` - DA with trained model

## JOSS Paper

Paper skeleton in `paper/paper.md` with `paper/paper.bib`.
