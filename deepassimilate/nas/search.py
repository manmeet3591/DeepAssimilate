# deepassimilate/nas/search.py
"""
Autoresearch-style neural architecture search for diffusion models.

Inspired by Karpathy's autoresearch: train each candidate for a fixed time
budget, evaluate on validation MSE, keep if improved, discard otherwise.

The search space covers diffusers UNet2DModel configurations:
  - block_out_channels (width and depth)
  - down/up block types (with/without attention)
  - layers_per_block
  - time_embedding_type
  - norm settings
"""

import csv
import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class NASConfig:
    """Configuration for architecture search.

    Args:
        time_budget_seconds: Wall-clock training time per candidate (default 300 = 5 min).
        max_experiments: Maximum number of architectures to try.
        img_size: Spatial size of data.
        in_channels: Number of input channels.
        num_train_timesteps: Diffusion training timesteps.
        batch_size: Training batch size.
        lr: Learning rate.
        device: Device string (auto-detected if None).
        results_dir: Directory to save results TSV and checkpoints.
        scheduler: Scheduler short name (default "ddpm" for fast NAS).
        search_space: Optional custom search space (list of config dicts).
    """

    time_budget_seconds: float = 300.0
    max_experiments: int = 20
    img_size: int = 64
    in_channels: int = 1
    num_train_timesteps: int = 1000
    batch_size: int = 16
    lr: float = 1e-4
    device: Optional[str] = None
    results_dir: str = "nas_results"
    scheduler: str = "ddpm"
    search_space: Optional[List[Dict[str, Any]]] = None


@dataclass
class NASResult:
    """Result of architecture search.

    Attributes:
        best_config: Config dict for the best architecture found.
        best_val_loss: Validation MSE of the best architecture.
        best_model_path: Path to saved checkpoint of best model.
        results_tsv: Path to the full results log.
        all_results: List of all experiment results.
    """

    best_config: Dict[str, Any]
    best_val_loss: float
    best_model_path: str
    results_tsv: str
    all_results: List[Dict[str, Any]]


def _default_search_space(img_size: int, in_channels: int) -> List[Dict[str, Any]]:
    """Generate default search space of UNet2D configurations."""

    candidates = []

    # Vary depth (number of blocks)
    for depth in [2, 3, 4]:
        # Vary width
        for base_ch in [32, 64, 128]:
            channels = tuple(min(base_ch * (2 ** i), 512) for i in range(depth))

            # Plain (no attention)
            candidates.append({
                "name": f"plain_d{depth}_w{base_ch}",
                "block_out_channels": channels,
                "down_block_types": ("DownBlock2D",) * depth,
                "up_block_types": ("UpBlock2D",) * depth,
                "layers_per_block": 2,
            })

            # With attention on middle blocks (only if depth >= 3)
            if depth >= 3:
                down_types = list(("DownBlock2D",) * depth)
                up_types = list(("UpBlock2D",) * depth)
                # Put attention on middle blocks
                for mid_idx in range(1, depth - 1):
                    down_types[mid_idx] = "AttnDownBlock2D"
                    up_types[depth - 1 - mid_idx] = "AttnUpBlock2D"
                candidates.append({
                    "name": f"attn_d{depth}_w{base_ch}",
                    "block_out_channels": channels,
                    "down_block_types": tuple(down_types),
                    "up_block_types": tuple(up_types),
                    "layers_per_block": 2,
                })

    # Vary layers_per_block
    for lpb in [1, 3]:
        candidates.append({
            "name": f"plain_d3_w64_lpb{lpb}",
            "block_out_channels": (64, 128, 256),
            "down_block_types": ("DownBlock2D",) * 3,
            "up_block_types": ("UpBlock2D",) * 3,
            "layers_per_block": lpb,
        })

    # Time embedding variations
    for te_type in ["fourier", "positional"]:
        candidates.append({
            "name": f"plain_d3_w64_te_{te_type}",
            "block_out_channels": (64, 128, 256),
            "down_block_types": ("DownBlock2D",) * 3,
            "up_block_types": ("UpBlock2D",) * 3,
            "layers_per_block": 2,
            "time_embedding_type": te_type,
        })

    return candidates


def _train_candidate(
    config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    nas_cfg: NASConfig,
    device: torch.device,
) -> Dict[str, Any]:
    """Train a single candidate architecture for the time budget.

    Returns dict with: val_loss, train_loss, num_params, num_steps, peak_memory, status.
    """
    from diffusers import UNet2DModel
    from ..schedulers import build_scheduler

    model_config = {k: v for k, v in config.items() if k != "name"}
    candidate_name = config.get("name", "unnamed")

    result = {
        "name": candidate_name,
        "config": config,
        "val_loss": float("inf"),
        "train_loss": float("inf"),
        "num_params": 0,
        "num_steps": 0,
        "peak_memory_mb": 0.0,
        "status": "crash",
        "error": "",
    }

    try:
        model = UNet2DModel(
            sample_size=nas_cfg.img_size,
            in_channels=nas_cfg.in_channels,
            out_channels=nas_cfg.in_channels,
            **model_config,
        ).to(device)

        result["num_params"] = sum(p.numel() for p in model.parameters())

        scheduler = build_scheduler(
            name=nas_cfg.scheduler,
            num_train_timesteps=nas_cfg.num_train_timesteps,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=nas_cfg.lr)
        loss_fn = nn.MSELoss()

        model.train()
        step = 0
        train_losses = []
        start_time = time.time()

        # Training loop with time budget
        while True:
            for batch in train_loader:
                if time.time() - start_time >= nas_cfg.time_budget_seconds:
                    break

                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                elif isinstance(batch, dict):
                    x = batch["input"]
                else:
                    x = batch

                x = x.to(device)

                t = torch.randint(
                    0, nas_cfg.num_train_timesteps, (x.shape[0],), device=device
                ).long()

                noise = torch.randn_like(x)
                noisy = scheduler.add_noise(x, noise, t)
                pred = model(noisy, t).sample
                loss = loss_fn(pred, noise)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_losses.append(loss.item())
                step += 1

            if time.time() - start_time >= nas_cfg.time_budget_seconds:
                break

        result["num_steps"] = step
        result["train_loss"] = float(np.mean(train_losses[-100:])) if train_losses else float("inf")

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                elif isinstance(batch, dict):
                    x = batch["input"]
                else:
                    x = batch

                x = x.to(device)
                t = torch.randint(
                    0, nas_cfg.num_train_timesteps, (x.shape[0],), device=device
                ).long()
                noise = torch.randn_like(x)
                noisy = scheduler.add_noise(x, noise, t)
                pred = model(noisy, t).sample
                val_loss = loss_fn(pred, noise)
                val_losses.append(val_loss.item())

        result["val_loss"] = float(np.mean(val_losses)) if val_losses else float("inf")
        result["status"] = "ok"

        # Memory tracking
        if torch.cuda.is_available() and device.type == "cuda":
            result["peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            torch.cuda.reset_peak_memory_stats(device)

        # Save model state
        result["model_state"] = model.state_dict()

    except Exception as e:
        result["status"] = "crash"
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def search_architecture(
    train_dataset: Dataset,
    val_dataset: Dataset,
    cfg: Optional[NASConfig] = None,
) -> NASResult:
    """Run autoresearch-style architecture search.

    Trains each candidate architecture for a fixed time budget,
    evaluates validation MSE, keeps the best.

    Args:
        train_dataset: PyTorch Dataset yielding [C, H, W] tensors.
        val_dataset: Validation Dataset.
        cfg: NASConfig. Uses defaults if None.

    Returns:
        NASResult with best architecture config, path to checkpoint, and full log.

    Example:
        >>> import deepassimilate as da
        >>> nas_result = da.search_architecture(
        ...     train_dataset=train_ds,
        ...     val_dataset=val_ds,
        ...     cfg=da.NASConfig(time_budget_seconds=300, max_experiments=10),
        ... )
        >>> print(f"Best: {nas_result.best_config['name']}, val_loss={nas_result.best_val_loss:.6f}")
        >>> # Use the best architecture for full training
        >>> model = da.build_unet_2d(
        ...     architecture="custom",
        ...     custom_builder=lambda **kw: da.build_model_from_config(nas_result.best_config),
        ... )
    """
    cfg = cfg or NASConfig()

    if cfg.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(cfg.device)

    os.makedirs(cfg.results_dir, exist_ok=True)
    results_tsv = os.path.join(cfg.results_dir, "results.tsv")

    # Build search space
    candidates = cfg.search_space or _default_search_space(cfg.img_size, cfg.in_channels)
    candidates = candidates[: cfg.max_experiments]

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # Write TSV header
    with open(results_tsv, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "name", "val_loss", "train_loss", "num_params",
            "num_steps", "peak_memory_mb", "status", "description",
        ])

    best_val_loss = float("inf")
    best_config = None
    best_model_state = None
    all_results = []

    print(f"Starting NAS: {len(candidates)} candidates, "
          f"{cfg.time_budget_seconds}s budget each")
    print(f"Device: {device}")
    print("-" * 60)

    for i, candidate in enumerate(candidates):
        name = candidate.get("name", f"candidate_{i}")
        print(f"\n[{i + 1}/{len(candidates)}] {name}")

        result = _train_candidate(candidate, train_loader, val_loader, cfg, device)

        # Log
        status = result["status"]
        val_loss = result["val_loss"]
        improved = val_loss < best_val_loss

        if improved and status == "ok":
            best_val_loss = val_loss
            best_config = candidate
            best_model_state = result.pop("model_state", None)
            status_label = "keep"
            print(f"  NEW BEST: val_loss={val_loss:.6f}, "
                  f"params={result['num_params']:,}, "
                  f"steps={result['num_steps']}")
        elif status == "ok":
            result.pop("model_state", None)
            status_label = "discard"
            print(f"  discard:  val_loss={val_loss:.6f} "
                  f"(best={best_val_loss:.6f}), "
                  f"params={result['num_params']:,}")
        else:
            result.pop("model_state", None)
            status_label = "crash"
            print(f"  CRASH: {result['error']}")

        # Write TSV row
        config_desc = {k: v for k, v in candidate.items() if k != "name"}
        with open(results_tsv, "a", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow([
                name,
                f"{val_loss:.6f}" if val_loss < float("inf") else "inf",
                f"{result['train_loss']:.6f}" if result["train_loss"] < float("inf") else "inf",
                result["num_params"],
                result["num_steps"],
                f"{result['peak_memory_mb']:.1f}",
                status_label,
                str(config_desc),
            ])

        all_results.append({
            "name": name,
            "val_loss": val_loss,
            "status": status_label,
            "config": candidate,
            **{k: v for k, v in result.items() if k not in ("config", "model_state")},
        })

    # Save best model
    best_model_path = os.path.join(cfg.results_dir, "best_model.pth")
    if best_model_state is not None:
        torch.save({
            "model_state_dict": best_model_state,
            "config": best_config,
            "val_loss": best_val_loss,
        }, best_model_path)
        print(f"\nBest model saved: {best_model_path}")

    print(f"\n{'=' * 60}")
    print(f"NAS COMPLETE")
    print(f"Best architecture: {best_config.get('name', 'unknown')}")
    print(f"Best val_loss: {best_val_loss:.6f}")
    print(f"Results: {results_tsv}")
    print(f"{'=' * 60}")

    return NASResult(
        best_config=best_config or {},
        best_val_loss=best_val_loss,
        best_model_path=best_model_path,
        results_tsv=results_tsv,
        all_results=all_results,
    )
