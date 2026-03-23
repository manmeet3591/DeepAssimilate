# deepassimilate/schedulers/factory.py
"""Scheduler factory supporting all diffusers noise schedulers."""

from typing import Dict, Any, Optional

SCHEDULER_MAP = {
    "heun_edm": "HeunDiscreteScheduler",
    "ddpm": "DDPMScheduler",
    "ddim": "DDIMScheduler",
    "euler": "EulerDiscreteScheduler",
    "euler_ancestral": "EulerAncestralDiscreteScheduler",
    "dpm_solver": "DPMSolverMultistepScheduler",
    "pndm": "PNDMScheduler",
    "lms": "LMSDiscreteScheduler",
}


def list_schedulers():
    """Return available scheduler short names."""
    return list(SCHEDULER_MAP.keys())


def build_scheduler(
    name: str,
    num_train_timesteps: int = 1000,
    scheduler_kwargs: Optional[Dict[str, Any]] = None,
):
    """Build a noise scheduler from diffusers.

    Args:
        name: Short name (see list_schedulers()) or full diffusers class name.
        num_train_timesteps: Number of training timesteps.
        scheduler_kwargs: Extra kwargs passed to the scheduler constructor.

    Returns:
        A diffusers scheduler instance.
    """
    import diffusers

    kwargs = dict(scheduler_kwargs or {})

    # Resolve short name -> class name
    class_name = SCHEDULER_MAP.get(name, name)

    scheduler_class = getattr(diffusers, class_name, None)
    if scheduler_class is None:
        raise ValueError(
            f"Unknown scheduler: '{name}'. "
            f"Available short names: {list_schedulers()}. "
            f"Or use a full diffusers class name like 'DDPMScheduler'."
        )

    return scheduler_class(num_train_timesteps=num_train_timesteps, **kwargs)
