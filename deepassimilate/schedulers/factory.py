# deepassimilate/schedulers/factory.py
from typing import Dict, Any, Optional
from diffusers import HeunDiscreteScheduler, DDPMScheduler, DDIMScheduler


def build_scheduler(
    name: str,
    num_train_timesteps: int = 1000,
    scheduler_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Build a noise scheduler from diffusers.

    Args:
        name: "heun_edm", "ddpm", or "ddim".
        num_train_timesteps: number of training timesteps.
        scheduler_kwargs: extra kwargs passed to the scheduler.
    """
    kwargs = dict(scheduler_kwargs or {})
    if name == "heun_edm":
        return HeunDiscreteScheduler(num_train_timesteps=num_train_timesteps, **kwargs)
    elif name == "ddpm":
        return DDPMScheduler(num_train_timesteps=num_train_timesteps, **kwargs)
    elif name == "ddim":
        return DDIMScheduler(num_train_timesteps=num_train_timesteps, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler: {name}")
