# deepassimilate/schedulers/distilled.py
from typing import Optional


def build_distilled_scheduler(
    base_scheduler,
    num_inference_steps: Optional[int],
):
    """
    Noise-scheduler-only 'distillation'.

    We don't touch UNet weights; we only change the timesteps used
    at inference (sampling / data assimilation).

    If num_inference_steps is None, we keep the original full schedule.
    Otherwise, we call scheduler.set_timesteps(num_inference_steps),
    letting diffusers choose an appropriate subset of timesteps.
    """
    if num_inference_steps is None:
        # Use the full training schedule
        base_scheduler.set_timesteps(base_scheduler.num_train_timesteps)
    else:
        base_scheduler.set_timesteps(num_inference_steps)
    return base_scheduler
