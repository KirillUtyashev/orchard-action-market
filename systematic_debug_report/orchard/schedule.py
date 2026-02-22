"""Schedule utilities for learning rate and epsilon decay."""

from orchard.enums import Schedule
from orchard.datatypes import ScheduleConfig


def compute_schedule_value(
    cfg: ScheduleConfig, step: int, total_steps: int | None = None
) -> float:
    """Compute current value given schedule config and step count.

    Used for both LR and epsilon.
    """
    if cfg.schedule == Schedule.NONE:
        return cfg.start
    elif cfg.schedule == Schedule.LINEAR:
        if total_steps is None:
            raise ValueError("total_steps required for linear schedule")
        fraction = min(step / total_steps, 1.0) if total_steps > 0 else 1.0
        return cfg.start + (cfg.end - cfg.start) * fraction
    elif cfg.schedule == Schedule.STEP:
        n_decays = step // cfg.step_size if cfg.step_size > 0 else 0
        return max(cfg.end, cfg.start * (cfg.step_factor ** n_decays))
    else:
        raise ValueError(f"Unknown schedule: {cfg.schedule}")
