"""Per-step timing instrumentation for trainer internals."""

from __future__ import annotations

import time
from enum import Enum, auto

import torch


class TimerSection(Enum):
    ENCODE = auto()
    TRAIN = auto()
    ACTION = auto()
    EVAL = auto()
    ENV = auto()
    # Fine-grained subsections (sum ≈ parent section)
    ACTION_ENV = auto()      # env after-state building inside greedy selection
    ACTION_ENCODE = auto()   # encode_all_agents_for_actions
    ACTION_FORWARD = auto()  # vmap forward (value computation)
    TRAIN_GRAD = auto()      # vmap grad_and_value call
    TRAIN_TRACE = auto()     # eligibility trace update loop
    TRAIN_V_NEXT = auto()    # V(s') forward pass
    TRAIN_PARAM = auto()     # parameter update loop
    # Note: EVAL_GREEDY/EVAL_HEURISTIC cannot be tracked with this flat timer
    # because the ACTION_* subsections fire inside them and overwrite the active
    # section before it accumulates.  Eval time is measured by wall-clock in train.py.


class Timer:
    """Accumulates wall-clock time per section. No-op when disabled."""

    def __init__(self, enabled: bool = False, gpu_sync: bool = False) -> None:
        self._enabled = enabled
        self._gpu_sync = gpu_sync
        self._accum: dict[TimerSection, float] = {s: 0.0 for s in TimerSection}
        self._active: TimerSection | None = None
        self._t0: float = 0.0
        self._step_count: int = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def step_begin(self) -> None:
        if not self._enabled:
            return
        self._step_count += 1

    def start(self, section: TimerSection) -> None:
        if not self._enabled:
            return
        if self._gpu_sync:
            torch.cuda.synchronize()
        self._active = section
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        if not self._enabled or self._active is None:
            return
        if self._gpu_sync:
            torch.cuda.synchronize()
        self._accum[self._active] += time.perf_counter() - self._t0
        self._active = None

    def report_and_reset(self) -> dict[TimerSection, float]:
        """Return average time per step for each section, then reset."""
        n = max(self._step_count, 1)
        result = {s: self._accum[s] / n for s in TimerSection}
        self._accum = {s: 0.0 for s in TimerSection}
        self._step_count = 0
        return result
