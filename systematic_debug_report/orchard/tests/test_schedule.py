"""Tests for scheduling utilities (LR and epsilon decay)."""

import pytest
from orchard.enums import Schedule
from orchard.schedule import compute_schedule_value
from orchard.datatypes import ScheduleConfig


class TestScheduleNone:
    def test_constant(self):
        cfg = ScheduleConfig(start=0.001, end=0.0001, schedule=Schedule.NONE)
        assert compute_schedule_value(cfg, 0) == 0.001
        assert compute_schedule_value(cfg, 5000) == 0.001
        assert compute_schedule_value(cfg, 10000) == 0.001


class TestScheduleLinear:
    def test_start(self):
        cfg = ScheduleConfig(start=1.0, end=0.0, schedule=Schedule.LINEAR)
        assert compute_schedule_value(cfg, 0, 100) == 1.0

    def test_midpoint(self):
        cfg = ScheduleConfig(start=1.0, end=0.0, schedule=Schedule.LINEAR)
        assert pytest.approx(compute_schedule_value(cfg, 50, 100)) == 0.5

    def test_end(self):
        cfg = ScheduleConfig(start=1.0, end=0.0, schedule=Schedule.LINEAR)
        assert compute_schedule_value(cfg, 100, 100) == 0.0

    def test_beyond_end(self):
        cfg = ScheduleConfig(start=1.0, end=0.0, schedule=Schedule.LINEAR)
        assert compute_schedule_value(cfg, 200, 100) == 0.0

    def test_requires_total_steps(self):
        cfg = ScheduleConfig(start=1.0, end=0.0, schedule=Schedule.LINEAR)
        with pytest.raises(ValueError, match="total_steps required"):
            compute_schedule_value(cfg, 50)


class TestScheduleStep:
    def test_no_decay_before_step(self):
        cfg = ScheduleConfig(
            start=1.0, end=0.1, schedule=Schedule.STEP,
            step_size=100, step_factor=0.5,
        )
        assert compute_schedule_value(cfg, 0) == 1.0
        assert compute_schedule_value(cfg, 99) == 1.0

    def test_one_decay(self):
        cfg = ScheduleConfig(
            start=1.0, end=0.1, schedule=Schedule.STEP,
            step_size=100, step_factor=0.5,
        )
        assert compute_schedule_value(cfg, 100) == 0.5

    def test_two_decays(self):
        cfg = ScheduleConfig(
            start=1.0, end=0.1, schedule=Schedule.STEP,
            step_size=100, step_factor=0.5,
        )
        assert compute_schedule_value(cfg, 200) == 0.25

    def test_floor_at_end(self):
        cfg = ScheduleConfig(
            start=1.0, end=0.3, schedule=Schedule.STEP,
            step_size=100, step_factor=0.5,
        )
        # After 2 decays: 1.0 * 0.25 = 0.25. Since 0.25 < 0.3, floor at 0.3.
        assert compute_schedule_value(cfg, 200) == 0.3