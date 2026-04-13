"""Tests for timing instrumentation: Timer, config, and timing.csv integration."""

import csv
import os
import tempfile
import time

import pytest

from orchard.trainer.timer import Timer, TimerSection


# ---------------------------------------------------------------------------
# TimerSection enum
# ---------------------------------------------------------------------------
class TestTimerSection:
    def test_env_section_exists(self):
        assert hasattr(TimerSection, "ENV")

    def test_all_sections_present(self):
        expected = {"ENCODE", "TRAIN", "ACTION", "EVAL", "ENV"}
        actual = {s.name for s in TimerSection}
        assert expected == actual


# ---------------------------------------------------------------------------
# Timer unit tests
# ---------------------------------------------------------------------------
class TestTimerDisabled:
    def test_disabled_is_noop(self):
        t = Timer(enabled=False)
        t.step_begin()
        t.start(TimerSection.ENV)
        time.sleep(0.01)
        t.stop()
        report = t.report_and_reset()
        # All zeros when disabled
        for section in TimerSection:
            assert report[section] == 0.0

    def test_disabled_report_returns_all_sections(self):
        t = Timer(enabled=False)
        report = t.report_and_reset()
        assert set(report.keys()) == set(TimerSection)


class TestTimerEnabled:
    def test_accumulates_time(self):
        t = Timer(enabled=True)
        t.step_begin()
        t.start(TimerSection.ENV)
        time.sleep(0.02)
        t.stop()
        report = t.report_and_reset()
        # Should have recorded ~20ms (1 step, so average = total)
        assert report[TimerSection.ENV] > 0.01
        # Other sections untouched
        assert report[TimerSection.TRAIN] == 0.0

    def test_step_begin_affects_average(self):
        t = Timer(enabled=True)
        # 2 steps, but only time once
        t.step_begin()
        t.step_begin()
        t.start(TimerSection.TRAIN)
        time.sleep(0.02)
        t.stop()
        report = t.report_and_reset()
        # Average over 2 steps → should be roughly half the raw time
        assert report[TimerSection.TRAIN] < 0.02

    def test_report_resets(self):
        t = Timer(enabled=True)
        t.step_begin()
        t.start(TimerSection.ACTION)
        time.sleep(0.01)
        t.stop()

        report1 = t.report_and_reset()
        assert report1[TimerSection.ACTION] > 0.0

        report2 = t.report_and_reset()
        assert report2[TimerSection.ACTION] == 0.0

    def test_multiple_sections_per_step(self):
        t = Timer(enabled=True)
        t.step_begin()

        t.start(TimerSection.ACTION)
        time.sleep(0.01)
        t.stop()

        t.start(TimerSection.ENV)
        time.sleep(0.01)
        t.stop()

        t.start(TimerSection.TRAIN)
        time.sleep(0.01)
        t.stop()

        report = t.report_and_reset()
        for s in [TimerSection.ACTION, TimerSection.ENV, TimerSection.TRAIN]:
            assert report[s] > 0.005


# ---------------------------------------------------------------------------
# Config / datatype tests
# ---------------------------------------------------------------------------
class TestTimingConfig:
    def test_logging_config_default(self):
        from orchard.datatypes import LoggingConfig
        cfg = LoggingConfig()
        assert cfg.timing_csv_freq == 0

    def test_logging_config_custom(self):
        from orchard.datatypes import LoggingConfig
        cfg = LoggingConfig(timing_csv_freq=1000)
        assert cfg.timing_csv_freq == 1000

    def test_config_parses_timing_csv_freq(self):
        from orchard.config import load_config

        yaml_str = """
env:
  height: 3
  width: 3
  n_agents: 2
  gamma: 0.99
  stochastic:
    spawn_prob: 0.0
    despawn_mode: none
model:
  encoder: blind_task_cnn_grid
  mlp_dims: [8]
train:
  total_steps: 10
  lr:
    start: 0.01
logging:
  timing_csv_freq: 500
"""
        fd, path = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, "w") as f:
            f.write(yaml_str)

        cfg = load_config(path)
        os.unlink(path)
        assert cfg.logging.timing_csv_freq == 500

    def test_config_override_timing_csv_freq(self):
        from orchard.config import load_config

        yaml_str = """
env:
  height: 3
  width: 3
  n_agents: 2
  gamma: 0.99
  stochastic:
    spawn_prob: 0.0
    despawn_mode: none
model:
  encoder: blind_task_cnn_grid
  mlp_dims: [8]
train:
  total_steps: 10
  lr:
    start: 0.01
"""
        fd, path = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, "w") as f:
            f.write(yaml_str)

        cfg = load_config(path, overrides=["logging.timing_csv_freq=2000"])
        os.unlink(path)
        assert cfg.logging.timing_csv_freq == 2000


# ---------------------------------------------------------------------------
# Integration: timing.csv produced by train()
# ---------------------------------------------------------------------------
_INTEGRATION_YAML = """
env:
  height: 4
  width: 4
  n_agents: 2
  n_tasks: 2
  n_task_types: 2
  gamma: 0.99
  r_picker: 1.0
  r_low: 0.0
  pick_mode: forced
  max_tasks_per_type: 2
  task_assignments: [[0], [1]]
  stochastic:
    spawn_prob: 0.1
    despawn_mode: probability
    despawn_prob: 0.05
model:
  encoder: blind_task_cnn_grid
  mlp_dims: [8]
  conv_specs: [[4, 3]]
train:
  learning_type: decentralized
  use_gpu: false
  td_lambda: 0.3
  total_steps: 10
  seed: 42
  heuristic: nearest_correct_task
  lr:
    start: 0.01
  epsilon:
    start: 0.3
eval:
  eval_steps: 5
  n_test_states: 2
logging:
  main_csv_freq: 10
  detail_csv_freq: 10
  output_dir: {output_dir}
  timing_csv_freq: {timing_csv_freq}
"""


def _run_integration(timing_csv_freq: int):
    """Run a short train() and return the run directory path."""
    from orchard.config import load_config
    from orchard.train import train

    tmpdir = tempfile.mkdtemp()
    yaml_str = _INTEGRATION_YAML.format(
        output_dir=tmpdir, timing_csv_freq=timing_csv_freq
    )
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(yaml_str)

    cfg = load_config(path)
    train(cfg)
    os.unlink(path)

    run_dirs = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
    assert len(run_dirs) == 1
    return os.path.join(tmpdir, run_dirs[0])


class TestTimingIntegration:
    def test_timing_csv_created_when_enabled(self):
        run_dir = _run_integration(timing_csv_freq=5)
        timing_path = os.path.join(run_dir, "timing.csv")
        assert os.path.exists(timing_path)

        with open(timing_path) as f:
            reader = csv.DictReader(f)
            assert set(reader.fieldnames) == {
                "step", "wall_time",
                "encode_ms", "train_ms", "action_ms", "env_ms", "eval_ms",
            }
            rows = list(reader)
            # 10 total steps / freq 5 = 2 rows
            assert len(rows) == 2
            assert rows[0]["step"] == "5"
            assert rows[1]["step"] == "10"

    def test_timing_values_are_positive(self):
        run_dir = _run_integration(timing_csv_freq=5)
        timing_path = os.path.join(run_dir, "timing.csv")

        with open(timing_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # These sections are always active during training
                assert float(row["encode_ms"]) > 0
                assert float(row["train_ms"]) > 0
                assert float(row["action_ms"]) > 0
                assert float(row["env_ms"]) > 0

    def test_no_timing_csv_when_disabled(self):
        run_dir = _run_integration(timing_csv_freq=0)
        timing_path = os.path.join(run_dir, "timing.csv")
        assert not os.path.exists(timing_path)

    def test_metrics_csv_still_works_with_timing(self):
        """Timing instrumentation should not break normal logging."""
        run_dir = _run_integration(timing_csv_freq=5)
        metrics_path = os.path.join(run_dir, "metrics.csv")
        assert os.path.exists(metrics_path)

        with open(metrics_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            assert "greedy_rps" in rows[0]
