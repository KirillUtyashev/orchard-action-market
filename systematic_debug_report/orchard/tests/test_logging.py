"""Tests for logging utilities."""

import csv
import tempfile
from pathlib import Path

from orchard.enums import TrainMode
from orchard.logging_ import CSVLogger, build_main_csv_fieldnames


class TestCSVLogger:
    def test_creates_file_with_header(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.csv"
            fields = ["step", "loss"]
            logger = CSVLogger(path, fields)
            logger.close()

            with open(path) as f:
                reader = csv.DictReader(f)
                assert reader.fieldnames == ["step", "loss"]
                rows = list(reader)
                assert len(rows) == 0

    def test_log_row(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.csv"
            fields = ["step", "loss"]
            logger = CSVLogger(path, fields)
            logger.log({"step": 1, "loss": 0.5})
            logger.log({"step": 2, "loss": 0.3})
            logger.close()

            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 2
                assert rows[0]["step"] == "1"
                assert rows[1]["loss"] == "0.3"


class TestFieldnames:
    def test_value_learning_fields(self):
        fields = build_main_csv_fieldnames(2, TrainMode.VALUE_LEARNING)
        assert "step" in fields
        assert "mae_agent_0" in fields
        assert "mae_agent_1" in fields
        assert "mae_avg" in fields
        assert "bias_avg" in fields
        assert "td_loss_avg" in fields

    def test_policy_learning_legacy_fields(self):
        fields = build_main_csv_fieldnames(2, TrainMode.POLICY_LEARNING)
        assert "greedy_rps" in fields
        assert "nearest_task_rps" in fields
        assert "greedy_pps" not in fields
        assert "nearest_pps" not in fields
        assert "td_loss_avg" in fields

    def test_policy_learning_task_spec_fields(self):
        fields = build_main_csv_fieldnames(
            4, TrainMode.POLICY_LEARNING, n_task_types=4,
            heuristic_name="nearest_correct_task",
        )
        assert "greedy_rps" in fields
        assert "greedy_correct_pps" in fields
        assert "greedy_wrong_pps" in fields
        assert "nearest_correct_task_rps" in fields
        assert "greedy_pps" not in fields  # legacy field should NOT appear

    def test_reward_learning_legacy_fields(self):
        fields = build_main_csv_fieldnames(2, TrainMode.REWARD_LEARNING)
        assert "mae_avg" in fields
        assert "mae_zero_avg" in fields
        assert "mae_picker_avg" in fields

    def test_reward_learning_task_spec_fields(self):
        fields = build_main_csv_fieldnames(
            4, TrainMode.REWARD_LEARNING, n_networks=4, n_task_types=4,
        )
        assert "mae_avg" in fields
        assert "mae_no_pick_avg" in fields
        assert "mae_my_task_avg" in fields
        assert "mae_other_task_avg" in fields
