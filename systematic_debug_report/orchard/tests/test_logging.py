"""Tests for logging utilities."""

import csv
import tempfile
from pathlib import Path

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
    def test_main_csv_fields(self):
        fields = build_main_csv_fieldnames("nearest_correct_task")
        
        assert "step" in fields
        assert "greedy_rps" in fields
        assert "greedy_correct_pps" in fields
        assert "greedy_wrong_pps" in fields
        assert "nearest_correct_task_rps" in fields
        assert "td_loss_avg" in fields