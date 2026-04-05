"""Tests for logging utilities."""

import csv
import tempfile
from pathlib import Path

from orchard.logging_ import (
    CSVLogger,
    build_action_prob_csv_fieldnames,
    build_following_rate_csv_fieldnames,
    build_influencer_csv_fieldnames,
    build_main_csv_fieldnames,
    build_phase1_policy_prob_csv_fieldnames,
    build_phase2_policy_prob_csv_fieldnames,
)

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

    def test_main_csv_fields_actor_critic_extensions(self):
        fields = build_main_csv_fieldnames(
            "nearest_correct_task",
            actor_critic=True,
            following_rates=True,
            influencer=True,
        )

        assert "actor_lr" in fields
        assert "actor_loss_mean" in fields
        assert "advantage_mean" in fields
        assert "policy_entropy_mean" in fields
        assert "alpha_mean" in fields
        assert "effective_follow_weight_mean" in fields
        assert "beta_mean" in fields
        assert "influencer_weight_mean" in fields

    def test_action_prob_csv_fields_exact(self):
        fields = build_action_prob_csv_fieldnames()

        assert fields == ["step", "wall_time", "left", "right", "up", "down", "stay"]

    def test_following_rate_csv_fields_exact(self):
        fields = build_following_rate_csv_fieldnames(2)

        assert fields == [
            "step",
            "wall_time",
            "alpha_to_0",
            "alpha_to_1",
            "lambda_to_0",
            "lambda_to_1",
            "weight_to_0",
            "weight_to_1",
            "lambda_to_influencer",
            "weight_to_influencer",
            "influencer_value",
        ]

    def test_influencer_csv_fields_exact(self):
        fields = build_influencer_csv_fieldnames(2)

        assert fields == [
            "step",
            "wall_time",
            "beta_to_actor_0",
            "beta_to_actor_1",
            "lambda_to_actor_0",
            "lambda_to_actor_1",
            "weight_to_actor_0",
            "weight_to_actor_1",
        ]

    def test_phase1_policy_prob_fields_exact(self):
        fields = build_phase1_policy_prob_csv_fieldnames(2)

        assert fields == [
            "step",
            "wall_time",
            "state_id",
            "actor_id",
            "state_json",
            "prob_up",
            "prob_down",
            "prob_left",
            "prob_right",
            "prob_stay",
            "prob_pick_0",
            "prob_pick_1",
        ]

    def test_phase2_policy_prob_fields_exact(self):
        fields = build_phase2_policy_prob_csv_fieldnames(2)

        assert fields == [
            "step",
            "wall_time",
            "state_id",
            "state_label",
            "actor_id",
            "state_json",
            "present_type_0",
            "present_type_1",
            "assigned_type_0",
            "assigned_type_1",
            "prob_up",
            "prob_down",
            "prob_left",
            "prob_right",
            "prob_stay",
            "prob_pick_0",
            "prob_pick_1",
        ]
