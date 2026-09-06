"""Tests that CLI resume flags are forwarded from main() to train().

Regression test for the bug where --resume-critic-only and --resume-actor-only
were parsed by argparse but silently dropped — never passed to train() — causing
the critic to be frozen at random initialisation instead of loading pretrained weights.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

import orchard.train as train_module
from orchard.train import main


def _run_main(argv: list[str]) -> MagicMock:
    """Patch train() and load_config(), run main() with the given argv, return the train mock."""
    fake_cfg = MagicMock()
    with (
        patch.object(train_module, "load_config", return_value=fake_cfg),
        patch.object(train_module, "train") as mock_train,
        patch.object(sys, "argv", ["orchard.train"] + argv),
    ):
        main()
    return mock_train


class TestCliResumeForwarding:
    """Every resume flag that argparse accepts must reach train()."""

    def test_resume_checkpoint_forwarded(self, tmp_path):
        ckpt = str(tmp_path / "final.pt")
        mock_train = _run_main(["--config", "cfg.yaml", "--resume", ckpt])
        mock_train.assert_called_once()
        _, kwargs = mock_train.call_args
        assert kwargs["resume_checkpoint"] == ckpt

    def test_resume_critic_only_forwarded(self, tmp_path):
        ckpt = str(tmp_path / "final.pt")
        mock_train = _run_main(["--config", "cfg.yaml", "--resume-critic-only", ckpt])
        mock_train.assert_called_once()
        _, kwargs = mock_train.call_args
        assert kwargs["resume_critic_only"] == ckpt, (
            "--resume-critic-only was parsed but not forwarded to train(). "
            "This is the bug that caused critics to be frozen at random init."
        )

    def test_resume_actor_only_forwarded(self, tmp_path):
        ckpt = str(tmp_path / "final.pt")
        mock_train = _run_main(["--config", "cfg.yaml", "--resume-actor-only", ckpt])
        mock_train.assert_called_once()
        _, kwargs = mock_train.call_args
        assert kwargs["resume_actor_only"] == ckpt

    def test_all_resume_flags_forwarded_together(self, tmp_path):
        full = str(tmp_path / "full.pt")
        critic = str(tmp_path / "critic.pt")
        actor = str(tmp_path / "actor.pt")
        mock_train = _run_main([
            "--config", "cfg.yaml",
            "--resume", full,
            "--resume-critic-only", critic,
            "--resume-actor-only", actor,
        ])
        _, kwargs = mock_train.call_args
        assert kwargs["resume_checkpoint"] == full
        assert kwargs["resume_critic_only"] == critic
        assert kwargs["resume_actor_only"] == actor

    def test_omitted_resume_flags_are_none(self):
        """When no resume flags are given, all three must be None, not absent."""
        mock_train = _run_main(["--config", "cfg.yaml"])
        _, kwargs = mock_train.call_args
        assert kwargs.get("resume_checkpoint") is None
        assert kwargs.get("resume_critic_only") is None
        assert kwargs.get("resume_actor_only") is None
