from pathlib import Path

import numpy as np
import torch

from debug.code.core.enums import DISCOUNT_FACTOR, NUM_AGENTS, PROBABILITY_APPLE
from debug.code.training.helpers import eval_performance, random_policy
from debug.code.training.learning_diagnostics import LearningDiagnosticsMixin
from debug.code.training.learning_eval import LearningEvalMixin
from debug.code.training.learning_loop import LearningLoopMixin
from debug.code.training.learning_setup import LearningSetupMixin
from debug.code.training.learning_stategen import LearningStateGenerationMixin
from debug.code.core.log import (
    CSVLogger,
    build_action_prob_csv_fieldnames,
    build_main_csv_fieldnames,
    build_value_track_csv_fieldnames,
    setup_logging,
)
from debug.code.env.reward import Reward
from debug.code.nn.value import Value


class Learning(
    LearningSetupMixin,
    LearningStateGenerationMixin,
    LearningDiagnosticsMixin,
    LearningEvalMixin,
    LearningLoopMixin,
):
    def __init__(self, exp_config):
        self.env = None
        self.encoder = None
        self.agent_controller = None
        self.rng_state = None

        self.exp_config = exp_config
        self.trajectory_length = exp_config.train.timesteps
        self.discount_factor = DISCOUNT_FACTOR

        self.agents = []
        self.critic_networks = []
        self._networks_for_eval = []

        self.reward_module = Reward(exp_config.reward.picker_r, NUM_AGENTS)
        self.theoretical_val = Value(
            exp_config.reward.picker_r,
            NUM_AGENTS,
            DISCOUNT_FACTOR,
            PROBABILITY_APPLE,
            exp_config.eval.variance,
        )

        self.num_eval_states = exp_config.eval.num_eval_states
        self.reward_eval_num_states = max(1, int(getattr(exp_config.eval, "reward_eval_num_states", 1000)))

        self.train_start_time = None
        self._last_eval_errors_by_agent = None

        self._init_logging_and_diagnostics()
        self._init_eval_buffers()
        self._load_weights_if_requested()

    def _init_logging_and_diagnostics(self) -> None:
        self.data_dir = setup_logging(self.exp_config)
        self.agent_positions_dir = self.data_dir / "agent_positions"
        self.agent_positions_dir.mkdir(parents=True, exist_ok=True)
        self.last_greedy_positions_path: Path | None = None
        self.last_greedy_eval_step: int | None = None

        reward_learning_mode = bool(self.exp_config.reward.reward_learning)

        main_fields = build_main_csv_fieldnames(reward_learning=reward_learning_mode)
        self.main_logger = CSVLogger(self.data_dir / "metrics.csv", main_fields)

        action_prob_fields = build_action_prob_csv_fieldnames()
        if reward_learning_mode:
            self.action_prob_loggers = {}
        else:
            self.action_prob_loggers = {
                agent_id: CSVLogger(
                    self.data_dir / f"action_probabilities_agent_{agent_id}.csv",
                    action_prob_fields,
                )
                for agent_id in range(NUM_AGENTS)
            }

        self.value_track_num_states = max(
            0,
            min(10, int(getattr(self.exp_config.eval, "value_track_num_states", 10))),
        )
        value_track_fields = build_value_track_csv_fieldnames(self.value_track_num_states)
        if reward_learning_mode:
            self.value_track_loggers = {}
        else:
            self.value_track_loggers = {
                agent_id: CSVLogger(
                    self.data_dir / f"tracked_state_values_agent_{agent_id}.csv",
                    value_track_fields,
                )
                for agent_id in range(NUM_AGENTS)
            }

        self.weight_samples_enabled = bool(getattr(self.exp_config.logging, "weight_samples_enabled", True))
        self.weight_samples_per_tensor = max(
            1,
            int(getattr(self.exp_config.logging, "weight_samples_per_tensor", 16)),
        )
        self.weight_samples_freq = int(getattr(self.exp_config.logging, "weight_samples_freq", 0))
        self.weight_sample_indices: dict[int, dict[str, np.ndarray]] = {}
        self.weight_sample_loggers: dict[int, CSVLogger] = {}

    def _init_eval_buffers(self) -> None:
        self.action_prob_num_states = max(0, int(getattr(self.exp_config.eval, "action_prob_num_states", 100)))
        self.action_prob_burnin = max(0, int(getattr(self.exp_config.eval, "action_prob_burnin", 500)))
        self.action_prob_stride = max(1, int(getattr(self.exp_config.eval, "action_prob_stride", 5)))
        self.action_prob_seed = 42069
        self.action_prob_eval_states = None
        self.value_track_states_by_agent = None

        self.careful_evals = []
        self.focus_actor_id = 0
        self.careful_distances = (2, 1)
        self.careful_actor_states = [None for _ in range(len(self.careful_distances))]
        self.careful_eval_steps = []
        self.careful_pred_history_actor0 = [
            [[] for _ in range(len(self.careful_distances))] for _ in range(NUM_AGENTS)
        ]

    def _load_weights_if_requested(self) -> None:
        if self.exp_config.train.load_weights:
            path = self.data_dir / "weights" / "weights.pt"
            ckpt = torch.load(path, map_location="cpu")
            self.crit_blobs = ckpt.get("critics", [])
