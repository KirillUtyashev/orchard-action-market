from pathlib import Path

import numpy as np
import torch

from debug.code.core.enums import DISCOUNT_FACTOR, PROBABILITY_APPLE
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
    build_pipeline_profile_csv_fieldnames,
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
        self.num_agents = int(exp_config.env.num_agents)
        self.width = int(exp_config.env.width)
        self.length = int(exp_config.env.length)
        self.supervised_enabled = bool(exp_config.reward.supervised)
        self.supervised_networks = []

        self.agents = []
        self.critic_networks = []
        self._networks_for_eval = []

        self.reward_module = Reward(exp_config.reward.picker_r, self.num_agents)
        self.theoretical_val = Value(
            exp_config.reward.picker_r,
            self.num_agents,
            DISCOUNT_FACTOR,
            PROBABILITY_APPLE,
            exp_config.eval.variance,
        )

        self.num_eval_states = exp_config.eval.num_eval_states
        self.reward_eval_num_states = max(1, int(getattr(exp_config.eval, "reward_eval_num_states", 1000)))
        self.reward_eval_zero_frac = min(
            1.0,
            max(0.0, float(getattr(exp_config.eval, "reward_eval_zero_frac", 0.25))),
        )
        self.supervised_eval_num_states = max(1, int(getattr(exp_config.eval, "supervised_eval_num_states", 1000)))

        self.train_start_time = None
        self._last_eval_errors_by_agent = None
        self.pipeline_profile_enabled = bool(getattr(self.exp_config.profiling, "enabled", False))
        self.pipeline_profile_include_eval = bool(getattr(self.exp_config.profiling, "include_eval", True))
        self.pipeline_profile_freq = int(getattr(self.exp_config.profiling, "csv_freq", 0))
        if self.pipeline_profile_freq <= 0:
            self.pipeline_profile_freq = int(self.exp_config.logging.main_csv_freq)
        self.pipeline_profile_stage_totals: dict[str, float] = {}
        self.pipeline_profile_stage_snapshot: dict[str, float] = {}
        self.pipeline_profile_last_step = 0
        self.pipeline_profile_last_wall_time = 0.0
        self.pipeline_profile_logger: CSVLogger | None = None
        self.cprofile_enabled = bool(getattr(self.exp_config.profiling, "cprofile", False))
        self.cprofile_sort_by = str(getattr(self.exp_config.profiling, "cprofile_sort_by", "cumulative"))
        self.cprofile_top_n = max(1, int(getattr(self.exp_config.profiling, "cprofile_top_n", 200)))

        self._validate_training_modes()
        self._init_logging_and_diagnostics()
        self._init_eval_buffers()
        self._load_weights_if_requested()

    def _validate_training_modes(self) -> None:
        if bool(getattr(self.exp_config.network, "self_centered_grid", False)):
            if self.exp_config.algorithm.centralized or not self.exp_config.network.CNN:
                raise ValueError(
                    "network.self_centered_grid=true is only supported for decentralized CNN critics."
                )
        if self.exp_config.reward.reward_learning and self.supervised_enabled:
            raise ValueError("reward.reward_learning and reward.supervised cannot both be enabled.")
        if self.supervised_enabled:
            if not str(getattr(self.exp_config.supervised, "weights_path", "")).strip():
                raise ValueError(
                    "reward.supervised=true requires supervised.weights_path to be set in the config."
                )
            if not tuple(getattr(self.exp_config.supervised, "mlp_dims", ())):
                raise ValueError("reward.supervised=true requires supervised.mlp_dims to be non-empty.")
            if bool(getattr(self.exp_config.supervised, "CNN", False)) and not list(
                getattr(self.exp_config.supervised, "conv_channels", [])
            ):
                raise ValueError(
                    "reward.supervised=true with supervised.CNN=true requires supervised.conv_channels."
                )

    def _init_logging_and_diagnostics(self) -> None:
        self.data_dir = setup_logging(self.exp_config)
        self.agent_positions_dir = self.data_dir / "agent_positions"
        self.agent_positions_dir.mkdir(parents=True, exist_ok=True)
        self.last_greedy_positions_path: Path | None = None
        self.last_greedy_eval_step: int | None = None

        reward_learning_mode = bool(self.exp_config.reward.reward_learning)

        main_fields = build_main_csv_fieldnames(
            reward_learning=reward_learning_mode,
            supervised=self.supervised_enabled,
        )
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
                for agent_id in range(self.num_agents)
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
                for agent_id in range(self.num_agents)
            }

        self.weight_samples_enabled = bool(getattr(self.exp_config.logging, "weight_samples_enabled", True))
        self.weight_samples_per_tensor = max(
            1,
            int(getattr(self.exp_config.logging, "weight_samples_per_tensor", 16)),
        )
        self.weight_samples_freq = int(getattr(self.exp_config.logging, "weight_samples_freq", 0))
        self.weight_sample_indices: dict[int, dict[str, np.ndarray]] = {}
        self.weight_sample_loggers: dict[int, CSVLogger] = {}
        if self.pipeline_profile_enabled:
            self.pipeline_profile_logger = CSVLogger(
                self.data_dir / "pipeline_profile.csv",
                build_pipeline_profile_csv_fieldnames(),
            )

    def _init_eval_buffers(self) -> None:
        self.action_prob_num_states = max(0, int(getattr(self.exp_config.eval, "action_prob_num_states", 100)))
        self.action_prob_burnin = max(0, int(getattr(self.exp_config.eval, "action_prob_burnin", 500)))
        self.action_prob_stride = max(1, int(getattr(self.exp_config.eval, "action_prob_stride", 5)))
        self.action_prob_seed = 42069
        self.action_prob_eval_states = None
        self.supervised_evaluation_states = None
        self.value_track_states_by_agent = None

        self.careful_evals = []
        self.focus_actor_id = 0
        self.careful_distances = (2, 1)
        self.careful_actor_states = [None for _ in range(len(self.careful_distances))]
        self.careful_eval_steps = []
        self.careful_pred_history_actor0 = [
            [[] for _ in range(len(self.careful_distances))] for _ in range(self.num_agents)
        ]

    def _load_weights_if_requested(self) -> None:
        if self.exp_config.train.load_weights:
            path = self.data_dir / "weights" / "weights.pt"
            ckpt = torch.load(path, map_location="cpu")
            self.crit_blobs = ckpt.get("critics", [])
