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
    build_following_rate_csv_fieldnames,
    build_influencer_csv_fieldnames,
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
        self.policy_networks = []
        self._networks_for_eval = []
        self.crit_blobs = []
        self.actor_blobs = []
        self.rate_blobs = []
        self.influencer_blob = None
        self.loaded_critic_checkpoint_path: Path | None = None
        self.loaded_critic_smoke_test_result = None
        self.external_influencer = None
        self._last_actor_training_stats = {
            "actor_loss_mean": None,
            "advantage_mean": None,
            "policy_entropy_mean": None,
        }
        self._last_following_rate_stats = {
            "alpha_mean": None,
            "alpha_positive_frac": None,
            "following_weight_mean": None,
            "active_follow_edges_mean": None,
            "beta_mean": None,
            "influencer_weight_mean": None,
            "follower_to_influencer_weight_mean": None,
            "effective_follow_weight_mean": None,
        }

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
        net_cfg = self._shared_network_cfg()
        if bool(getattr(net_cfg, "self_centered_grid", False)):
            if self.exp_config.algorithm.centralized or not net_cfg.CNN:
                raise ValueError(
                    "critic_network.self_centered_grid=true and actor_network.self_centered_grid=true "
                    "are only supported for decentralized CNN runs."
                )
        if bool(getattr(self.exp_config.algorithm, "actor_critic", False)):
            if bool(self.exp_config.algorithm.centralized):
                raise ValueError("algorithm.actor_critic=true is only supported for decentralized training in v1.")
            if bool(self.exp_config.reward.reward_learning):
                raise ValueError("algorithm.actor_critic=true is not supported with reward.reward_learning=true.")
        if bool(getattr(self.exp_config.algorithm, "following_rates", False)):
            if not bool(getattr(self.exp_config.algorithm, "actor_critic", False)):
                raise ValueError("algorithm.following_rates=true requires algorithm.actor_critic=true.")
            if bool(self.exp_config.algorithm.centralized):
                raise ValueError("algorithm.following_rates=true is only supported for decentralized training.")
            if self.num_agents < 2:
                raise ValueError("algorithm.following_rates=true requires env.num_agents >= 2.")
            budget = float(getattr(self.exp_config.train, "following_rate_budget", 0.0))
            rho = float(getattr(self.exp_config.train, "following_rate_rho", 0.0))
            realloc_freq = int(getattr(self.exp_config.train, "following_rate_reallocation_freq", 0))
            if budget < 0.0:
                raise ValueError("train.following_rate_budget must be >= 0.")
            if not (0.0 < rho <= 1.0):
                raise ValueError("train.following_rate_rho must be in (0, 1].")
            if realloc_freq <= 0:
                raise ValueError("train.following_rate_reallocation_freq must be >= 1.")
            if bool(getattr(self.exp_config.algorithm, "influencer", False)):
                influencer_budget = float(getattr(self.exp_config.train, "influencer_budget", 0.0))
                if influencer_budget < 0.0:
                    raise ValueError("train.influencer_budget must be >= 0.")
        elif bool(getattr(self.exp_config.algorithm, "influencer", False)):
            raise ValueError("algorithm.influencer=true requires algorithm.following_rates=true.")
        elif bool(getattr(self.exp_config.train, "fixed_following_rates", False)):
            raise ValueError("train.fixed_following_rates=true requires algorithm.following_rates=true.")
        if bool(getattr(self.exp_config.train, "load_policy_weights", False)) and not bool(
            getattr(self.exp_config.algorithm, "actor_critic", False)
        ):
            raise ValueError("train.load_policy_weights=true requires algorithm.actor_critic=true.")
        if bool(getattr(self.exp_config.train, "load_policy_weights", False)) and not bool(
            getattr(self.exp_config.train, "load_weights", False)
        ):
            if not str(getattr(self.exp_config.train, "critic_weights_path", "")).strip():
                raise ValueError(
                    "train.load_policy_weights=true requires train.critic_weights_path or train.load_weights=true."
                )
        if bool(getattr(self.exp_config.train, "freeze_critics", False)) and not self._should_load_critics():
            raise ValueError(
                "train.freeze_critics=true requires train.critic_weights_path, train.load_weights=true, "
                "or train.load_policy_weights=true."
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
            actor_critic=bool(getattr(self.exp_config.algorithm, "actor_critic", False)),
            following_rates=bool(getattr(self.exp_config.algorithm, "following_rates", False)),
            influencer=bool(getattr(self.exp_config.algorithm, "influencer", False)),
            critic_smoke=bool(getattr(self.exp_config.train, "freeze_critics", False)) and self._should_load_critics(),
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

        if reward_learning_mode or not bool(getattr(self.exp_config.algorithm, "following_rates", False)):
            self.following_rate_loggers = {}
        else:
            follow_fields = build_following_rate_csv_fieldnames(self.num_agents)
            self.following_rate_loggers = {
                agent_id: CSVLogger(
                    self.data_dir / f"following_rates_agent_{agent_id}.csv",
                    follow_fields,
                )
                for agent_id in range(self.num_agents)
            }
        if reward_learning_mode or not bool(getattr(self.exp_config.algorithm, "influencer", False)):
            self.influencer_logger = None
        else:
            self.influencer_logger = CSVLogger(
                self.data_dir / "external_influencer.csv",
                build_influencer_csv_fieldnames(self.num_agents),
            )

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

    def _should_load_critics(self) -> bool:
        return (
            bool(getattr(self.exp_config.train, "load_weights", False))
            or bool(getattr(self.exp_config.train, "load_policy_weights", False))
            or bool(
            str(getattr(self.exp_config.train, "critic_weights_path", "")).strip()
            )
        )

    def _should_load_policy_weights(self) -> bool:
        return bool(getattr(self.exp_config.train, "load_weights", False)) or bool(
            getattr(self.exp_config.train, "load_policy_weights", False)
        )

    def _resolve_critic_checkpoint_path(self) -> Path:
        raw_path = str(getattr(self.exp_config.train, "critic_weights_path", "")).strip()
        if raw_path:
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
            if path.is_dir():
                path = path / "weights.pt"
            if not path.exists():
                raise FileNotFoundError(f"Critic checkpoint not found: {path}")
            return path
        return self.data_dir / "weights" / "weights.pt"

    def _load_weights_if_requested(self) -> None:
        if self._should_load_critics():
            path = self._resolve_critic_checkpoint_path()
            ckpt = torch.load(path, map_location="cpu")
            self.loaded_critic_checkpoint_path = path
            self.crit_blobs = ckpt.get("critics", [])
            if self._should_load_policy_weights():
                self.actor_blobs = ckpt.get("actors", [])
            else:
                self.actor_blobs = []
            if bool(getattr(self.exp_config.train, "load_weights", False)):
                self.rate_blobs = ckpt.get("following_rate_agents", [])
                self.influencer_blob = ckpt.get("external_influencer")
            else:
                self.rate_blobs = []
                self.influencer_blob = None
