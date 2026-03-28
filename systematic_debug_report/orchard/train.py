"""Training loop and CLI entry point."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
sys.path.append("../")

import argparse
import os
import time

import torch

import orchard.encoding as encoding
from orchard.config import load_config
from orchard.enums import (
    LearningType, PickMode, StoppingCondition, TDTarget, TrainMode,
)
from orchard.env import create_env
from orchard.eval import (
    Action,
    collect_after_state_test_states,
    collect_on_policy_test_states,
    collect_reward_test_states,
    compute_reward_ground_truth,
    evaluate_policy_learning,
    evaluate_reward_learning,
    evaluate_value_learning,
    precompute_ground_truth,
)
from orchard.logging_ import (
    CSVLogger,
    build_detail_csv_fieldnames,
    build_main_csv_fieldnames,
    finalize_logging,
    setup_logging,
)
from orchard.model import create_networks
from orchard.policy import (
    ACTION_PRIORITY,
    ValueNetwork,
    argmax_a_Q_team,
    argmax_a_Q_team_batched,
    argmax_a_Q_team_gpu,
    epsilon_greedy,
    epsilon_greedy_batched,
    epsilon_greedy_gpu,
    get_all_actions,
    get_phase2_actions,
    heuristic_action,
    init_vmap,
    make_pick_action,
    refresh_vmap,
)
from orchard.schedule import compute_schedule_value
from orchard.seed import rng, set_all_seeds
from orchard.datatypes import EncoderOutput, ExperimentConfig, State

def _train_all_agents(
    networks: list[ValueNetwork],
    s_encs: list[EncoderOutput],
    rewards: tuple[float, ...],
    discount: float,
    s_next_encs: list[EncoderOutput],
    env_step: int,
) -> tuple[float, int]:
    """Train all agent networks on one transition. Returns (total_loss, count)."""
    total_loss = 0.0
    for i in range(len(networks)):
        total_loss += networks[i].train_step(
            s_enc=s_encs[i], reward=rewards[i],
            discount=discount, s_next_enc=s_next_encs[i],
            env_step=env_step,
        )
    return total_loss, len(networks)


def _train_all_agents_gpu(
    batched_trainer: "BatchedTrainer",
    grids_t: "torch.Tensor",
    scalars_t: "torch.Tensor",
    rewards: tuple[float, ...],
    discount: float,
    grids_next: "torch.Tensor",
    scalars_next: "torch.Tensor",
    env_step: int,
) -> tuple[float, int]:
    """GPU-batched training. Same semantics as _train_all_agents."""
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    total_loss = batched_trainer.td_lambda_step_batched(
        grids_t, scalars_t, rewards_t, discount,
        grids_next, scalars_next, env_step,
    )
    return total_loss, batched_trainer.n

def _encode_all(state: State, n_networks: int, n_agents: int, centralized: bool) -> list[EncoderOutput]:
    """Encode state for each network."""
    if centralized:
        return [encoding.encode(state, 0)]
    return [encoding.encode(state, i) for i in range(n_agents)]


def _encode_all_gpu(state: State) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU path: encode state for all agents, returning stacked tensors."""
    return encoding.encode_all_agents(state)


def _team_rewards(rewards: tuple[float, ...], centralized: bool) -> tuple[float, ...]:
    """Convert per-agent rewards to per-network rewards."""
    if centralized:
        return (sum(rewards),)
    return rewards


def _zero_rewards(n_networks: int) -> tuple[float, ...]:
    """Zero rewards tuple of correct length."""
    return tuple(0.0 for _ in range(n_networks))

def _save_checkpoint(
    networks: list[ValueNetwork],
    step: int,
    path: Path,
) -> None:
    """Save network and optimizer state dicts."""
    ckpt = {
        "step": step,
        "networks": [net.state_dict() for net in networks],
        "optimizers": [net.optimizer.state_dict() for net in networks],
    }
    torch.save(ckpt, path)
    

def train(cfg: ExperimentConfig, resume_checkpoint: str | None = None) -> None:
    """Main training loop."""
    start_time = time.time()

    # --- Setup ---
    set_all_seeds(cfg.train.seed)
    env = create_env(cfg.env)
    encoding.init_encoder(cfg.model.input_type, cfg.env, cfg.model.k_nearest,
                          use_vec_encode=cfg.train.use_vec_encode)

    n_agents = cfg.env.n_agents
    centralized = cfg.train.learning_type == LearningType.CENTRALIZED
    n_networks = 1 if centralized else n_agents
    heuristic = cfg.train.heuristic
    n_task_types = cfg.env.n_task_types
    pick_mode = cfg.env.pick_mode

    _cached_zero_rewards = tuple(0.0 for _ in range(n_networks))
    networks = create_networks(
            cfg.model, cfg.env, cfg.train.lr, cfg.train.total_steps,
            nstep=cfg.train.nstep, td_lambda=cfg.train.td_lambda,
            train_method=cfg.train.train_method, n_networks=n_networks,
        )

    # --- Resume from checkpoint (weights only) ---
    if resume_checkpoint is not None:
        ckpt = torch.load(resume_checkpoint, map_location="cpu", weights_only=True)
        state_dicts = ckpt["networks"]
        if len(state_dicts) != len(networks):
            raise ValueError(
                f"Checkpoint has {len(state_dicts)} networks but current config "
                f"expects {len(networks)}. Check learning_type (centralized vs decentralized)."
            )
        for net, sd in zip(networks, state_dicts):
            net.load_state_dict(sd, strict=True)
        print(f"Loaded pretrained weights from: {resume_checkpoint}")
        print(f"  Checkpoint was at step {ckpt.get('step', '?')}. Training restarts from step 0.")
    
    # --- Initialize vmap (if enabled) ---
    use_vmap = cfg.train.use_vmap and not centralized
    if use_vmap:
        init_vmap(networks)
        print("vmap-batched action selection enabled")

    # --- Initialize GPU-batched training (if enabled) ---
    use_gpu_batched = cfg.train.use_gpu_batched and not centralized
    batched_trainer = None
    if use_gpu_batched:
        from orchard.batched_training import BatchedTrainer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batched_trainer = BatchedTrainer(
            networks, td_lambda=cfg.train.td_lambda,
            lr_schedule=cfg.train.lr, total_steps=cfg.train.total_steps,
            device=device,
        )
        print(f"GPU-batched TD(λ) training enabled (device={device})")
        if device == "cuda" or device.startswith("cuda"):
            alloc = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            print(f"  VRAM: {alloc:.0f}MB allocated, {reserved:.0f}MB reserved, {total:.0f}MB total ({alloc/total*100:.1f}% used)")

    run_dir = setup_logging(cfg)
    _save_checkpoint(networks, 0, run_dir / "checkpoints" / "step_0.pt")

    # --- CSV loggers ---
    heuristic_name = heuristic.name.lower()
    main_fields = build_main_csv_fieldnames(
        n_agents, cfg.train.mode, n_networks=n_networks,
        centralized=centralized, n_task_types=n_task_types,
        heuristic_name=heuristic_name,
    )
    main_logger = CSVLogger(run_dir / "metrics.csv", main_fields)

    detail_fields = build_detail_csv_fieldnames(n_agents, networks, n_networks=n_networks)
    detail_logger = CSVLogger(run_dir / "details.csv", detail_fields)

    # --- Test states + ground truth ---
    test_states: list | None = None
    ground_truth: list | None = None
    reward_categories: list | None = None
    if cfg.train.mode == TrainMode.VALUE_LEARNING:
        if cfg.train.td_target == TDTarget.PRE_ACTION:
            test_states = collect_on_policy_test_states(env, cfg.eval.n_test_states, heuristic=heuristic)
        else:
            test_states = collect_after_state_test_states(env, cfg.eval.n_test_states, heuristic=heuristic)
        ground_truth = precompute_ground_truth(
            test_states, env, cfg.eval, cfg.train.td_target, heuristic=heuristic
        )
    elif cfg.train.mode == TrainMode.REWARD_LEARNING:
        test_states = collect_reward_test_states(env, cfg.eval.n_test_states)
        ground_truth, reward_categories = compute_reward_ground_truth(
            test_states, cfg.env, n_networks, centralized
        )
    s_t = env.init_state()
    
    # --- Sample states for value diagnostics (policy_learning) ---
    sample_states: list[State] | None = None
    sample_logger: CSVLogger | None = None
    if cfg.train.mode == TrainMode.POLICY_LEARNING:
        n_sample = 100
        set_all_seeds(9999)
        sample_states = []
        s_tmp = env.init_state()
        for _ in range(n_sample * 3):
            # Phase 1: random move
            move_acts = get_all_actions(cfg.env)
            a_tmp = move_acts[rng.randint(0, len(move_acts) - 1)]
            s_moved = env.apply_action(s_tmp, a_tmp)

            if cfg.train.td_target == TDTarget.AFTER_STATE:
                if s_moved.is_agent_on_task(s_moved.actor):
                    # Phase 2: auto-pick (FORCED) or random (CHOICE)
                    if pick_mode == PickMode.FORCED:
                        s_after, _ = env.resolve_pick(s_moved)
                    else:
                        p2 = get_phase2_actions(s_moved, cfg.env)
                        p2_act = p2[rng.randint(0, len(p2) - 1)]
                        s_after, _ = env.resolve_pick(
                            s_moved,
                            pick_type=p2_act.pick_type() if p2_act.is_pick() else None,
                        )
                    if len(sample_states) < n_sample:
                        sample_states.append(s_after)
                    s_tmp = env.advance_actor(env.spawn_and_despawn(s_after))
                else:
                    if len(sample_states) < n_sample:
                        sample_states.append(s_moved)
                    s_tmp = env.advance_actor(env.spawn_and_despawn(s_moved))
            else:
                if len(sample_states) < n_sample:
                    sample_states.append(s_tmp)
                # step() handles FORCED auto-pick; CHOICE just moves
                s_tmp = env.step(s_tmp, a_tmp).s_t_next

        set_all_seeds(cfg.train.seed)
        s_t = env.init_state()

        # Column headers: 5 move actions + T pick actions for CHOICE
        all_actions = get_all_actions(cfg.env)
        if cfg.env.pick_mode == PickMode.CHOICE:
            for t_idx in range(cfg.env.n_task_types):
                all_actions.append(make_pick_action(t_idx))
        sample_fields = ["step", "wall_time",
                         "value_mean", "value_std", "value_min", "value_max", "value_range"]
        for act in all_actions:
            sample_fields.append(f"action_{act.name}")
        for ni in range(n_networks):
            for si in range(n_sample):
                sample_fields.append(f"net{ni}_s{si}")
        sample_logger = CSVLogger(run_dir / "sample_values.csv", sample_fields)

    # --- After-state TD bookkeeping ---
    prev_after_enc: list[EncoderOutput] | None = None
    prev_after_grids: torch.Tensor | None = None   # GPU path
    prev_after_scalars: torch.Tensor | None = None  # GPU path
    prev_reward: tuple[float, ...] | None = None
    prev_discount: float | None = None

    # --- Running loss accumulator ---
    td_loss_accum: float = 0.0
    td_loss_count: int = 0
    running_max_pps: float = float("-inf")
    running_max_rps: float = float("-inf")
    running_min_mae: float = float("inf")
    steps_since_improvement: int = 0
    action: Action = Action.STAY

    # --- Timing debug ---
    time_debug = cfg.train.time_debug
    if time_debug:
        import time as _time
        _gpu_sync = use_gpu_batched and torch.cuda.is_available()
        def _tick():
            if _gpu_sync:
                torch.cuda.synchronize()
            return _time.perf_counter()
        _t_buckets = {"sync": 0.0, "action": 0.0, "env": 0.0, "encode": 0.0,
                      "train": 0.0, "eval": 0.0, "detail": 0.0, "reset": 0.0}
        _t_step_count = 0
        timing_fields = ["step"] + [f"t_{k}" for k in _t_buckets] + ["t_total"]
        timing_logger = CSVLogger(run_dir / "timing.csv", timing_fields)
        print(f"time_debug enabled (csv_freq={cfg.train.time_csv_freq}, gpu_sync={_gpu_sync})")

    # --- Main loop ---
    for t in range(cfg.train.total_steps):
        if time_debug:
            _t0 = _tick()

        # --- Sync (CPU-only paths need vmap refresh; GPU path needs no sync) ---
        if not use_gpu_batched and use_vmap:
            refresh_vmap()

        if time_debug:
            _t1 = _tick()
            _t_buckets["sync"] += _t1 - _t0
        if cfg.train.mode == TrainMode.VALUE_LEARNING:
            move_action = heuristic_action(s_t, cfg.env, heuristic, phase2=False)
        elif cfg.train.mode == TrainMode.POLICY_LEARNING:
            assert cfg.train.policy_learning is not None
            epsilon = compute_schedule_value(
                cfg.train.policy_learning.epsilon, t, cfg.train.total_steps
            )
            if use_gpu_batched:
                move_action = epsilon_greedy_gpu(s_t, batched_trainer, env, epsilon,
                                                 phase2=False)
            elif cfg.train.batch_actions:
                move_action = epsilon_greedy_batched(s_t, networks, env, epsilon,
                                                     use_vmap=use_vmap, phase2=False)
            else:
                move_action = epsilon_greedy(s_t, networks, env, epsilon, phase2=False)
        elif cfg.train.mode == TrainMode.REWARD_LEARNING:
            move_actions = get_all_actions(cfg.env)
            move_action = move_actions[rng.randint(0, len(move_actions) - 1)]

        assert move_action.is_move(), f"Expected move action in phase 1, got {move_action}"

        if time_debug:
            _t2 = _tick()

        # --- Apply move ---
        s_moved = env.apply_action(s_t, move_action)
        on_task = s_moved.is_agent_on_task(s_moved.actor)

        # --- Phase-2 action selection (only if on a task cell) ---
        if on_task:
            if pick_mode == PickMode.FORCED:
                # Auto-pick: find the single task at this cell
                tau = s_moved.task_type_at(s_moved.agent_positions[s_moved.actor])
                pick_action = make_pick_action(tau)
            elif cfg.train.mode == TrainMode.VALUE_LEARNING:
                pick_action = heuristic_action(s_moved, cfg.env, heuristic, phase2=True)
            elif cfg.train.mode == TrainMode.POLICY_LEARNING:
                if use_gpu_batched:
                    pick_action = epsilon_greedy_gpu(s_moved, batched_trainer, env, epsilon,
                                                     phase2=True)
                elif cfg.train.batch_actions:
                    pick_action = epsilon_greedy_batched(s_moved, networks, env, epsilon,
                                                         use_vmap=use_vmap, phase2=True)
                else:
                    pick_action = epsilon_greedy(s_moved, networks, env, epsilon, phase2=True)
            else:  # REWARD_LEARNING
                p2_acts = get_phase2_actions(s_moved, cfg.env)
                pick_action = p2_acts[rng.randint(0, len(p2_acts) - 1)]

            s_picked, pick_rewards = env.resolve_pick(
                s_moved,
                pick_type=pick_action.pick_type() if pick_action.is_pick() else None,
            )
            s_next = env.advance_actor(env.spawn_and_despawn(s_picked))
        else:
            s_picked = s_moved
            pick_rewards = _zero_rewards(n_networks)
            pick_action = Action.STAY  # unused sentinel
            s_next = env.advance_actor(env.spawn_and_despawn(s_moved))

        if time_debug:
            _t3 = _tick()
            # Phase-2 has both action + env interleaved; attribute action to action, env to env
            # For simplicity, everything from _t1 to _t2 is action (phase 1),
            # _t2 to _t3 includes apply_action + phase2 action + resolve_pick + advance
            _t_buckets["action"] += _t2 - _t1
            _t_buckets["env"] += _t3 - _t2

        # --- TD updates ---
        if time_debug:
            _te_accum = 0.0  # encode time this step
            _tt_accum = 0.0  # train time this step

        if cfg.train.td_target == TDTarget.PRE_ACTION:
            if use_gpu_batched:
                if time_debug: _ta = _tick()
                pre_g, pre_s = _encode_all_gpu(s_t)
                if time_debug: _tb = _tick(); _te_accum += _tb - _ta
                if on_task:
                    if time_debug: _ta = _tick()
                    move_g, move_s = _encode_all_gpu(s_moved)
                    if time_debug: _tb = _tick(); _te_accum += _tb - _ta

                    if time_debug: _ta = _tick()
                    loss, count = _train_all_agents_gpu(batched_trainer, pre_g, pre_s,
                                                         _cached_zero_rewards, cfg.env.gamma,
                                                         move_g, move_s, t)
                    if time_debug: _tb = _tick(); _tt_accum += _tb - _ta
                    td_loss_accum += loss; td_loss_count += count

                    if time_debug: _ta = _tick()
                    next_g, next_s = _encode_all_gpu(s_next)
                    if time_debug: _tb = _tick(); _te_accum += _tb - _ta

                    if time_debug: _ta = _tick()
                    loss, count = _train_all_agents_gpu(batched_trainer, move_g, move_s,
                                                         _team_rewards(pick_rewards, centralized),
                                                         1.0, next_g, next_s, t)
                    if time_debug: _tb = _tick(); _tt_accum += _tb - _ta
                    td_loss_accum += loss; td_loss_count += count
                else:
                    if time_debug: _ta = _tick()
                    next_g, next_s = _encode_all_gpu(s_next)
                    if time_debug: _tb = _tick(); _te_accum += _tb - _ta

                    if time_debug: _ta = _tick()
                    loss, count = _train_all_agents_gpu(batched_trainer, pre_g, pre_s,
                                                         _cached_zero_rewards, cfg.env.gamma,
                                                         next_g, next_s, t)
                    if time_debug: _tb = _tick(); _tt_accum += _tb - _ta
                    td_loss_accum += loss; td_loss_count += count
            else:
                if time_debug: _ta = _tick()
                pre_enc = _encode_all(s_t, n_networks, n_agents, centralized)
                if time_debug: _tb = _tick(); _te_accum += _tb - _ta
                if on_task:
                    if time_debug: _ta = _tick()
                    move_enc = _encode_all(s_moved, n_networks, n_agents, centralized)
                    if time_debug: _tb = _tick(); _te_accum += _tb - _ta

                    if time_debug: _ta = _tick()
                    loss, count = _train_all_agents(networks, pre_enc, _zero_rewards(n_networks),
                                                    cfg.env.gamma, move_enc, t)
                    if time_debug: _tb = _tick(); _tt_accum += _tb - _ta
                    td_loss_accum += loss; td_loss_count += count

                    if time_debug: _ta = _tick()
                    next_enc = _encode_all(s_next, n_networks, n_agents, centralized)
                    if time_debug: _tb = _tick(); _te_accum += _tb - _ta

                    if time_debug: _ta = _tick()
                    loss, count = _train_all_agents(networks, move_enc,
                                                    _team_rewards(pick_rewards, centralized),
                                                    1.0, next_enc, t)
                    if time_debug: _tb = _tick(); _tt_accum += _tb - _ta
                    td_loss_accum += loss; td_loss_count += count
                else:
                    if time_debug: _ta = _tick()
                    next_enc = _encode_all(s_next, n_networks, n_agents, centralized)
                    if time_debug: _tb = _tick(); _te_accum += _tb - _ta

                    if time_debug: _ta = _tick()
                    loss, count = _train_all_agents(networks, pre_enc, _zero_rewards(n_networks),
                                                    cfg.env.gamma, next_enc, t)
                    if time_debug: _tb = _tick(); _tt_accum += _tb - _ta
                    td_loss_accum += loss; td_loss_count += count

        else:
            s_moved_enc_state = replace(s_moved, phase2_pending=True) if on_task else s_moved

            if use_gpu_batched:
                if time_debug: _ta = _tick()
                move_g, move_s = _encode_all_gpu(s_moved_enc_state)
                if time_debug: _tb = _tick(); _te_accum += _tb - _ta

                if prev_after_grids is not None:
                    assert prev_reward is not None and prev_discount is not None
                    if time_debug: _ta = _tick()
                    loss, count = _train_all_agents_gpu(batched_trainer, prev_after_grids, prev_after_scalars,
                                                         prev_reward, prev_discount, move_g, move_s, t)
                    if time_debug: _tb = _tick(); _tt_accum += _tb - _ta
                    td_loss_accum += loss; td_loss_count += count

                if on_task:
                    if time_debug: _ta = _tick()
                    pick_g, pick_s = _encode_all_gpu(s_picked)
                    if time_debug: _tb = _tick(); _te_accum += _tb - _ta
                    pick_discount = 0.0 if cfg.train.mode == TrainMode.REWARD_LEARNING else 1.0
                    if time_debug: _ta = _tick()
                    loss, count = _train_all_agents_gpu(batched_trainer, move_g, move_s,
                                                         _team_rewards(pick_rewards, centralized),
                                                         pick_discount, pick_g, pick_s, t)
                    if time_debug: _tb = _tick(); _tt_accum += _tb - _ta
                    td_loss_accum += loss; td_loss_count += count
                    prev_after_grids, prev_after_scalars = pick_g, pick_s
                else:
                    prev_after_grids, prev_after_scalars = move_g, move_s
            else:
                if time_debug: _ta = _tick()
                move_after_enc = _encode_all(s_moved_enc_state, n_networks, n_agents, centralized)
                if time_debug: _tb = _tick(); _te_accum += _tb - _ta

                if prev_after_enc is not None:
                    assert prev_reward is not None and prev_discount is not None
                    if time_debug: _ta = _tick()
                    loss, count = _train_all_agents(networks, prev_after_enc, prev_reward,
                                                    prev_discount, move_after_enc, t)
                    if time_debug: _tb = _tick(); _tt_accum += _tb - _ta
                    td_loss_accum += loss; td_loss_count += count

                if on_task:
                    if time_debug: _ta = _tick()
                    pick_after_enc = _encode_all(s_picked, n_networks, n_agents, centralized)
                    if time_debug: _tb = _tick(); _te_accum += _tb - _ta
                    pick_discount = 0.0 if cfg.train.mode == TrainMode.REWARD_LEARNING else 1.0
                    if time_debug: _ta = _tick()
                    loss, count = _train_all_agents(networks, move_after_enc,
                                                    _team_rewards(pick_rewards, centralized),
                                                    pick_discount, pick_after_enc, t)
                    if time_debug: _tb = _tick(); _tt_accum += _tb - _ta
                    td_loss_accum += loss; td_loss_count += count
                    prev_after_enc = pick_after_enc
                else:
                    prev_after_enc = move_after_enc

            prev_reward = _zero_rewards(n_networks)
            prev_discount = cfg.env.gamma

        if time_debug:
            _t_buckets["encode"] += _te_accum
            _t_buckets["train"] += _tt_accum

        s_t = s_next

        # --- Reset (value_learning only) ---
        if time_debug:
            _tr0 = _tick()
        if cfg.train.mode == TrainMode.VALUE_LEARNING:
            assert cfg.train.value_learning is not None
            if (t + 1) % cfg.train.value_learning.reset_freq == 0:
                if use_gpu_batched:
                    batched_trainer.reset_traces()
                else:
                    for net in networks:
                        net.flush_nstep()
                        net.reset_traces()
                s_t = env.init_state()
                if cfg.train.td_target == TDTarget.AFTER_STATE:
                    prev_after_enc = None
                    prev_after_grids = None
                    prev_after_scalars = None
        if time_debug:
            _t_buckets["reset"] += _tick() - _tr0

        # --- Main CSV logging ---
        if (t + 1) % cfg.logging.main_csv_freq == 0:
            if time_debug:
                _tev0 = _tick()
            wall_time = time.time() - start_time
            row: dict[str, float | int | str] = {
                "step": t + 1,
                "wall_time": round(wall_time, 3),
            }

            # Value learning metrics
            if cfg.train.mode == TrainMode.VALUE_LEARNING and test_states is not None and ground_truth is not None:
                val_metrics = evaluate_value_learning(
                    networks, cfg.env, test_states, ground_truth,
                    batched_trainer=batched_trainer if use_gpu_batched else None,
                )
                row.update(val_metrics)

            # Policy learning metrics
            if cfg.train.mode == TrainMode.POLICY_LEARNING:
                pol_metrics = evaluate_policy_learning(
                    networks, env, cfg.eval.eval_steps,
                    batch_actions=cfg.train.batch_actions,
                    heuristic=heuristic,
                    batched_trainer=batched_trainer if use_gpu_batched else None,
                )
                row.update(pol_metrics)

            # Reward learning metrics
            if cfg.train.mode == TrainMode.REWARD_LEARNING and test_states is not None:
                assert ground_truth is not None
                assert reward_categories is not None
                reward_metrics = evaluate_reward_learning(
                    networks, cfg.env, test_states, ground_truth,
                    reward_categories, centralized,
                    batched_trainer=batched_trainer if use_gpu_batched else None,
                )
                row.update(reward_metrics)

            # TD loss
            avg_loss = td_loss_accum / max(td_loss_count, 1)
            row["td_loss_avg"] = round(avg_loss, 8)
            td_loss_accum = 0.0
            td_loss_count = 0

            main_logger.log(row)
            
            # --- Early stopping ---
            if (
                cfg.train.stopping_condition == StoppingCondition.RUNNING_MAX_RPS
                and "greedy_rps" in row
            ):
                rps = float(row["greedy_rps"])
                if rps > running_max_rps + cfg.train.improvement_threshold:
                    running_max_rps = rps
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += cfg.logging.main_csv_freq
                if steps_since_improvement >= cfg.train.patience_steps and (t + 1) >= cfg.train.min_steps_before_stop:
                    print(f"\nEarly stop at step {t+1}: running max RPS {running_max_rps:.6f} "
                          f"unchanged for {cfg.train.patience_steps} steps.")
                    break

            if (
                cfg.train.stopping_condition == StoppingCondition.RUNNING_MIN_MAE
                and "mae_avg" in row
            ):
                mae = float(row["mae_avg"])
                if mae < running_min_mae - cfg.train.improvement_threshold:
                    running_min_mae = mae
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += cfg.logging.main_csv_freq
                if steps_since_improvement >= cfg.train.patience_steps and (t + 1) >= cfg.train.min_steps_before_stop:
                    print(f"\nEarly stop at step {t+1}: running min MAE {running_min_mae:.6f} "
                        f"unchanged for {cfg.train.patience_steps} steps.")
                    break

            # Print progress
            print(f"\n--- Step {t + 1} ({wall_time:.1f}s) ---")
            if "greedy_rps" in row:
                msg = f"  Greedy RPS: {row['greedy_rps']:.4f}"
                if "greedy_correct_pps" in row:
                    msg += (f"  Correct PPS: {row['greedy_correct_pps']:.4f}"
                            f"  Wrong PPS: {row['greedy_wrong_pps']:.4f}")
                print(msg)
                h_key = f"{heuristic_name}_rps"
                if h_key in row:
                    print(f"  {heuristic_name} RPS: {row[h_key]:.4f}")
            if "mae_avg" in row:
                msg = f"  MAE avg: {row['mae_avg']:.4f}"
                if "bias_avg" in row:
                    msg += f"  Bias avg: {row['bias_avg']:.4f}"
                if "mape_avg" in row:
                    msg += f"  MAPE avg: {row['mape_avg']:.4f}"
                print(msg)

            if time_debug:
                _t_buckets["eval"] += _tick() - _tev0

        # --- Detail CSV logging ---
        if (t + 1) % cfg.logging.detail_csv_freq == 0:
            if time_debug:
                _td0 = _tick()
            if use_gpu_batched:
                batched_trainer.sync_to_networks()
            wall_time = time.time() - start_time
            detail_row: dict[str, float | int | str] = {
                "step": t + 1,
                "wall_time": round(wall_time, 3),
            }

            try:
                import resource
                ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            except ImportError:
                ram_mb = 0.0
            detail_row["ram_mb"] = round(ram_mb, 1)

            if use_gpu_batched and torch.cuda.is_available():
                detail_row["vram_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024**2, 1)
                detail_row["vram_peak_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
                detail_row["vram_total_mb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 1)

            detail_row["current_lr"] = networks[0].optimizer.param_groups[0]["lr"]

            if cfg.train.mode == TrainMode.POLICY_LEARNING:
                assert cfg.train.policy_learning is not None
                detail_row["current_epsilon"] = compute_schedule_value(
                    cfg.train.policy_learning.epsilon, t + 1, cfg.train.total_steps
                )
            else:
                detail_row["current_epsilon"] = 0.0

            for agent_idx in range(n_networks):
                w_norms = networks[agent_idx].get_weight_norms()
                g_norms = networks[agent_idx].get_grad_norms()
                for name in w_norms:
                    detail_row[f"weight_norm_agent_{agent_idx}_{name}"] = round(w_norms[name], 6)
                for name in g_norms:
                    detail_row[f"grad_norm_agent_{agent_idx}_{name}"] = round(g_norms[name], 6)

            detail_row["td_loss_step"] = round(avg_loss if td_loss_count == 0 else td_loss_accum / max(td_loss_count, 1), 8)

            if test_states is not None:
                preds: list[float] = []
                with torch.no_grad():
                    if use_gpu_batched:
                        for s in test_states[:10]:
                            grids, scalars = encoding.encode_all_agents(s)
                            vals = batched_trainer.forward_single_batched(grids, scalars)
                            for i in range(n_networks):
                                preds.append(vals[i].item())
                    else:
                        for s in test_states[:10]:
                            for i in range(n_networks):
                                preds.append(networks[i](encoding.encode(s, i)).item())
                if preds:
                    detail_row["value_pred_mean"] = round(sum(preds) / len(preds), 6)
                    mean = sum(preds) / len(preds)
                    var = sum((p - mean) ** 2 for p in preds) / len(preds)
                    detail_row["value_pred_std"] = round(var ** 0.5, 6)

            detail_logger.log(detail_row)

            # --- Sample value logging ---
            if sample_states is not None and sample_logger is not None:
                sample_row: dict[str, float | int | str] = {
                    "step": t + 1,
                    "wall_time": round(time.time() - start_time, 3),
                }
                all_vals: list[float] = []
                _log_actions = get_all_actions(cfg.env)
                if cfg.env.pick_mode == PickMode.CHOICE:
                    for t_idx in range(cfg.env.n_task_types):
                        _log_actions.append(make_pick_action(t_idx))
                action_counts: dict[str, int] = {f"action_{act.name}": 0 for act in _log_actions}

                with torch.no_grad():
                    for si, ss in enumerate(sample_states):
                        if use_gpu_batched:
                            grids, scalars = encoding.encode_all_agents(ss)
                            vals = batched_trainer.forward_single_batched(grids, scalars)
                            for ni in range(n_networks):
                                v = vals[ni].item()
                                sample_row[f"net{ni}_s{si}"] = round(v, 6)
                                all_vals.append(v)
                            ga = argmax_a_Q_team_gpu(ss, batched_trainer, env)
                        else:
                            for ni in range(n_networks):
                                v = networks[ni](encoding.encode(ss, ni)).item()
                                sample_row[f"net{ni}_s{si}"] = round(v, 6)
                                all_vals.append(v)
                            ga = argmax_a_Q_team_batched(ss, networks, env) if cfg.train.batch_actions else argmax_a_Q_team(ss, networks, env)
                        action_counts[f"action_{ga.name}"] += 1

                sample_row["value_mean"] = round(sum(all_vals) / len(all_vals), 6)
                mean = sum(all_vals) / len(all_vals)
                var = sum((v - mean) ** 2 for v in all_vals) / len(all_vals)
                sample_row["value_std"] = round(var ** 0.5, 6)
                sample_row["value_min"] = round(min(all_vals), 6)
                sample_row["value_max"] = round(max(all_vals), 6)
                sample_row["value_range"] = round(max(all_vals) - min(all_vals), 6)
                for k, cnt in action_counts.items():
                    sample_row[k] = cnt / len(sample_states)

                sample_logger.log(sample_row)

            if time_debug:
                _t_buckets["detail"] += _tick() - _td0

        # --- Timing CSV ---
        if time_debug:
            _t_step_count += 1
            if (t + 1) % cfg.train.time_csv_freq == 0:
                timing_row = {"step": t + 1}
                t_total = 0.0
                for k in _t_buckets:
                    avg = _t_buckets[k] / _t_step_count
                    timing_row[f"t_{k}"] = round(avg, 6)
                    t_total += avg
                timing_row["t_total"] = round(t_total, 6)
                timing_logger.log(timing_row)
                for k in _t_buckets:
                    _t_buckets[k] = 0.0
                _t_step_count = 0

        # --- Periodic checkpoints ---
        if cfg.eval.checkpoint_freq > 0 and (t + 1) % cfg.eval.checkpoint_freq == 0:
            if use_gpu_batched:
                batched_trainer.sync_to_networks()
            _save_checkpoint(networks, t + 1, run_dir / "checkpoints" / f"step_{t + 1}.pt")
    
    # --- Final checkpoint ---
    if use_gpu_batched:
        batched_trainer.sync_to_networks()
    _save_checkpoint(networks, cfg.train.total_steps, run_dir / "checkpoints" / "final.pt")
    
    # --- Cleanup ---
    main_logger.close()
    detail_logger.close()
    if sample_logger is not None:
        sample_logger.close()
    if time_debug:
        timing_logger.close()
    finalize_logging(run_dir, start_time)
    print(f"\nRun saved to: {run_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Orchard RL Training")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint (.pt) to load pretrained weights from.",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Override config values (dot notation): key=value",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    train(cfg, resume_checkpoint=args.resume)


if __name__ == "__main__":
    main()
