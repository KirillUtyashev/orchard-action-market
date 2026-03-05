import json
import os
import random
import time
from pathlib import Path

import numpy as np
import multiprocessing as mp

import torch

from debug.code.encoders import CenConcatEncoder, CenEntityEncoder, \
    CenGridEncoder, \
    DecEntityEncoder, DecGridEncoder
from debug.code.log import CSVLogger, build_main_csv_fieldnames, \
    finalize_logging, \
    setup_logging
from debug.code.td_lambda import TDLambda
from utils import ten
from debug.code.controllers import AgentControllerDecentralized, AgentControllerCentralized
from debug.code.monte_carlo import generate_careful_distance_series, \
    generate_initial_state_full, \
    generate_initial_state_supervised

from debug.code.simple_agent import SimpleAgent
from debug.code.enums import (
    NUM_AGENTS,
    W,
    L,
    PROBABILITY_APPLE,
    DISCOUNT_FACTOR,
    DEVICE,
    data_dir
)

from debug.code.environment import Orchard
from debug.code.helpers import env_step, eval_performance, make_env, \
    random_policy, \
    set_all_seeds, teleport, \
    env_step
from debug.code.reward import Reward
from debug.code.value import Value
from debug.code.value_function import VNetwork
import matplotlib.pyplot as plt


def _worker_generate_state(args):
    reward_module, run_id, seed, discount_factor, p_apple, d_apple, q_agent, tau = args
    state = generate_initial_state_full(
        reward_module=reward_module,
        run_id=run_id,
        seed=seed,
        discount_factor=discount_factor,
        p_apple=p_apple,
        d_apple=d_apple,
        q_agent=q_agent,
        tau=tau,
        save=True
    )
    return state


def _worker_generate_careful(arg):
    reward_module, seed, discount_factor, p_apple, d_apple, agent_id, distance = arg
    res = generate_careful_distance_series(
        reward_module=reward_module,
        seed=seed,
        discount_factor=discount_factor,
        p_apple=p_apple,
        d_apple=d_apple,
        distances=(distance,),
        self_id=agent_id
    )
    return res


def _state_path(picker_r: float, run_id: int) -> Path:
    return data_dir / "states" / "full" / f"init_state_reward_{picker_r}_{run_id}.npz"


def _careful_state_path(agent_id: int, seed: int, distance: int):
    out_dir = data_dir / "states" / "careful"
    return out_dir / f"careful_agent{agent_id}_seed{seed}_d{distance}.npz"


def _load_state_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as f:
        return f["dict"].item()


class Learning:
    def __init__(self, exp_config):
        self.env = None
        self.encoder = None          # replaces view_controller
        self.agent_controller = None
        self.rng_state = None
        self.agents = []
        self.critic_networks = []
        self.reward_module = Reward(exp_config.reward.picker_r, NUM_AGENTS)
        self.trajectory_length = exp_config.train.timesteps
        self.exp_config = exp_config

        self.num_eval_states = exp_config.eval.num_eval_states

        self.theoretical_val = Value(
            exp_config.reward.picker_r, NUM_AGENTS, DISCOUNT_FACTOR,
            PROBABILITY_APPLE, exp_config.eval.variance
        )

        self._networks_for_eval = []
        self.eval_history = []

        self.data_dir = setup_logging(self.exp_config)
        main_fields = build_main_csv_fieldnames()
        self.main_logger = CSVLogger(self.data_dir / "metrics.csv", main_fields)

        self._last_eval_errors_by_state = None
        self.discount_factor = DISCOUNT_FACTOR

        self.careful_evals = []
        self.focus_actor_id = 0
        self.careful_distances = (2, 1)
        self.careful_actor_states = [None for _ in range(len(self.careful_distances))]
        self.careful_eval_steps = []
        self.careful_pred_history_actor0 = [
            [[] for _ in range(len(self.careful_distances))]
            for _ in range(NUM_AGENTS)
        ]

        if self.exp_config.train.load_weights:
            path = self.data_dir / "weights" / "weights.pt"
            ckpt = torch.load(path, map_location="cpu")
            self.crit_blobs = ckpt.get("critics", [])

    def _generate_evaluation_states(self, p_apple, d_apple, sequential: bool = False, processes: int = 8):
        start = time.time()

        out_dir = data_dir / "states" / "full"
        out_dir.mkdir(parents=True, exist_ok=True)

        num = self.num_eval_states
        results = [None] * num
        missing_args = []
        missing_indices = []

        for i in range(num):
            path = _state_path(self.reward_module.picker_r, i)
            if path.exists():
                results[i] = _load_state_npz(path)
            else:
                missing_indices.append(i)
                missing_args.append((
                    self.reward_module, i, i, self.discount_factor, p_apple, d_apple,
                    self.exp_config.algorithm.q_agent, self.exp_config.env.apple_life
                ))

        if missing_args:
            if sequential:
                generated = [_worker_generate_state(arg) for arg in missing_args]
            else:
                with mp.Pool(processes=processes) as pool:
                    generated = pool.map(_worker_generate_state, missing_args)
            for idx, state in zip(missing_indices, generated):
                results[idx] = state

        self.evaluation_states = results

        careful_dir = data_dir / "states" / "careful"
        careful_dir.mkdir(parents=True, exist_ok=True)

        distances = self.careful_distances
        careful_seed = 42069
        self.careful_evals = [[] for _ in range(NUM_AGENTS)]
        missing_args, missing_keys = [], []

        for agent_id in range(NUM_AGENTS):
            for d in distances:
                path = _careful_state_path(agent_id, careful_seed, d)
                if path.exists():
                    self.careful_evals[agent_id].append(_load_state_npz(path))
                else:
                    self.careful_evals[agent_id].append(None)
                    missing_keys.append((agent_id, d, path))
                    missing_args.append((
                        self.reward_module, careful_seed, self.discount_factor,
                        p_apple, d_apple, agent_id, d
                    ))

        if missing_args:
            if sequential:
                generated = [_worker_generate_careful(arg) for arg in missing_args]
            else:
                with mp.Pool(processes=processes) as pool:
                    generated = pool.map(_worker_generate_careful, missing_args)
            for (agent_id, d, path), item in zip(missing_keys, generated):
                j = distances.index(d)
                self.careful_evals[agent_id][j] = item

        self.careful_actor_states = self.careful_evals[0]
        print(f"Generated/loaded {num} eval states in {time.time() - start:.3f}s")

    def _generate_evaluation_states_reward_learning(self):
        if not hasattr(self, 'evaluation_states'):
            self.evaluation_states = {'Z1': [], 'Y11': [], 'Y10': []}

        for state_type in ['Z1', 'Y11', 'Y10']:
            for _ in range(self.num_eval_states):
                state_dict, agent_positions = generate_initial_state_supervised(
                    self.reward_module, state_type, save=False
                )
                state_dict["agent_positions"] = agent_positions
                self.evaluation_states[state_type].append(state_dict)

    # -----------------------------------------------------------------------
    # Encoder factory — replaces the view_controller block in build_experiment
    # -----------------------------------------------------------------------

    def _build_encoder(self):
        cfg = self.exp_config
        k = cfg.reward.top_k_num_apples

        if cfg.algorithm.centralized:
            if cfg.network.CNN:
                self.encoder = CenGridEncoder(W, W, NUM_AGENTS)
            elif cfg.algorithm.concat:
                dec = DecEntityEncoder(W, W, NUM_AGENTS, k)
                self.encoder = CenConcatEncoder(dec)
            else:
                self.encoder = CenEntityEncoder(W, W, NUM_AGENTS, k)
        else:
            if cfg.network.CNN:
                self.encoder = DecGridEncoder(W, W, NUM_AGENTS)
            else:
                self.encoder = DecEntityEncoder(W, W, NUM_AGENTS, k)

    # -----------------------------------------------------------------------
    # Network init now derives dims from the encoder
    # -----------------------------------------------------------------------

    def _init_critic_networks(self):
        cfg = self.exp_config

        if cfg.algorithm.centralized:
            self.critic_networks.append(VNetwork(
                self.encoder, 1,
                cfg.train.alpha,
                self.discount_factor,
                mlp_dims=tuple(cfg.network.mlp_dims),
                num_training_steps=self.trajectory_length,
                lam=self.exp_config.train.lmda,
                schedule=cfg.train.schedule_lr,
                conv_channels=cfg.network.conv_channels,
                kernel_size=cfg.network.kernel_size,
            ))
        else:
            for i in range(NUM_AGENTS):
                nn = VNetwork(
                    self.encoder, 1,
                    cfg.train.alpha,
                    self.discount_factor,
                    mlp_dims=tuple(cfg.network.mlp_dims),
                    lam=self.exp_config.train.lmda,
                    num_training_steps=self.trajectory_length,
                    schedule=cfg.train.schedule_lr,
                    conv_channels=cfg.network.conv_channels,
                    kernel_size=cfg.network.kernel_size,
                )
                if cfg.train.load_weights:
                    nn.import_net_state(self.crit_blobs[i]["blob"])
                self.critic_networks.append(nn)

    def _init_agents_for_training(self):
        policy_fn = teleport(W) if not self.exp_config.algorithm.random_policy else random_policy
        for i in range(NUM_AGENTS):
            net = self.critic_networks[0] if self.exp_config.algorithm.centralized else self.critic_networks[i]
            self.agents.append(SimpleAgent(policy_fn, i, net))

    # -----------------------------------------------------------------------
    # build_experiment: encoder built first, then networks (order matters)
    # -----------------------------------------------------------------------

    def build_experiment(self):
        self._build_encoder()            # must come before _init_critic_networks
        self._init_critic_networks()
        self._init_agents_for_training()

        p_apple = self.exp_config.algorithm.q_agent / (W ** 2)
        d_apple = 1 / self.exp_config.env.apple_life

        if self.exp_config.algorithm.random_policy:
            self._generate_evaluation_states(p_apple, d_apple)
        elif self.exp_config.reward.reward_learning:
            self._generate_evaluation_states_reward_learning()

        if self.exp_config.algorithm.centralized:
            self.agent_controller = AgentControllerCentralized(
                self.agents, self.encoder, self.discount_factor, self.exp_config.train.epsilon
            )
        else:
            self.agent_controller = AgentControllerDecentralized(
                self.agents, self.encoder, self.discount_factor, self.exp_config.train.epsilon
            )

        self.env = Orchard(W, L, NUM_AGENTS, self.reward_module, p_apple, d_apple)
        self.env.set_positions()
        self._networks_for_eval = self.critic_networks

    # -----------------------------------------------------------------------
    # Training loop: view_controller(...) → encoder.encode(...)
    # -----------------------------------------------------------------------

    def step_and_collect_observation(self) -> None:
        if self.exp_config.algorithm.random_policy:
            self.evaluate_networks(step=0, plot=True, store_last=True)
        elif self.exp_config.reward.reward_learning:
            self.evaluate_networks_reward(step=0, plot=True, store_last=True)
        else:
            self.eval_performance(0)

        curr_state = dict(self.env.get_state())
        curr_state["actor_id"] = 0
        actor_idx = 0

        for sec in range(self.trajectory_length):
            if self.exp_config.algorithm.random_policy or self.exp_config.reward.reward_learning:
                new_pos = random_policy(curr_state["agent_positions"][actor_idx])
            else:
                new_pos = self.agent_controller.agent_get_action(self.env, actor_idx)

            s_moved, s_next, pick_rewards, on_apple, next_actor_idx = env_step(
                self.env, actor_idx, new_pos, NUM_AGENTS
            )

            if self.exp_config.algorithm.centralized:
                enc_t = self.encoder.encode(curr_state, 0)   # agent_idx ignored
                enc_moved = self.encoder.encode(s_moved, 0)
                enc_next = self.encoder.encode(s_next, 0)
                net = self.critic_networks[0]
                reward = sum(pick_rewards)

                if on_apple:
                    net.add_experience(enc_t,     enc_moved, 0,      discount_factor=self.discount_factor)
                    net.add_experience(enc_moved, enc_next,  reward, discount_factor=1.0)
                else:
                    net.add_experience(enc_t, enc_next, reward, discount_factor=self.discount_factor)

                net.train()
            else:
                for i in range(len(self.critic_networks)):
                    enc_t = self.encoder.encode(curr_state, i)
                    enc_moved = self.encoder.encode(s_moved, i)
                    enc_next = self.encoder.encode(s_next, i)

                    if on_apple:
                        self.critic_networks[i].add_experience(enc_t,     enc_moved, 0,               discount_factor=self.discount_factor)
                        self.critic_networks[i].add_experience(enc_moved, enc_next,  pick_rewards[i], discount_factor=1.0)
                    else:
                        self.critic_networks[i].add_experience(enc_t, enc_next, pick_rewards[i], discount_factor=self.discount_factor)

                    self.critic_networks[i].train()

            curr_state = s_next
            actor_idx = next_actor_idx

            if (sec + 1) % self.exp_config.logging.main_csv_freq == 0:
                print(f"Running evaluation at step {sec + 1}/{self.trajectory_length}")
                if not self.exp_config.reward.reward_learning:
                    if self.exp_config.algorithm.random_policy:
                        plot = sec == self.trajectory_length - 1
                        self.evaluate_networks(step=(sec + 1), plot=plot, store_last=True)
                    else:
                        self.eval_performance(sec + 1)
                else:
                    plot = sec == self.trajectory_length - 1
                    self.evaluate_networks_reward(step=(sec + 1), plot=plot, store_last=True)

    # eval_performance, save/restore rng unchanged
    def eval_performance(self, step):
        self.save_rng_state()
        set_all_seeds(42069)
        self.agent_controller.epsilon = 0
        with torch.no_grad():
            results = eval_performance(agent_controller=self.agent_controller, reward_module=self.reward_module,
                                       d_apple=self.env.d_apple, p_apple=self.env.p_apple)
        results["step"] = step
        results["current_lr"] = self.critic_networks[0].get_lr()
        self.agent_controller.epsilon = self.exp_config.train.epsilon
        self.main_logger.log(results)
        self.restore_rng_state()

    def restore_rng_state(self):
        if self.rng_state is not None:
            random.setstate(self.rng_state["python"])
            np.random.set_state(self.rng_state["numpy"])
            torch.set_rng_state(self.rng_state["torch"])

    def save_rng_state(self):
        self.rng_state = {
            "python": random.getstate(),
            "numpy":  np.random.get_state(),
            "torch":  torch.get_rng_state(),
        }

    def evaluate_networks(self, *, step: int | None = None, plot: bool = False, store_last: bool = True):
        errors_by_agent = {i: [] for i in range(NUM_AGENTS)}
        ape_by_agent = {i: [] for i in range(NUM_AGENTS)}
        in_ci_by_agent = {i: [] for i in range(NUM_AGENTS)}
        eps = 1e-8

        for eval_state in self.evaluation_states:
            mc_values = eval_state["mc"]
            ci_low = eval_state.get("ci95_low", None)
            ci_high = eval_state.get("ci95_high", None)
            rewards = self.reward_module.get_reward(
                eval_state, eval_state["actor_id"],
                eval_state["agent_positions"][eval_state["actor_id"]],
                eval_state["mode"],
            )

            for i in range(NUM_AGENTS):
                input_ = self.view_controller(eval_state, i)
                if not self.exp_config.algorithm.centralized:
                    pred = float(self.critic_networks[eval_state["actor_id"]].get_value_function(input_))
                else:
                    pred = float(self.critic_networks[0].get_value_function(input_))

                true = float(mc_values[i]) if not self.exp_config.reward.reward_learning else float(rewards[i])
                err = true - pred
                errors_by_agent[i].append(err)
                ape_by_agent[i].append((abs(err) / abs(true)) * 100.0 if abs(true) > eps else abs(err))

                if (not self.exp_config.reward.reward_learning) and (ci_low is not None) and (ci_high is not None):
                    lo, hi = float(ci_low[i]), float(ci_high[i])
                    in_ci_by_agent[i].append(1.0 if (pred >= lo and pred <= hi) else 0.0)

        if store_last:
            self._last_eval_errors_by_agent = ape_by_agent

        all_ape = [x for xs in ape_by_agent.values() for x in xs]
        mae_pct_overall = float(np.mean(all_ape)) if all_ape else float("nan")
        mae_pct_by_agent = {i: (float(np.mean(xs)) if xs else None) for i, xs in ape_by_agent.items()}
        all_in_ci = [x for xs in in_ci_by_agent.values() for x in xs]
        coverage_overall = float(np.mean(all_in_ci)) if all_in_ci else None
        coverage_by_agent = {i: (float(np.mean(xs)) if xs else None) for i, xs in in_ci_by_agent.items()}

        if step is not None:
            self.eval_history.append({
                "step": int(step),
                "mae_pct_overall": mae_pct_overall,
                "mae_pct_by_agent": mae_pct_by_agent,
                "coverage_overall": coverage_overall,
                "coverage_by_agent": coverage_by_agent,
            })

        careful_pred_this_eval = np.full((NUM_AGENTS, len(self.careful_distances)), np.nan, dtype=float)

        for j, d in enumerate(self.careful_distances):
            item = self.careful_actor_states[j]
            if item is None:
                continue
            st = item["init_state"] if isinstance(item, dict) and "init_state" in item else item
            if st.get("actor_id", None) != 0:
                raise RuntimeError(f"Expected actor_id=0, got {st.get('actor_id')} at distance {d}")

            for eval_agent_id in range(NUM_AGENTS):
                input_ = self.view_controller(st, eval_agent_id)
                if not self.exp_config.algorithm.centralized:
                    pred = float(self.critic_networks[eval_agent_id].get_value_function(input_))
                else:
                    pred = float(self.critic_networks[0].get_value_function(input_))
                careful_pred_this_eval[eval_agent_id, j] = pred

        if step is not None:
            self.careful_eval_steps.append(int(step))
            for eval_agent_id in range(NUM_AGENTS):
                for j in range(len(self.careful_distances)):
                    self.careful_pred_history_actor0[eval_agent_id][j].append(
                        float(careful_pred_this_eval[eval_agent_id, j])
                    )

        return errors_by_agent

    def evaluate_networks_reward(self, *, step: int | None = None, plot: bool = False, store_last: bool = True):
        errors_by_state = {"Z0": [], "Z1": [], "Y11": [], "Y10": [], "Y00": [], "Y01": []}
        ape_by_state = {k: [] for k in errors_by_state}
        eps = 1e-8

        for state in self.evaluation_states:
            for eval_state in self.evaluation_states[state]:
                theoretical_values = self.theoretical_val.theoretical_value(
                    eval_state, eval_state["actor_id"], eval_state["agent_positions"], eval_=True
                )
                rewards = self.reward_module.get_reward(
                    eval_state, eval_state["actor_id"],
                    eval_state["agent_positions"][eval_state["actor_id"]],
                    eval_state["mode"],
                )

                for i in range(NUM_AGENTS):
                    input_ = self.view_controller(eval_state, i)
                    if self.exp_config.algorithm.eligibility or self.exp_config.algorithm.forward:
                        obs = ten(input_, DEVICE).view(-1)
                        pred = float(self.critic_networks[i].get_value_function(obs).cpu().item())
                    else:
                        pred = float(self.critic_networks[i].get_value_function(input_))

                    true = float(rewards[i]) if self.exp_config.reward.reward_learning else float(theoretical_values[i])
                    err = true - pred

                    agent_state = state
                    if eval_state["actor_id"] != i:
                        agent_state = {"Z1": "Z0", "Y11": "Y01"}.get(state, "Y00")

                    errors_by_state[agent_state].append(err)
                    ape_by_state[agent_state].append(
                        (abs(err) / abs(true)) * 100.0 if abs(true) > eps else abs(err)
                    )

        if store_last:
            self._last_eval_errors_by_state = ape_by_state

        mae_pct_by_state = {}
        all_apes = []
        for k, v in ape_by_state.items():
            if v:
                mae_pct_by_state[k] = float(np.mean(v))
                all_apes.extend(v)
            else:
                mae_pct_by_state[k] = None

        mae_pct_overall = float(np.mean(all_apes)) if all_apes else None

        if step is not None:
            self.eval_history.append({
                "step": int(step),
                "mae_pct_overall": mae_pct_overall,
                "mae_pct_by_state": mae_pct_by_state,
            })

        return errors_by_state

    def train(self):
        start_time = time.time()
        self.build_experiment()
        self.step_and_collect_observation()
        self.save_networks(self.data_dir / "weights")
        finalize_logging(self.data_dir, start_time)

    def save_networks(self, path: Path) -> None:
        os.makedirs(path, exist_ok=True)
        print("Saving networks: ", random.getstate()[1][0])
        payload = {"critics": []}
        for crit in self.critic_networks:
            payload["critics"].append({"blob": crit.export_net_state()})

        dst = path / f"weights.pt"
        torch.save(payload, dst)
