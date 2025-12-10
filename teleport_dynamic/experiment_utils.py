"""
Experiment Utilities for Reward and Value Learning

Contains shared functions for:
- Test set generation (centralized and decentralized)
- Evaluation metrics computation
- CSV logging for experiment results
- Benchmark checking
"""

import os
import csv
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable, Any
from dataclasses import dataclass

from tadd_helpers.env_functions import State, init_empty_state
from teleport_dynamic.base_value_model import BaseValueModelV2

import psutil
import os


def get_current_ram_mb() -> float:
    """Returns current process RAM usage in MB."""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


# =============================================================================
# Test Set Generation - Centralized
# =============================================================================


@dataclass
class CentralizedTestCase:
    """A single test case for centralized reward/value learning."""

    state: State
    acting_agent_idx: int
    true_reward: float
    category: str  # "picker" or "zero"


def generate_centralized_test_set(
    height: int,
    width: int,
    num_agents: int,
    num_apples: int,
    samples_per_category: int,
    fixed_apples: np.ndarray,
    seed: Optional[int] = None,
) -> Dict[str, List[CentralizedTestCase]]:
    """
    Generate test set for centralized reward learning.
    """
    if seed is not None:
        np.random.seed(seed)

    test_sets = {"picker": [], "zero": []}

    # Helper to create base state with fixed apples
    def create_base_state() -> State:
        s = init_empty_state(height, width, num_agents)
        if fixed_apples is not None:
            s.apples = fixed_apples.copy()  # <--- USE FIXED MAP
        else:
            raise ValueError(
                "fixed_apples must be provided for centralized test set generation"
            )
        return s

    # Generate PICKER cases (actor on apple)
    while len(test_sets["picker"]) < samples_per_category:
        s = create_base_state()
        apple_positions = list(zip(*np.where(s.apples > 0)))

        if len(apple_positions) == 0:
            continue

        # Pick random actor and place on apple
        actor_idx = np.random.randint(0, num_agents)
        apple_pos = apple_positions[np.random.randint(len(apple_positions))]
        s.set_agent_position(actor_idx, np.array(apple_pos))

        # Randomize other agent positions (can be anywhere, including on apples)
        for i in range(num_agents):
            if i != actor_idx:
                r = np.random.randint(0, height)
                c = np.random.randint(0, width)
                s.set_agent_position(i, np.array([r, c]))

        test_sets["picker"].append(
            CentralizedTestCase(
                state=s, acting_agent_idx=actor_idx, true_reward=1.0, category="picker"
            )
        )

    # Generate ZERO cases (actor NOT on apple)
    while len(test_sets["zero"]) < samples_per_category:
        s = create_base_state()
        apple_positions = set(zip(*np.where(s.apples > 0)))
        non_apple_positions = [
            (r, c)
            for r in range(height)
            for c in range(width)
            if (r, c) not in apple_positions
        ]

        if len(non_apple_positions) == 0:
            continue

        # Pick random actor and place NOT on apple
        actor_idx = np.random.randint(0, num_agents)
        pos = non_apple_positions[np.random.randint(len(non_apple_positions))]
        s.set_agent_position(actor_idx, np.array(pos))

        # Randomize other agent positions
        for i in range(num_agents):
            if i != actor_idx:
                r = np.random.randint(0, height)
                c = np.random.randint(0, width)
                s.set_agent_position(i, np.array([r, c]))

        test_sets["zero"].append(
            CentralizedTestCase(
                state=s, acting_agent_idx=actor_idx, true_reward=0.0, category="zero"
            )
        )

    return test_sets


# =============================================================================
# Test Set Generation - Decentralized
# =============================================================================


@dataclass
class DecentralizedTestCase:
    """A single test case for decentralized reward/value learning."""

    state: State
    acting_agent_idx: int
    self_agent_idx: int  # The perspective agent
    true_reward: float
    category: str


def generate_decentralized_test_set(
    height: int,
    width: int,
    num_agents: int,
    num_apples: int,
    samples_per_category: int,
    fixed_apples: np.ndarray,
    reward_func: Callable[[State, int], Dict[int, float]],
    seed: Optional[int] = None,
) -> Dict[str, List[DecentralizedTestCase]]:
    """
    Generate test set for decentralized reward learning.
    """
    if seed is not None:
        np.random.seed(seed)

    if num_agents < 2:
        raise ValueError("Decentralized requires at least 2 agents")

    test_sets = {
        "self_picker": [],
        "bystander": [],
        "zero_actor_miss": [],
        "zero_self_on_apple": [],
    }

    def create_base_state() -> State:
        s = init_empty_state(height, width, num_agents)
        if fixed_apples is not None:
            s.apples = fixed_apples.copy()  # <--- USE FIXED MAP
        else:
            raise ValueError(
                "fixed_apples must be provided for centralized test set generation"
            )
        return s

    def get_apple_positions(s: State) -> List[Tuple[int, int]]:
        return list(zip(*np.where(s.apples > 0)))

    def get_non_apple_positions(s: State) -> List[Tuple[int, int]]:
        apple_set = set(get_apple_positions(s))
        return [
            (r, c)
            for r in range(height)
            for c in range(width)
            if (r, c) not in apple_set
        ]

    # SELF_PICKER: self is actor AND self is on apple
    while len(test_sets["self_picker"]) < samples_per_category:
        s = create_base_state()
        apple_positions = get_apple_positions(s)
        if not apple_positions:
            continue

        self_idx = np.random.randint(0, num_agents)
        actor_idx = self_idx  # Self is the actor

        # Place self on apple
        apple_pos = apple_positions[np.random.randint(len(apple_positions))]
        s.set_agent_position(self_idx, np.array(apple_pos))

        # Randomize others
        for i in range(num_agents):
            if i != self_idx:
                r, c = np.random.randint(0, height), np.random.randint(0, width)
                s.set_agent_position(i, np.array([r, c]))

        rewards = reward_func(s, actor_idx)
        test_sets["self_picker"].append(
            DecentralizedTestCase(
                state=s,
                acting_agent_idx=actor_idx,
                self_agent_idx=self_idx,
                true_reward=rewards[self_idx],
                category="self_picker",
            )
        )

    # BYSTANDER: other is actor AND other is on apple
    while len(test_sets["bystander"]) < samples_per_category:
        s = create_base_state()
        apple_positions = get_apple_positions(s)
        if not apple_positions:
            continue

        self_idx = np.random.randint(0, num_agents)
        # Pick a different agent as actor
        other_agents = [i for i in range(num_agents) if i != self_idx]
        actor_idx = np.random.choice(other_agents)

        # Place actor (other) on apple
        apple_pos = apple_positions[np.random.randint(len(apple_positions))]
        s.set_agent_position(actor_idx, np.array(apple_pos))

        # Randomize all others including self
        for i in range(num_agents):
            if i != actor_idx:
                r, c = np.random.randint(0, height), np.random.randint(0, width)
                s.set_agent_position(i, np.array([r, c]))

        rewards = reward_func(s, actor_idx)
        test_sets["bystander"].append(
            DecentralizedTestCase(
                state=s,
                acting_agent_idx=actor_idx,
                self_agent_idx=self_idx,
                true_reward=rewards[self_idx],
                category="bystander",
            )
        )

    # ZERO_ACTOR_MISS: actor (anyone) is NOT on apple
    while len(test_sets["zero_actor_miss"]) < samples_per_category:
        s = create_base_state()
        non_apple_positions = get_non_apple_positions(s)
        if not non_apple_positions:
            continue

        self_idx = np.random.randint(0, num_agents)
        actor_idx = np.random.randint(0, num_agents)

        # Place actor NOT on apple
        pos = non_apple_positions[np.random.randint(len(non_apple_positions))]
        s.set_agent_position(actor_idx, np.array(pos))

        # Randomize others
        for i in range(num_agents):
            if i != actor_idx:
                r, c = np.random.randint(0, height), np.random.randint(0, width)
                s.set_agent_position(i, np.array([r, c]))

        rewards = reward_func(s, actor_idx)
        test_sets["zero_actor_miss"].append(
            DecentralizedTestCase(
                state=s,
                acting_agent_idx=actor_idx,
                self_agent_idx=self_idx,
                true_reward=rewards[self_idx],
                category="zero_actor_miss",
            )
        )

    # ZERO_SELF_ON_APPLE: self is on apple but OTHER is acting (and other is not on apple)
    while len(test_sets["zero_self_on_apple"]) < samples_per_category:
        s = create_base_state()
        apple_positions = get_apple_positions(s)
        non_apple_positions = get_non_apple_positions(s)
        if not apple_positions or not non_apple_positions:
            continue

        self_idx = np.random.randint(0, num_agents)
        other_agents = [i for i in range(num_agents) if i != self_idx]
        actor_idx = np.random.choice(other_agents)

        # Place self on apple
        apple_pos = apple_positions[np.random.randint(len(apple_positions))]
        s.set_agent_position(self_idx, np.array(apple_pos))

        # Place actor NOT on apple
        non_apple_pos = non_apple_positions[np.random.randint(len(non_apple_positions))]
        s.set_agent_position(actor_idx, np.array(non_apple_pos))

        # Randomize remaining agents
        for i in range(num_agents):
            if i != self_idx and i != actor_idx:
                r, c = np.random.randint(0, height), np.random.randint(0, width)
                s.set_agent_position(i, np.array([r, c]))

        rewards = reward_func(s, actor_idx)
        test_sets["zero_self_on_apple"].append(
            DecentralizedTestCase(
                state=s,
                acting_agent_idx=actor_idx,
                self_agent_idx=self_idx,
                true_reward=rewards[self_idx],
                category="zero_self_on_apple",
            )
        )

    return test_sets


# =============================================================================
# Evaluation Functions
# =============================================================================


@dataclass
class EvalResult:
    """Results from evaluating a model on a test set category."""

    category: str
    mean_abs_error: float
    max_abs_error: float
    mean_pct_error: float  # Percentage
    max_pct_error: float  # Percentage
    num_samples: int


def evaluate_centralized_model(
    model: BaseValueModelV2,
    test_sets: Dict[str, List[CentralizedTestCase]],
) -> Dict[str, EvalResult]:
    """
    Evaluate centralized model on test sets.
    """
    results = {}

    for category, test_cases in test_sets.items():
        if len(test_cases) == 0:
            continue

        errors = []
        for tc in test_cases:
            pred = model.get_value(tc.state, tc.acting_agent_idx)
            err = abs(pred - tc.true_reward)
            errors.append(err)

        errors = np.array(errors)

        # Robust percentage calculation
        target_val = test_cases[0].true_reward
        if abs(target_val) < 1e-9:
            # Target is 0: pct error = absolute error * 100
            pct_errors = errors * 100.0
        else:
            pct_errors = (errors / abs(target_val)) * 100.0

        results[category] = EvalResult(
            category=category,
            mean_abs_error=float(np.mean(errors)),
            max_abs_error=float(np.max(errors)),
            mean_pct_error=float(np.mean(pct_errors)),
            max_pct_error=float(np.max(pct_errors)),
            num_samples=len(test_cases),
        )

    return results


def evaluate_decentralized_models(
    models: List[BaseValueModelV2],
    test_sets: Dict[str, List[DecentralizedTestCase]],
) -> Dict[str, Dict[int, EvalResult]]:
    """
    Evaluate decentralized models on test sets.
    Returns: Dict of category -> Dict of agent_idx -> EvalResult
    """
    num_agents = len(models)
    results = {}

    for category, test_cases in test_sets.items():
        if len(test_cases) == 0:
            continue

        results[category] = {}

        # Group test cases by self_agent_idx
        agent_cases = {i: [] for i in range(num_agents)}
        for tc in test_cases:
            agent_cases[tc.self_agent_idx].append(tc)

        for agent_idx, cases in agent_cases.items():
            if len(cases) == 0:
                continue

            errors = []
            for tc in cases:
                pred = models[agent_idx].get_value(tc.state, tc.acting_agent_idx)
                err = abs(pred - tc.true_reward)
                errors.append(err)

            errors = np.array(errors)

            # Robust percentage calculation
            target_val = cases[0].true_reward
            if abs(target_val) < 1e-9:
                pct_errors = errors * 100.0
            else:
                pct_errors = (errors / abs(target_val)) * 100.0

            results[category][agent_idx] = EvalResult(
                category=category,
                mean_abs_error=float(np.mean(errors)) if len(errors) > 0 else 0.0,
                max_abs_error=float(np.max(errors)) if len(errors) > 0 else 0.0,
                mean_pct_error=(
                    float(np.mean(pct_errors)) if len(pct_errors) > 0 else 0.0
                ),
                max_pct_error=float(np.max(pct_errors)) if len(pct_errors) > 0 else 0.0,
                num_samples=len(cases),
            )

    return results


# =============================================================================
# Benchmark Checking
# =============================================================================


def check_benchmark_centralized(
    results: Dict[str, EvalResult],
    soft_threshold: float = 1.0,
    hard_threshold: float = 1.0,
) -> Tuple[bool, bool]:
    """Check if centralized model meets benchmarks."""
    if len(results) == 0:
        return False, False

    max_mean = max(r.mean_pct_error for r in results.values())
    max_max = max(r.max_pct_error for r in results.values())

    soft_passed = max_mean < soft_threshold
    hard_passed = max_max < hard_threshold

    return soft_passed, hard_passed


def check_benchmark_decentralized(
    results: Dict[str, Dict[int, EvalResult]],
    soft_threshold: float = 1.0,
    hard_threshold: float = 1.0,
) -> Tuple[bool, bool]:
    """Check if all decentralized models meet benchmarks."""
    if len(results) == 0:
        return False, False

    all_means = []
    all_maxes = []

    for category_results in results.values():
        for agent_result in category_results.values():
            all_means.append(agent_result.mean_pct_error)
            all_maxes.append(agent_result.max_pct_error)

    if len(all_means) == 0:
        return False, False

    max_mean = max(all_means)
    max_max = max(all_maxes)

    soft_passed = max_mean < soft_threshold
    hard_passed = max_max < hard_threshold

    return soft_passed, hard_passed


# =============================================================================
# CSV Logging
# =============================================================================


def append_experiment_result(
    csv_path: str,
    experiment_type: str,
    model_type: str,
    centralized: bool,
    grid_size: str,
    num_agents: int,
    num_apples: int,
    reward_scheme: str,
    model_config: str,
    soft_benchmark_step: Optional[int],
    hard_benchmark_step: Optional[int],
    final_step: int,
    final_mean_error: float,
    final_max_error: float,
    wall_time_seconds: float,
    notes: str = "",
):
    """Append a single experiment result to CSV file."""
    file_exists = os.path.exists(csv_path)

    fieldnames = [
        "timestamp",
        "experiment_type",
        "model_type",
        "centralized",
        "grid_size",
        "num_agents",
        "num_apples",
        "reward_scheme",
        "model_config",
        "soft_benchmark_step",
        "hard_benchmark_step",
        "final_step",
        "final_mean_error_pct",
        "final_max_error_pct",
        "wall_time_seconds",
        "notes",
    ]

    row = {
        "timestamp": datetime.now().isoformat(),
        "experiment_type": experiment_type,
        "model_type": model_type,
        "centralized": centralized,
        "grid_size": grid_size,
        "num_agents": num_agents,
        "num_apples": num_apples,
        "reward_scheme": reward_scheme,
        "model_config": model_config,
        "soft_benchmark_step": soft_benchmark_step if soft_benchmark_step else "N/A",
        "hard_benchmark_step": hard_benchmark_step if hard_benchmark_step else "N/A",
        "final_step": final_step,
        "final_mean_error_pct": f"{final_mean_error:.4f}",
        "final_max_error_pct": f"{final_max_error:.4f}",
        "wall_time_seconds": f"{wall_time_seconds:.1f}",
        "notes": notes,
    }

    # Ensure directory exists
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def format_eval_results_for_log(
    results: Dict[str, EvalResult],
    step: int,
    loss: float,
) -> str:
    """Format centralized evaluation results as a log line."""
    parts = [f"Step {step:<6} | Loss {loss:.6f}"]

    for category, result in sorted(results.items()):
        parts.append(
            f"{category}: {result.mean_pct_error:.2f}%/{result.max_pct_error:.2f}%"
        )

    return " | ".join(parts)


def format_decen_eval_results_for_log(
    results: Dict[str, Dict[int, EvalResult]],
    step: int,
    loss: float,
    num_agents: int,
) -> str:
    """Format decentralized evaluation results as a log line."""
    parts = [f"Step {step:<6} | Loss {loss:.6f}"]

    for category in sorted(results.keys()):
        agent_results = results[category]
        if len(agent_results) == 0:
            continue

        means = [r.mean_pct_error for r in agent_results.values()]
        maxes = [r.max_pct_error for r in agent_results.values()]

        avg_mean = np.mean(means)
        worst_max = np.max(maxes)

        parts.append(f"{category}: {avg_mean:.2f}%/{worst_max:.2f}%")

    return " | ".join(parts)
