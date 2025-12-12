"""
Experiment Utilities for Reward and Value Learning

Contains shared functions for:
- Test set generation (centralized and decentralized, reward and value)
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
from teleport_dynamic.analytical import get_exact_value_centralized

import psutil


def get_current_ram_mb() -> float:
    """Returns current process RAM usage in MB."""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


# =============================================================================
# Test Case Data Classes
# =============================================================================


@dataclass
class CentralizedTestCase:
    """A single test case for centralized reward/value learning."""

    state: State
    acting_agent_idx: int
    true_reward: float
    true_value: Optional[float]  # None for reward learning
    category: str


@dataclass
class DecentralizedTestCase:
    """A single test case for decentralized reward/value learning."""

    state: State
    acting_agent_idx: int
    self_agent_idx: int
    true_reward: float
    true_value: Optional[float]  # None for reward learning
    category: str


# =============================================================================
# Test Set Generation - Centralized Reward Learning
# =============================================================================


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
    Categories: picker (actor on apple), zero (actor not on apple)
    """
    if seed is not None:
        np.random.seed(seed)

    test_sets = {"picker": [], "zero": []}

    def create_base_state() -> State:
        s = init_empty_state(height, width, num_agents)
        if fixed_apples is not None:
            s.apples = fixed_apples.copy()
        else:
            raise ValueError("fixed_apples must be provided")
        return s

    # PICKER cases (actor on apple)
    while len(test_sets["picker"]) < samples_per_category:
        s = create_base_state()
        apple_positions = list(zip(*np.where(s.apples > 0)))
        if not apple_positions:
            continue

        actor_idx = np.random.randint(0, num_agents)
        apple_pos = apple_positions[np.random.randint(len(apple_positions))]
        s.set_agent_position(actor_idx, np.array(apple_pos))

        for i in range(num_agents):
            if i != actor_idx:
                r = np.random.randint(0, height)
                c = np.random.randint(0, width)
                s.set_agent_position(i, np.array([r, c]))

        test_sets["picker"].append(
            CentralizedTestCase(
                state=s,
                acting_agent_idx=actor_idx,
                true_reward=1.0,
                true_value=None,
                category="picker",
            )
        )

    # ZERO cases (actor NOT on apple)
    while len(test_sets["zero"]) < samples_per_category:
        s = create_base_state()
        apple_positions = set(zip(*np.where(s.apples > 0)))
        non_apple_positions = [
            (r, c)
            for r in range(height)
            for c in range(width)
            if (r, c) not in apple_positions
        ]
        if not non_apple_positions:
            continue

        actor_idx = np.random.randint(0, num_agents)
        pos = non_apple_positions[np.random.randint(len(non_apple_positions))]
        s.set_agent_position(actor_idx, np.array(pos))

        for i in range(num_agents):
            if i != actor_idx:
                r = np.random.randint(0, height)
                c = np.random.randint(0, width)
                s.set_agent_position(i, np.array([r, c]))

        test_sets["zero"].append(
            CentralizedTestCase(
                state=s,
                acting_agent_idx=actor_idx,
                true_reward=0.0,
                true_value=None,
                category="zero",
            )
        )

    return test_sets


# =============================================================================
# Test Set Generation - Centralized Value Learning
# =============================================================================


def generate_centralized_value_test_set(
    height: int,
    width: int,
    num_agents: int,
    num_apples: int,
    samples_per_category: int,
    fixed_apples: np.ndarray,
    gamma: float,
    seed: Optional[int] = None,
) -> Dict[str, List[CentralizedTestCase]]:
    """
    Generate test set for centralized value learning.
    Uses get_exact_value_centralized for true values.
    """
    # Import the NEW analytical function and the centralized reward function
    from teleport_dynamic.analytical import get_exact_value_centralized
    from teleport_dynamic.rewards_centralized import get_reward_centralized

    if seed is not None:
        np.random.seed(seed)

    test_sets = {"picker": [], "zero": []}

    def create_base_state() -> State:
        s = init_empty_state(height, width, num_agents)
        if fixed_apples is not None:
            s.apples = fixed_apples.copy()
        else:
            raise ValueError("fixed_apples must be provided")
        return s

    # Note: reward_func_wrapper is DELETED. We use get_reward_centralized directly.

    # PICKER cases (actor on apple)
    while len(test_sets["picker"]) < samples_per_category:
        s = create_base_state()
        apple_positions = list(zip(*np.where(s.apples > 0)))
        if not apple_positions:
            continue

        actor_idx = np.random.randint(0, num_agents)
        apple_pos = apple_positions[np.random.randint(len(apple_positions))]
        s.set_agent_position(actor_idx, np.array(apple_pos))

        for i in range(num_agents):
            if i != actor_idx:
                r = np.random.randint(0, height)
                c = np.random.randint(0, width)
                s.set_agent_position(i, np.array([r, c]))

        # --- UPDATED CALL ---
        # Uses the new specific function, passing the reward function directly
        true_val = get_exact_value_centralized(
            s, actor_idx, get_reward_centralized, gamma
        )

        test_sets["picker"].append(
            CentralizedTestCase(
                state=s,
                acting_agent_idx=actor_idx,
                true_reward=1.0,
                true_value=true_val,
                category="picker",
            )
        )

    # ZERO cases (actor NOT on apple)
    while len(test_sets["zero"]) < samples_per_category:
        s = create_base_state()
        apple_positions = set(zip(*np.where(s.apples > 0)))
        non_apple_positions = [
            (r, c)
            for r in range(height)
            for c in range(width)
            if (r, c) not in apple_positions
        ]
        if not non_apple_positions:
            continue

        actor_idx = np.random.randint(0, num_agents)
        pos = non_apple_positions[np.random.randint(len(non_apple_positions))]
        s.set_agent_position(actor_idx, np.array(pos))

        for i in range(num_agents):
            if i != actor_idx:
                r = np.random.randint(0, height)
                c = np.random.randint(0, width)
                s.set_agent_position(i, np.array([r, c]))

        # --- UPDATED CALL ---
        true_val = get_exact_value_centralized(
            s, actor_idx, get_reward_centralized, gamma
        )

        test_sets["zero"].append(
            CentralizedTestCase(
                state=s,
                acting_agent_idx=actor_idx,
                true_reward=0.0,
                true_value=true_val,
                category="zero",
            )
        )

    return test_sets


# =============================================================================
# Test Set Generation - Decentralized Reward Learning
# =============================================================================


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
    Categories: self_picker, bystander, zero_actor_miss, zero_self_on_apple
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
            s.apples = fixed_apples.copy()
        else:
            raise ValueError("fixed_apples must be provided")
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
        actor_idx = self_idx

        apple_pos = apple_positions[np.random.randint(len(apple_positions))]
        s.set_agent_position(self_idx, np.array(apple_pos))

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
                true_value=None,
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
        other_agents = [i for i in range(num_agents) if i != self_idx]
        actor_idx = np.random.choice(other_agents)

        apple_pos = apple_positions[np.random.randint(len(apple_positions))]
        s.set_agent_position(actor_idx, np.array(apple_pos))

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
                true_value=None,
                category="bystander",
            )
        )

    # ZERO_ACTOR_MISS: actor misses (no apple), self anywhere
    while len(test_sets["zero_actor_miss"]) < samples_per_category:
        s = create_base_state()
        non_apple_positions = get_non_apple_positions(s)
        if not non_apple_positions:
            continue

        self_idx = np.random.randint(0, num_agents)
        actor_idx = np.random.randint(0, num_agents)

        pos = non_apple_positions[np.random.randint(len(non_apple_positions))]
        s.set_agent_position(actor_idx, np.array(pos))

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
                true_value=None,
                category="zero_actor_miss",
            )
        )

    # ZERO_SELF_ON_APPLE: self on apple but NOT actor (other misses)
    while len(test_sets["zero_self_on_apple"]) < samples_per_category:
        s = create_base_state()
        apple_positions = get_apple_positions(s)
        non_apple_positions = get_non_apple_positions(s)
        if not apple_positions or not non_apple_positions:
            continue

        self_idx = np.random.randint(0, num_agents)
        other_agents = [i for i in range(num_agents) if i != self_idx]
        if not other_agents:
            continue
        actor_idx = np.random.choice(other_agents)

        # Place self on apple
        apple_pos = apple_positions[np.random.randint(len(apple_positions))]
        s.set_agent_position(self_idx, np.array(apple_pos))

        # Place actor NOT on apple
        non_apple_pos = non_apple_positions[np.random.randint(len(non_apple_positions))]
        s.set_agent_position(actor_idx, np.array(non_apple_pos))

        # Randomize others
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
                true_value=None,
                category="zero_self_on_apple",
            )
        )

    return test_sets


# =============================================================================
# Test Set Generation - Decentralized Value Learning
# =============================================================================


def generate_decentralized_value_test_set(
    height: int,
    width: int,
    num_agents: int,
    num_apples: int,
    samples_per_category: int,
    fixed_apples: np.ndarray,
    reward_func: Callable[[State, int], Dict[int, float]],
    gamma: float,
    seed: Optional[int] = None,
) -> Dict[str, List[DecentralizedTestCase]]:
    """
    Generate test set for decentralized value learning.
    Uses analytical solution for true values.

    Categories: self_picker, bystander, zero_actor_miss, zero_self_on_apple
    """
    from teleport_dynamic.analytical import get_exact_value

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
            s.apples = fixed_apples.copy()
        else:
            raise ValueError("fixed_apples must be provided")
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

    # SELF_PICKER
    while len(test_sets["self_picker"]) < samples_per_category:
        s = create_base_state()
        apple_positions = get_apple_positions(s)
        if not apple_positions:
            continue

        self_idx = np.random.randint(0, num_agents)
        actor_idx = self_idx

        apple_pos = apple_positions[np.random.randint(len(apple_positions))]
        s.set_agent_position(self_idx, np.array(apple_pos))

        for i in range(num_agents):
            if i != self_idx:
                r, c = np.random.randint(0, height), np.random.randint(0, width)
                s.set_agent_position(i, np.array([r, c]))

        rewards = reward_func(s, actor_idx)
        true_val = get_exact_value(s, actor_idx, self_idx, reward_func, gamma)

        test_sets["self_picker"].append(
            DecentralizedTestCase(
                state=s,
                acting_agent_idx=actor_idx,
                self_agent_idx=self_idx,
                true_reward=rewards[self_idx],
                true_value=true_val,
                category="self_picker",
            )
        )

    # BYSTANDER
    while len(test_sets["bystander"]) < samples_per_category:
        s = create_base_state()
        apple_positions = get_apple_positions(s)
        if not apple_positions:
            continue

        self_idx = np.random.randint(0, num_agents)
        other_agents = [i for i in range(num_agents) if i != self_idx]
        actor_idx = np.random.choice(other_agents)

        apple_pos = apple_positions[np.random.randint(len(apple_positions))]
        s.set_agent_position(actor_idx, np.array(apple_pos))

        for i in range(num_agents):
            if i != actor_idx:
                r, c = np.random.randint(0, height), np.random.randint(0, width)
                s.set_agent_position(i, np.array([r, c]))

        rewards = reward_func(s, actor_idx)
        true_val = get_exact_value(s, actor_idx, self_idx, reward_func, gamma)

        test_sets["bystander"].append(
            DecentralizedTestCase(
                state=s,
                acting_agent_idx=actor_idx,
                self_agent_idx=self_idx,
                true_reward=rewards[self_idx],
                true_value=true_val,
                category="bystander",
            )
        )

    # ZERO_ACTOR_MISS
    while len(test_sets["zero_actor_miss"]) < samples_per_category:
        s = create_base_state()
        non_apple_positions = get_non_apple_positions(s)
        if not non_apple_positions:
            continue

        self_idx = np.random.randint(0, num_agents)
        actor_idx = np.random.randint(0, num_agents)

        pos = non_apple_positions[np.random.randint(len(non_apple_positions))]
        s.set_agent_position(actor_idx, np.array(pos))

        for i in range(num_agents):
            if i != actor_idx:
                r, c = np.random.randint(0, height), np.random.randint(0, width)
                s.set_agent_position(i, np.array([r, c]))

        rewards = reward_func(s, actor_idx)
        true_val = get_exact_value(s, actor_idx, self_idx, reward_func, gamma)

        test_sets["zero_actor_miss"].append(
            DecentralizedTestCase(
                state=s,
                acting_agent_idx=actor_idx,
                self_agent_idx=self_idx,
                true_reward=rewards[self_idx],
                true_value=true_val,
                category="zero_actor_miss",
            )
        )

    # ZERO_SELF_ON_APPLE
    while len(test_sets["zero_self_on_apple"]) < samples_per_category:
        s = create_base_state()
        apple_positions = get_apple_positions(s)
        non_apple_positions = get_non_apple_positions(s)
        if not apple_positions or not non_apple_positions:
            continue

        self_idx = np.random.randint(0, num_agents)
        other_agents = [i for i in range(num_agents) if i != self_idx]
        if not other_agents:
            continue
        actor_idx = np.random.choice(other_agents)

        apple_pos = apple_positions[np.random.randint(len(apple_positions))]
        s.set_agent_position(self_idx, np.array(apple_pos))

        non_apple_pos = non_apple_positions[np.random.randint(len(non_apple_positions))]
        s.set_agent_position(actor_idx, np.array(non_apple_pos))

        for i in range(num_agents):
            if i != self_idx and i != actor_idx:
                r, c = np.random.randint(0, height), np.random.randint(0, width)
                s.set_agent_position(i, np.array([r, c]))

        rewards = reward_func(s, actor_idx)
        true_val = get_exact_value(s, actor_idx, self_idx, reward_func, gamma)

        test_sets["zero_self_on_apple"].append(
            DecentralizedTestCase(
                state=s,
                acting_agent_idx=actor_idx,
                self_agent_idx=self_idx,
                true_reward=rewards[self_idx],
                true_value=true_val,
                category="zero_self_on_apple",
            )
        )

    return test_sets


# =============================================================================
# Evaluation Results
# =============================================================================


@dataclass
class EvalResult:
    """Evaluation result for a single category."""

    category: str
    mean_abs_error: float
    max_abs_error: float
    mean_pct_error: float
    max_pct_error: float
    mean_signed_pct_error: float
    num_samples: int


def evaluate_centralized_model(
    model: BaseValueModelV2,
    test_sets: Dict[str, List[CentralizedTestCase]],
    use_value: bool = False,
) -> Dict[str, EvalResult]:
    """
    Evaluate centralized model on test sets.
    Calculates both Absolute Error (Accuracy) and Signed Error (Bias).
    """
    results = {}

    for category, test_cases in test_sets.items():
        if len(test_cases) == 0:
            continue

        # 1. Collect Predictions and Targets
        preds = []
        targets = []

        for tc in test_cases:
            p = model.get_value(tc.state, tc.acting_agent_idx)
            t = tc.true_value if use_value else tc.true_reward
            if t is None:
                raise ValueError(f"Target is None for {category}")
            preds.append(p)
            targets.append(t)

        preds = np.array(preds)
        targets = np.array(targets)

        # 2. Calculate Signed and Absolute Errors
        signed_errors = preds - targets
        abs_errors = np.abs(signed_errors)

        # 3. Calculate Percentages (Relative to Mean Target Magnitude)
        mean_target_mag = np.mean(np.abs(targets))

        if mean_target_mag < 1e-9:
            # Avoid division by zero for zero-target cases
            pct_signed_errors = signed_errors * 100.0
            pct_abs_errors = abs_errors * 100.0
        else:
            pct_signed_errors = (signed_errors / mean_target_mag) * 100.0
            pct_abs_errors = (abs_errors / mean_target_mag) * 100.0

        results[category] = EvalResult(
            category=category,
            mean_abs_error=float(np.mean(abs_errors)),
            max_abs_error=float(np.max(abs_errors)),
            mean_pct_error=float(np.mean(pct_abs_errors)),
            max_pct_error=float(np.max(pct_abs_errors)),
            mean_signed_pct_error=float(np.mean(pct_signed_errors)),  # Bias
            num_samples=len(test_cases),
        )

    return results


def evaluate_decentralized_models(
    models: List[BaseValueModelV2],
    test_sets: Dict[str, List[DecentralizedTestCase]],
    use_value: bool = False,
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

        # Group by self_agent_idx
        agent_cases = {i: [] for i in range(num_agents)}
        for tc in test_cases:
            agent_cases[tc.self_agent_idx].append(tc)

        for agent_idx, cases in agent_cases.items():
            if len(cases) == 0:
                continue

            # 1. Collect Predictions and Targets
            preds = []
            targets = []

            for tc in cases:
                p = models[agent_idx].get_value(tc.state, tc.acting_agent_idx)
                t = tc.true_value if use_value else tc.true_reward
                if t is None:
                    raise ValueError(f"Target is None for {category}")
                preds.append(p)
                targets.append(t)

            preds = np.array(preds)
            targets = np.array(targets)

            # 2. Calculate Signed and Absolute Errors
            signed_errors = preds - targets
            abs_errors = np.abs(signed_errors)

            # 3. Calculate Percentages
            mean_target_mag = np.mean(np.abs(targets))

            if mean_target_mag < 1e-9:
                pct_signed_errors = signed_errors * 100.0
                pct_abs_errors = abs_errors * 100.0
            else:
                pct_signed_errors = (signed_errors / mean_target_mag) * 100.0
                pct_abs_errors = (abs_errors / mean_target_mag) * 100.0

            results[category][agent_idx] = EvalResult(
                category=category,
                mean_abs_error=(
                    float(np.mean(abs_errors)) if len(abs_errors) > 0 else 0.0
                ),
                max_abs_error=float(np.max(abs_errors)) if len(abs_errors) > 0 else 0.0,
                mean_pct_error=(
                    float(np.mean(pct_abs_errors)) if len(pct_abs_errors) > 0 else 0.0
                ),
                max_pct_error=(
                    float(np.max(pct_abs_errors)) if len(pct_abs_errors) > 0 else 0.0
                ),
                mean_signed_pct_error=(
                    float(np.mean(pct_signed_errors))
                    if len(pct_signed_errors) > 0
                    else 0.0
                ),  # Bias
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
    num_parameters: int,
    kernel_size: Optional[int] = None,  # None for MLP
    learning_method: str = "reward",  # "reward", "td0", "td_lambda"
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
        "learning_method",
        "model_config",
        "num_parameters",
        "kernel_size",
        "soft_benchmark_step",
        "hard_benchmark_step",
        "final_step",
        "final_mean_error_pct",
        "final_max_error_pct",
        "wall_time_seconds",
        "converged",
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
        "learning_method": learning_method,
        "model_config": model_config,
        "num_parameters": num_parameters,
        "kernel_size": kernel_size if kernel_size else "N/A",
        "soft_benchmark_step": soft_benchmark_step if soft_benchmark_step else "N/A",
        "hard_benchmark_step": hard_benchmark_step if hard_benchmark_step else "N/A",
        "final_step": final_step,
        "final_mean_error_pct": f"{final_mean_error:.4f}",
        "final_max_error_pct": f"{final_max_error:.4f}",
        "wall_time_seconds": f"{wall_time_seconds:.1f}",
        "converged": hard_benchmark_step is not None,
        "notes": notes,
    }

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# =============================================================================
# Log Formatting
# =============================================================================


def format_eval_results_for_log(
    results: Dict[str, EvalResult],
    step: int,
    loss: float,
    lr: float,
) -> str:
    """Format centralized evaluation results as a log line."""
    # Added LR to the header
    parts = [f"Step {step:<6} | Loss {loss:.6f} | LR {lr:.2e}"]

    for category, result in sorted(results.items()):
        parts.append(
            f"{category}: {result.mean_pct_error:.2f}% (bias {result.mean_signed_pct_error:+.2f}%) / {result.max_pct_error:.2f}%"
        )

    return " | ".join(parts)


def format_decen_eval_results_for_log(
    results: Dict[str, Dict[int, EvalResult]],
    step: int,
    loss: float,
    num_agents: int,
    lr: float,
) -> str:
    """Format decentralized evaluation results as a log line."""
    # Added LR to the header
    parts = [f"Step {step:<6} | Loss {loss:.6f} | LR {lr:.2e}"]

    for category in sorted(results.keys()):
        agent_results = results[category]
        if len(agent_results) == 0:
            continue

        means = [r.mean_pct_error for r in agent_results.values()]
        maxes = [r.max_pct_error for r in agent_results.values()]
        biases = [r.mean_signed_pct_error for r in agent_results.values()]

        avg_mean = np.mean(means)
        worst_max = np.max(maxes)
        avg_bias = np.mean(biases)

        parts.append(
            f"{category}: {avg_mean:.2f}% (bias {avg_bias:+.2f}%) / {worst_max:.2f}%"
        )

    return " | ".join(parts)
