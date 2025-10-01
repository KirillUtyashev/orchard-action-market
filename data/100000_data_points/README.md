
# Documentation for `synthetic_rl_datasets.pkl`

This document describes the structure and contents of the `synthetic_rl_datasets.pkl` file.

## High-Level Summary

The pickle file contains a single Python `dictionary`. This dictionary holds multiple generated datasets, each corresponding to a different environment configuration (grid size and number of agents).

- **Purpose:** To provide clean, synthetic states for training and debugging a reinforcement learning agent.
- **Generated on:** 2025-09-30

---

## Data Structure

The data is a nested dictionary with the following hierarchy:

`[dataset_name]` -> `[condition]` -> `[list_of_samples]`

### 1. Top-Level Keys (`dataset_name`)

The main keys represent the different environment configurations. The datasets found in this file are:
- `6x6_2_agents`
- `9x9_4_agents`
- `12x12_7_agents`

### 2. Second-Level Keys (`condition`)

Each dataset is a dictionary containing two keys, representing the two conditions under which data was generated:

- `picks_apple`: A list of states where an agent is guaranteed to start on the same cell as an apple (reward = 1).
- `doesnt_pick_apple`: A list of states where no agent starts on a cell with an apple (reward = 0).

### 3. Dataset Properties

- **`6x6_2_agents`**:
  - Grid Shape: `(6, 6)`
  - `picks_apple`: Contains 100000 samples.
  - `doesnt_pick_apple`: Contains 100000 samples.
- **`9x9_4_agents`**:
  - Grid Shape: `(9, 9)`
  - `picks_apple`: Contains 100000 samples.
  - `doesnt_pick_apple`: Contains 100000 samples.
- **`12x12_7_agents`**:
  - Grid Shape: `(12, 12)`
  - `picks_apple`: Contains 100000 samples.
  - `doesnt_pick_apple`: Contains 100000 samples.

---

## Sample Structure

Each sample in the list is a `dictionary` with the following keys and properties:

- **`reward`**:
  - **Type:** `int`
  - **Description:** The reward for the given state. Is `1` if the agent picks an apple, `0` otherwise.

- **`agent_pos`**:
  - **Type:** `tuple`
  - **Description:** A tuple `(row, col)` indicating the position of the acting agent.

- **`state`**:
  - **Type:** `dict`
  - **Description:** A dictionary containing the grid representations of the environment. It has two keys:
    - **`'agents'`**:
        - **Type:** `numpy.ndarray`
        - **Dtype:** `int8`
        - **Values:** `1` for a cell containing an agent, `0` otherwise.
    - **`'apples'`**:
        - **Type:** `numpy.ndarray`
        - **Dtype:** `int8`
        - **Values:** An integer representing the number of apples in a cell (can be >= 0).
