# %%
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys

# Add the project root to the system path to allow imports from the orchard package.
sys.path.append("../")

from tests.cnn.centralized.generate_synthetic_states import (
    generate_synthetic_state_with_agent_picks_apple,
    generate_synthetic_state_with_agent_doesnt_pick_apple,
)
from config import GRAPHS_DIR

print("Imports successful.")

# %%
# --- Configuration for the datasets ---
datasets_config = [
    {"name": "6x6_2_agents", "width": 6, "height": 6, "num_agents": 2},
    {"name": "9x9_4_agents", "width": 9, "height": 9, "num_agents": 4},
    {"name": "12x12_7_agents", "width": 12, "height": 12, "num_agents": 7},
]

# --- Generation Parameters ---
# This value controls the general density of apples. Adjust as needed.
S_TARGET = 1

# Number of synthetic states to generate for each condition (picks vs. doesn't pick)
NUM_SAMPLES_PER_SET = 1000000

# The name of the file where the final dataset will be saved
OUTPUT_FILENAME = GRAPHS_DIR / "synthetic_rl_datasets.pkl"

print(
    f"Generating {NUM_SAMPLES_PER_SET} samples for {len(datasets_config)} configurations."
)

# %%
all_datasets = {}
print(NUM_SAMPLES_PER_SET)
for config in datasets_config:
    dataset_name = config["name"]
    print(f"--- Generating dataset: {dataset_name} ---")

    # --- Sub-dataset 1: Agent picks an apple (reward = 1) ---
    picks_apple_data = []
    for i in range(NUM_SAMPLES_PER_SET):
        state, agent_pos, reward = generate_synthetic_state_with_agent_picks_apple(
            num_agents=config["num_agents"],
            width=config["width"],
            height=config["height"],
            s_target=S_TARGET,
        )
        # We store everything in a dictionary for clarity
        picks_apple_data.append(
            {"state": state, "agent_pos": agent_pos, "reward": reward}
        )

    # --- Sub-dataset 2: Agent does NOT pick an apple (reward = 0) ---
    no_apple_data = []
    for i in range(NUM_SAMPLES_PER_SET):
        state, agent_pos, reward = (
            generate_synthetic_state_with_agent_doesnt_pick_apple(
                num_agents=config["num_agents"],
                width=config["width"],
                height=config["height"],
                s_target=S_TARGET,
            )
        )
        no_apple_data.append({"state": state, "agent_pos": agent_pos, "reward": reward})

    # Store the two sub-datasets under the main dataset's name
    all_datasets[dataset_name] = {
        "picks_apple": picks_apple_data,
        "doesnt_pick_apple": no_apple_data,
    }

print("\n--- ✅ All datasets generated successfully! ---")

# %%


# %%
with open(OUTPUT_FILENAME, "wb") as f:
    pickle.dump(all_datasets, f)

print(f"All generated data has been saved to '{OUTPUT_FILENAME}'")

# %%
import pickle
import numpy as np
from pathlib import Path

# --- Define your filenames ---
# Assuming GRAPHS_DIR is defined correctly, e.g.: GRAPHS_DIR = Path("generated_data")
PICKLE_FILENAME = GRAPHS_DIR / "synthetic_rl_datasets.pkl"
HTML_FILENAME = GRAPHS_DIR / "synthetic_rl_datasets_VIEWABLE.html"

# --- 1. Load the data from the Pickle file ---
print(f"Loading data from '{PICKLE_FILENAME}'...")
with open(PICKLE_FILENAME, "rb") as f:
    all_data = pickle.load(f)


# --- 2. Helper function to convert a matrix into an HTML table ---
def matrix_to_html_table(matrix):
    # Use a simple color scale for the apple grid
    def get_color(value):
        if value == 0:
            return "#FFFFFF"  # White
        if value == 1:
            return "#FFCDD2"  # Light Red
        if value < 4:
            return "#E57373"  # Medium Red
        return "#B71C1C; color: white;"  # Dark Red

    html = '<table style="border-collapse: collapse; margin: 10px; font-family: monospace;">'
    for row in matrix:
        html += "<tr>"
        for cell in row:
            style = f"border: 1px solid #ccc; width: 25px; height: 25px; text-align: center; background-color: {get_color(cell)}"
            html += f'<td style="{style}">{cell if cell > 0 else ""}</td>'
        html += "</tr>"
    html += "</table>"
    return html


# --- 3. Build the HTML content ---
print("Generating HTML content...")
html_content = """
<html>
<head>
    <title>Synthetic State Viewer</title>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        .entry { border: 1px solid #ddd; border-radius: 5px; margin-bottom: 20px; padding: 15px; background-color: #f9f9f9; }
        .grids { display: flex; flex-direction: row; }
        h1, h2, h3 { color: #333; }
    </style>
</head>
<body>
    <h1>Synthetic RL Dataset Viewer</h1>
"""

for dataset_name, data in all_data.items():
    html_content += f"<h2>Dataset: {dataset_name}</h2>"
    for condition, samples in data.items():
        html_content += f"<h3>Condition: {condition}</h3>"
        # Let's just show the first 20 samples to keep the file size reasonable
        for i, sample in enumerate(samples[:20]):
            state = sample["state"]
            html_content += f"""
            <div class="entry">
                <h4>Sample #{i} &nbsp;&nbsp;|&nbsp;&nbsp; Reward: {sample['reward']} &nbsp;&nbsp;|&nbsp;&nbsp; Agent Pos: {sample['agent_pos']}</h4>
                <div class="grids">
                    <div>
                        <p><b>Agents Grid:</b></p>
                        {matrix_to_html_table(state['agents'])}
                    </div>
                    <div>
                        <p><b>Apples Grid:</b></p>
                        {matrix_to_html_table(state['apples'])}
                    </div>
                </div>
            </div>
            """

html_content += "</body></html>"

# --- 4. Write the HTML file ---
with open(HTML_FILENAME, "w") as f:
    f.write(html_content)

print("\n✅ Success! You can now open the new file in your file explorer:")
print(f"--> {HTML_FILENAME}")

# %%
import pickle
import numpy as np
from pathlib import Path

# --- Configuration ---
# Assuming GRAPHS_DIR is defined and contains your pickle file
PICKLE_FILENAME = GRAPHS_DIR / "synthetic_rl_datasets.pkl"
DOCS_FILENAME = GRAPHS_DIR / "README.md"

# --- 1. Load the data to inspect it ---
print(f"Loading data from '{PICKLE_FILENAME}' to generate documentation...")
with open(PICKLE_FILENAME, "rb") as f:
    all_data = pickle.load(f)

# --- 2. Start building the Markdown string ---
# Inspect the first sample to learn about the structure
first_dataset_key = next(iter(all_data))
first_condition_key = next(iter(all_data[first_dataset_key]))
first_sample = all_data[first_dataset_key][first_condition_key][0]

# Extract properties from the first sample
reward_type = type(first_sample["reward"]).__name__
agent_pos_type = type(first_sample["agent_pos"]).__name__

# Start writing the content
readme_content = f"""
# Documentation for `synthetic_rl_datasets.pkl`

This document describes the structure and contents of the `{PICKLE_FILENAME.name}` file.

## High-Level Summary

The pickle file contains a single Python `dictionary`. This dictionary holds multiple generated datasets, each corresponding to a different environment configuration (grid size and number of agents).

- **Purpose:** To provide clean, synthetic states for training and debugging a reinforcement learning agent.
- **Generated on:** {np.datetime64('now', 'D')}

---

## Data Structure

The data is a nested dictionary with the following hierarchy:

`[dataset_name]` -> `[condition]` -> `[list_of_samples]`

### 1. Top-Level Keys (`dataset_name`)

The main keys represent the different environment configurations. The datasets found in this file are:
"""

for key in all_data.keys():
    readme_content += f"- `{key}`\n"

#
# --- THIS IS THE CORRECTED BLOCK ---
# It now uses triple quotes """ to handle the multi-line string.
#
readme_content += """
### 2. Second-Level Keys (`condition`)

Each dataset is a dictionary containing two keys, representing the two conditions under which data was generated:

- `picks_apple`: A list of states where an agent is guaranteed to start on the same cell as an apple (reward = 1).
- `doesnt_pick_apple`: A list of states where no agent starts on a cell with an apple (reward = 0).
"""

# Add statistics for each dataset
readme_content += "\n### 3. Dataset Properties\n\n"
for name, data in all_data.items():
    picks_count = len(data["picks_apple"])
    no_picks_count = len(data["doesnt_pick_apple"])
    # Get shape from the first sample of this specific dataset
    sample_state = data["picks_apple"][0]["state"]
    grid_shape = sample_state["agents"].shape

    readme_content += f"- **`{name}`**:\n"
    readme_content += f"  - Grid Shape: `{grid_shape}`\n"
    readme_content += f"  - `picks_apple`: Contains {picks_count} samples.\n"
    readme_content += f"  - `doesnt_pick_apple`: Contains {no_picks_count} samples.\n"


readme_content += f"""
---

## Sample Structure

Each sample in the list is a `dictionary` with the following keys and properties:

- **`reward`**:
  - **Type:** `{reward_type}`
  - **Description:** The reward for the given state. Is `1` if the agent picks an apple, `0` otherwise.

- **`agent_pos`**:
  - **Type:** `{agent_pos_type}`
  - **Description:** A tuple `(row, col)` indicating the position of the acting agent.

- **`state`**:
  - **Type:** `dict`
  - **Description:** A dictionary containing the grid representations of the environment. It has two keys:
    - **`'agents'`**:
        - **Type:** `numpy.ndarray`
        - **Dtype:** `{first_sample['state']['agents'].dtype}`
        - **Values:** `1` for a cell containing an agent, `0` otherwise.
    - **`'apples'`**:
        - **Type:** `numpy.ndarray`
        - **Dtype:** `{first_sample['state']['apples'].dtype}`
        - **Values:** An integer representing the number of apples in a cell (can be >= 0).
"""

# --- 3. Write the content to the README.md file ---
print(f"Writing documentation to '{DOCS_FILENAME}'...")
with open(DOCS_FILENAME, "w") as f:
    f.write(readme_content)

print("\n✅ Success! A README.md file has been generated in your data directory.")
