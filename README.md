# Orchard Action Market

Research code for multi-agent reinforcement learning in a grid-based orchard environment. The implementation supports value learning and actor-critic training, centralized or decentralized critics, stochastic task generation, checkpointing, evaluation, and rollout visualization.

The runnable project lives in `orchard_rl/`; its Python package is `orchard`.

## Getting started

Use Python 3.10 or newer. The checked-in dependency file pins a PyTorch build for CUDA 12.6, so the standard setup is intended for a Linux machine with a compatible NVIDIA driver.

From a local checkout of the repository:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r orchard_rl/requirements.txt

For macOS or a CPU-only environment, install the packages listed in `orchard_rl/requirements.txt` using the PyTorch build appropriate for your platform instead of `torch==2.11.0+cu126`. The default reference configuration sets `train.use_gpu: false`.

## Run a training experiment

Commands should be run from `orchard_rl/` so that the local `orchard` package and relative output paths resolve correctly.

    cd orchard_rl
    python -m orchard.train --config orchard/configs/reference.yaml

Configuration values can be overridden from the command line with dot notation:

    python -m orchard.train \
      --config orchard/configs/reference.yaml \
      --override train.total_steps=10000 train.lr.start=0.001

Runs are written beneath the configured `logging.output_dir` (by default, `output/runs/`). Each run contains its resolved configuration, CSV metrics, timing data, and model checkpoints.

To resume from a checkpoint:

    python -m orchard.train \
      --config orchard/configs/reference.yaml \
      --resume output/runs/<run>/checkpoints/final.pt

The CLI also accepts `--resume-critic-only` and `--resume-actor-only` for partial warm starts.

## Tests

From `orchard_rl/` with the virtual environment active:

    pytest

The test configuration in `pytest.ini` discovers the suite under `orchard/tests/` and adds the current directory to Python's import path.

## Visualize a rollout

    python -m orchard.viz orchard/configs/reference.yaml --steps 200

Add `--checkpoint <path>` to visualize a trained model. More options are documented in `orchard/viz/README.md` and available through:

    python -m orchard.viz --help

## Repository layout

- `README.md` — project setup and usage
- `orchard_rl/requirements.txt` — pinned Python dependencies
- `orchard_rl/pytest.ini` — test discovery and import configuration
- `orchard_rl/orchard/` — environment, models, trainers, and command-line tools
- `orchard_rl/orchard/configs/` — experiment configuration templates
- `orchard_rl/orchard/tests/` — unit and integration tests
- `orchard_rl/orchard/viz/` — rollout rendering tools
- `orchard_rl/*_eval.py` — evaluation command wrappers

Start with `orchard_rl/orchard/configs/reference.yaml`; it documents every supported setting and the available training modes.
