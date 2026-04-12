import os
import tempfile
import csv
from orchard.config import load_config
from orchard.train import train

def _write_config(yaml_str: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(yaml_str)
    return path

BASE_CONFIG = """
env:
  height: 4
  width: 4
  n_agents: 2
  n_tasks: 2
  n_task_types: 2
  gamma: 0.99
  r_picker: 1.0
  r_low: 0.0
  pick_mode: forced
  max_tasks_per_type: 2
  task_assignments: [[0], [1]]
  stochastic:
    spawn_prob: 0.1
    despawn_mode: probability
    despawn_prob: 0.05
model:
  encoder: blind_task_cnn_grid
  mlp_dims: [16]
  conv_specs: [[4, 3]]
train:
  learning_type: decentralized
  use_gpu: false
  td_lambda: 0.3
  total_steps: 5
  seed: 42
  heuristic: nearest_correct_task
  lr:
    start: 0.01
  epsilon:
    start: 0.3
eval:
  eval_steps: 5
  n_test_states: 2
logging:
  main_csv_freq: 5
  detail_csv_freq: 5
  output_dir: {output_dir}
"""

def test_end_to_end_training_loop():
    """Runs the main train() loop for 5 steps to verify no crashes and proper artifact generation."""
    tmpdir = tempfile.mkdtemp()
    yaml_str = BASE_CONFIG.format(output_dir=tmpdir)
    path = _write_config(yaml_str)
    cfg = load_config(path)
    
    # Run the full training loop (CpuTrainer path due to use_gpu: false)
    train(cfg)
    os.unlink(path)

    # Check that output directory was created
    run_dirs = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
    assert len(run_dirs) == 1
    run_dir = os.path.join(tmpdir, run_dirs[0])

    # Check for metadata
    assert os.path.exists(os.path.join(run_dir, "metadata.yaml"))
    
    # Check for metrics.csv and verify fields
    metrics_path = os.path.join(run_dir, "metrics.csv")
    assert os.path.exists(metrics_path)
    with open(metrics_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert "greedy_rps" in rows[0]
        assert "td_loss_avg" in rows[0]

    # Check for details.csv
    details_path = os.path.join(run_dir, "details.csv")
    assert os.path.exists(details_path)
    with open(details_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1

    # Check for checkpoints
    assert os.path.exists(os.path.join(run_dir, "checkpoints", "step_0.pt"))
    assert os.path.exists(os.path.join(run_dir, "checkpoints", "final.pt"))