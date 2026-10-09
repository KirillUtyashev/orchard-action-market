# Learned actor with a swappable critic

A shared, learned policy network (actor) for the orchard environment, trained against
either a **decentralized** critic (one value network `V_i` per agent) or a **centralized**
critic (one network `V_cen`). The actor code is identical in both cases; the critic is
chosen by one config field.

Everything here is **opt-in**. With `train.algorithm.learned_actor.enabled: false`
(the default) the repository behaves exactly as before.

## 1. What it does

Each turn one agent acts: it moves (up / down / left / right / stay), and if it lands on a
cell with a task it eligibly can pick, it makes a second decision (pick a task type, or
stay). Both decisions are made by the same actor.

**Actor**
- One network shared by all agents (parameter sharing). The encoder input already contains
  the acting agent's one-hot id and the pick-phase flag, so the network knows who acts and
  in which phase.
- One masked head over `5 + T` actions (5 moves, `T` pick actions). Illegal actions get
  probability exactly 0 (pick is illegal with no eligible task on the cell). A single head
  was chosen because the existing `PolicyNetwork` already has one and action indices are
  shared between phases.
- Execution is independent: at run time the actor uses only the observation.
- Training samples from the masked softmax (no epsilon). Evaluation reports greedy
  (argmax) and sampled performance.

**Critic** (`critic.py`, interface `TeamCritic.team_values(state, afterstates) -> (B,)`)
- Decentralized: `sum_i V_i(afterstate)`. Network `i` is trained with TD(lambda) on the
  individual reward `r_i`.
- Centralized: `V_cen(afterstate)`, trained on the team reward `sum_i r_i`.
- `sum_i r_i` is the team reward, so both critics estimate the same objective
  (`sum_i V_i = V_cen = V_team`).
- Values used in the actor loss are computed under `no_grad`. The actor loss never
  produces a gradient in any critic parameter.

**Example (decentralized).** Agent A picks a task and agent B receives the reward.
`env.resolve_pick` returns the per-agent reward vector with `r_B != 0`. Network `V_B` is
updated with `r_B`. When A later chooses an action, its value estimate is
`r_team + discount * sum_i V_i(afterstate)`, which includes `V_B`, so B's benefit shapes
A's policy. Note that the trainer sees all rewards at once, i.e. this implementation
assumes every agent's reward is shared with everyone. Communication-limited sharing is the
separate `following_rates` extension and is **not** part of this path.

**Advantage** (`train.algorithm.learned_actor.advantage`)
- `afterstate` (default): `Q(s,a) = r_team(a) + discount * V_team(afterstate_a)` for every
  legal action; baseline `b(s) = sum_a pi(a|s) Q(s,a)`; `A(s,a) = Q(s,a) - b(s)`.
  The afterstate value does not contain the transition reward, so `r` is added exactly once.
- `td_error`: `A = r_team + discount * V_team(afterstate_sampled) - V_team(previous afterstate)`
  for the sampled action only (the critic's own TD error). Skipped on the first decision.

**Update**
- Experience is collected for one round (all `N` agents act once); the actor takes one SGD
  step at the end of the round, so every sample is on-policy for the current weights.
- Loss: `-mean(log pi(a|s) * A) - entropy_coef * mean(entropy)`.
- Advantages are divided by their per-batch RMS (`normalize_advantage: true`) so that the
  actor's effective learning rate does not depend on the reward scale (sigma_b up to 2048).
  RMS (not a z-score) keeps sign and mean and is defined for a batch of one.
- Actor and critic have separate learning rates (`train.actor_lr` and `train.lr`).

**Discounting** is round-based and reuses the value path's rule: with
`train.discount_method: round_steps`, gamma is applied once per round (to the first
agent's move decision); every other move decision uses 1.0, and pick decisions always use
1.0. A centralized critic requires `round_steps` (validated in `config.py`).

## 2. Quick start

Use the repo's Python environment (see `orchard_rl/requirements.txt`). Run from `orchard_rl/`.
Training is CPU-only; `train.use_gpu` is ignored for this path.

```bash
# Smoke run, decentralized critic
python -m orchard.train --config orchard/configs/reference.yaml --override \
  train.total_steps=10000 train.discount_method=round_steps \
  train.algorithm.learned_actor.enabled=true train.learning_type=decentralized

# Same, centralized critic (only the critic changes)
python -m orchard.train --config orchard/configs/reference.yaml --override \
  train.total_steps=10000 train.discount_method=round_steps \
  train.algorithm.learned_actor.enabled=true train.learning_type=centralized

# Sanity point (11 agents, 9x9, radii 3, sigma_a=1, sigma_b=8), 5 seeds, both critics
./run_learned_actor_sanity.sh both 3000000     # or: dec | cen
python summarize_learned_actor.py output/learned_actor_sanity
```

`summarize_learned_actor.py` reports, per critic type, the mean and sd across seeds of
"mean of the final 10 evaluations" of team reward per step (greedy, sampled, nearest).

**Debug aid: train the actor against a known-good critic.** Train a value-based critic
(`train.algorithm.name=value`, same env/model, `train.discount_method=round_steps`, same
`learning_type`), then:

```bash
python -m orchard.train --config orchard/configs/learned_actor_sanity.yaml \
  --resume-critic-only path/to/step_N.pt --override train.freeze_critic=true
```

**Check the Q computation.** Evaluates `argmax_a Q(s,a)` with a critic checkpoint. It
should reach roughly the value-based greedy result; if it does not, the problem is the
critic, the discounting or the masking rather than the actor update.

```bash
python check_q_greedy.py --config orchard/configs/reference.yaml --critic path/to/step_N.pt
```

All outputs go to `output/` (git-ignored). Do not commit or push anything under `output/`.

## 3. Configuration

```yaml
train:
  learning_type: decentralized      # decentralized | centralized  (selects the critic)
  discount_method: round_steps      # required for centralized
  lr: {start: 1e-3, ...}            # critic learning rate
  actor_lr: {start: 3e-3, ...}      # actor learning rate (separate)
  freeze_critic: false              # true = train only the actor
  warmup_steps: 0
  algorithm:
    name: actor_critic
    learned_actor:
      enabled: true                 # default false
      advantage: afterstate         # afterstate | td_error
      entropy_coef: 0.01
      normalize_advantage: true     # false = raw advantages
      eval_sampled: true            # also report sampled-policy performance
actor_model: null                   # optional separate actor architecture; defaults to `model`
```

Not supported together with `learned_actor` (config validation rejects them):
`following_rates`, `influencer`, `comm_only_teammates`, `train_only_teammates`.

`metrics.csv` gains `sampled_rps` and `sampled_team_rps`, plus the usual `actor_lr`,
`actor_loss_mean`, `advantage_mean`, `policy_entropy_mean`.

## 4. Code map

| File | Role |
|---|---|
| `learned_actor/actor.py` | `SharedActor`: masked policy + batched update with entropy bonus |
| `learned_actor/critic.py` | `TeamCritic`, `DecentralizedTeamCritic`, `CentralizedTeamCritic` |
| `trainer/learned_actor.py` | `LearnedActorTrainer`: step, advantages, critic TD, evaluation, checkpoints |
| `trainer/__init__.py` | `_create_learned_actor_trainer` (builds the actor before the critics) |
| `config.py`, `datatypes.py` | `LearnedActorConfig` parsing and validation |
| `tests/test_learned_actor.py` | tests, see below |
| `configs/learned_actor_sanity.yaml` | 11-agent sanity configuration |

The actor is created before the critics so that its initial weights are identical for the
decentralized and centralized runs with the same seed.

## 5. Tests

`pytest` from `orchard_rl/` (all existing tests still pass). `test_learned_actor.py` covers:
1. Reward consistency: the decentralized TD targets sum to the centralized target
   (`sum_i r_i`), for both `independent` and `circulant_all_to_all` rewards.
2. Masking: no pick without a task; illegal actions have probability 0 and are never sampled.
3. Gradient isolation: after an actor update no critic parameter has a gradient; attached
   advantages are rejected.
4. Critic switch: dec vs cen changes only the critic; actor class and initial parameters are identical.
5. The afterstate advantage equals a manual `Q - baseline` and counts the reward once;
   all (dec/cen) x (afterstate/td_error) combinations train; round discounting.

## 6. Status and known issues

**Implementation: complete and tested. Sanity comparison: not reached yet.**

On a 4-agent setup (reference.yaml; the nearest-task heuristic gets about 0.43 team reward
per step) the learned actor did not get close to the heuristic within 200k steps:

| Run | Team reward / step |
|---|---|
| Value-based critic, 400k steps (greedy) | about 0.42 |
| `argmax_a Q(s,a)` with that critic (`check_q_greedy.py`) | about 0.40 |
| Learned actor, critic trained jointly, 200k steps | greedy about 0.04, sampled about 0.065 |
| Learned actor, frozen good critic, 200k steps | best about 0.08 to 0.10 |

Findings: the Q computation, advantage sign, discounting and masking are consistent (the
Q-argmax matches the value path). The actor learns the pick decision (agreement with the
Q-argmax is 1.0) but not navigation (agreement about 0.28, close to chance). Several
variants did not help: no normalization, actor lr from 1e-3 to 1e-1 (high lr collapses the
entropy to a poor deterministic policy), a larger actor network, a running-RMS
normalization, and per-state normalization. These experimental variants are not in the
code. The 4-agent runs may simply be too short (the value-based critic alone needed about
300k steps); the 11-agent, 5-seed sanity runs have **not** been done. Round-based
discounting makes the Q gap between moves tiny far from tasks, which is one suspected cause
of slow navigation learning.

Other things to know:
- **Critic divergence at large reward scale.** With the reference critic lr (9e-3), the
  sanity configuration (11 agents, sigma_b = 8) produced NaN within about 100 to 250 steps
  on some seeds, and the value-based path does too. `learned_actor_sanity.yaml` therefore
  uses a critic lr of 3e-4 (1e-3 still diverged on 3 of 10 runs after 4k to 55k steps; 3e-4 and 1e-4 passed those points on the same seeds). The trainer
  raises a clear error if the critic loss becomes non-finite.
- CPU only; the per-step cost grows with the number of agents (decentralized critics).
- "Model width 8" in the task description is read as `conv_specs [[8,3]]` and
  `mlp_dims [8]`; please confirm.
- Spec items still open: advantage form (default `afterstate`), reporting sampled
  performance (done, on by default), and the deliverable level (implementation + sanity).
- The `td_error` advantage uses the previous afterstate as `V_team(s)`, because the
  critic is only defined on afterstates.

## 7. Repository hygiene

Work lives on the local branch `learned-actor`; nothing has been pushed. `orchard_rl/output/`
is git-ignored. Do not add papers or PDFs to the repo, and do not change `requirements.txt`
without asking.
