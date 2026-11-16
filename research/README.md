# Motivation pilots (MATE-JAX)

Heuristic studies used to motivate forecast-driven allocation. MATE-4v8-9, greedy targets,
obstacle transmittance 0, 1000 steps; 64 (pilot 1) or 32 (pilot 2) parallel episodes, mean ± s.e.

| Camera team | Coverage |
| --- | ---: |
| Greedy cameras (MATE `GreedyCameraAgent`, decentralized) | 0.607 ± 0.006 |
| Greedy, aim at constant-velocity extrapolation (lead 5 / 10 / 20 steps) | 0.604 / 0.601 / 0.572 |
| Greedy, aim at the true future position (lead 5 / 10 / 20 / 40) | 0.618 / 0.614 / 0.577 / 0.439 |
| Centralized greedy max-coverage planner, true *current* positions | 0.617 ± 0.007 |
| + score each pose at its reach time t + τ (true future) | 0.659 ± 0.005 |
| + average over [t + τ, t + τ + 10] (true future) | 0.688 ± 0.006 |
| same, constant-velocity forecast from true positions | 0.688 ± 0.006 |
| same, team memory of observed targets only, no extrapolation | 0.287 ± 0.006 |
| same, team memory of observed targets + constant-velocity extrapolation | 0.622 ± 0.008 |

MATE-8v8-9 (same protocol, 32 episodes): planner with current positions 0.671 ± 0.005;
reach + window (true future) 0.781 ± 0.004; team memory + constant velocity 0.710 ± 0.006.

Random Waypoint targets (`PILOT_TARGETS=waypoint`, MATE-4v8-9, 32 episodes): current positions
0.796 ± 0.008; true future (reach + window) 0.824 ± 0.007; constant-velocity forecast from true
positions 0.778 ± 0.011 (worse than no forecast); team memory + constant velocity 0.735 ± 0.011.
Probabilistic version (8 particles, N(cv path, (0.5 · speed · h)²)): full information 0.809 ± 0.011,
team memory only 0.799 ± 0.012 (+0.064 over the point forecast from the same memory).

Take-aways: aiming a single camera ahead barely helps; forecasts pay off when used for
*allocation* with the actuation latency τ of rotation/zoom (+11.5 % over the same planner without
forecasts); with greedy targets a constant-velocity model is enough *when targets are observed*,
and ~0.066 coverage is lost to partial observability. With stochastic (Random Waypoint) motion a
point forecast is harmful, so the forecast has to be probabilistic.

```bash
python3 research/pilot_aim_ahead.py . MATE-4v8-9.yaml
python3 research/pilot_allocation.py . MATE-4v8-9.yaml now,reach,reach+win,cv+win,mem-static+win,mem-cv+win
PILOT_TARGETS=waypoint python3 research/pilot_allocation.py . MATE-4v8-9.yaml now,reach+win,cv+win,mem-cv+win,cvp+win,mem-cvp+win
```

Caveats: heuristic planners, one seed of 32 parallel episodes per row, transmittance 0; the
probabilistic rows are within ~1 s.e. of each other except memory point vs. memory particles.

# FORESIGHT prototype (non-learned baseline)

`mate_jax/foresight/` implements the decision layer of the proposed method without any learning:

| Module | What it does |
| --- | --- |
| `grid.py` | 32 x 32 grid (62.5 m cells). Separable transition matrices of a *cell-integrated* Gaussian (uniform mass within a cell, shifted by the velocity, plus Gaussian noise): sub-cell shifts are exact and `h = 0` is the identity, so there is no numerical diffusion. |
| `belief.py` | One belief map per target (and per camera without communication). Prediction: displacement `v·a + N(0, s(a)²)` with `s(a)² = σ_w²·a + σ_v²·a²` (random walk + velocity uncertainty, `σ_v = speed/√2` for an unknown heading at the known speed), applied recursively with exact variance increments. Update: detections collapse the map; cells inside a field of view without a detection lose `p_d = 0.9` of their mass (negative information). Velocity from consecutive detections. Multi-step forecasts in closed form for all horizons at once. |
| `atlas.py` | Per-episode visibility atlas of every camera over 2 x 2 samples per cell (line of sight through the obstacles seen by the team, maximum sight range), and the free-space mask. |
| `options.py` | Pose options (72 orientations x 6 zoom levels), reach times `τ` under the rotation/zoom rate limits, and sector coverage of every option from bearing histograms + circular prefix sums (`O(cells + options)` per camera instead of `O(cells x options)`). |
| `planner.py` | `ForesightCameraPolicy`: belief update → forecasts for `h ∈ [0, 50]` (stride 2) → cameras choose, one after another, the option maximizing the expected number of covered targets during `[t + τ, t + τ + 10)`; the chosen sector is discounted from the expected counts of the next cameras (probabilistic OR, greedy ≥ 1/2-optimal for this submodular objective) → rate-limited move towards the pose. Uses the joint camera observation only (poses + detections shared in the team, i.e. the replicated filter of the proposal); `communication='none'` gives every camera its own belief and no coordination. |

```python
import mate_jax
from mate_jax.agents import GreedyTargetPolicy
from mate_jax.evaluation import evaluate_coverage
from mate_jax.foresight import ForesightCameraPolicy

env = mate_jax.make('MATE-4v8-9.yaml')
cameras = ForesightCameraPolicy.from_env(env)          # options: horizon, window, communication, ...
metrics = evaluate_coverage(env, cameras, GreedyTargetPolicy.from_env(env), num_envs=32, num_steps=1000)
metrics['coverage_rate'].mean()
```

Additional options of `ForesightCameraPolicy`:

* `objective='path'` (default): an option's value integrates coverage along the rate-limited path
  to it, i.e. cameras keep scanning while they turn (`'window'` / `'cumulative'` are the weaker
  alternatives measured below);
* `refine=True`: per-agent post-processing; each camera tries 9 small corrections of its planned
  rotation/zoom and keeps the one covering most of its detected targets at their exact,
  one-step-predicted positions (redundant targets weigh 0.1, ties keep the plan);
* `communication='hybrid'`, `message_drop=p`: a shared team memory that only receives the packets
  that are not lost, plus one private memory per camera; for every target a camera plans with the
  more recent of the two.

`mate_jax.evaluation.evaluate_coverage` also returns per-camera coverage, unique coverage,
rotation/zoom effort and redundancy (`research/eval_foresight.py --per-agent`).

## Results

Paired evaluation: target trajectories do not depend on the cameras, so the same seeds give paired
episodes and differences are measured per episode. MATE-4v8-9 (obstacle transmittance 0.1), 1000
steps per episode, Δ = paired difference to greedy cameras (mean ± s.e.).

**Objective and ablations** (greedy targets, 32 episodes; the ablations use the `window` objective
that was the default at that time):

| Camera team | Coverage | Δ vs greedy |
| --- | ---: | ---: |
| Greedy cameras | 0.620 | — |
| FORESIGHT, `window` objective | 0.661 | +0.040 ± 0.008 |
| ↳ no forecast | 0.670 | +0.050 ± 0.008 |
| ↳ no reach time | 0.678 | +0.058 ± 0.010 |
| ↳ no negative information | 0.646 | +0.025 ± 0.009 |
| ↳ point forecast (σ = 0) | 0.673 | +0.053 ± 0.008 |
| FORESIGHT, `path` objective, γ = 0.95 | **0.712** | **+0.091 ± 0.007** |
| FORESIGHT, `path` objective, γ = 0.98 | 0.702 | +0.082 ± 0.008 |

**New proposals** (`path` objective, 16 episodes per row, `results/foresight_new_proposals_4v8.json`):

| Camera team | Greedy targets | Δ | Random Waypoint | Δ |
| --- | ---: | ---: | ---: | ---: |
| Greedy cameras | 0.622 | — | 0.718 | — |
| FORESIGHT (team memory) | 0.712 | +0.090 | 0.821 | +0.103 |
| + per-agent refinement | 0.717 | +0.095 | 0.829 | +0.111 |
| no communication (private memories only) | 0.584 | −0.038 | 0.748 | +0.030 |
| team memory, 50 % packet loss | 0.691 | +0.069 | 0.808 | +0.090 |
| hybrid memory, 50 % packet loss | 0.698 | +0.076 | 0.815 | +0.097 |
| team memory, 90 % packet loss | 0.524 | −0.098 | 0.727 | +0.009 |
| hybrid memory, 90 % packet loss | 0.669 | +0.047 | 0.801 | +0.083 |

Direct paired differences (16 episodes):

| Comparison | Greedy targets | Random Waypoint |
| --- | ---: | ---: |
| refinement − FORESIGHT | +0.005 ± 0.006 | +0.008 ± 0.001 (94 % of episodes) |
| hybrid − team memory, 50 % loss | +0.007 ± 0.006 | +0.007 ± 0.003 |
| hybrid − team memory, 90 % loss | +0.145 ± 0.015 (100 %) | +0.074 ± 0.007 (100 %) |
| team memory − no communication | +0.128 ± 0.016 (100 %) | +0.073 ± 0.009 (100 %) |

Per camera (greedy targets): without communication every camera still sees ~22 % of the targets,
but its unique coverage drops from 0.11–0.16 to ~0.08 and redundancy rises from 0.17 to 0.24
(0.31 → 0.43 for Random Waypoint): coordination, not information, is what communication buys.
Sequential greedy favours the first cameras (0.25 → 0.20 coverage from camera 0 to camera 3;
0.35 → 0.24 for Random Waypoint).

Take-aways:

* The reach time only helps when the option value integrates coverage along the turning path
  (`window` +0.040 → `path` +0.091); counting no coverage while turning (`cumulative`) is worse.
* Refinement on exact detections gives a small but consistent gain (significant for Random
  Waypoint targets).
* Hybrid memory degrades gracefully under packet loss, whereas a pure shared memory collapses
  below the greedy baseline at 90 % loss.
* Open issues: forecasts did not beat the current-belief planner under the `window` objective;
  this ablation has to be repeated under `path`. Also to do: camera-order effects (shuffle or
  global greedy), 8v8-9 and more episodes (to be run on the GPU machine).

# FORESIGHT-HRL (learned option + transmission scheduling)

High level per camera every 5 steps: one of the top-8 FORESIGHT options x transmit or not, over a
4-slot slotted-ALOHA channel; reward = coverage - 0.1 tx - 0.05 actuation. MAPPO / IPPO learners,
512 envs, 1e7 env steps, 3 seeds, MATE-4v8-9 (`research/run_learning_curves.sh` on the GPU machine,
logs in `results/curves/`). Rules (`research/eval_hrl_rules.py --env-steps 2500`, greedy option +
fixed schedule, 16 deterministic episodes of 500 decisions) in `results/hrl_rules_4v8_h500.json`.
Learned numbers: stochastic training policy, mean over the last two training episodes.

| Method | Goal-directed CR / tx / reward | RWP CR / tx / reward |
| --- | --- | --- |
| Greedy, always transmit | 0.704 / 1.00 / 0.585 | 0.800 / 1.00 / 0.680 |
| Greedy, event-triggered | 0.704 / 0.70 / 0.616 | 0.804 / 0.68 / 0.716 |
| Greedy, periodic (1/4) | 0.701 / 0.25 / 0.658 | 0.800 / 0.25 / 0.755 |
| Greedy, never transmit | 0.628 / 0 / 0.612 | 0.800 / 0 / 0.783 |
| Flat MAPPO | 0.393 | 0.485 |
| Flat IPPO | 0.353 | 0.589 |
| FORESIGHT-HRL (MAPPO) | 0.690 / 0.27 / 0.637 | 0.768 / 0.05 / 0.739 |
| FORESIGHT-HRL (IPPO) | 0.693 / 0.46 / 0.620 | 0.769 / 0.06 / 0.739 |

Take-aways: +0.18-0.30 coverage over flat MARL; learns to transmit only where it helps (27 % vs 5 %,
-73 % / -95 % radio energy vs always-on); still below the collision-free periodic schedule on reward.
All training environments reset together, so with RWP targets the per-update coverage ripples with
the episode phase; average over whole episodes. Next: deterministic evaluation of saved policies,
learned slot choice / collision-aware reward, lambda_c sweep.
