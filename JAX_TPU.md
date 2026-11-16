# MATE-JAX: MATE on JAX, Flax and TPU

`mate_jax/` is a pure-JAX port of the Multi-Agent Tracking Environment plus a Flax/Optax MAPPO
trainer that runs data-parallel over TPU cores (also GPU/CPU). The original NumPy package `mate/`
is untouched and still works; both share the same YAML configs in `mate/assets/`.

> **Tóm tắt (Tiếng Việt).** `mate_jax` viết lại toàn bộ môi trường MATE bằng JAX thuần: `reset`/`step`
> là hàm thuần với shape tĩnh nên `jax.jit`, `jax.vmap` hàng nghìn môi trường song song và chia
> (shard) lên các lõi TPU được. Kèm theo là các agent heuristic (greedy/random) viết bằng JAX làm đối
> thủ, wrapper single-team (toạ độ tương đối, rescale observation, frame-skip, reward coverage), mô hình
> Flax và trainer MAPPO dùng `shard_map` + `lax.pmean` (data-parallel, chạy được trên TPU pod nhiều host).
> Chạy nhanh: `python3 -m mate_jax.train --team camera`. Trên TPU thêm `--dtype bfloat16 --prng-impl rbg`.

## Layout

| File | Content |
| --- | --- |
| `mate_jax/config.py` | YAML/JSON loader (same files and defaults as `mate.read_config`, no `gym`) |
| `mate_jax/constants.py` | Constants and observation-space bounds (NumPy, baked into XLA) |
| `mate_jax/geometry.py` | Vectorized kernels: target/obstacle collision (`obstruct_step`), exact line-of-sight |
| `mate_jax/env.py` | `MultiAgentTracking` (functional) and the `EnvState` pytree |
| `mate_jax/transforms.py` | `RelativeCoordinates` + `RescaledObservation` in JAX |
| `mate_jax/agents.py` | JAX team policies: `Greedy{Camera,Target}Policy`, `Random*`, `RandomWaypointTargetPolicy`, `Static*` |
| `mate_jax/wrappers.py` | `AutoResetEnv`, `SingleTeamEnv` (`mate.MultiCamera` / `mate.MultiTarget` analogue) |
| `mate_jax/models.py` | Flax actor / centralized critic, Gaussian & categorical heads |
| `mate_jax/sharding.py` | Device mesh, `shard_map` compat, multi-host replication |
| `mate_jax/mappo.py` | `MAPPOTrainer` (rollout + GAE + PPO in one XLA program), checkpoints, evaluation |
| `mate_jax/train.py` | CLI: `python3 -m mate_jax.train` |
| `mate_jax/benchmark.py` | CLI: `python3 -m mate_jax.benchmark` (NumPy vs JAX throughput) |
| `mate_jax/bridge.py` | Convert `mate.MultiAgentTracking` ⇄ `EnvState` (parity tests, rendering) |
| `mate_jax/evaluation.py` | `evaluate_coverage(env, camera_policy, target_policy)` for any pair of JAX team policies |
| `mate_jax/foresight/` | FORESIGHT prototype: belief maps, visibility atlas, pose options, greedy planner (see `research/README.md`) |
| `tests/jax/` | Unit, parity and sharded-training tests |
| `research/` | Motivation pilots for forecast-driven allocation (see `research/README.md`) |
| `scripts/jax.*.sh` | Launchers for a single host / TPU VM and multi-host TPU pod slices |

## Installation

```bash
# CPU / generic (Python >= 3.10)
pip install -r requirements-jax.txt
# Cloud TPU VM
pip install "jax[tpu]" -r requirements-jax.txt
# or as extras of the package: pip install ".[jax]" / ".[tpu]" / ".[cuda]"
```

`mate_jax` does not import `gym`. Only the parity tests and `bridge.py` need the original package, which
requires `gym < 0.26` (it uses the `RandomState` API); `gym==0.23.1` works with Python 3.11.

## Environment API

```python
import jax
import mate_jax

env = mate_jax.make('MATE-4v8-9.yaml')            # same configs and overrides as `mate.make`
(camera_obs, target_obs), state = jax.jit(env.reset)(jax.random.PRNGKey(0))
camera_action, target_action = env.sample_actions(jax.random.PRNGKey(1), state)
timestep, state = jax.jit(env.step)(jax.random.PRNGKey(2), state, camera_action, target_action)
timestep.camera_reward, timestep.target_reward, timestep.done, timestep.info['coverage_rate']
env.state_vector(state)                            # the global state, same layout as `mate`

# 4096 environments in parallel
keys = jax.random.split(jax.random.PRNGKey(0), 4096)
obs, states = jax.vmap(env.reset)(keys)
```

Observations, global states, rewards, `done` and the `info` metrics have exactly the same layout and
semantics as `mate.MultiAgentTracking`. A single-team training interface with an opponent policy:

```python
from mate_jax.agents import GreedyTargetPolicy
from mate_jax.wrappers import SingleTeamEnv

team_env = SingleTeamEnv(env, team='camera', opponent=GreedyTargetPolicy.from_env(env),
                         frame_skip=5, horizon=500, reward='coverage')
obs, global_state, team_state = team_env.reset(key)        # obs: (4, 126), rescaled & relative
obs, global_state, reward, done, info, team_state = team_env.step(key, team_state, actions)
```

## Training (MAPPO)

```bash
# camera team vs. greedy targets (default hyper-parameters follow examples/mappo/camera)
python3 -m mate_jax.train --team camera --config MATE-4v8-9.yaml --num-envs 1024 --total-timesteps 2e7

# target team vs. greedy cameras
python3 -m mate_jax.train --team target --reward raw

# single TPU VM (all local cores), with the recommended TPU flags
python3 -m mate_jax.train --team camera --num-envs 4096 --dtype bfloat16 --prng-impl rbg \
    --compilation-cache ~/.cache/mate_jax/xla --save-dir results/camera

# TPU pod slice (multi-host): same command on every worker
TPU_NAME=my-tpu ZONE=us-central2-b bash scripts/jax.tpu-pod.sh --team camera --num-envs 16384
```

Useful flags: `--opponent {greedy,random,static}`, `--action-type {discrete,continuous}`,
`--discrete-levels 5`, `--frame-skip 5`, `--horizon 500`, `--hiddens 512 256`, `--num-devices N`,
`--updates-per-call` (updates fused into one jitted call), `--log-file metrics.jsonl`.
Python API: `mate_jax.mappo.MAPPOTrainer(MAPPOConfig(...))`; a trained actor can be turned into an
opponent policy with `trainer.policy(params['actor'])` (e.g. for self-play / PSRO-style training).

## Design for TPU

* **Pure functions with static shapes.** No Python-side state, no data-dependent shapes; every
  `while`/`for` of the original became `lax.scan` / `fori_loop` / vectorized math. Entity counts are
  compile-time constants, booleans/ints are `bool`/`int32` (TPU native), floats default to `float32`.
* **Everything in one XLA program.** Rollout (`lax.scan` over time × `vmap` over envs), GAE and all
  PPO epochs/minibatches run inside one jitted call; `--updates-per-call` fuses several iterations,
  the runner state is donated (`donate_argnums`) so HBM buffers are reused.
* **Data parallelism with `shard_map`.** Mesh axis `data` over all (global) devices. Each core owns
  `num_envs / num_devices` environments, shuffles minibatches *locally* (no cross-device all-to-all)
  and gradients/metrics are averaged with `lax.pmean` (an ICI all-reduce). Parameters and optimizer
  states are replicated; multi-host pods use `jax.distributed.initialize()` (`--multi-host`).
* **Cheap resets.** The rejection sampling of entity placement draws `num_reset_candidates` (32)
  candidates per entity at once instead of up to 500 sequential retries, places all targets in
  parallel, and the training wrapper caches one pre-sampled episode start per environment and rollout
  fragment (`SingleTeamEnv(cache_resets=True)`) instead of computing a reset on every step.
* **TPU knobs.** `--dtype bfloat16` (MXU matmuls, float32 params), `--prng-impl rbg` (hardware RNG,
  much faster than threefry on TPU), persistent compilation cache, small fixed loops unrolled.

## Verified results (this machine: Apple M-series CPU, no accelerator)

Tests: `python3 -m pytest tests/jax` → 35 passed (14 environment, 7 parity, 4 trainer, 10 FORESIGHT).

Parity with the NumPy implementation (`tests/jax/test_parity.py`, float64): starting from identical
states, one step of both implementations gives identical target positions (collisions included),
camera poses, observations, global states and rewards; `Obstacle.obstruct` matches on 2000 random
rays; observation transforms match `mate.RelativeCoordinates` + `mate.RescaledObservation` exactly.
Camera visibility agrees on > 99.9 % of camera–target/camera–camera pairs (see differences below).

Environment throughput, `MATE-4v8-9`, random actions, auto-reset every step (`python3 -m mate_jax.benchmark`):

| Implementation | env steps / s | vs NumPy |
| --- | ---: | ---: |
| `mate` (NumPy, 1 env) | 1,255 | 1× |
| `mate_jax`, 1 env | 6,599 | 5× |
| `mate_jax`, 256 envs | 40,345 | 32× |
| `mate_jax`, 4096 envs | 94,657 | 75× |
| `mate_jax`, 4096 envs, 4 virtual CPU devices (`shard_map`) | 186,699 | 158× |

MAPPO, camera team vs. greedy targets (coverage reward, 256 envs, 256-256 MLPs, CPU): about 58k env
steps/s end-to-end (rollout + update). In a 2M agent-step (10M env-step) run the coverage rate went
from 0.28 (random cameras) to 0.43 (stochastic policy) / 0.47 (deterministic evaluation) in 7 minutes.
A 2-process × 2-device `jax.distributed` run (simulated pod) kept parameters identical on all hosts.
TPU numbers were not measured here.

## Differences from the original implementation

* **Exact occlusion.** `mate` approximates each camera's visible boundary by sampling rays and linear
  interpolation (`scipy.interpolate.interp1d`); `mate_jax` uses an exact ray–circle test. They only
  disagree for targets lying almost exactly on an obstacle's silhouette (< 0.1 % of pairs in tests).
* **Random streams.** JAX PRNG keys instead of NumPy `RandomState`; trajectories are not bit-identical
  for the same seed, but every random choice follows the same distribution. The rejection sampling of
  reset positions is truncated at `num_reset_candidates` (default 32) instead of 500 retries.
* **Collision order.** A target step is obstructed by obstacles then cameras in index order (the
  original iterates over a Python `set`); this only matters when one step hits two obstacles at once.
* **Communication.** The message API (`send_messages` / `receive_messages`, arbitrary Python objects)
  has no JAX equivalent. The JAX greedy agents model their messages as instantaneous team-wide sharing.
* **Precision.** `float32` by default (`jax_enable_x64` gives `float64` as in `mate`); `is_colliding`
  uses an absolute tolerance of `1e-3` in `float32`.
* **Rendering.** There is no JAX renderer. Copy a state into the original environment and use its
  pyglet viewer: `mate_jax.bridge.sync_numpy_env(np_env, state); np_env.render()` (the state copy is
  tested; the rendering itself was not exercised here because it needs a display).
* **Truncation.** In `SingleTeamEnv`, reaching the training `horizon` is treated as `done` (no
  bootstrapping), as in common JAX PPO implementations.
