# MATE-JAX, hindsight bounds and FORESIGHT

Code and results of the paper

> **Trace-driven evaluation of camera network controllers with certified hindsight bounds**
> Nguyen Van Son, Phenikaa University.
> Submitted to *Simulation Modelling Practice and Theory* (Elsevier).

## Summary

Controllers of camera sensor networks are usually compared only with each other in simulation, so we do
not know how far they are from the coverage that could be achieved. This repository implements a trace-driven
simulation method that gives such a reference. The targets move independently of the cameras, so their
trajectories are recorded once and replayed for every controller. An offline optimizer then sees the whole
recorded trajectory. Exact dynamic programs over a lattice of rate-limited camera poses give a feasible
clairvoyant schedule (its coverage is the *hindsight bound*, a lower bound on the clairvoyant optimum), and
a Lagrangian relaxation gives an upper bound on the clairvoyant optimum. Oracle planners show whether a
planner lacks information or a better decision structure. The simulator is MATE-JAX, a GPU
re-implementation of the MATE benchmark, checked against the original by replaying recorded actions. As a
case study the repository contains FORESIGHT, a planner in which every camera plans its pose trajectory by
a Viterbi recursion, shares its plan with a short message and best-responds to the plans of the others.

## What is in this repository

| Path | Content |
| --- | --- |
| `mate/` | The original MATE environment (NumPy), from <https://github.com/XuehaiPan/mate>. Three configurations were added: `mate/assets/MATE-8v16-9.yaml`, `MATE-16v16-9.yaml`, `MATE-16v32-9.yaml`. |
| `mate_jax/` | MATE-JAX: the environment in pure JAX (`env.py`, `agents.py`, `wrappers.py`, ...), a Flax MAPPO trainer (`mappo.py`, `train.py`) and other MARL baselines (`value_marl.py`, `value_marl_ext.py`, `hitmac.py`). See [`JAX_TPU.md`](JAX_TPU.md). |
| `mate_jax/foresight/` | The FORESIGHT planner: belief maps (`belief.py`, `grid.py`), visibility atlas (`atlas.py`), pose options (`options.py`), planner and trajectory coordination (`planner.py`), channel model (`channel.py`), hierarchical RL variant (`hrl.py`). |
| `mate_jax/hindsight/` | Learned forecaster (`forecaster.py`), privileged / teacher / student models for learning from hindsight (`privileged.py`, `teacher.py`, `student.py`), environments with communication (`comm_env.py`, `env.py`). |
| `research/` | Experiment scripts. The main ones are `eval_foresight.py` (controllers on paired episodes), `clairvoyant.py` (hindsight bound and Lagrangian bound), `exact_pair.py` (exact optimum for two cameras), `headroom.py` (oracle planners), `bench_foresight.py` (latency), `parity_multistep.py` (MATE-JAX vs. MATE), `train_forecaster.py` (learned forecaster). `run_paper_experiments.sh` runs the experiments of the paper. See also [`research/README.md`](research/README.md). |
| `results/paper/` | The JSON result files used in the paper. |
| `results/` (other) | Smaller result files read by the table scripts (learning curves `*.jsonl`, policy evaluations, communication study) and the two trained forecasters (`results/forecaster_{greedy,waypoint}/forecaster.msgpack`, used by the `traj-2-forecast` variant). `results/simpat_chain/*.sh` are the scripts of the last group of experiments. |
| `paper/` | The scripts that write the tables and figures of the paper from `results/`, and `regenerate.sh` to run them. |
| `tests/jax/` | Unit tests, parity tests with MATE, trainer and FORESIGHT tests. |
| `scripts/` | Upstream MATE training scripts (RLlib) and `jax.*.sh` launchers for MATE-JAX. |
| `docs/` | Upstream MATE documentation. The upstream README is in [`docs/MATE_README.md`](docs/MATE_README.md). |

Trained checkpoints of the learned baselines and the training logs are not included (see
[Not included](#not-included)).

## Installation

The experiments of the paper were run with Python 3.12 on an NVIDIA GB10 GPU, with JAX 0.12.0.1 and
Flax 0.12.10.

```bash
git clone <this repository> mate-jax-hindsight
cd mate-jax-hindsight
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -U "jax[cuda12]"          # JAX with CUDA on an NVIDIA GPU; pick the CUDA extra that matches your driver (JAX install guide)
pip install -r requirements-jax.txt   # numpy, pyyaml, jax, flax, optax
```

`mate_jax` does not need `gym`. Only `mate_jax/bridge.py`, `mate_jax/benchmark.py`,
`research/parity_multistep.py` and `tests/jax/test_parity.py` use the original `mate` package, which needs
`gym < 0.26` (install it with `pip install -e .` and a matching `gym` version, see
[`JAX_TPU.md`](JAX_TPU.md)). The table scripts in `paper/` need `numpy`, `scipy` and `matplotlib`.

Check the installation:

```bash
python3 -c "import mate_jax; print(mate_jax.make('MATE-4v8-9.yaml'))"
python3 -m pytest tests/jax
```

## Running the experiments

Run all commands from the repository root. Every script uses the same episode keys, so the coverage of all
controllers, oracle planners and bounds is measured on the same (paired) target trajectories: 32 episodes
of 1000 steps, seed 0, unless stated otherwise.

### All experiments of the paper

[`research/run_paper_experiments.sh`](research/run_paper_experiments.sh) contains the commands. Choose the
parts with `PARTS`; finished outputs are skipped.

```bash
PARTS="main loss scale headroom bound learned bench" bash research/run_paper_experiments.sh
PARTS="published tradeoff burst delay scale8 scale2 benchscale" bash research/run_paper_experiments.sh
```

Other variables of the script: `PYTHON` (default `python3`), `OUT` (default `results/paper`), `ENVS`
(default 32), `STEPS` (default 1000). The last group of experiments (latency on an idle machine, extra
seeds, exact two-camera optimum, opaque obstacles, oracle with blurred forecasts, sensitivity to target
speed, model noise and grid size) is in two scripts:

```bash
bash results/simpat_chain/chain1.sh    # latency first, then coverage jobs
bash results/simpat_chain/chain2.sh    # waits until the latency part of chain1.sh is done
```

On the GPU the scripts set, for example, `XLA_FLAGS=--xla_gpu_autotune_level=0` and
`XLA_PYTHON_CLIENT_MEM_FRACTION` so that several jobs share one GPU.

### Single experiments

Controllers on paired episodes (option-level planning `foresight`, rule-based cameras `greedy`, trajectory
coordination with one, two or three rounds `traj-1` / `traj-2` / `traj-3`, without communication
`traj-2-none`, with the learned forecaster `traj-2-forecast`, published intents `pub-period-4`, ...; the
list is `VARIANTS` in `research/eval_foresight.py`):

```bash
python3 research/eval_foresight.py --configs MATE-4v8-9.yaml --targets greedy waypoint \
    --num-envs 32 --num-steps 1000 --seeds 0 \
    --variants foresight greedy hybrid traj-1 traj-2 traj-3 traj-2-hybrid traj-2-none traj-2-forecast \
    --reference foresight --out results/paper/main_4v8.json
```

More variants can be given as JSON with `--variant-json` (keyword arguments of `ForesightCameraPolicy`),
and the environment can be changed with `--env-json`, e.g. `'{"target": {"step_size": 30.0}}'`.

Hindsight bound (block-coordinate ascent of exact per-camera dynamic programs) and Lagrangian upper bound:

```bash
python3 research/clairvoyant.py --config MATE-4v8-9.yaml --targets greedy waypoint \
    --num-envs 32 --num-steps 1000 --key-steps 1000 --rounds 3 --dual-iterations 150 \
    --out results/paper/bound_dual_4v8.json
```

With `--dual-iterations 0` (the default) only the hindsight bound is computed. Exact optimum of two cameras,
to check that hindsight bound <= exact optimum <= Lagrangian bound:

```bash
python3 research/exact_pair.py --config MATE-4v8-9.yaml --targets greedy waypoint --pairs 0,1 2,3 \
    --num-envs 32 --num-steps 1000 --key-steps 1000 --dual-iterations 150 --out results/paper/exact_pair_4v8.json
```

Oracle planners (true current or future target positions instead of the belief maps):

```bash
python3 research/headroom.py --config MATE-4v8-9.yaml --targets greedy waypoint \
    --modes foresight oracle-future --num-envs 32 --num-steps 1000 --key-steps 1000 \
    --planner-json '{"objective": "trajectory", "coordination_rounds": 2, "levels": [30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180]}' \
    --out results/paper/headroom_traj2_4v8.json
```

Planning latency (batch 1, 7 timed runs per variant):

```bash
python3 research/bench_foresight.py --config MATE-4v8-9.yaml --batch-sizes 1 --num-steps 200 --repeats 7 \
    --variants greedy foresight traj-2 pub-4 pub-4-hybrid --out results/paper/bench_idle_4v8.json
```

Multi-step parity of MATE-JAX with the original MATE (needs `mate` and `gym < 0.26`; the options below
match the stored `results/paper/parity_multistep.json`):

```bash
python3 research/parity_multistep.py --configs MATE-4v8-9.yaml MATE-8v8-9.yaml --episodes 32 \
    --factorial-episodes 32 --out results/paper/parity_multistep.json
```

Learned forecaster (writes `forecaster.msgpack` to `--save-dir`; `eval_foresight.py` reads it from
`results/forecaster_{targets}/forecaster.msgpack`). The two trained forecasters of the paper are included,
so this step is only needed to train new ones:

```bash
python3 research/train_forecaster.py --targets greedy --save-dir results/forecaster_greedy --out results/forecaster_greedy.json
python3 research/train_forecaster.py --targets waypoint --save-dir results/forecaster_waypoint --out results/forecaster_waypoint.json
```

### Additional checks

These scripts produce the continuous-pose bounds, the exact optima of small networks, Supplementary Section S11 and the other checks of the paper.

```bash
# Upper bounds for continuous poses (Lagrangian bound over cells of poses), opaque obstacles; --refine 2 halves the cells
python3 research/clairvoyant.py --config MATE-4v8-9.yaml --targets greedy waypoint --num-envs 32 --num-steps 1000 \
    --key-steps 1000 --rounds 3 --dual-iterations 150 --cells --env-json '{"obstacle": {"transmittance": 0.0}}' \
    --out results/paper/bound_cells_4v8.json
# Exact optimum of pairs over cells (splits the width of the bracket)
python3 research/exact_pair.py --config MATE-4v8-9.yaml --targets greedy waypoint --pairs 0,1 2,3 --num-envs 32 \
    --num-steps 1000 --key-steps 1000 --dual-iterations 150 --cells --env-json '{"obstacle": {"transmittance": 0.0}}' \
    --out results/paper/exact_pair_cells_4v8.json

# Exact optimum for three cameras on a coarser lattice (20 and 10 degrees per step), against both bounds
python3 research/exact_triple.py --config MATE-4v8-9.yaml --targets greedy waypoint --triples 0,1,2 1,2,3 \
    --num-envs 32 --num-steps 1000 --key-steps 1000 --dual-iterations 150 --self-test --out results/paper/exact_triple_4v8.json

# Blurred oracle for option-level planning (default planner of headroom.py)
python3 research/headroom.py --config MATE-4v8-9.yaml --targets greedy waypoint --modes foresight oracle-future \
    --blur-sigmas 0.5 1 --num-envs 32 --num-steps 1000 --key-steps 1000 --out results/paper/headroom_blur_option_4v8.json

# Replay of uniformly random actions in MATE and MATE-JAX (needs gym and the original MATE)
python3 research/parity_multistep.py --configs MATE-4v8-9.yaml MATE-8v8-9.yaml --transmittances 0.0 --episodes 32 \
    --num-steps 1000 --horizons 1 10 100 1000 --precisions float64 float32 --closed-loop-episodes 0 --actions random \
    --out results/paper/parity_random.json

# Penalized upper bounds (two penalty forms) and the predictability of the targets given the full state
python3 research/penalty_bound.py --penalty lags --targets greedy waypoint --num-envs 4 --num-steps 300 --key-steps 1000 \
    --samples 16 --chunk 50 --dual-iterations 100 --tune-envs 4 --tune-samples 16 --tune-iterations 60 \
    --lambdas 0.05 0.1 0.25 0.5 --profiles 0 0,1,3,7 0,1,3,7,15,31 --out results/paper/penalty_lags_4v8.json
python3 research/penalty_bound.py --penalty path --targets greedy waypoint --num-envs 4 --num-steps 300 --key-steps 1000 \
    --path-horizon 20 --children 8 --samples 8 --chunk 25 --dual-iterations 100 --tune-envs 4 --tune-samples 8 \
    --tune-iterations 60 --path-grid 0,0.05 0,0.1 0,0.25 0,0.5 0.1,0.1 --out results/paper/penalty_path_4v8.json
python3 research/predictability.py --targets greedy waypoint --num-envs 8 --out results/paper/predictability_4v8.json
```

### Learned baselines

The multi-agent reinforcement learning baselines are trained and evaluated by these scripts (see the
comments at the top of each script for the options):

* `research/run_learning_curves.sh`: MAPPO, IPPO and FORESIGHT-HRL learning curves (`mate_jax.train`).
* `research/run_value_baselines.sh`: QMIX, QPLEX, RiskQ, QFree and MACPF (`research/train_value.py`).
* `research/eval_learned.py`: evaluation of saved learned controllers on the paired episodes.
* `research/run_hitmac_anneal.sh`: HiT-MAC (`research/train_hitmac.py`), evaluated by `research/eval_pivot_extra.py`.
* `research/run_amortized_scale.sh`, `research/run_amortized_16v32.sh`: a learned (amortized) planner
  trained from the planner by DAgger (`research/train_hindsight.py`), evaluated by `research/eval_amortized.py`.
* `research/eval_init.py`: coverage of the initial policies (step 0 of the learning curves).

## Regenerating the tables

```bash
pip install numpy scipy matplotlib
bash paper/regenerate.sh
```

The script runs `paper/make_figures.py`, `paper/make_figures_v2.py` and `paper/make_simpat_additions.py`
(unchanged copies of the scripts used for the manuscript) on `results/` and writes the LaTeX tables and
macros to `paper/generated/` and the figures to `paper/figures/`. The numbers are the same as in the paper;
only some bold / underline marks in the tables were set by hand in the manuscript. The scripts expect the
folder layout of the author's working copy, so `regenerate.sh` runs them in a temporary folder that links
to `results/`.

## Not included

* Checkpoints of the learned baselines and of the learned planner (about 255 MB of `*.msgpack` files),
  training logs (`*.log`), status files, and result folders of early screening runs that the paper does not
  use. They can be recreated with the scripts above.
* Launcher scripts that were specific to the machines used for the experiments.

## Citation

Citation will be added after publication.

The MATE benchmark itself:

```bibtex
@inproceedings{pan2022mate,
  title     = {{MATE}: Benchmarking Multi-Agent Reinforcement Learning in Distributed Target Coverage Control},
  author    = {Xuehai Pan and Mickel Liu and Fangwei Zhong and Yaodong Yang and Song-Chun Zhu and Yizhou Wang},
  booktitle = {Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year      = {2022},
  url       = {https://openreview.net/forum?id=SyoUVEyzJbE}
}
```

## Licence

MIT licence, see [`LICENSE`](LICENSE). The MATE environment (`mate/`, `examples/`, `docs/`, `scripts/`
except `scripts/jax.*.sh`) is Copyright (c) 2022 Xuehai Pan; the additions (`mate_jax/`, `research/`,
`tests/jax/`, `paper/`, `results/`) are Copyright (c) 2026 Nguyen Van Son.

## Contact

Nguyen Van Son, Phenikaa University: son.nguyenvan@phenikaa-uni.edu.vn
