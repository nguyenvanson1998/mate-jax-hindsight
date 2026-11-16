#!/usr/bin/env bash
# Speed benchmarks of MATE-JAX on the local accelerator (writes results/benchmarks.log, unbuffered).
#   bash research/run_benchmarks.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python3}"
mkdir -p results
{
	echo "=== environment throughput (random actions, auto-reset every step)"
	"${PYTHON}" -u -m mate_jax.benchmark --batch-sizes 1 256 4096 16384 65536 --num-steps 200 --numpy-steps 10
	echo "=== camera team policies in closed loop (MATE-4v8-9, greedy targets)"
	"${PYTHON}" -u research/bench_foresight.py --batch-sizes 16 64 256 1024 --num-steps 200 --variants greedy foresight hybrid
	echo "=== MAPPO training (camera team, 2048 envs, 256-256 MLP)"
	"${PYTHON}" -u -m mate_jax.train --team camera --num-envs 2048 --num-steps 25 --total-timesteps 1.024e6 --hiddens 256 256 --updates-per-call 10
	echo "BENCHMARKS_DONE"
} 2>&1 | tee results/benchmarks.log
