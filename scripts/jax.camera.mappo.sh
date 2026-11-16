#!/usr/bin/env bash

# MAPPO (JAX + Flax) for the camera team vs. greedy targets on all local accelerators
# (a single TPU VM, e.g. v4-8 / v5e-8, a GPU box, or CPU).
#
#   bash scripts/jax.camera.mappo.sh [extra arguments for `python3 -m mate_jax.train`]

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

EXTRA_ARGS=()
if python3 -c 'import jax, sys; sys.exit(jax.default_backend() != "tpu")' 2>/dev/null; then
	EXTRA_ARGS+=(--dtype bfloat16 --prng-impl rbg)
fi

python3 -m mate_jax.train \
	--team camera --opponent greedy --reward coverage \
	--config MATE-4v8-9.yaml --frame-skip 5 --horizon 500 \
	--num-envs 2048 --num-steps 25 --num-minibatches 8 --update-epochs 4 \
	--total-timesteps 1E8 --lr 5E-4 --ent-coef 0.01 \
	--compilation-cache "${HOME}/.cache/mate_jax/xla" \
	--log-file "results/jax.camera.mappo.jsonl" --save-dir "results/jax.camera.mappo" \
	"${EXTRA_ARGS[@]}" "$@"
