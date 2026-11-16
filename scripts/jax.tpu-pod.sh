#!/usr/bin/env bash

# Run MATE-JAX MAPPO on a multi-host Cloud TPU pod slice (e.g. v4-32, v5e-64).
# Every host runs the same program; `--multi-host` calls `jax.distributed.initialize()`,
# environments are sharded over all TPU cores of the slice, gradients are all-reduced over ICI.
#
#   TPU_NAME=my-tpu ZONE=us-central2-b bash scripts/jax.tpu-pod.sh --team camera --num-envs 16384
#
# The repository must be present at the same path on every worker, e.g.:
#   gcloud compute tpus tpu-vm scp --recurse --worker=all --zone "${ZONE}" . "${TPU_NAME}":~/mate

set -euo pipefail

: "${TPU_NAME:?Set TPU_NAME to the TPU VM name}"
: "${ZONE:?Set ZONE to the TPU zone}"
REMOTE_DIR="${REMOTE_DIR:-~/mate}"
ARGS="$*"

gcloud compute tpus tpu-vm ssh "${TPU_NAME}" --zone "${ZONE}" --worker=all --command "
	set -e
	cd ${REMOTE_DIR}
	pip install -q 'jax[tpu]' -r requirements-jax.txt
	export PYTHONPATH=\${PWD}
	python3 -m mate_jax.train --multi-host --dtype bfloat16 --prng-impl rbg \
		--compilation-cache \${HOME}/.cache/mate_jax/xla \
		--log-file results/jax.pod.jsonl --save-dir results/jax.pod ${ARGS}
"
