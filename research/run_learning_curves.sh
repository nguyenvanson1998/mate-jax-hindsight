#!/usr/bin/env bash
# Learning curves of MARL baselines on MATE-4v8-9 (camera team vs. greedy or Random Waypoint targets).
# ALGOS: mappo, ippo (baselines); mappofs, ippofs (with FORESIGHT advice); hrl (FORESIGHT-HRL:
# learned option + transmission, MAPPO critic), hrlippo (IPPO critic), hrlnotx (options only).
# Each run: 2M agent steps = 10M environment steps (frame skip 5), deterministic evaluation on
# 16 episodes x 1000 environment steps every EVAL_INTERVAL updates. Logs: ${CURVES_DIR}/*.jsonl
# (default results/curves); with SAVE_DIR set, the final parameters of every run go to SAVE_DIR/<run>.
#   SEEDS="0 1 2" bash research/run_learning_curves.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python3}"
SEEDS="${SEEDS:-0 1 2}"
TARGETS="${TARGETS:-greedy waypoint}"
ALGOS="${ALGOS:-mappo ippo}"
NUM_ENVS="${NUM_ENVS:-512}"
TOTAL="${TOTAL:-2e6}"
EVAL_INTERVAL="${EVAL_INTERVAL:-8}"
EXTRA="${EXTRA:-}"
CURVES_DIR="${CURVES_DIR:-results/curves}"
SAVE_DIR="${SAVE_DIR:-}"
mkdir -p "${CURVES_DIR}"
for targets in ${TARGETS}; do
	for algo in ${ALGOS}; do
		critic=centralized
		advisor=none
		hrl=""
		case "${algo}" in ippo* | hrlippo*) critic=local ;; esac
		case "${algo}" in *fs) advisor=foresight ;; esac
		case "${algo}" in hrl*) hrl="--hierarchical" ;; esac
		case "${algo}" in hrlnotx) hrl="--hierarchical --no-learn-transmission" ;; esac
		for seed in ${SEEDS}; do
			run="${algo}_${targets}_s${seed}"
			log="${CURVES_DIR}/${run}.jsonl"
			save=""
			[ -n "${SAVE_DIR}" ] && save="--save-dir ${SAVE_DIR}/${run}"
			# Skip finished runs (and, with SAVE_DIR, only those whose parameters were saved).
			[ -n "${SAVE_DIR}" ] && [ ! -s "${SAVE_DIR}/${run}/params.msgpack" ] && rm -f "${log}"
			[ -s "${log}" ] && tail -1 "${log}" | grep -q "\"update\": $(${PYTHON} -c "print(int(float('${TOTAL}')) // (${NUM_ENVS} * 25))")," && continue
			rm -f "${log}"
			echo "=== ${algo} vs ${targets} targets, seed ${seed}"
			"${PYTHON}" -u -m mate_jax.train --team camera --opponent "${targets}" --critic "${critic}" --advisor "${advisor}" ${hrl} \
				--compilation-cache "${HOME}/.cache/mate_jax/xla" \
				--seed "${seed}" --num-envs "${NUM_ENVS}" --num-steps 25 --total-timesteps "${TOTAL}" \
				--hiddens 256 256 --updates-per-call 4 --eval-interval "${EVAL_INTERVAL}" \
				--eval-envs 16 --eval-env-steps 1000 --log-file "${log}" ${save} ${EXTRA}
		done
	done
done
echo "CURVES_DONE"
