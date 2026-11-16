#!/usr/bin/env bash
# Communication-control MARL over FORESIGHT-T (published intents): MAPPO with the hindsight critic
# (exogenous future target positions), MAPPO without it, and IPPO; 3 seeds x 2 target models.
# Runs on the CPU (the planner compiles in a minute there); PARALLEL runs at a time.
#   PARALLEL=3 bash research/run_comm_rl.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python3}"
OUT="${OUT:-results/comm}"
TOTAL="${TOTAL:-5e5}"
PARALLEL="${PARALLEL:-3}"
SEEDS="${SEEDS:-0 1 2}"
mkdir -p "${OUT}/curves" "${OUT}/policies" "${OUT}/logs"

jobs=()
for targets in greedy waypoint; do
	for algo in mappo-h mappo ippo; do
		for seed in ${SEEDS}; do
			jobs+=("${algo} ${targets} ${seed}")
		done
	done
done

run_one() {
	local algo="$1" targets="$2" seed="$3"
	local name="${algo}_${targets}_s${seed}" critic=centralized extra=""
	[ -s "${OUT}/policies/${name}/params.msgpack" ] && return
	case "${algo}" in
	mappo) extra="--no-comm-hindsight" ;;
	ippo) critic=local; extra="--no-comm-hindsight" ;;
	esac
	JAX_PLATFORMS=cpu "${PYTHON}" -u -m mate_jax.train --team camera --opponent "${targets}" --comm --critic "${critic}" ${extra} \
		--frame-skip 5 --horizon 100 --seed "${seed}" --num-envs 128 --num-steps 10 --total-timesteps "${TOTAL}" \
		--hiddens 128 128 --updates-per-call 6 --eval-interval 0 --no-final-eval \
		--log-file "${OUT}/curves/${name}.jsonl" --save-dir "${OUT}/policies/${name}" > "${OUT}/logs/${name}.log" 2>&1
	echo "done ${name}"
}
export -f run_one
export PYTHON OUT TOTAL
printf '%s\n' "${jobs[@]}" | xargs -P "${PARALLEL}" -I{} bash -c 'run_one {}'

# Rules and learned policies on the same 16 episodes of 200 decisions (1000 steps).
JAX_PLATFORMS=cpu "${PYTHON}" -u research/eval_comm.py --targets greedy waypoint --runs "${OUT}"/policies/* \
	--num-envs 16 --decisions 200 --out "${OUT}/eval.json" > "${OUT}/logs/eval.log" 2>&1
echo "COMM_RL_DONE"
