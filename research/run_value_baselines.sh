#!/usr/bin/env bash
# Value-factorization baselines (QMIX, QPLEX, RiskQ, QFree, MACPF) on the camera team, 3 seeds x 2 target
# models, PARALLEL runs at a time; afterwards all learned controllers are evaluated on the paired episodes.
#   PARALLEL=3 bash research/run_value_baselines.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python3}"
TOTAL="${TOTAL:-5000000}"
PARALLEL="${PARALLEL:-3}"
ALGOS="${ALGOS:-qmix qplex riskq qfree macpf}"
TARGETS="${TARGETS:-greedy waypoint}"
CURVES="${CURVES:-results/curves_value}"
POLICIES="${POLICIES:-results/policies_value}"
LOGS="${LOGS:-results/logs_value}"
mkdir -p "${CURVES}" "${POLICIES}" "${LOGS}"

jobs=()
for algo in ${ALGOS}; do for targets in ${TARGETS}; do for seed in 0 1 2; do jobs+=("${algo} ${targets} ${seed}"); done; done; done
run_one() {
	local algo="$1" targets="$2" seed="$3" name="$1_$2_s$3"
	[ -s "${POLICIES}/${name}/params.msgpack" ] && return
	rm -f "${CURVES}/${name}.jsonl"
	"${PYTHON}" -u research/train_value.py --algo "${algo}" --opponent "${targets}" --seed "${seed}" \
		--total-timesteps "${TOTAL}" --num-envs 128 --steps-per-call 500 --eval-interval 10 \
		--log "${CURVES}/${name}.jsonl" --save-dir "${POLICIES}/${name}" > "${LOGS}/${name}.log" 2>&1
	echo "done ${name}"
}
export -f run_one
export PYTHON TOTAL CURVES POLICIES LOGS
printf '%s\n' "${jobs[@]}" | xargs -P "${PARALLEL}" -I{} bash -c 'run_one {}'
echo "VALUE_DONE"
