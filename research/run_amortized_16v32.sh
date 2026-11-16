#!/usr/bin/env bash
# Learned FORESIGHT on MATE-16v32-9 with the budget of the four-camera study (40 DAgger iterations), then the paired
# evaluation. Finished runs are skipped, so the script can be started again after an interruption.
#   PARALLEL=6 bash research/run_amortized_16v32.sh
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION:-0.12}" PY="${PYTHON:-python3}"
OUT=results/amortized_16v32_i40
mkdir -p $OUT results/paper
one() {
	local t=$1 seed=$2 name=16v32_${1}_s${2}
	[ -s "$OUT/$name/params.msgpack" ] && return
	rm -f "$OUT/$name.jsonl"
	$PY -u research/train_hindsight.py --config MATE-16v32-9.yaml --target planner --lookahead 2 --targets "$t" --seed "$seed" \
		--iterations 40 --envs 16 --steps 500 --subsample 8 --batch-size 256 --betas 1,0.7,0.5,0.3,0.2,0.1,0 --max-samples 400000 \
		--epochs 4 --eval-envs 8 --eval-steps 1000 --eval-every 5 --log "$OUT/$name.jsonl" --save-dir "$OUT/$name" > "$OUT/$name.log" 2>&1
	echo "done $name"
}
export -f one
export OUT
for seed in 0 1 2; do for t in greedy waypoint; do echo "$t $seed"; done; done | xargs -P "${PARALLEL:-6}" -I{} bash -c 'one {}'
[ "$(ls $OUT/*/params.msgpack | wc -l)" = 6 ] || { echo TRAINING_INCOMPLETE >> $OUT/done.log; exit 1; }
$PY -u research/eval_amortized.py --runs $OUT/16v32_*_s? --out results/paper/amortized_16v32_i40.json > results/eval_amortized_16v32_i40.log 2>&1
echo AMORTIZED_16V32_DONE >> $OUT/done.log
