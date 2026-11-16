#!/usr/bin/env bash
# Missing results of the scale study: 8v8 variants, network metrics of learned FORESIGHT with four cameras, and
# learned FORESIGHT at larger networks (same labelled camera-states per step as with four cameras).
#   PARALLEL=3 MEM_FRACTION=0.10 SEEDS="0 1 2" bash research/run_amortized_scale.sh
# TRAIN_ONLY=1 skips the evaluations, for a second machine that trains some of the seeds.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION:-0.10}" PY="${PYTHON:-python3}"
PARALLEL="${PARALLEL:-3}"
SEEDS="${SEEDS:-0 1 2}"
OUT=results/amortized_scale
mkdir -p $OUT results/paper
wait_memory() { while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 15 ]; do sleep 60; done; }
export -f wait_memory
[ -n "${TRAIN_ONLY:-}" ] || ( wait_memory; PYTHON=$PY PARTS=scale8 bash research/run_paper_experiments.sh > results/scale8.log 2>&1
  $PY -u research/eval_amortized.py --runs results/amortized/*_s? --out results/paper/amortized_4v8.json > results/eval_amortized_4v8.log 2>&1
  $PY -u research/eval_pivot_extra.py --hitmac results/hitmac_anneal/*/ --out results/paper/hitmac_metrics.json > results/hitmac_metrics.log 2>&1 ) &
one() {
	local config=$1 t=$2 seed=$3 sub=4 batch=512 name=${1}_${2}_s${3}
	[ "$config" = 16v32 ] && { sub=8; batch=256; }
	[ -s "$OUT/$name/params.msgpack" ] && return
	wait_memory
	rm -f "$OUT/$name.jsonl"
	$PY -u research/train_hindsight.py --config "MATE-$config-9.yaml" --target planner --lookahead 2 --targets "$t" --seed "$seed" \
		--iterations 20 --envs 16 --steps 500 --subsample $sub --batch-size $batch --betas 1,0.7,0.5,0.3,0.2,0.1,0 --max-samples 400000 \
		--epochs 4 --eval-envs 8 --eval-steps 1000 --eval-every 5 --log "$OUT/$name.jsonl" --save-dir "$OUT/$name" > "$OUT/$name.log" 2>&1
	echo "done $name"
}
export -f one
export OUT
for seed in ${SEEDS}; do for config in 8v8 8v16 16v32; do for t in greedy waypoint; do echo "$config $t $seed"; done; done; done \
	| xargs -P "$PARALLEL" -I{} bash -c 'one {}'
wait
[ -n "${TRAIN_ONLY:-}" ] && { echo AMORTIZED_TRAIN_DONE >> $OUT/done.log; exit 0; }
for config in 8v8 8v16 16v32; do
	wait_memory
	$PY -u research/eval_amortized.py --runs $OUT/${config}_*_s? --out results/paper/amortized_$config.json > results/eval_amortized_$config.log 2>&1
done
echo AMORTIZED_SCALE_DONE >> $OUT/done.log
