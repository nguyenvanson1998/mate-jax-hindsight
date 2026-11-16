#!/usr/bin/env bash
# Planner-prior students (3 seeds x 2 target types) saving every validated checkpoint (eval seed 1, 16 episodes);
# then the checkpoint-ensemble size K is chosen on a second validation set (seed 2), and the test (seed 0) is run.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY="${PYTHON:-python3}"
LITE='{"samples_per_axis":1,"horizon":20,"window":8,"stride":4}'
D=results/student_ens
mkdir -p $D
student() {
	local targets=$1 seed=$2
	$PY -u research/train_hindsight.py --target privileged --targets $targets --seed $seed --steps 500 \
		--betas 1,0.7,0.5,0.3,0.2,0.1,0 --max-samples 400000 --eval-envs 16 --eval-steps 1000 --eval-every 2 --eval-seed 1 \
		--planner-json "$LITE" --iterations 30 --envs 8 --epochs 4 --prior \
		--log $D/${targets}_s${seed}.jsonl --save-dir $D/${targets}_s${seed} > $D/${targets}_s${seed}.log 2>&1
}
for job in greedy:0 greedy:1 greedy:2 waypoint:0 waypoint:1; do student ${job%:*} ${job#*:} & done
wait
student waypoint 2
for k in 1 3 5; do $PY -u research/eval_pivot_extra.py --students $D/*/ --seed 2 --num-envs 16 --ensemble $k --student-only --out $D/val2_k$k.json > $D/val2_k$k.log 2>&1 & done
wait
for k in 1 3 5; do $PY -u research/eval_pivot_extra.py --students $D/*/ --seed 0 --num-envs 32 --ensemble $k --student-only --out $D/test_k$k.json > $D/test_k$k.log 2>&1 & done
wait
echo ENS_DONE > $D/done.log
