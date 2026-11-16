#!/usr/bin/env bash
# Student screening on validation episodes disjoint from the test episodes (eval seed 1), best checkpoint kept.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY="${PYTHON:-python3}"
LITE='{"samples_per_axis":1,"horizon":20,"window":8,"stride":4}'
mkdir -p results/student_screen2
run() {
	local name=$1; shift
	$PY -u research/train_hindsight.py --target privileged --targets greedy --seed 0 --steps 500 \
		--betas 1,0.7,0.5,0.3,0.2,0.1,0 --max-samples 400000 --eval-envs 8 --eval-steps 1000 --eval-every 2 --eval-seed 1 \
		--planner-json "$LITE" --iterations 30 --envs 8 --epochs 4 "$@" \
		--log results/student_screen2/${name}.jsonl --save-dir results/student_screen2/${name} > results/student_screen2/${name}.log 2>&1
}
run base & run motion --motion & run prior --prior & run prior_motion --prior --motion & wait
echo SCREEN2_DONE > results/student_screen2/done.log
