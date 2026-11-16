#!/usr/bin/env bash
# GD screening of student variants on validation episodes (eval seed 1, 16 episodes, disjoint from test), best checkpoint kept.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY="${PYTHON:-python3}"
LITE='{"samples_per_axis":1,"horizon":20,"window":8,"stride":4}'
mkdir -p results/student_screen3
run() {
	local name=$1; shift
	$PY -u research/train_hindsight.py --target privileged --targets greedy --seed 0 --steps 500 \
		--betas 1,0.7,0.5,0.3,0.2,0.1,0 --max-samples 400000 --eval-envs 16 --eval-steps 1000 --eval-every 2 --eval-seed 1 \
		--planner-json "$LITE" --epochs 4 --prior "$@" \
		--log results/student_screen3/${name}.jsonl --save-dir results/student_screen3/${name} > results/student_screen3/${name}.log 2>&1
}
run prior --iterations 30 --envs 8 &
run full_own --iterations 30 --envs 8 --student-planner-json '{}' &
run more_data --iterations 40 --envs 16 &
run full_own_lr1e4 --iterations 30 --envs 8 --student-planner-json '{}' --learning-rate 1e-4 &
wait
echo SCREEN3_DONE > results/student_screen3/done.log
