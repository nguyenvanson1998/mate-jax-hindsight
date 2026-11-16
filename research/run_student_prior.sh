#!/usr/bin/env bash
# Chosen student (planner prior; selected on validation episodes of eval seed 1), best validation checkpoint.
# Usage: bash research/run_student_prior.sh "<targets>:<seed> ..." [extra screening]
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY="${PYTHON:-python3}"
LITE='{"samples_per_axis":1,"horizon":20,"window":8,"stride":4}'
mkdir -p results/student_prior results/student_screen2
student() {
	local targets=$1 seed=$2 dir=$3 name=$4; shift 4
	$PY -u research/train_hindsight.py --target privileged --targets $targets --seed $seed --steps 500 \
		--betas 1,0.7,0.5,0.3,0.2,0.1,0 --max-samples 400000 --eval-envs 8 --eval-steps 1000 --eval-every 2 --eval-seed 1 \
		--planner-json "$LITE" --iterations 30 --envs 8 --epochs 4 --prior "$@" \
		--log results/$dir/${name}.jsonl --save-dir results/$dir/${name} > results/$dir/${name}.log 2>&1
}
for job in $1; do t=${job%:*}; s=${job#*:}; student $t $s student_prior ${t}_s${s} & done
if [ -n "$2" ]; then student greedy 0 student_screen2 prior_lr1e4 --learning-rate 1e-4 & fi
wait
echo PRIOR_RUNS_DONE > results/student_prior/done_$(hostname).log
