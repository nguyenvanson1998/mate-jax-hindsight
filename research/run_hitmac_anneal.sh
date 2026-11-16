#!/usr/bin/env bash
# HiT-MAC with linear decay of the learning rate and entropy bonus (converging PPO); 3 at a time; then the test evaluation.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY="${PYTHON:-python3}"
mkdir -p results/hitmac_anneal results/paper
hitmac() {
	local targets=$1 seed=$2 name=${1}_s${2}
	$PY -u research/train_hitmac.py --targets $targets --seed $seed --lr 3e-4 --anneal --total-timesteps 3e7 --eval-every 25 \
		--log results/hitmac_anneal/${name}.jsonl --save-dir results/hitmac_anneal/${name} > results/hitmac_anneal/${name}.log 2>&1
}
hitmac greedy 0 & hitmac waypoint 0 & hitmac greedy 1 & wait
hitmac waypoint 1 & hitmac greedy 2 & hitmac waypoint 2 & wait
$PY -u research/eval_pivot_extra.py --hitmac results/hitmac_anneal/*/ --out results/paper/pivot_hitmac_anneal.json > results/pivot_hitmac_anneal.log 2>&1
echo HITMAC_DONE >> results/pivot_hitmac_anneal.log
