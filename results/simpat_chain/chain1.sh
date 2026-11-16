#!/usr/bin/env bash
# SIMPAT revision, chain 1 (NVIDIA GB10 machine). Run from the repository root:  nohup bash results/simpat_chain/chain1.sh &
# 1. latency on the idle machine (nothing else running), 2. coverage jobs in two streams.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export XLA_FLAGS=--xla_gpu_autotune_level=0
P="${PYTHON:-python3}"
OUT=results/paper
LOG=results/simpat_chain
LEVELS='[30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180]'
TRAJ="\"objective\": \"trajectory\", \"levels\": ${LEVELS}"
SYNC="${TRAJ}, \"coordination_rounds\": 2, \"communication\": \"hybrid\""
COMMON="--num-envs 32 --num-steps 1000"

run() {  # output file, command...
	local out="$1"; shift
	[ -s "${out}" ] && { echo "=== ${out}: done"; return; }
	echo "=== ${out} $(date +%H:%M:%S)"
	"$@" || echo "!!! ${out} failed"
}

# ---- 1. latency, batch 1, 7 timed runs per variant, machine otherwise idle
echo "LATENCY_START $(date)" > ${LOG}/status
for config in 4v8 8v8 8v16 16v32; do
	XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 run "${OUT}/bench_idle_${config}.json" ${P} -u research/bench_foresight.py \
		--config "MATE-${config}-9.yaml" --batch-sizes 1 --num-steps 200 --repeats 7 \
		--variants greedy foresight traj-2 pub-4 pub-4-hybrid --out "${OUT}/bench_idle_${config}.json"
done > ${LOG}/latency.log 2>&1
echo "LATENCY_DONE $(date)" >> ${LOG}/status

# ---- 2. coverage jobs
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.28
(
	# loss settings at the exact effective rates of the saturated cells, feasible bursts of 5, non-uniform without cut-off
	variants='{'
	for p in 0.6625 0.71 0.8 0.8333333; do variants+="\"iid-${p}\": {${SYNC}, \"message_drop\": ${p}},"; done
	for b in 5 20; do variants+="\"burst${b}-0.8\": {${SYNC}, \"message_drop\": 0.8, \"loss_model\": \"burst\", \"burst_length\": ${b}},"; done
	variants+="\"spread03-0.75\": {${SYNC}, \"message_drop\": 0.75, \"loss_spread\": 0.3},"
	variants+="\"spread01-0.9\": {${SYNC}, \"message_drop\": 0.9, \"loss_spread\": 0.1}}"
	run "${OUT}/burst_extra_4v8.json" ${P} -u research/eval_foresight.py --configs MATE-4v8-9.yaml --targets greedy waypoint ${COMMON} --seeds 0 \
		--variants traj-2-hybrid --variant-json "${variants}" --reference traj-2-hybrid --out "${OUT}/burst_extra_4v8.json"
	# option-level planning on the 31 zoom levels of the trajectory lattice
	run "${OUT}/zoom_4v8.json" ${P} -u research/eval_foresight.py --configs MATE-4v8-9.yaml --targets greedy waypoint ${COMMON} --seeds 0 \
		--variants foresight --variant-json "{\"option-31\": {\"levels\": ${LEVELS}}}" --reference foresight --out "${OUT}/zoom_4v8.json"
) > ${LOG}/stream_a.log 2>&1 &
(
	# 96 more paired episodes for the small differences (forecaster, third round, published intents)
	run "${OUT}/main_4v8_seeds123.json" ${P} -u research/eval_foresight.py --configs MATE-4v8-9.yaml --targets greedy waypoint ${COMMON} --seeds 1 2 3 \
		--variants traj-2 traj-2-forecast traj-3 pub-period-4 traj-1 --reference traj-2 --out "${OUT}/main_4v8_seeds123.json"
) > ${LOG}/stream_b.log 2>&1 &
wait
echo "CHAIN1_DONE $(date)" >> ${LOG}/status
