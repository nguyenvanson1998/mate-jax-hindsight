#!/usr/bin/env bash
# SIMPAT revision, chain 2 (NVIDIA GB10 machine): verification of the bounds, visibility rule, oracle diagnostic, sensitivity.
# Starts after the latency measurement of chain 1 has finished (it needs an idle machine).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export XLA_FLAGS=--xla_gpu_autotune_level=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.14
P="${PYTHON:-python3}"
OUT=results/paper
LOG=results/simpat_chain
LEVELS='[30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180]'
TRAJ="\"objective\": \"trajectory\", \"levels\": ${LEVELS}"
TWO="${TRAJ}, \"coordination_rounds\": 2"
COMMON="--num-envs 32 --num-steps 1000 --seeds 0"
OPAQUE='{"obstacle": {"transmittance": 0.0}}'

run() {  # output file, command...
	local out="$1"; shift
	[ -s "${out}" ] && { echo "=== ${out}: done"; return; }
	echo "=== ${out} $(date +%H:%M:%S)"
	"$@" || echo "!!! ${out} failed"
}

until grep -q LATENCY_DONE ${LOG}/status 2>/dev/null; do sleep 60; done
echo "CHAIN2_START $(date)" >> ${LOG}/status

(
	# exact optimum of two cameras against the hindsight bound and the Lagrangian bound
	run "${OUT}/exact_pair_4v8.json" ${P} -u research/exact_pair.py --config MATE-4v8-9.yaml --targets greedy waypoint --pairs 0,1 2,3 \
		--num-envs 32 --num-steps 1000 --key-steps 1000 --dual-iterations 150 --out "${OUT}/exact_pair_4v8.json"
	# controllers and bounds scored with the same visibility rule (opaque obstacles)
	run "${OUT}/main_4v8_opaque.json" ${P} -u research/eval_foresight.py --configs MATE-4v8-9.yaml --targets greedy waypoint ${COMMON} \
		--env-json "${OPAQUE}" --variants foresight greedy traj-1 traj-2 traj-2-forecast --reference foresight --out "${OUT}/main_4v8_opaque.json"
	run "${OUT}/bound_dual_4v8_opaque.json" ${P} -u research/clairvoyant.py --config MATE-4v8-9.yaml --targets greedy waypoint \
		--num-envs 32 --num-steps 1000 --key-steps 1000 --rounds 3 --dual-iterations 150 --env-json "${OPAQUE}" --out "${OUT}/bound_dual_4v8_opaque.json"
	# why do true future positions not help with RWP targets? oracle with blurred future cells
	run "${OUT}/headroom_blur_4v8.json" ${P} -u research/headroom.py --config MATE-4v8-9.yaml --targets greedy waypoint \
		--modes foresight oracle-future --blur-sigmas 0.5 1 --num-envs 32 --num-steps 1000 --key-steps 1000 \
		--planner-json "{${TWO}}" --out "${OUT}/headroom_blur_4v8.json"
	# scenario sensitivity: target speed (default step size 20 m)
	for v in 10 30 40; do
		run "${OUT}/sens_speed${v}_4v8.json" ${P} -u research/eval_foresight.py --configs MATE-4v8-9.yaml --targets greedy waypoint ${COMMON} \
			--env-json "{\"target\": {\"step_size\": ${v}.0}}" --variants foresight greedy traj-2 --reference foresight --out "${OUT}/sens_speed${v}_4v8.json"
		run "${OUT}/bound_speed${v}_4v8.json" ${P} -u research/clairvoyant.py --config MATE-4v8-9.yaml --targets greedy waypoint \
			--num-envs 32 --num-steps 1000 --key-steps 1000 --rounds 3 --dual-iterations 150 \
			--env-json "{\"target\": {\"step_size\": ${v}.0}}" --out "${OUT}/bound_speed${v}_4v8.json"
	done
) > ${LOG}/stream_c.log 2>&1 &
(
	# planner sensitivity: lookahead, motion-model noise, detection probability (reference: defaults)
	variants='{'
	for h in 0 10 20 60; do variants+="\"lookahead-$((h + 10))\": {${TWO}, \"horizon\": ${h}},"; done
	variants+="\"noise-half\": {${TWO}, \"sigma_walk_ratio\": 0.125, \"sigma_velocity_known_ratio\": 0.15, \"sigma_velocity_unknown_ratio\": 0.3536},"
	variants+="\"noise-double\": {${TWO}, \"sigma_walk_ratio\": 0.5, \"sigma_velocity_known_ratio\": 0.6, \"sigma_velocity_unknown_ratio\": 1.4142},"
	variants+="\"detect-0.7\": {${TWO}, \"detection_prob\": 0.7},"
	variants+="\"detect-0.99\": {${TWO}, \"detection_prob\": 0.99}}"
	run "${OUT}/sens_model_4v8.json" ${P} -u research/eval_foresight.py --configs MATE-4v8-9.yaml --targets greedy waypoint ${COMMON} \
		--variants traj-2 --variant-json "${variants}" --reference traj-2 --out "${OUT}/sens_model_4v8.json"
	# grid resolution of the belief maps (default 32 x 32)
	for g in 16 24 48; do
		run "${OUT}/sens_grid${g}_4v8.json" ${P} -u research/eval_foresight.py --configs MATE-4v8-9.yaml --targets greedy waypoint ${COMMON} \
			--grid-size ${g} --variants foresight traj-2 --reference foresight --out "${OUT}/sens_grid${g}_4v8.json"
	done
) > ${LOG}/stream_d.log 2>&1 &
wait
echo "CHAIN2_DONE $(date)" >> ${LOG}/status
