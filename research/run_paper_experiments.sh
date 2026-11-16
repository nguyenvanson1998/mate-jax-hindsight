#!/usr/bin/env bash
# All experiments of the trajectory-coordination manuscript, on 32 paired episodes (seed 0, 1000 steps).
# Every script uses the same episode keys, so all coverage numbers are paired across methods, the
# oracle variants (headroom.py) and the hindsight bound (clairvoyant.py).
#   PARTS="main loss scale headroom bound learned bench" bash research/run_paper_experiments.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python3}"
OUT="${OUT:-results/paper}"
ENVS="${ENVS:-32}"
STEPS="${STEPS:-1000}"
PARTS="${PARTS:-main loss scale headroom bound learned bench}"
LEVELS='[30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180]'
COMMON="--num-envs ${ENVS} --num-steps ${STEPS} --seeds 0"
TRAJ="\"objective\": \"trajectory\", \"levels\": ${LEVELS}"
SYNC="${TRAJ}, \"coordination_rounds\": 2, \"communication\": \"hybrid\""
PUB="${TRAJ}, \"coordination\": \"published\", \"communication\": \"hybrid\""

mkdir -p "${OUT}"

run() {  # name, command...
	local name="$1"
	shift
	[ -s "${OUT}/${name}.json" ] && { echo "=== ${name}: done"; return; }
	echo "=== ${name}"
	"$@" || echo "!!! ${name} failed"
}

for part in ${PARTS}; do
	case "${part}" in
	main)
		run main_4v8 "${PYTHON}" -u research/eval_foresight.py --configs MATE-4v8-9.yaml --targets greedy waypoint ${COMMON} \
			--variants foresight greedy hybrid traj-1 traj-2 traj-3 traj-2-hybrid traj-2-none traj-2-forecast \
			--reference foresight --out "${OUT}/main_4v8.json"
		;;
	loss)
		variants='{'
		for p in 0.25 0.5 0.75 0.9; do
			variants+="\"foresight-team-${p}\": {\"message_drop\": ${p}},"
			variants+="\"foresight-hybrid-${p}\": {\"communication\": \"hybrid\", \"message_drop\": ${p}},"
			variants+="\"traj-team-${p}\": {\"objective\": \"trajectory\", \"levels\": ${LEVELS}, \"coordination_rounds\": 2, \"message_drop\": ${p}},"
			variants+="\"traj-hybrid-${p}\": {\"objective\": \"trajectory\", \"levels\": ${LEVELS}, \"coordination_rounds\": 2, \"communication\": \"hybrid\", \"message_drop\": ${p}},"
		done
		variants="${variants%,}}"
		run loss_4v8 "${PYTHON}" -u research/eval_foresight.py --configs MATE-4v8-9.yaml --targets greedy waypoint ${COMMON} \
			--variants foresight traj-2 --variant-json "${variants}" --reference foresight --out "${OUT}/loss_4v8.json"
		;;
	scale)
		run main_8v8 "${PYTHON}" -u research/eval_foresight.py --configs MATE-8v8-9.yaml --targets greedy waypoint ${COMMON} \
			--variants foresight greedy hybrid traj-1 traj-2 traj-2-hybrid --reference foresight --out "${OUT}/main_8v8.json"
		;;
	headroom)
		for config in 4v8 8v8; do
			run "headroom_option_${config}" "${PYTHON}" -u research/headroom.py --config "MATE-${config}-9.yaml" \
				--targets greedy waypoint --num-envs "${ENVS}" --num-steps "${STEPS}" --key-steps "${STEPS}" \
				--out "${OUT}/headroom_option_${config}.json"
			run "headroom_traj2_${config}" "${PYTHON}" -u research/headroom.py --config "MATE-${config}-9.yaml" \
				--targets greedy waypoint --modes foresight oracle-future --num-envs "${ENVS}" --num-steps "${STEPS}" \
				--key-steps "${STEPS}" --planner-json "{\"objective\": \"trajectory\", \"coordination_rounds\": 2, \"levels\": ${LEVELS}}" \
				--out "${OUT}/headroom_traj2_${config}.json"
		done
		;;
	bound)
		for config in 4v8 8v8; do
			run "bound_${config}" "${PYTHON}" -u research/clairvoyant.py --config "MATE-${config}-9.yaml" --targets greedy waypoint \
				--num-envs "${ENVS}" --num-steps "${STEPS}" --key-steps "${STEPS}" --rounds 3 --out "${OUT}/bound_${config}.json"
		done
		;;
	published)  # decentralized coordination through published intents
		run published_4v8 "${PYTHON}" -u research/eval_foresight.py --configs MATE-4v8-9.yaml --targets greedy waypoint ${COMMON} \
			--variants traj-2 pub-always pub-period-4 pub-event-1 pub-always-hybrid --reference traj-2 \
			--out "${OUT}/published_4v8.json"
		;;
	learned)  # planners on the episode length of the learned methods' evaluation (500 decisions x 5 steps)
		run learned_protocol_4v8 "${PYTHON}" -u research/eval_foresight.py --configs MATE-4v8-9.yaml --targets greedy waypoint \
			--num-envs 16 --num-steps 2500 --seeds 1 --variants foresight greedy traj-2 --reference foresight \
			--out "${OUT}/learned_protocol_4v8.json"
		;;
	bench)
		run bench_4v8 "${PYTHON}" -u research/bench_foresight.py --config MATE-4v8-9.yaml --batch-sizes 1 32 \
			--num-steps 200 --variants greedy foresight traj-1 traj-2 traj-3 --out "${OUT}/bench_4v8.json"
		;;
	tradeoff)  # coverage vs. traffic: detection packets every R steps, intents every P steps (published, hybrid memory)
		variants='{'
		for r in 1 2 4 8; do
			for p in 1 4 16; do
				variants+="\"net-r${r}-p${p}\": {${PUB}, \"report_period\": ${r}, \"publish_period\": ${p}},"
			done
		done
		variants+="\"net-none\": {${TRAJ}, \"coordination_rounds\": 2, \"communication\": \"none\"}}"
		run tradeoff_4v8 "${PYTHON}" -u research/eval_foresight.py --configs MATE-4v8-9.yaml --targets greedy waypoint ${COMMON} \
			--variants net-r1-p1 --variant-json "${variants}" --reference net-r1-p1 --out "${OUT}/tradeoff_4v8.json"
		;;
	burst)  # bursty (Gilbert-Elliott) and non-uniform loss; hybrid memory, synchronous and published intents
		variants='{'
		for p in 0.25 0.5 0.75 0.9; do
			variants+="\"iid-${p}\": {${SYNC}, \"message_drop\": ${p}},"
			for b in 5 20; do
				variants+="\"burst${b}-${p}\": {${SYNC}, \"message_drop\": ${p}, \"loss_model\": \"burst\", \"burst_length\": ${b}},"
			done
			variants+="\"spread-${p}\": {${SYNC}, \"message_drop\": ${p}, \"loss_spread\": 0.8},"
			variants+="\"pub-burst20-${p}\": {${PUB}, \"publish_period\": 4, \"message_drop\": ${p}, \"loss_model\": \"burst\", \"burst_length\": 20, \"intent_loss\": true},"
		done
		variants="${variants%,}}"
		run burst_4v8 "${PYTHON}" -u research/eval_foresight.py --configs MATE-4v8-9.yaml --targets greedy waypoint ${COMMON} \
			--variants traj-2-hybrid --variant-json "${variants}" --reference traj-2-hybrid --out "${OUT}/burst_4v8.json"
		;;
	delay)  # latency of detection packets (team / hybrid memory) and of published intents
		variants='{'
		for d in 1 2 5 10 20; do
			variants+="\"delay-hybrid-${d}\": {${SYNC}, \"message_delay\": ${d}},"
			variants+="\"delay-team-${d}\": {${TRAJ}, \"coordination_rounds\": 2, \"message_delay\": ${d}},"
			variants+="\"delay-intent-${d}\": {${PUB}, \"publish_period\": 4, \"intent_delay\": ${d}},"
		done
		variants+="\"delay-intent-0\": {${PUB}, \"publish_period\": 4}}"
		run delay_4v8 "${PYTHON}" -u research/eval_foresight.py --configs MATE-4v8-9.yaml --targets greedy waypoint ${COMMON} \
			--variants traj-2-hybrid --variant-json "${variants}" --reference traj-2-hybrid --out "${OUT}/delay_4v8.json"
		;;
	scale8)  # the variants of scale2 that the first 8v8 run did not include (foresight: reproduction check)
		run main_8v8_extra "${PYTHON}" -u research/eval_foresight.py --configs MATE-8v8-9.yaml --targets greedy waypoint ${COMMON} \
			--variants foresight pub-period-4 traj-2-none --reference foresight --out "${OUT}/main_8v8_extra.json"
		;;
	scale2)  # larger networks at the 1:2 camera-to-target ratio of MATE-4v8
		for config in 8v16 16v32; do
			run "main_${config}" "${PYTHON}" -u research/eval_foresight.py --configs "MATE-${config}-9.yaml" --targets greedy waypoint ${COMMON} \
				--variants foresight greedy traj-1 traj-2 pub-period-4 traj-2-none --reference foresight --out "${OUT}/main_${config}.json"
			run "bound_${config}" "${PYTHON}" -u research/clairvoyant.py --config "MATE-${config}-9.yaml" --targets greedy waypoint \
				--num-envs "${ENVS}" --num-steps "${STEPS}" --key-steps "${STEPS}" --rounds 3 --out "${OUT}/bound_${config}.json"
		done
		;;
	benchscale)
		for config in 4v8 8v8 8v16 16v32; do
			run "bench_scale_${config}" "${PYTHON}" -u research/bench_foresight.py --config "MATE-${config}-9.yaml" --batch-sizes 1 \
				--num-steps 200 --variants greedy foresight traj-2 pub-4 pub-4-hybrid --out "${OUT}/bench_scale_${config}.json"
		done
		;;
	esac
done
echo "PAPER_EXPERIMENTS_DONE"
