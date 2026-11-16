"""Rule-based high levels of FORESIGHT-HRL: greedy option + a fixed transmission rule.

Same evaluation protocol as the learning curves (16 episodes x 1000 environment steps).
Usage: python3 research/eval_hrl_rules.py --targets greedy waypoint --out results/hrl_rules_4v8.json
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mate_jax  # noqa: E402
from mate_jax.foresight import ForesightCameraPolicy  # noqa: E402
from mate_jax.foresight.hrl import LITE_PLANNER, TRANSMIT_RULES, ForesightHRLEnv, evaluate_rule  # noqa: E402
from mate_jax.mappo import OPPONENTS  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', default='MATE-4v8-9.yaml')
    parser.add_argument('--targets', nargs='+', default=['greedy', 'waypoint'])
    parser.add_argument('--rules', nargs='+', default=list(TRANSMIT_RULES))
    parser.add_argument('--comm-cost', type=float, default=0.1)
    parser.add_argument('--actuation-cost', type=float, default=0.05)
    parser.add_argument('--channel-slots', type=int, default=4)
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--env-steps', type=int, default=1000)
    parser.add_argument('--frame-skip', type=int, default=5)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    env = mate_jax.make(args.config)
    results = []
    for targets in args.targets:
        planner = ForesightCameraPolicy.from_env(
            env, communication='hybrid', channel_slots=args.channel_slots, **LITE_PLANNER
        )
        hrl = ForesightHRLEnv(
            env,
            opponent=OPPONENTS[('camera', targets)].from_env(env),
            planner=planner,
            frame_skip=args.frame_skip,
            horizon=args.env_steps // args.frame_skip,
            comm_cost=args.comm_cost,
            actuation_cost=args.actuation_cost,
        )
        for rule in args.rules:
            start = time.perf_counter()
            metrics = evaluate_rule(hrl, rule, num_envs=args.num_envs, num_steps=args.env_steps // args.frame_skip)
            row = {'targets': targets, 'rule': rule, **metrics, 'seconds': time.perf_counter() - start}
            results.append(row)
            print(json.dumps(row), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='UTF-8') as file:
            json.dump(results, file, indent=2)


if __name__ == '__main__':
    main()
