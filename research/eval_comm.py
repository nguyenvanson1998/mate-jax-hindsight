"""Evaluate communication rules and learned communication policies on the same episodes.

Rules: always / never / periodic-4 / event (see :func:`mate_jax.hindsight.comm_env.comm_rule_actions`).
Learned policies: run directories written by ``mate_jax.train --comm --save-dir`` (deterministic and
stochastic). The episodes are identical for every method (seed, number of environments and horizon).
Usage: python3 research/eval_comm.py --targets greedy --runs results/policies/comm_* --out results/comm_eval.json
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax  # noqa: E402
from flax import serialization  # noqa: E402

import mate_jax  # noqa: E402
from mate_jax.hindsight.comm_env import COMM_RULES, CommEnv, comm_rule_actions, evaluate_comm  # noqa: E402
from mate_jax.mappo import OPPONENTS, MAPPOConfig, MAPPOTrainer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', default='MATE-4v8-9.yaml')
    parser.add_argument('--targets', nargs='+', default=['greedy', 'waypoint'])
    parser.add_argument('--rules', nargs='*', default=list(COMM_RULES))
    parser.add_argument('--runs', nargs='*', default=[])
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--decisions', type=int, default=200, help='decisions per episode (x frame skip steps)')
    parser.add_argument('--frame-skip', type=int, default=5)
    parser.add_argument('--packet-cost', type=float, default=0.05)
    parser.add_argument('--intent-cost', type=float, default=0.04)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--full', action='store_true', help='full planner re-planning every step (default: lite)')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    env = mate_jax.make(args.config)
    results = []

    def record(row):
        results.append(row)
        print(json.dumps(row), flush=True)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, 'w', encoding='UTF-8') as file:
                json.dump(results, file, indent=2)

    for targets in args.targets:
        comm_env = CommEnv(
            env, opponent=OPPONENTS[('camera', targets)].from_env(env), frame_skip=args.frame_skip,
            horizon=args.decisions, packet_cost=args.packet_cost, intent_cost=args.intent_cost,
            lite=not args.full, replan_every=1 if args.full else args.frame_skip,
        )
        for rule in args.rules:
            start = time.perf_counter()
            select = lambda k, o, s, rule=rule: jax.vmap(lambda x: comm_rule_actions(comm_env, rule, x))(s)  # noqa: E731
            metrics = evaluate_comm(comm_env, select, num_envs=args.num_envs, num_steps=args.decisions, seed=args.seed)
            record({'targets': targets, 'method': rule, **metrics, 'seconds': time.perf_counter() - start})
        for directory in args.runs:
            with open(os.path.join(directory, 'config.json'), encoding='UTF-8') as file:
                config = json.load(file)
            if config['opponent'] != targets:
                continue
            config.update(horizon=args.decisions, num_devices=1)
            trainer = MAPPOTrainer(MAPPOConfig(**config))
            with open(os.path.join(directory, 'params.msgpack'), 'rb') as file:
                params = serialization.msgpack_restore(file.read())['actor']
            dist = trainer.distribution
            for mode in ('deterministic', 'stochastic'):
                def select(key, obs, state, mode=mode):
                    logits, _ = trainer.actor.apply(params, trainer.actor_input(obs))
                    return dist.mode(logits, None) if mode == 'deterministic' else dist.sample(key, logits, None)

                start = time.perf_counter()
                metrics = evaluate_comm(trainer.team_env, select, num_envs=args.num_envs, num_steps=args.decisions, seed=args.seed)
                record({
                    'targets': targets, 'method': os.path.basename(os.path.normpath(directory)), 'mode': mode,
                    'hindsight': config.get('comm_hindsight', True), 'seed': config['seed'], **metrics,
                    'seconds': time.perf_counter() - start,
                })


if __name__ == '__main__':
    main()
