"""Train the HiT-MAC coordinator (mate_jax.hitmac) and log learning curves.

Evaluation: deterministic goal maps on ``--eval-envs`` episodes of 1000 steps with the key layout of
eval_foresight.py (``--eval-seed 0`` gives the paired evaluation episodes).
Usage: python3 research/train_hitmac.py --targets greedy --total-timesteps 1e7 --log results/hitmac/greedy_s0.jsonl
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax  # noqa: E402
import numpy as np  # noqa: E402
from flax import serialization  # noqa: E402

import mate_jax  # noqa: E402
from mate_jax import agents as A  # noqa: E402
from mate_jax.evaluation import make_coverage_evaluator  # noqa: E402
from mate_jax.hitmac import HitMacPolicy, HitMacTrainer  # noqa: E402

TARGETS = {'greedy': A.GreedyTargetPolicy, 'waypoint': A.RandomWaypointTargetPolicy}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', default='MATE-4v8-9.yaml')
    parser.add_argument('--targets', default='greedy', choices=sorted(TARGETS))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-envs', type=int, default=128)
    parser.add_argument('--decisions', type=int, default=16, help='coordinator decisions per rollout')
    parser.add_argument('--total-timesteps', type=float, default=1e7, help='environment steps')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--eval-every', type=int, default=20, help='updates between evaluations')
    parser.add_argument('--eval-envs', type=int, default=16)
    parser.add_argument('--eval-seed', type=int, default=1)
    parser.add_argument('--anneal', action='store_true', help='linear decay of the learning rate and entropy bonus to 0')
    parser.add_argument('--patience', type=int, default=0, help='stop after this many evaluations without improvement (0: never)')
    parser.add_argument('--log', default=None)
    parser.add_argument('--save-dir', default=None)
    args = parser.parse_args()

    env = mate_jax.make(args.config)
    target_policy = TARGETS[args.targets].from_env(env)
    probe = HitMacTrainer(env, target_policy, num_envs=args.num_envs, decisions=args.decisions)
    updates = int(np.ceil(args.total_timesteps / probe.env_steps_per_update()))
    trainer = HitMacTrainer(env, target_policy, num_envs=args.num_envs, decisions=args.decisions, lr=args.lr, seed=args.seed,
                            anneal_updates=updates if args.anneal else 0)
    params, opt_state, envs = trainer.init()

    def eval_coverage(params):
        policy = HitMacPolicy.from_env(env, params=params)
        run = make_coverage_evaluator(env, policy, target_policy, num_envs=args.eval_envs, num_steps=1000)
        return float(np.mean(run(args.eval_seed)['coverage_rate']))

    log = open(args.log, 'a', encoding='UTF-8') if args.log else None
    key = jax.random.PRNGKey(args.seed + 1)
    best = (-1.0, params)  # early stopping on the validation episodes (eval seed != the test seed 0)
    stale = 0
    row = {'update': 0, 'env_steps': 0, 'eval_coverage': eval_coverage(params)}
    print(json.dumps(row), flush=True)
    if log:
        log.write(json.dumps(row) + '\n')
    start = time.perf_counter()
    for update in range(1, updates + 1):
        key, k_roll, k_upd = jax.random.split(key, 3)
        envs, batch, last_value = trainer._rollout(params, envs, k_roll)
        params, opt_state, stats = trainer._update(params, opt_state, batch, last_value, k_upd, (update - 1) / updates)
        row = {'update': update, 'env_steps': update * trainer.env_steps_per_update(), **trainer.summary(batch, stats),
               'seconds': time.perf_counter() - start}
        if update % args.eval_every == 0 or update == updates:
            row['eval_coverage'] = eval_coverage(params)
            if row['eval_coverage'] > best[0]:
                best, stale = (row['eval_coverage'], jax.device_get(params)), 0
            else:
                stale += 1
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        if log:
            log.write(json.dumps(row) + '\n')
            log.flush()
        if args.patience and stale >= args.patience:
            print(json.dumps({'early_stop': update, 'best_eval_coverage': best[0]}), flush=True)
            break
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, 'params.msgpack'), 'wb') as file:  # best on validation
            file.write(serialization.to_bytes(best[1]))
        with open(os.path.join(args.save_dir, 'final.msgpack'), 'wb') as file:
            file.write(serialization.to_bytes(jax.device_get(params)))
        with open(os.path.join(args.save_dir, 'config.json'), 'w', encoding='UTF-8') as file:
            json.dump(vars(args), file, indent=2)


if __name__ == '__main__':
    main()
