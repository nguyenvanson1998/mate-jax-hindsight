"""How predictable are the targets given the full simulator state?

From recorded states of the episodes of the paper, the simulator is branched with fresh random keys and the target
positions of the branches are compared after h steps: the spread (root mean square distance to the mean position)
and the share of targets whose branches end in different grid cells of the planner (62.5 m). If the spread is small,
a controller that knew the full state could predict the targets almost exactly, and the clairvoyant bound is close
to the best value of such a controller; penalties computed from the full state can then not tighten it.

Usage: python3 research/predictability.py --targets greedy waypoint --num-envs 8 --out results/paper/predictability_4v8.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import clairvoyant as C  # noqa: E402
import mate_jax  # noqa: E402
from penalty_bound import make_rollout, record_full  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', default='MATE-4v8-9.yaml')
    parser.add_argument('--targets', nargs='+', default=['greedy', 'waypoint'])
    parser.add_argument('--num-envs', type=int, default=8)
    parser.add_argument('--key-steps', type=int, default=1000)
    parser.add_argument('--every', type=int, default=10, help='branch every this many steps')
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--horizons', type=int, nargs='+', default=[1, 5, 10, 20, 50])
    parser.add_argument('--cell', type=float, default=62.5)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    env = mate_jax.make(args.config)
    keys = jax.random.split(jax.random.PRNGKey(0), args.num_envs)
    length = max(args.horizons)
    results = []
    for targets in args.targets:
        policy = C.TARGETS[targets].from_env(env)
        _, origins, _ = record_full(env, policy, keys, args.key_steps)
        rollout = make_rollout(env, policy, length)
        index = jnp.arange(0, args.key_steps - length, args.every)

        @jax.jit
        def branch(carries, k):
            ks = jax.random.split(k, carries[0].target_location.shape[0] * args.samples).reshape(-1, args.samples, 2)
            return jax.vmap(lambda c, kk: jax.vmap(lambda k1: rollout(c, k1))(kk))(carries, ks)  # (U, K, L, T, 2)

        spread, split_cell = {h: [] for h in args.horizons}, {h: [] for h in args.horizons}
        for e in range(args.num_envs):
            carries = jax.tree_util.tree_map(lambda a: a[e][index], origins)
            pos = np.asarray(branch(carries, jax.random.PRNGKey(1000 + e)))  # (U, K, L, T, 2)
            for h in args.horizons:
                p = pos[:, :, h - 1]  # (U, K, T, 2)
                mean = p.mean(1, keepdims=True)
                spread[h].append(np.sqrt(((p - mean) ** 2).sum(-1).mean(1)).ravel())  # per state and target
                cells = np.floor((p + 1000.0) / args.cell).astype(int)
                different = np.any(np.any(cells != cells[:, :1], axis=-1), axis=1)  # (U, T)
                split_cell[h].append(different.ravel())
        row = {'targets': targets, 'episodes': args.num_envs, 'samples': args.samples, 'every': args.every,
               'step_size': float(env.config.target_step_size)}
        for h in args.horizons:
            s = np.concatenate(spread[h])
            row[f'spread_mean_{h}'] = float(s.mean())
            row[f'spread_median_{h}'] = float(np.median(s))
            row[f'split_share_{h}'] = float(np.concatenate(split_cell[h]).mean())
        results.append(row)
        print(json.dumps(row), flush=True)
    if args.out:
        with open(args.out, 'w', encoding='UTF-8') as file:
            json.dump(results, file, indent=2)


if __name__ == '__main__':
    main()
