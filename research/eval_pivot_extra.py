"""Paired evaluation (32 episodes of 1000 steps, seed 0, the keys of eval_foresight.py) of
(1) the communication-free students distilled from the communicating PIVOT teacher (research/train_hindsight.py
    --target privileged), executed greedily, and the same teacher, and
(2) the final HiT-MAC coordinators (research/train_hitmac.py).
Usage: python3 research/eval_pivot_extra.py --students results/pivot_student/* --hitmac results/hitmac/* --out results/paper/pivot_extra.json
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
from mate_jax.hindsight.privileged import PrivilegedDistiller  # noqa: E402
from mate_jax.hitmac import HitMacNet, HitMacPolicy, PAIR_FEATURES  # noqa: E402

TARGETS = {'greedy': A.GreedyTargetPolicy, 'waypoint': A.RandomWaypointTargetPolicy}


def record(rows, row):
    rows.append(row)
    print(json.dumps({k: v for k, v in row.items() if k != 'per_episode_coverage'}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', default='MATE-4v8-9.yaml')
    parser.add_argument('--students', nargs='*', default=[])
    parser.add_argument('--hitmac', nargs='*', default=[])
    parser.add_argument('--num-envs', type=int, default=32)
    parser.add_argument('--num-steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0, help='episode seed (0: test; 1: the validation episodes of training)')
    parser.add_argument('--ensemble', type=int, default=0,
                        help='students: average the K checkpoints with the highest training-time validation coverage')
    parser.add_argument('--student-only', action='store_true')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    env = mate_jax.make(args.config)
    rows = []
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.num_envs)
    for directory in args.students:
        with open(os.path.join(directory, 'config.json'), encoding='UTF-8') as file:
            config = json.load(file)
        targets = config['targets']
        lite = json.loads(config['planner_json'])
        own = json.loads(config['student_planner_json']) if config.get('student_planner_json') is not None else lite
        distiller = PrivilegedDistiller(env, TARGETS[targets].from_env(env), num_steps=args.num_steps, key_steps=args.num_steps,
                                        hiddens=config['hiddens'], teacher_kwargs=lite, student_kwargs=own,
                                        layer_norm=config.get('layer_norm', False), motion=config.get('motion', False), prior=config.get('prior', False))
        params, _ = distiller.init(jax.random.PRNGKey(0))
        name = 'best.msgpack' if config.get('eval_seed', 0) != 0 else 'params.msgpack'  # checkpoint chosen on validation
        if args.ensemble:  # top-K checkpoints by validation coverage (eval seed != 0), stacked
            with open(config['log'], encoding='UTF-8') as file:
                logged = [json.loads(line) for line in file]
            ranked = sorted((r for r in logged if 'eval_coverage' in r), key=lambda r: -r['eval_coverage'])[: args.ensemble]
            members = []
            for r in ranked:
                with open(os.path.join(directory, 'checkpoints', f"{r['iteration']:03d}.msgpack"), 'rb') as file:
                    members.append(serialization.from_bytes(params, file.read()))
            params = jax.tree_util.tree_map(lambda *x: np.stack(x), *members)
            distiller.ensemble = True
        else:
            with open(os.path.join(directory, name), 'rb') as file:
                params = serialization.from_bytes(params, file.read())
        for mode, beta in (('student', 0.0),) + ((('teacher', 1.0),) if not args.student_only else ()):
            start = time.perf_counter()
            _, _, coverage, _ = distiller.rollout(params, keys, None, beta)
            coverage = np.asarray(coverage)
            record(rows, {'targets': targets, 'variant': f'privileged-{mode}', 'ensemble': args.ensemble, 'seed': args.seed, 'run': os.path.basename(os.path.normpath(directory)),
                          'coverage': float(coverage.mean()), 'coverage_se': float(coverage.std(ddof=1) / np.sqrt(coverage.size)),
                          'per_episode_coverage': coverage.tolist(), 'seconds': time.perf_counter() - start})
    for directory in args.hitmac:
        with open(os.path.join(directory, 'config.json'), encoding='UTF-8') as file:
            config = json.load(file)
        targets = config['targets']
        template = HitMacNet().init(jax.random.PRNGKey(0), np.zeros((env.num_cameras * env.num_targets, PAIR_FEATURES), np.float32))
        with open(os.path.join(directory, 'params.msgpack'), 'rb') as file:
            params = serialization.from_bytes(template, file.read())
        policy = HitMacPolicy.from_env(env, params=params)
        start = time.perf_counter()
        metrics = make_coverage_evaluator(env, policy, TARGETS[targets].from_env(env), args.num_envs, args.num_steps)(0)
        coverage = np.asarray(metrics['coverage_rate'])
        record(rows, {'targets': targets, 'variant': 'hitmac', 'run': os.path.basename(os.path.normpath(directory)),
                      'coverage': float(coverage.mean()), 'coverage_se': float(coverage.std(ddof=1) / np.sqrt(coverage.size)),
                      'per_episode_coverage': coverage.tolist(), 'redundancy': float(np.mean(metrics['redundancy'])),
                      'per_camera': {k: np.asarray(metrics[k]).mean(0).tolist() for k in
                                     ('camera_coverage', 'camera_unique_coverage', 'camera_rotation', 'camera_zoom')},
                      'seconds': time.perf_counter() - start})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='UTF-8') as file:
        json.dump(rows, file, indent=2)


if __name__ == '__main__':
    main()
