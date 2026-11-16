"""Evaluate learned controllers (MAPPO / IPPO and value-factorization baselines) on the paired episodes.

Each run directory holds ``params.msgpack`` and ``config.json``. The controllers act every ``frame_skip``
steps (as in training) on the same episodes as ``eval_foresight.py`` (seed, number of environments, steps),
so that their coverage is paired with the planners'. Rows follow the format of eval_foresight.py.
Usage: python3 research/eval_learned.py --runs results/policies_long/* results/policies_value/* --out results/paper/learned_4v8.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from flax import serialization  # noqa: E402

import mate_jax  # noqa: E402
from mate_jax.evaluation import FrameSkipPolicy, evaluate_coverage  # noqa: E402
from mate_jax.mappo import OPPONENTS, MAPPOConfig, MAPPOTrainer  # noqa: E402


def load_policy(directory):
    with open(os.path.join(directory, 'config.json'), encoding='UTF-8') as file:
        config = json.load(file)
    with open(os.path.join(directory, 'params.msgpack'), 'rb') as file:
        params = serialization.msgpack_restore(file.read())
    if 'algo' in config:  # value-factorization baseline
        from mate_jax.value_marl import ValueConfig, ValueTrainer  # pylint: disable=import-outside-toplevel

        fields = ValueConfig().to_dict()
        trainer = ValueTrainer(ValueConfig(**{k: v for k, v in config.items() if k in fields}))
        transform = trainer.team_env.transform

        def decide(key, observation):  # pylint: disable=unused-argument
            q = trainer.utilities(params, transform(observation))
            return trainer.env_action(jnp.argmax(q, axis=-1))

        return config['algo'], config['opponent'], config['seed'], trainer.num_agents, config['frame_skip'], decide
    config.update(num_devices=1)
    trainer = MAPPOTrainer(MAPPOConfig(**config))
    actor = params['actor']
    transform = trainer.team_env.transform
    name = 'ippo' if config['critic'] == 'local' else 'mappo'

    def decide(key, observation):  # pylint: disable=unused-argument
        out, log_std = trainer.actor.apply(actor, trainer.actor_input(transform(observation)))
        return trainer.env_action(trainer.distribution.mode(out, log_std))

    return name, config['opponent'], config['seed'], trainer.num_agents, config['frame_skip'], decide


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', default='MATE-4v8-9.yaml')
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--num-envs', type=int, default=32)
    parser.add_argument('--num-steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    env = mate_jax.make(args.config)
    results = []
    for directory in sorted(args.runs):
        if not os.path.exists(os.path.join(directory, 'params.msgpack')):
            continue
        start = time.perf_counter()
        algo, targets, seed, num_agents, frame_skip, decide = load_policy(directory)
        policy = FrameSkipPolicy(decide, num_agents, frame_skip)
        metrics = evaluate_coverage(
            env, policy, OPPONENTS[('camera', targets)].from_env(env), seed=args.seed, num_envs=args.num_envs,
            num_steps=args.num_steps,
        )
        coverage = np.asarray(metrics['coverage_rate'])
        row = {
            'config': args.config, 'targets': targets, 'variant': algo, 'run': os.path.basename(os.path.normpath(directory)),
            'train_seed': seed, 'coverage': float(coverage.mean()),
            'coverage_se': float(coverage.std(ddof=1) / np.sqrt(coverage.size)),
            'per_episode_coverage': coverage.tolist(), 'real_coverage': float(np.mean(metrics['real_coverage_rate'])),
            'per_camera': {k: np.asarray(metrics[k]).mean(0).tolist() for k in ('camera_coverage', 'camera_unique_coverage', 'camera_rotation', 'camera_zoom')},
            'redundancy': float(np.mean(metrics['redundancy'])), 'seconds': time.perf_counter() - start,
        }
        results.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ('per_episode_coverage', 'per_camera')}), flush=True)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, 'w', encoding='UTF-8') as file:
                json.dump(results, file, indent=2)


if __name__ == '__main__':
    main()
