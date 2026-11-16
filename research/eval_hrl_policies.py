"""Evaluate saved FORESIGHT-HRL (and flat MAPPO / IPPO) policies with the protocol of ``eval_hrl_rules.py``.

Each run directory holds ``params.msgpack`` and ``config.json`` written by ``mate_jax.train --save-dir``.
The high-level environment is rebuilt from the run's configuration with the evaluation horizon, so the
policies see the same target trajectories as the rule-based high levels (16 episodes x 2500 steps, seed 1).
Usage: python3 research/eval_hrl_policies.py --runs results/policies/hrl_* --out results/hrl_policies_4v8_h500.json
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flax import serialization  # noqa: E402

from mate_jax.foresight.hrl import evaluate_actions  # noqa: E402
from mate_jax.mappo import MAPPOConfig, MAPPOTrainer  # noqa: E402


def load_run(directory, horizon):
    with open(os.path.join(directory, 'config.json'), encoding='UTF-8') as file:
        config = json.load(file)
    config['horizon'] = horizon
    config['num_devices'] = 1
    trainer = MAPPOTrainer(MAPPOConfig(**config))
    with open(os.path.join(directory, 'params.msgpack'), 'rb') as file:
        params = serialization.msgpack_restore(file.read())['actor']
    return config, trainer, params


def policy_select(trainer, params, deterministic):
    dist = trainer.distribution

    def select(key, obs, state):  # pylint: disable=unused-argument
        logits, log_std = trainer.actor.apply(params, trainer.actor_input(obs))
        action = dist.mode(logits, log_std) if deterministic else dist.sample(key, logits, log_std)
        return trainer.env_action(action)  # identity for FORESIGHT-HRL, pan/zoom commands for flat learners

    return select


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--runs', nargs='+', required=True, help='run directories (--save-dir of mate_jax.train)')
    parser.add_argument('--modes', nargs='+', default=['deterministic'], choices=['deterministic', 'stochastic'])
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--env-steps', type=int, default=2500)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    results = []
    for directory in args.runs:
        name = os.path.basename(os.path.normpath(directory))
        with open(os.path.join(directory, 'config.json'), encoding='UTF-8') as file:
            num_steps = args.env_steps // json.load(file)['frame_skip']
        config, trainer, params = load_run(directory, horizon=num_steps)
        for mode in args.modes:
            start = time.perf_counter()
            metrics = evaluate_actions(
                trainer.team_env,
                policy_select(trainer, params, deterministic=mode == 'deterministic'),
                num_envs=args.num_envs,
                num_steps=num_steps,
                seed=args.seed,
            )
            row = {
                'run': name, 'targets': config['opponent'], 'critic': config['critic'], 'seed': config['seed'],
                'mode': mode, **metrics, 'seconds': time.perf_counter() - start,
            }
            results.append(row)
            print(json.dumps(row), flush=True)
            if args.out:  # write after every run, so that partial results survive an interruption
                os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
                with open(args.out, 'w', encoding='UTF-8') as file:
                    json.dump(results, file, indent=2)


if __name__ == '__main__':
    main()
