"""Train an off-policy value-factorization baseline (VDN / QMIX / QPLEX / ...) on the camera team.

Usage: python3 research/train_value.py --algo qmix --opponent greedy --seed 0 --log results/curves_value/qmix_greedy_s0.jsonl
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax  # noqa: E402
from flax import serialization  # noqa: E402

from mate_jax.value_marl import ValueConfig, ValueTrainer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    defaults = ValueConfig()
    for field, value in defaults.to_dict().items():
        flag = '--' + field.replace('_', '-')
        if isinstance(value, list):
            parser.add_argument(flag, type=int, nargs='+', default=value)
        elif isinstance(value, bool):
            parser.add_argument(flag, type=lambda s: s.lower() in ('1', 'true', 'yes'), default=value)
        else:
            parser.add_argument(flag, type=type(value), default=value)
    parser.add_argument('--log', default=None)
    parser.add_argument('--save-dir', default=None)
    args = parser.parse_args()
    config = ValueConfig(**{k: v for k, v in vars(args).items() if k in defaults.to_dict()})
    print(f'{jax.default_backend()} backend; {config.algo} vs {config.opponent} targets, seed {config.seed}', flush=True)
    if args.log:
        os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    trainer = ValueTrainer(config)
    runner = trainer.train(log_file=args.log)
    final = {'eval_coverage_final': trainer.evaluate(runner.params, num_envs=16, num_steps=200)}
    print(json.dumps(final), flush=True)
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, 'params.msgpack'), 'wb') as file:
            file.write(serialization.to_bytes(jax.device_get(runner.params)))
        with open(os.path.join(args.save_dir, 'config.json'), 'w', encoding='UTF-8') as file:
            json.dump({**config.to_dict(), **final}, file, indent=2)


if __name__ == '__main__':
    main()
