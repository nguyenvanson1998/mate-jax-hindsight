"""Coverage of the initial (untrained) policies of MARL runs, with the evaluation used during their training.

The parameters at step 0 are a deterministic function of the run's seed, so this adds the missing first point of
every learning curve. Usage: python3 research/eval_init.py --runs results/policies_final/* --out init.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mate_jax.mappo import MAPPOConfig, MAPPOTrainer  # noqa: E402


def initial_coverage(directory):
    with open(os.path.join(directory, 'config.json'), encoding='UTF-8') as file:
        config = json.load(file)
    if 'algo' in config:  # value-factorization baseline: same evaluation as ValueTrainer.train
        from mate_jax.value_marl import ValueConfig, ValueTrainer  # pylint: disable=import-outside-toplevel

        fields = ValueConfig().to_dict()
        trainer = ValueTrainer(ValueConfig(**{k: v for k, v in config.items() if k in fields}))
        return trainer.evaluate(trainer.init(config['seed']).params)
    config.update(num_devices=1)
    trainer = MAPPOTrainer(MAPPOConfig(**config))
    cfg = trainer.config
    params = trainer.init(cfg.seed).actor_state.params  # same evaluation as MAPPOTrainer's training loop
    return trainer.evaluate(params, num_envs=cfg.eval_envs, num_steps=max(1, cfg.eval_env_steps // cfg.frame_skip))['coverage_rate']


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    results = {}
    for directory in args.runs:
        name = os.path.basename(os.path.normpath(directory))
        results[name] = initial_coverage(directory)
        print(name, round(results[name], 4), flush=True)
        with open(args.out, 'w', encoding='UTF-8') as file:
            json.dump(results, file, indent=2)


if __name__ == '__main__':
    main()
