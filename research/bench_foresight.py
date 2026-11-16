"""Throughput of the FORESIGHT planner (and of greedy cameras) in closed loop with the environment.

Usage (from the repository root):
    python3 research/bench_foresight.py --config MATE-4v8-9.yaml --batch-sizes 16 64 256 --num-steps 200
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mate_jax  # noqa: E402
from mate_jax import agents as A  # noqa: E402
from mate_jax.evaluation import make_coverage_evaluator  # noqa: E402
from mate_jax.foresight import ForesightCameraPolicy  # noqa: E402
from mate_jax.sharding import describe_devices  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', default='MATE-4v8-9.yaml')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[16, 64, 256])
    parser.add_argument('--num-steps', type=int, default=200)
    parser.add_argument('--variants', nargs='+', default=['greedy', 'foresight', 'hybrid'])
    parser.add_argument('--repeats', type=int, default=1, help='timed runs after the compilation run (median reported)')
    parser.add_argument('--out', default=None, help='JSON file for the measurements')
    args = parser.parse_args()

    print(describe_devices(), flush=True)
    env = mate_jax.make(args.config)
    targets = A.GreedyTargetPolicy.from_env(env)
    cameras = {
        'greedy': A.GreedyCameraPolicy.from_env(env),
        'foresight': ForesightCameraPolicy.from_env(env),
        'refine': ForesightCameraPolicy.from_env(env, refine=True),
        'hybrid': ForesightCameraPolicy.from_env(env, communication='hybrid', message_drop=0.5),
    }
    levels = tuple(float(v) for v in range(30, 181, 5))
    for rounds in (1, 2, 3):
        cameras[f'traj-{rounds}'] = ForesightCameraPolicy.from_env(
            env, objective='trajectory', levels=levels, coordination_rounds=rounds
        )
    cameras['pub-4'] = ForesightCameraPolicy.from_env(
        env, objective='trajectory', levels=levels, coordination='published', publish_period=4
    )
    cameras['pub-4-hybrid'] = ForesightCameraPolicy.from_env(
        env, objective='trajectory', levels=levels, coordination='published', publish_period=4, communication='hybrid'
    )
    rows = []
    for variant in args.variants:
        for batch in args.batch_sizes:
            evaluate = make_coverage_evaluator(env, cameras[variant], targets, num_envs=batch, num_steps=args.num_steps)
            start = time.perf_counter()
            evaluate(0)
            first = time.perf_counter() - start  # includes compilation
            runs = []
            for repeat in range(args.repeats):  # different episodes; the compiled function is reused
                start = time.perf_counter()
                metrics = evaluate(1 + repeat)
                runs.append(time.perf_counter() - start)
            elapsed = sorted(runs)[len(runs) // 2] if len(runs) % 2 else sum(sorted(runs)[len(runs) // 2 - 1:len(runs) // 2 + 1]) / 2
            print(
                f'{variant:>10s} B={batch:>5d}: {batch * args.num_steps / elapsed:10,.0f} env steps/s '
                f'({elapsed / args.num_steps * 1e3:7.2f} ms per team decision step, compile+run {first:5.1f}s, '
                f'coverage {metrics["coverage_rate"].mean():.3f})',
                flush=True,
            )
            rows.append({
                'config': args.config, 'variant': variant, 'batch': batch, 'env_steps_per_second': batch * args.num_steps / elapsed,
                'ms_per_step': elapsed / args.num_steps * 1e3, 'compile_and_run_seconds': first,
                'ms_per_step_runs': [run / args.num_steps * 1e3 for run in runs],
            })
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='UTF-8') as file:
            json.dump(rows, file, indent=2)


if __name__ == '__main__':
    main()
