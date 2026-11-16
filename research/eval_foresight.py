"""Evaluate the FORESIGHT planner and its ablations against greedy cameras.

Usage (from the repository root):
    python3 research/eval_foresight.py --configs MATE-4v8-9.yaml MATE-8v8-9.yaml \
        --targets greedy waypoint --num-envs 32 --num-steps 1000 --out results/foresight.json
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import mate_jax  # noqa: E402
from mate_jax import agents as A  # noqa: E402
from mate_jax.evaluation import evaluate_coverage  # noqa: E402
from mate_jax.foresight import ForesightCameraPolicy  # noqa: E402
from mate_jax.hindsight.forecaster import LearnedForecastPlanner, load_forecaster  # noqa: E402


TRAJECTORY_LEVELS = tuple(float(v) for v in range(30, 181, 5))
VARIANTS = {
    'greedy': None,  # mate.GreedyCameraAgent (JAX port)
    'foresight': {},  # path-integrated objective, gamma = 0.95
    'window': {'objective': 'window'},
    'no-forecast': {'use_forecast': False},
    'no-reach': {'reach_aware': False},
    'no-negative-info': {'negative_information': False},
    'no-communication': {'communication': 'none'},
    'hybrid': {'communication': 'hybrid'},
    'implicit': {'communication': 'none', 'implicit': True},  # no packets: team plan imagined on the private belief
    'refine': {'refine': True},
    'point-forecast': {'sigma_walk_ratio': 0.0, 'sigma_velocity_known_ratio': 0.0, 'sigma_velocity_unknown_ratio': 0.0},
    # Trajectory-level coordination (Viterbi over the rate-limit lattice, 5-degree zoom levels) with R rounds.
    'traj-1': {'objective': 'trajectory', 'levels': TRAJECTORY_LEVELS},
    'traj-2': {'objective': 'trajectory', 'levels': TRAJECTORY_LEVELS, 'coordination_rounds': 2},
    'traj-3': {'objective': 'trajectory', 'levels': TRAJECTORY_LEVELS, 'coordination_rounds': 3},
    'traj-2-hybrid': {'objective': 'trajectory', 'levels': TRAJECTORY_LEVELS, 'coordination_rounds': 2, 'communication': 'hybrid'},
    'traj-2-none': {'objective': 'trajectory', 'levels': TRAJECTORY_LEVELS, 'coordination_rounds': 2, 'communication': 'none'},
    # Decentralized coordination through published trajectories (intent messages).
    'pub-always': {'objective': 'trajectory', 'levels': TRAJECTORY_LEVELS, 'coordination': 'published'},
    'pub-period-4': {'objective': 'trajectory', 'levels': TRAJECTORY_LEVELS, 'coordination': 'published', 'publish_period': 4},
    'pub-event-1': {'objective': 'trajectory', 'levels': TRAJECTORY_LEVELS, 'coordination': 'published', 'publish_threshold': 1.0},
    'pub-always-hybrid': {
        'objective': 'trajectory', 'levels': TRAJECTORY_LEVELS, 'coordination': 'published', 'communication': 'hybrid',
    },
    # ``forecaster``: a saved ForecastNet ('{targets}' is replaced by the target model).
    'traj-2-forecast': {
        'objective': 'trajectory', 'levels': TRAJECTORY_LEVELS, 'coordination_rounds': 2,
        'forecaster': 'results/forecaster_{targets}/forecaster.msgpack',
    },
}
TARGETS = {'greedy': A.GreedyTargetPolicy, 'waypoint': A.RandomWaypointTargetPolicy, 'random': A.RandomTargetPolicy}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--configs', nargs='+', default=['MATE-4v8-9.yaml', 'MATE-8v8-9.yaml'])
    parser.add_argument('--targets', nargs='+', default=['greedy', 'waypoint'], choices=sorted(TARGETS))
    parser.add_argument('--variants', nargs='+', default=None, help=f'subset of {list(VARIANTS)}')
    parser.add_argument(
        '--variant-json',
        default=None,
        help='extra variants as JSON, e.g. \'{"cum-0.9": {"discount": 0.9}}\' (ForesightCameraPolicy kwargs)',
    )
    parser.add_argument('--num-envs', type=int, default=32)
    parser.add_argument('--num-steps', type=int, default=1000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--grid-size', type=int, default=32)
    parser.add_argument('--reference', default=None, help='variant for paired differences (default: first)')
    parser.add_argument('--per-agent', action='store_true', help='print per-camera metrics')
    parser.add_argument('--env-json', default='{}', help='environment overrides, e.g. \'{"target": {"step_size": 30.0}}\'')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    variants = dict(VARIANTS)
    names = list(args.variants) if args.variants else []
    if args.variant_json:
        extra = json.loads(args.variant_json)
        variants.update(extra)
        names += [name for name in extra if name not in names]
    if not names:
        names = list(VARIANTS)

    reference = args.reference or names[0]
    if reference in names:  # evaluate the reference first
        names = [reference] + [name for name in names if name != reference]
    print('| config | targets | variant | coverage ± s.e. | paired Δ vs ' + reference + ' ± s.e. | real coverage | time |')
    print('| --- | --- | --- | ---: | ---: | ---: | ---: |')

    results = []
    for config in args.configs:
        env = mate_jax.make(config, **json.loads(args.env_json))
        for target in args.targets:
            target_policy = TARGETS[target].from_env(env)
            per_episode = {}
            for variant in names:
                kwargs = variants[variant]
                if kwargs is None:
                    camera_policy = A.GreedyCameraPolicy.from_env(env)
                else:
                    kwargs = dict(kwargs)
                    forecaster = kwargs.pop('forecaster', None)
                    if forecaster is None:
                        camera_policy = ForesightCameraPolicy.from_env(env, grid_size=args.grid_size, **kwargs)
                    else:
                        camera_policy = LearnedForecastPlanner.from_env(env, grid_size=args.grid_size, **kwargs)
                        camera_policy = camera_policy.replace(
                            forecaster_params=load_forecaster(forecaster.format(targets=target), camera_policy)
                        )
                coverage, real, agents = [], [], []
                start = time.perf_counter()
                for seed in args.seeds:
                    metrics = evaluate_coverage(
                        env, camera_policy, target_policy, seed=seed, num_envs=args.num_envs, num_steps=args.num_steps
                    )
                    coverage.append(metrics['coverage_rate'])
                    real.append(metrics['real_coverage_rate'])
                    agents.append(metrics)
                coverage, real = np.concatenate(coverage), np.concatenate(real)
                per_episode[variant] = coverage
                # Target trajectories do not depend on the cameras: same seeds give paired episodes.
                delta = coverage - per_episode.get(reference, coverage)
                row = {
                    'config': config,
                    'targets': target,
                    'variant': variant,
                    'coverage': float(coverage.mean()),
                    'coverage_se': float(coverage.std(ddof=1) / np.sqrt(coverage.size)),
                    'paired_delta': float(delta.mean()),
                    'paired_delta_se': float(delta.std(ddof=1) / np.sqrt(delta.size)),
                    'per_episode_coverage': coverage.tolist(),
                    'real_coverage': float(real.mean()),
                    'per_camera': {
                        key: np.concatenate([m[key] for m in agents]).mean(0).tolist()
                        for key in ('camera_coverage', 'camera_unique_coverage', 'camera_rotation', 'camera_zoom', 'camera_bytes')
                        if key in agents[0]
                    },
                    'redundancy': float(np.concatenate([m['redundancy'] for m in agents]).mean()),
                    'episodes': int(coverage.size),
                    'seconds': time.perf_counter() - start,
                }
                results.append(row)
                print(
                    f'| {config} | {target} | {variant} | {row["coverage"]:.3f} ± {row["coverage_se"]:.3f} '
                    f'| {row["paired_delta"]:+.3f} ± {row["paired_delta_se"]:.3f} '
                    f'| {row["real_coverage"]:.3f} | {row["seconds"]:.0f}s |',
                    flush=True,
                )
                if args.per_agent:
                    per_camera = row['per_camera']
                    print(
                        f'|   per camera: coverage {np.round(per_camera["camera_coverage"], 3).tolist()} '
                        f'unique {np.round(per_camera["camera_unique_coverage"], 3).tolist()} '
                        f'|rot| {np.round(per_camera["camera_rotation"], 2).tolist()} '
                        f'redundancy {row["redundancy"]:.3f} |',
                        flush=True,
                    )
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='UTF-8') as file:
            json.dump(results, file, indent=2)


if __name__ == '__main__':
    main()
