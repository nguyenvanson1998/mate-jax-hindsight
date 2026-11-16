"""Multi-step parity of MATE-JAX with the original NumPy MATE (requires ``gym`` and ``mate``).

Trace replay (open loop): the original environment runs in closed loop with its rule-based agents
(``GreedyCameraAgent``, ``GreedyTargetAgent``). MATE-JAX starts from the same initial state and receives the
recorded joint actions, without any re-synchronization. After 1, 10, 100 and 1000 steps we report the largest
deviation of the target positions and camera poses, the share of camera-target visibilities that differ, and
the difference of the coverage rate.

Closed loop: MATE-JAX runs its own ports of the rule-based agents with its own randomness; the coverage of its
episodes is compared with the coverage of the original episodes above (two independent samples).

Usage (from the repository root):
    python3 research/parity_multistep.py --configs MATE-4v8-9.yaml --episodes 16 --out results/paper/parity.json
"""

import argparse
import contextlib
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

# The original MATE uses NumPy aliases that NumPy 2 removed.
for _alias, _type in (('bool8', np.bool_), ('float_', np.float64), ('int_', np.int64), ('complex_', np.complex128)):
    if not hasattr(np, _alias):
        setattr(np, _alias, _type)


def trace_replay(config, transmittance, episodes, num_steps, x64, horizons, actions='greedy'):
    import gym
    import jax
    import jax.experimental

    import mate
    import mate_jax
    from mate.agents import GreedyCameraAgent, GreedyTargetAgent
    from mate_jax.bridge import state_from_numpy_env

    gym.logger.set_level(40)
    overrides = {'obstacle': {'transmittance': float(transmittance)}}
    np_env = mate.MultiAgentTracking(config, **overrides)
    records = []  # per episode: arrays over steps
    enable_x64 = jax.enable_x64 if hasattr(jax, 'enable_x64') else jax.experimental.enable_x64  # moved in recent JAX
    with (enable_x64(True) if x64 else contextlib.nullcontext()):
        env = mate_jax.make(config, **overrides)
        step = jax.jit(env.step)
        for episode in range(episodes):
            # one agent object per entity (``spawn`` deep-copies a generator, which fails with recent NumPy)
            camera_agents = [GreedyCameraAgent(seed=1000 * episode + i) for i in range(np_env.num_cameras)]
            target_agents = [GreedyTargetAgent(seed=1000 * episode + 500 + i) for i in range(np_env.num_targets)]
            camera_obs, target_obs = np_env.reset(seed=episode)
            mate.group_reset(camera_agents, camera_obs)
            mate.group_reset(target_agents, target_obs)
            camera_infos = target_infos = None
            state = state_from_numpy_env(env, np_env)
            rng = np.random.default_rng(10007 * episode + 17)  # random actions (``--actions random``)
            cam_space, tgt_space = np_env.camera_action_space, np_env.target_action_space
            rec = {k: [] for k in ('position', 'orientation', 'viewing', 'mismatch', 'coverage_np', 'coverage_jax')}
            for t in range(num_steps):
                if actions == 'random':  # uniform over the action boxes: exercises poses and collisions the agents avoid
                    camera_action = rng.uniform(cam_space.low, cam_space.high, (np_env.num_cameras,) + cam_space.shape)
                    target_action = rng.uniform(tgt_space.low, tgt_space.high, (np_env.num_targets,) + tgt_space.shape)
                else:
                    camera_action = np.asarray(mate.group_step(np_env, camera_agents, camera_obs, camera_infos), dtype=np.float64)
                    target_action = np.asarray(mate.group_step(np_env, target_agents, target_obs, target_infos), dtype=np.float64)
                (camera_obs, target_obs), _, done, (camera_infos, target_infos) = np_env.step((camera_action, target_action))
                _, state = step(jax.random.PRNGKey(episode * 100003 + t), state, camera_action, target_action)

                positions = np.asarray([x.location for x in np_env.targets])
                orientation = np.asarray([c.orientation for c in np_env.cameras])
                viewing = np.asarray([c.viewing_angle for c in np_env.cameras])
                d_orientation = np.abs(np.asarray(state.camera_orientation) - orientation)
                rec['position'].append(float(np.abs(np.asarray(state.target_location) - positions).max()))
                rec['orientation'].append(float(np.minimum(d_orientation, 360.0 - d_orientation).max()))
                rec['viewing'].append(float(np.abs(np.asarray(state.camera_viewing_angle) - viewing).max()))
                rec['mismatch'].append(float((np.asarray(state.camera_target_view_mask) != np_env.camera_target_view_mask).mean()))
                rec['coverage_np'].append(float(np.asarray(np_env.tracked_bits, dtype=bool).mean()))
                rec['coverage_jax'].append(float(np.asarray(state.tracked_bits).mean()))
                if done:
                    break
            records.append({k: np.asarray(v) for k, v in rec.items()})

    rows = []
    for horizon in horizons:
        usable = [r for r in records if r['position'].size >= horizon]
        if not usable:
            continue
        diff = np.asarray([r['coverage_jax'][:horizon].mean() - r['coverage_np'][:horizon].mean() for r in usable])
        rows.append({
            'config': config, 'transmittance': transmittance, 'precision': 'float64' if x64 else 'float32', 'actions': actions,
            'steps': horizon, 'episodes': len(usable),
            'max_position_error': float(max(r['position'][:horizon].max() for r in usable)),
            'max_orientation_error': float(max(r['orientation'][:horizon].max() for r in usable)),
            'max_viewing_error': float(max(r['viewing'][:horizon].max() for r in usable)),
            'visibility_mismatch_rate': float(np.mean([r['mismatch'][:horizon].mean() for r in usable])),
            'coverage_difference': float(diff.mean()),
            'coverage_difference_se': float(diff.std(ddof=1) / np.sqrt(diff.size)) if diff.size > 1 else 0.0,
            'coverage_difference_max_abs': float(np.abs(diff).max()),
            'coverage_numpy': float(np.mean([r['coverage_np'][:horizon].mean() for r in usable])),
        })
    numpy_coverage = [float(r['coverage_np'].mean()) for r in records if r['position'].size >= num_steps]
    lengths = [int(r['position'].size) for r in records]
    return rows, numpy_coverage, lengths


def closed_loop_jax(config, transmittance, episodes, num_steps):
    import mate_jax
    from mate_jax import agents as A
    from mate_jax.evaluation import evaluate_coverage

    env = mate_jax.make(config, obstacle={'transmittance': float(transmittance)})
    metrics = evaluate_coverage(
        env, A.GreedyCameraPolicy.from_env(env), A.GreedyTargetPolicy.from_env(env), seed=0, num_envs=episodes, num_steps=num_steps
    )
    return [float(x) for x in np.asarray(metrics['coverage_rate'])]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--configs', nargs='+', default=['MATE-4v8-9.yaml'])
    parser.add_argument('--transmittances', type=float, nargs='+', default=[0.0, 0.1])
    parser.add_argument('--episodes', type=int, default=16)
    parser.add_argument('--num-steps', type=int, default=1000)
    parser.add_argument('--horizons', type=int, nargs='+', default=[1, 10, 100, 1000])
    parser.add_argument('--precisions', nargs='+', default=['float64', 'float32'], choices=['float64', 'float32'])
    parser.add_argument('--closed-loop-episodes', type=int, default=32)
    parser.add_argument('--actions', default='greedy', choices=['greedy', 'random'],
                        help='replayed actions: those of the original rule-based agents, or uniform random actions')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    result = {'trace_replay': [], 'closed_loop': []}
    for config in args.configs:
        for transmittance in args.transmittances:
            numpy_coverage = None
            for precision in args.precisions:
                start = time.perf_counter()
                rows, coverage, lengths = trace_replay(
                    config, transmittance, args.episodes, args.num_steps, precision == 'float64', args.horizons, args.actions
                )
                numpy_coverage = coverage
                result['trace_replay'] += rows
                for row in rows:
                    print(
                        f'{config} tau={transmittance} {precision} K={row["steps"]:5d} (n={row["episodes"]}): '
                        f'pos {row["max_position_error"]:.2e} m, orient {row["max_orientation_error"]:.2e} deg, '
                        f'view {row["max_viewing_error"]:.2e} deg, mask mismatch {row["visibility_mismatch_rate"]:.2e}, '
                        f'coverage diff {row["coverage_difference"]:+.5f} +- {row["coverage_difference_se"]:.5f} '
                        f'(max |.| {row["coverage_difference_max_abs"]:.5f})', flush=True,
                    )
                print(f'  episode lengths: min {min(lengths)}, {time.perf_counter() - start:.0f} s', flush=True)
            if args.closed_loop_episodes > 0:
                jax_coverage = closed_loop_jax(config, transmittance, args.closed_loop_episodes, args.num_steps)
                row = {
                    'config': config, 'transmittance': transmittance, 'numpy_coverage': numpy_coverage,
                    'jax_coverage': jax_coverage,
                }
                result['closed_loop'].append(row)
                a, b = np.asarray(numpy_coverage), np.asarray(jax_coverage)
                if a.size > 1:
                    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
                    print(
                        f'{config} tau={transmittance} closed loop: NumPy {a.mean():.4f} (n={a.size}), '
                        f'JAX {b.mean():.4f} (n={b.size}), difference {b.mean() - a.mean():+.4f} +- {se:.4f}', flush=True,
                    )
            if args.out:
                os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
                with open(args.out, 'w', encoding='UTF-8') as file:
                    json.dump(result, file, indent=2)


if __name__ == '__main__':
    main()
