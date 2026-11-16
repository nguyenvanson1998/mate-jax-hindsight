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


def trace_replay(config, transmittance, episodes, num_steps, x64, horizons):
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
    with (jax.experimental.enable_x64() if x64 else contextlib.nullcontext()):
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
            rec = {k: [] for k in ('position', 'position_targets', 'orientation', 'viewing', 'mismatch', 'coverage_np', 'coverage_jax')}
            for t in range(num_steps):
                camera_action = np.asarray(mate.group_step(np_env, camera_agents, camera_obs, camera_infos), dtype=np.float64)
                target_action = np.asarray(mate.group_step(np_env, target_agents, target_obs, target_infos), dtype=np.float64)
                (camera_obs, target_obs), _, done, (camera_infos, target_infos) = np_env.step((camera_action, target_action))
                _, state = step(jax.random.PRNGKey(episode * 100003 + t), state, camera_action, target_action)

                positions = np.asarray([x.location for x in np_env.targets])
                orientation = np.asarray([c.orientation for c in np_env.cameras])
                viewing = np.asarray([c.viewing_angle for c in np_env.cameras])
                d_orientation = np.abs(np.asarray(state.camera_orientation) - orientation)
                error = np.linalg.norm(np.asarray(state.target_location) - positions, axis=-1)  # (T,) metres
                rec['position'].append(float(error.max()))
                rec['position_targets'].append(error)
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
            'config': config, 'transmittance': transmittance, 'precision': 'float64' if x64 else 'float32',
            'steps': horizon, 'episodes': len(usable),
            'max_position_error': float(max(r['position'][:horizon].max() for r in usable)),
            # share of target trajectories that leave the original one by more than 1 m, and the median final error
            'diverged_target_share': float(np.mean([(r['position_targets'][:horizon].max(0) > 1.0).mean() for r in usable])),
            'median_position_error': float(np.median(np.concatenate([r['position_targets'][horizon - 1] for r in usable]))),
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


def factorial_numpy_env(config, transmittance, episodes, num_steps):
    """Coverage per episode of the ORIGINAL environment when the cameras and / or the targets are driven by the
    JAX ports of the rule-based agents (fed with the original observations) instead of the original agents."""

    import gym
    import jax
    import jax.experimental
    import jax.numpy as jnp

    import mate
    import mate_jax
    from mate.agents import GreedyCameraAgent, GreedyTargetAgent
    from mate_jax import agents as A

    gym.logger.set_level(40)
    overrides = {'obstacle': {'transmittance': float(transmittance)}}
    np_env = mate.MultiAgentTracking(config, **overrides)
    result = {}
    with jax.experimental.enable_x64():
        env = mate_jax.make(config, **overrides)
        camera_policy, target_policy = A.GreedyCameraPolicy.from_env(env), A.GreedyTargetPolicy.from_env(env)
        camera_act, target_act = jax.jit(camera_policy.act), jax.jit(target_policy.act)
        for camera_kind in ('numpy', 'jax'):
            for target_kind in ('numpy', 'jax'):
                coverage = []
                for episode in range(episodes):
                    camera_agents = [GreedyCameraAgent(seed=1000 * episode + i) for i in range(np_env.num_cameras)]
                    target_agents = [GreedyTargetAgent(seed=1000 * episode + 500 + i) for i in range(np_env.num_targets)]
                    camera_obs, target_obs = np_env.reset(seed=episode)
                    mate.group_reset(camera_agents, camera_obs)
                    mate.group_reset(target_agents, target_obs)
                    camera_infos = target_infos = None
                    key = jax.random.PRNGKey(episode)
                    k_camera, k_target = jax.random.split(key)
                    camera_carry = camera_policy.init(k_camera, jnp.asarray(camera_obs))
                    target_carry = target_policy.init(k_target, jnp.asarray(target_obs))
                    tracked = []
                    for t in range(num_steps):
                        k_camera, k_target = jax.random.split(jax.random.fold_in(key, t + 1))
                        if camera_kind == 'numpy':
                            camera_action = mate.group_step(np_env, camera_agents, camera_obs, camera_infos)
                        else:
                            camera_action, camera_carry = camera_act(k_camera, camera_carry, jnp.asarray(camera_obs))
                        if target_kind == 'numpy':
                            target_action = mate.group_step(np_env, target_agents, target_obs, target_infos)
                        else:
                            target_action, target_carry = target_act(k_target, target_carry, jnp.asarray(target_obs))
                        (camera_obs, target_obs), _, done, (camera_infos, target_infos) = np_env.step(
                            (np.asarray(camera_action, np.float64), np.asarray(target_action, np.float64))
                        )
                        tracked.append(float(np.asarray(np_env.tracked_bits, dtype=bool).mean()))
                        if done:
                            break
                    if len(tracked) == num_steps:
                        coverage.append(float(np.mean(tracked)))
                result[f'cameras-{camera_kind}_targets-{target_kind}'] = coverage
                print(f'{config} tau={transmittance} original env, cameras {camera_kind:5s} targets {target_kind:5s}: '
                      f'{np.mean(coverage):.4f} +- {np.std(coverage, ddof=1) / np.sqrt(len(coverage)):.4f} (n={len(coverage)})', flush=True)
    return result


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
    parser.add_argument('--factorial-episodes', type=int, default=0, help='2 x 2 agent factorial on the original environment')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    result = {'trace_replay': [], 'closed_loop': [], 'factorial': []}
    for config in args.configs:
        for transmittance in args.transmittances:
            numpy_coverage = None
            for precision in args.precisions:
                start = time.perf_counter()
                rows, coverage, lengths = trace_replay(
                    config, transmittance, args.episodes, args.num_steps, precision == 'float64', args.horizons
                )
                numpy_coverage = coverage
                result['trace_replay'] += rows
                for row in rows:
                    print(
                        f'{config} tau={transmittance} {precision} K={row["steps"]:5d} (n={row["episodes"]}): '
                        f'pos {row["max_position_error"]:.2e} m (diverged {row["diverged_target_share"]:.4f}, median {row["median_position_error"]:.1e}), orient {row["max_orientation_error"]:.2e} deg, '
                        f'view {row["max_viewing_error"]:.2e} deg, mask mismatch {row["visibility_mismatch_rate"]:.2e}, '
                        f'coverage diff {row["coverage_difference"]:+.5f} +- {row["coverage_difference_se"]:.5f} '
                        f'(max |.| {row["coverage_difference_max_abs"]:.5f})', flush=True,
                    )
                print(f'  episode lengths: min {min(lengths)}, {time.perf_counter() - start:.0f} s', flush=True)
            if args.factorial_episodes > 0:
                result['factorial'].append({
                    'config': config, 'transmittance': transmittance,
                    **factorial_numpy_env(config, transmittance, args.factorial_episodes, args.num_steps),
                })
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
