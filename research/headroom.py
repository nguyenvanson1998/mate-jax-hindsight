"""How much coverage is left above FORESIGHT? Oracle variants of the same planner on paired episodes.

Target trajectories do not depend on the cameras, so a first pass with any camera team records them,
and a second pass gives the planner privileged information instead of its belief maps:

* ``foresight``: the planner with its belief maps (hybrid memory, ideal channel);
* ``oracle-now``: true current target positions, no forecast (perfect, instantaneous information);
* ``oracle-future``: true target positions at every planning horizon (clairvoyant);
* ``ceiling``: fraction of targets that some camera could see from some pose (range and occlusion only);
* ``teleport``: the best joint coverage if every camera could jump to any pose in every step (exact over
  72 orientations x 12 zoom levels, by dynamic programming over target subsets): an upper bound for any
  camera team up to the pose discretization, which isolates what the rate limits cost.

Usage: python3 research/headroom.py --targets greedy waypoint --out results/headroom_4v8.json
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
from flax import struct  # noqa: E402

import mate_jax  # noqa: E402
from mate_jax import agents as A  # noqa: E402
from mate_jax.foresight import ForesightCameraPolicy  # noqa: E402
from mate_jax.constants import MAX_CAMERA_VIEWING_ANGLE  # noqa: E402
from mate_jax.geometry import arctan2_deg, line_of_sight, normalize_angle, safe_norm  # noqa: E402


TARGETS = {'greedy': A.GreedyTargetPolicy, 'waypoint': A.RandomWaypointTargetPolicy}
RECORD_EXTRA = 51


@struct.dataclass
class OraclePlanner(ForesightCameraPolicy):
    """FORESIGHT whose planning counts are replaced by ``carry['oracle_counts']`` (C, H, N)."""

    def _planning_counts(self, carry):
        if 'oracle_counts' in carry:
            return carry['oracle_counts']
        return super()._planning_counts(carry)


def cell_counts(planner, positions):
    """(..., T, 2) positions -> (..., N) target counts per grid cell."""

    grid = planner.grid
    index = jnp.clip(jnp.floor((positions + 1000.0) / grid.cell).astype(jnp.int32), 0, grid.size - 1)
    flat = index[..., 0] * grid.size + index[..., 1]
    return jax.nn.one_hot(flat, grid.size * grid.size, dtype=jnp.float32).sum(-2)


def teleport_optimum(env, state, num_orientations=72, num_levels=12):
    """Fraction of targets covered by the best joint pose of all cameras (no rate limits, no transmittance)."""

    cfg = env.config
    num_targets = env.num_targets
    relative = state.target_location[None] - state.camera_location[:, None]  # (C, T, 2)
    distance = safe_norm(relative)
    angle = arctan2_deg(relative[..., 1], relative[..., 0])
    visible = line_of_sight(
        state.camera_location[:, None, :], state.target_location[None, :, :], distance,
        state.obstacle_location, state.obstacle_radius,
    )
    orientation = jnp.arange(num_orientations) * (360.0 / num_orientations) - 180.0  # (O,)
    viewing = jnp.linspace(cfg.camera_min_viewing_angle, MAX_CAMERA_VIEWING_ANGLE, num_levels)  # (V,)
    sight = jnp.sqrt(env.camera_area_product / viewing)
    offset = jnp.abs(normalize_angle(angle[:, None, None, :] - orientation[None, :, None, None]))  # (C, O, 1, T)
    covers = (
        (2.0 * offset <= viewing[None, None, :, None])
        & (distance[:, None, None, :] <= sight[None, None, :, None])
        & visible[:, None, None, :]
    )
    masks = jnp.sum(covers.astype(jnp.int32) << jnp.arange(num_targets), axis=-1).reshape(env.num_cameras, -1)
    subsets = jnp.arange(2**num_targets)
    reachable = subsets == 0
    for camera in range(env.num_cameras):
        union = subsets[:, None] | masks[camera][None, :]  # (2^T, P)
        reachable = jnp.zeros_like(reachable).at[union.ravel()].max(
            jnp.repeat(reachable, masks.shape[1])
        )
    popcount = jnp.sum((subsets[:, None] >> jnp.arange(num_targets)) & 1, axis=-1)
    return jnp.max(jnp.where(reachable, popcount, 0)) / num_targets


def blur_counts(planner, counts, sigma):
    """(..., N) cell counts smoothed by a normalized Gaussian of ``sigma`` cells (the total mass is kept)."""

    size = planner.grid.size
    radius = max(1, int(np.ceil(3.0 * sigma)))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel = jnp.asarray(kernel / kernel.sum(), jnp.float32)
    maps = counts.reshape(counts.shape[:-1] + (size, size))
    for axis in (-2, -1):
        pad = [(0, 0)] * maps.ndim
        pad[axis] = (radius, radius)
        padded = jnp.pad(maps, pad, mode='edge')
        maps = sum(kernel[k] * jnp.take(padded, jnp.arange(k, k + size), axis=axis) for k in range(2 * radius + 1))
    return maps.reshape(counts.shape)


def make_runner(env, target_policy, planner, mode, num_steps, record_steps, blur_sigma=0.0):
    """Compiled ``run(keys) -> metrics`` over a batch of episodes (same keys as ``evaluate_coverage``)."""

    horizons = np.asarray(planner.horizons)

    def episode(key, future):
        k_reset, k_camera, k_target, k_loop = jax.random.split(key, 4)
        (camera_obs, target_obs), state = env.reset(k_reset)
        camera_carry = planner.init(k_camera, camera_obs)
        if mode != 'foresight':
            camera_carry['oracle_counts'] = jnp.zeros(
                (env.num_cameras, horizons.size, planner.grid.size**2), jnp.float32
            )
        target_carry = target_policy.init(k_target, target_obs)
        max_range = env.config.camera_max_sight_range

        def body(carry, inputs):
            k, t = inputs
            state, camera_obs, target_obs, camera_carry, target_carry = carry
            k_camera, k_target, k_env = jax.random.split(k, 3)
            if mode == 'oracle-now':
                counts = cell_counts(planner, state.target_location)
                camera_carry['oracle_counts'] = jnp.broadcast_to(counts, camera_carry['oracle_counts'].shape)
            elif mode.startswith('oracle-future'):
                ahead = jnp.take(future, jnp.minimum(t + horizons, record_steps - 1), axis=0)  # (H, T, 2)
                counts = cell_counts(planner, ahead)
                if blur_sigma > 0.0:
                    counts = blur_counts(planner, counts, blur_sigma)
                camera_carry['oracle_counts'] = jnp.broadcast_to(counts, camera_carry['oracle_counts'].shape)
            camera_action, camera_carry = planner.act(k_camera, camera_carry, camera_obs)
            target_action, target_carry = target_policy.act(k_target, target_carry, target_obs)
            location = state.target_location
            timestep, state = env.step(k_env, state, camera_action, target_action)
            # Ceiling: some camera has the target within its maximum range and in line of sight.
            distance = safe_norm(state.target_location[None] - state.camera_location[:, None])
            visible = line_of_sight(
                state.camera_location[:, None, :], state.target_location[None, :, :], distance,
                state.obstacle_location, state.obstacle_radius,
            )
            ceiling = jnp.any((distance <= max_range) & visible, axis=0).astype(jnp.float32).mean()
            teleport = teleport_optimum(env, state) if mode == 'foresight' else jnp.zeros(())
            carry = (state, timestep.camera_observation, timestep.target_observation, camera_carry, target_carry)
            return carry, (timestep.info['coverage_rate'], ceiling, teleport, location)

        carry = (state, camera_obs, target_obs, camera_carry, target_carry)
        steps = jnp.arange(record_steps)
        _, (coverage, ceiling, teleport, location) = jax.lax.scan(
            body, carry, (jax.random.split(k_loop, record_steps), steps)
        )
        return coverage[:num_steps].mean(), ceiling[:num_steps].mean(), teleport[:num_steps].mean(), location

    return jax.jit(jax.vmap(episode))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', default='MATE-4v8-9.yaml')
    parser.add_argument('--targets', nargs='+', default=['greedy', 'waypoint'])
    parser.add_argument('--modes', nargs='+', default=['foresight', 'oracle-now', 'oracle-future'])
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--num-steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--key-steps', type=int, default=None,
                        help='episode key length (default num_steps + 51; num_steps pairs with eval_foresight.py)')
    parser.add_argument('--planner-json', default='{}', help='extra ForesightCameraPolicy keyword arguments')
    parser.add_argument('--blur-sigmas', type=float, nargs='*', default=[],
                        help='also run oracle-future with the true future cells blurred by these widths (cells)')
    parser.add_argument('--env-json', default='{}', help='environment overrides, e.g. \'{"target": {"step_size": 30.0}}\'')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    env = mate_jax.make(args.config, **json.loads(args.env_json))
    kwargs = json.loads(args.planner_json)
    planner = OraclePlanner.from_env(env, **kwargs)
    # Fixed length (that of the default 50-step lookahead), so that the episode keys, and hence the
    # target trajectories, are the same for every planner; longer lookaheads are clamped at the end.
    record_steps = args.key_steps or args.num_steps + RECORD_EXTRA
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.num_envs)
    results = []
    for targets in args.targets:
        target_policy = TARGETS[targets].from_env(env)
        future = None
        per_episode = {}
        modes = ['foresight'] + [m for m in args.modes if m != 'foresight'] + [f'oracle-future-blur-{s:g}' for s in args.blur_sigmas]
        for mode in modes:
            start = time.perf_counter()
            sigma = float(mode.rsplit('-', 1)[1]) if mode.startswith('oracle-future-blur-') else 0.0
            run = make_runner(env, target_policy, planner, mode, args.num_steps, record_steps, blur_sigma=sigma)
            dummy = jnp.zeros((args.num_envs, record_steps, env.num_targets, 2), jnp.float32)
            coverage, ceiling, teleport, location = jax.device_get(run(keys, dummy if future is None else future))
            if future is None:
                future = jnp.asarray(location)
            else:  # paired episodes: the targets must not depend on the cameras
                assert np.allclose(location, np.asarray(future), atol=1e-3), 'target trajectories differ'
            per_episode[mode] = coverage
            delta = coverage - per_episode['foresight']
            row = {
                'targets': targets, 'mode': mode, 'coverage': float(coverage.mean()),
                'coverage_se': float(coverage.std(ddof=1) / np.sqrt(coverage.size)),
                'delta_vs_foresight': float(delta.mean()),
                'delta_se': float(delta.std(ddof=1) / np.sqrt(delta.size)),
                'ceiling': float(ceiling.mean()), 'seconds': time.perf_counter() - start,
                'per_episode_coverage': np.asarray(coverage).tolist(),
                **({'teleport': float(teleport.mean())} if mode == 'foresight' else {}),
            }
            results.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != 'per_episode_coverage'}), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='UTF-8') as file:
            json.dump(results, file, indent=2)


if __name__ == '__main__':
    main()
