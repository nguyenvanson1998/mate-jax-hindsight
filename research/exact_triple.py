"""Exact clairvoyant optimum for three cameras, to check the hindsight and Lagrangian bounds with more overlap.

The joint dynamic program of ``exact_pair.py`` grows as (number of poses)^3, which is far too large on the
lattice of the paper (72 x 62 poses per camera). We therefore use a coarser variant of the benchmark, in which the
cameras turn and zoom by larger steps (``--env-json``, e.g. 20 degrees and 10 degrees per step: 18 x 17 poses per
camera). Its lattice is coarser, but the bounds are computed on the same lattice, so the check is exact for this
variant: on the recorded episodes we compare, per episode,

    L   = block-coordinate ascent (the hindsight bound of clairvoyant.py, restricted to the three cameras),
    OPT = exact joint optimum over all joint lattice schedules (this file),
    U   = Lagrangian bound (clairvoyant.py, restricted to the three cameras),

and check L <= OPT <= U.

Usage: python3 research/exact_triple.py --config MATE-4v8-9.yaml --targets greedy waypoint --triples 0,1,2 \
           --env-json '{"camera": {"rotation_step": 20.0, "zooming_step": 10.0}}' \
           --num-envs 32 --num-steps 1000 --key-steps 1000 --dual-iterations 150 --out results/paper/exact_triple_4v8.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import clairvoyant as C  # noqa: E402
import exact_pair as P  # noqa: E402
import mate_jax  # noqa: E402
from mate_jax.constants import MAX_CAMERA_VIEWING_ANGLE  # noqa: E402
from mate_jax.foresight import ForesightCameraPolicy  # noqa: E402
from mate_jax.geometry import arctan2_deg, line_of_sight, normalize_angle, safe_norm  # noqa: E402


NEG = -1e9


def make_triple_dp(env, num_steps):
    """Compiled ``exact(location (3, 2), theta0 (3,), view0 (3,), targets (S, T, 2), obstacles, mute) -> value``:
    the largest number of covered target-steps of three cameras over all joint lattice schedules. ``mute`` (3,)
    switches cameras off (used by the self-test)."""

    cfg = env.config
    rotation, zooming = cfg.camera_rotation_step, cfg.camera_zooming_step
    num_orientations = int(round(360.0 / rotation))
    num_views = int((MAX_CAMERA_VIEWING_ANGLE - cfg.camera_min_viewing_angle) / zooming) + 2
    num_poses = num_orientations * num_views

    def lattice(location, theta0, view0, targets, obstacle_location, obstacle_radius):
        orientation = theta0 + rotation * jnp.arange(num_orientations)
        j0 = jnp.floor((view0 - cfg.camera_min_viewing_angle) / zooming + 1e-6).astype(jnp.int32)
        viewing = view0 + zooming * (jnp.arange(num_views) - j0)
        valid = (viewing >= cfg.camera_min_viewing_angle - 1e-6) & (viewing <= MAX_CAMERA_VIEWING_ANGLE + 1e-6)
        sight = jnp.sqrt(env.camera_area_product / jnp.maximum(viewing, 1e-3))
        relative = targets - location
        distance = safe_norm(relative)
        angle = arctan2_deg(relative[..., 1], relative[..., 0])
        visible = jax.vmap(
            lambda d, p: line_of_sight(location[None, None, :], p[None], d[None], obstacle_location, obstacle_radius)[0]
        )(distance, targets)

        def covers(t):  # (O * V, T) float
            offset = jnp.abs(normalize_angle(angle[t][None, :] - orientation[:, None]))
            mask = (
                (2.0 * offset[:, None, :] <= viewing[None, :, None])
                & (distance[t][None, None, :] <= sight[None, :, None])
                & visible[t][None, None, :]
            )
            return mask.reshape(num_poses, -1).astype(jnp.float32)

        return covers, valid, j0

    def best_predecessor(value, o_axis, v_axis):
        """Maximum over the 9 lattice moves of one camera (orientation wraps, viewing angle is bounded)."""

        pad = [(0, 0)] * value.ndim
        pad[v_axis] = (1, 1)
        padded = jnp.pad(value, pad, constant_values=NEG)
        best = jnp.full_like(value, NEG)
        for do in (-1, 0, 1):
            rolled = jnp.roll(padded, -do, axis=o_axis)
            for dv in (-1, 0, 1):
                index = [slice(None)] * value.ndim
                index[v_axis] = slice(1 + dv, 1 + dv + num_views)
                best = jnp.maximum(best, rolled[tuple(index)])
        return best

    def exact(location, theta0, view0, targets, obstacle_location, obstacle_radius, mute):
        parts = [lattice(location[k], theta0[k], view0[k], targets, obstacle_location, obstacle_radius) for k in range(3)]
        (cov_a, valid_a, j0_a), (cov_b, valid_b, j0_b), (cov_c, valid_c, j0_c) = parts
        shape = (num_orientations, num_views) * 3
        valid = (valid_a[None, :, None, None, None, None] & valid_b[None, None, None, :, None, None]
                 & valid_c[None, None, None, None, None, :])
        start = jnp.full(shape, NEG, jnp.float32).at[0, j0_a, 0, j0_b, 0, j0_c].set(0.0)

        def forward(value, t):
            best = best_predecessor(best_predecessor(best_predecessor(value, 0, 1), 2, 3), 4, 5)
            a, b, c = cov_a(t) * (1.0 - mute[0]), cov_b(t) * (1.0 - mute[1]), cov_c(t) * (1.0 - mute[2])
            # targets covered by A or B or C, by inclusion and exclusion (coverage indicators are 0 or 1)
            gain = (a.sum(-1)[:, None, None] + b.sum(-1)[None, :, None] + c.sum(-1)[None, None, :]
                    - (a @ b.T)[:, :, None] - (a @ c.T)[:, None, :] - (b @ c.T)[None, :, :]
                    + jnp.einsum('pj,qj,rj->pqr', a, b, c))
            return jnp.where(valid, best + gain.reshape(shape), NEG), None

        final, _ = jax.lax.scan(forward, start, jnp.arange(num_steps))
        return final.max()

    print(f'lattice: {num_orientations} x {num_views} = {num_poses} poses per camera, {num_poses ** 3:.2e} joint states',
          flush=True)
    return jax.jit(exact)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', default='MATE-4v8-9.yaml')
    parser.add_argument('--targets', nargs='+', default=['greedy', 'waypoint'])
    parser.add_argument('--triples', nargs='+', default=['0,1,2'], help='camera triples, e.g. 0,1,2 1,2,3')
    parser.add_argument('--env-json', default='{"camera": {"rotation_step": 20.0, "zooming_step": 10.0}}',
                        help='environment overrides; the default coarsens the pose lattice')
    parser.add_argument('--num-envs', type=int, default=32)
    parser.add_argument('--num-steps', type=int, default=1000)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--key-steps', type=int, default=None, help='as in clairvoyant.py')
    parser.add_argument('--dual-iterations', type=int, default=150)
    parser.add_argument('--self-test', action='store_true', help='check the joint program against exact_pair.py')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    env = mate_jax.make(args.config, **json.loads(args.env_json))
    planner = ForesightCameraPolicy.from_env(env)
    record_steps = args.key_steps or args.num_steps + int(np.max(planner.horizons)) + 1
    C.RECORD_STEPS[0] = record_steps
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.num_envs)
    dp = C.make_camera_dp(env, args.num_steps)
    exact = make_triple_dp(env, args.num_steps)
    scale = args.num_steps * env.num_targets
    results = []
    for targets in args.targets:
        state, after = C.record(env, C.TARGETS[targets].from_env(env), keys, record_steps)
        after = after[:, : args.num_steps]
        for triple in args.triples:
            triple = tuple(int(x) for x in triple.split(','))
            sub = P.pair_state(state, triple)
            start = time.perf_counter()
            if args.self_test:  # camera C muted: the triple program must equal the exact pair program
                pair = P.make_pair_dp(env, args.num_steps)
                for e in range(min(args.num_envs, 2)):
                    joint = float(exact(sub.camera_location[e], sub.camera_orientation[e], sub.camera_viewing_angle[e],
                                        after[e], sub.obstacle_location[e], sub.obstacle_radius[e], jnp.array([0.0, 0.0, 1.0])))
                    two = float(pair(sub.camera_location[e][:2], sub.camera_orientation[e][:2], sub.camera_viewing_angle[e][:2],
                                     after[e], sub.obstacle_location[e], sub.obstacle_radius[e], 0.0))
                    print(f'self-test {targets} {triple} episode {e}: triple with C muted {joint}, pair {two}', flush=True)
                    assert abs(joint - two) < 1e-3
            lower = P.block_coordinate(dp, sub, after, args.rounds)
            optimum = []
            for e in range(args.num_envs):
                optimum.append(float(exact(sub.camera_location[e], sub.camera_orientation[e], sub.camera_viewing_angle[e],
                                           after[e], sub.obstacle_location[e], sub.obstacle_radius[e], jnp.zeros(3))))
                if e == 0:
                    print(f'  first episode done after {time.perf_counter() - start:.0f} s', flush=True)
            optimum = np.asarray(optimum, np.float64)
            row = {
                'config': args.config, 'env_json': args.env_json, 'targets': targets, 'triple': list(triple),
                'episodes': args.num_envs, 'num_steps': args.num_steps,
                'lower': (lower / scale).tolist(), 'optimum': (optimum / scale).tolist(),
            }
            if args.dual_iterations:
                upper, _ = C.dual_bound(dp, sub, after, lower, args.dual_iterations)
                row['upper'] = np.asarray(upper, np.float64).tolist()
            row['seconds'] = time.perf_counter() - start
            results.append(row)
            lo, op = np.asarray(row['lower']), np.asarray(row['optimum'])
            message = (f'{args.config} {targets} triple {triple}: L {lo.mean():.4f}  OPT {op.mean():.4f}  '
                       f'OPT-L mean {np.mean(op - lo):.4f} max {np.max(op - lo):.4f} min {np.min(op - lo):+.5f}')
            if 'upper' in row:
                up = np.asarray(row['upper'])
                message += f'  U {up.mean():.4f}  U-OPT mean {np.mean(up - op):.4f} max {np.max(up - op):.4f} min {np.min(up - op):+.5f}'
            print(message + f'  ({row["seconds"]:.0f} s)', flush=True)
            if args.out:
                os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
                with open(args.out, 'w', encoding='UTF-8') as file:
                    json.dump(results, file, indent=2)


if __name__ == '__main__':
    main()
