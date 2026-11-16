"""Exact clairvoyant optimum for two cameras, to verify the hindsight bound and the Lagrangian bound.

For two cameras the joint schedule can be optimized exactly: a dynamic program over (time, pose of camera A,
pose of camera B) on the same rate-limited pose lattice as ``clairvoyant.py`` (72 orientations x 62 viewing
angles per camera, one lattice step per axis and time step, anchored at the initial poses), with the reward
"number of targets covered by A or B". On the recorded episodes of the paper (same keys) we take two cameras of
the network and compare, per episode,

    L   = block-coordinate ascent (the hindsight bound of clairvoyant.py, restricted to the two cameras),
    OPT = exact joint optimum (this file),
    U   = Lagrangian bound (clairvoyant.py, restricted to the two cameras),

and check L <= OPT <= U.

Usage: python3 research/exact_pair.py --config MATE-4v8-9.yaml --targets greedy waypoint --pairs 0,1 2,3 \
           --num-envs 32 --num-steps 1000 --key-steps 1000 --dual-iterations 150 --out results/paper/exact_pair_4v8.json
"""

import argparse
import json
import os
import sys
import time
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import clairvoyant as C  # noqa: E402
import mate_jax  # noqa: E402
from mate_jax.constants import MAX_CAMERA_VIEWING_ANGLE  # noqa: E402
from mate_jax.foresight import ForesightCameraPolicy  # noqa: E402
from mate_jax.geometry import arctan2_deg, line_of_sight, normalize_angle, safe_norm  # noqa: E402


NEG = -1e9


def make_pair_dp(env, num_steps, cells=False):
    """Compiled ``exact(location (2, 2), theta0 (2,), view0 (2,), targets (S, T, 2), obstacles, mute_b) -> value``:
    the largest number of covered target-steps of two cameras over all joint lattice schedules."""

    cfg = env.config
    rotation, zooming = cfg.camera_rotation_step, cfg.camera_zooming_step
    num_orientations = int(round(360.0 / rotation))
    num_views = int((MAX_CAMERA_VIEWING_ANGLE - cfg.camera_min_viewing_angle) / zooming) + 2

    def lattice(location, theta0, view0, targets, obstacle_location, obstacle_radius):
        orientation = theta0 + rotation * jnp.arange(num_orientations)
        j0 = jnp.floor((view0 - cfg.camera_min_viewing_angle) / zooming + 1e-6).astype(jnp.int32)
        viewing = view0 + zooming * (jnp.arange(num_views) - j0)
        if cells:
            valid = jnp.maximum(viewing, cfg.camera_min_viewing_angle) <= jnp.minimum(viewing + zooming, MAX_CAMERA_VIEWING_ANGLE) + 1e-6
        else:
            valid = (viewing >= cfg.camera_min_viewing_angle - 1e-6) & (viewing <= MAX_CAMERA_VIEWING_ANGLE + 1e-6)
        sight = jnp.sqrt(env.camera_area_product / jnp.maximum(viewing, 1e-3))
        relative = targets - location
        distance = safe_norm(relative)
        angle = arctan2_deg(relative[..., 1], relative[..., 0])
        visible = jax.vmap(
            lambda d, p: line_of_sight(location[None, None, :], p[None], d[None], obstacle_location, obstacle_radius)[0]
        )(distance, targets)

        def covers(t):  # (O * V, T) float
            if cells:  # optimistic coverage of the cell [o, o + rotation) x [v, v + zooming), as in clairvoyant.py
                view_lo = jnp.maximum(viewing, cfg.camera_min_viewing_angle)
                view_hi = jnp.minimum(viewing + zooming, MAX_CAMERA_VIEWING_ANGLE)
                start = normalize_angle(angle[t][None, :] - orientation[:, None])
                inside = (start >= 0.0) & (start <= rotation)
                offset = jnp.where(inside, 0.0, jnp.minimum(jnp.abs(start), jnp.abs(normalize_angle(start - rotation))))
                widest = env.camera_area_product / jnp.maximum(jnp.square(distance[t]), 1e-6)
                need = jnp.maximum(view_lo[None, :, None], 2.0 * offset[:, None, :])
                mask = (need <= jnp.minimum(view_hi[None, :, None], widest[None, None, :]) + 1e-6) & visible[t][None, None, :]
            else:
                offset = jnp.abs(normalize_angle(angle[t][None, :] - orientation[:, None]))
                mask = (
                    (2.0 * offset[:, None, :] <= viewing[None, :, None])
                    & (distance[t][None, None, :] <= sight[None, :, None])
                    & visible[t][None, None, :]
                )
            return mask.reshape(num_orientations * num_views, -1).astype(jnp.float32)

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

    def exact(location, theta0, view0, targets, obstacle_location, obstacle_radius, mute_b):
        covers_a, valid_a, j0_a = lattice(location[0], theta0[0], view0[0], targets, obstacle_location, obstacle_radius)
        covers_b, valid_b, j0_b = lattice(location[1], theta0[1], view0[1], targets, obstacle_location, obstacle_radius)
        shape = (num_orientations, num_views, num_orientations, num_views)
        valid = valid_a[None, :, None, None] & valid_b[None, None, None, :]
        start = jnp.full(shape, NEG, jnp.float32).at[0, j0_a, 0, j0_b].set(0.0)

        def forward(value, t):
            best = best_predecessor(best_predecessor(value, 0, 1), 2, 3)
            a = covers_a(t)
            b = covers_b(t) * (1.0 - mute_b)
            gain = a.sum(-1)[:, None] + b.sum(-1)[None, :] - a @ b.T  # targets covered by A or B
            return jnp.where(valid, best + gain.reshape(shape), NEG), None

        final, _ = jax.lax.scan(forward, start, jnp.arange(num_steps))
        return final.max()

    return jax.jit(exact)


def pair_state(state, pair):
    index = jnp.asarray(pair)
    return types.SimpleNamespace(
        camera_location=state.camera_location[:, index], camera_orientation=state.camera_orientation[:, index],
        camera_viewing_angle=state.camera_viewing_angle[:, index], obstacle_location=state.obstacle_location,
        obstacle_radius=state.obstacle_radius,
    )


def block_coordinate(dp, sub, after, rounds):
    """Hindsight bound of clairvoyant.py for the cameras of ``sub``: covered target-steps per episode."""

    num_envs, num_steps, num_targets = after.shape[0], after.shape[1], after.shape[2]
    num_cameras = sub.camera_location.shape[1]
    masks = jnp.zeros((num_envs, num_cameras, num_steps, num_targets), bool)
    for round_index in range(rounds):
        for camera in range(num_cameras):
            others = jnp.any(masks[:, :camera], axis=1) if round_index == 0 else jnp.any(jnp.delete(masks, camera, axis=1), axis=1)
            mask, _, _ = dp(
                sub.camera_location[:, camera], sub.camera_orientation[:, camera], sub.camera_viewing_angle[:, camera],
                after, (~others).astype(jnp.float32), sub.obstacle_location, sub.obstacle_radius,
            )
            masks = masks.at[:, camera].set(mask)
    return np.asarray(jnp.any(masks, axis=1).sum((1, 2)), np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', default='MATE-4v8-9.yaml')
    parser.add_argument('--targets', nargs='+', default=['greedy', 'waypoint'])
    parser.add_argument('--pairs', nargs='+', default=['0,1'], help='camera pairs, e.g. 0,1 2,3')
    parser.add_argument('--num-envs', type=int, default=32)
    parser.add_argument('--num-steps', type=int, default=1000)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--key-steps', type=int, default=None, help='as in clairvoyant.py')
    parser.add_argument('--dual-iterations', type=int, default=150)
    parser.add_argument('--self-test', action='store_true', help='check the joint program against the single-camera one')
    parser.add_argument('--cells', action='store_true',
                        help='continuous poses: exact joint optimum and Lagrangian bound over cells (optimistic coverage)')
    parser.add_argument('--env-json', default='{}', help='environment overrides')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    env = mate_jax.make(args.config, **json.loads(args.env_json))
    planner = ForesightCameraPolicy.from_env(env)
    record_steps = args.key_steps or args.num_steps + int(np.max(planner.horizons)) + 1
    C.RECORD_STEPS[0] = record_steps
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.num_envs)
    dp = C.make_camera_dp(env, args.num_steps)
    exact = make_pair_dp(env, args.num_steps, cells=args.cells)
    cell_dp = C.make_camera_dp(env, args.num_steps, cells=True) if args.cells else None
    scale = args.num_steps * env.num_targets
    results = []
    for targets in args.targets:
        state, after = C.record(env, C.TARGETS[targets].from_env(env), keys, record_steps)
        after = after[:, : args.num_steps]
        for pair in args.pairs:
            pair = tuple(int(x) for x in pair.split(','))
            sub = pair_state(state, pair)
            start = time.perf_counter()
            if args.self_test:  # camera B muted: the joint optimum must equal the single-camera optimum of A
                ones = jnp.ones(after.shape[:3], jnp.float32)
                _, _, single = dp(
                    sub.camera_location[:, 0], sub.camera_orientation[:, 0], sub.camera_viewing_angle[:, 0], after, ones,
                    sub.obstacle_location, sub.obstacle_radius,
                )
                joint = [
                    float(exact(sub.camera_location[e], sub.camera_orientation[e], sub.camera_viewing_angle[e], after[e],
                                sub.obstacle_location[e], sub.obstacle_radius[e], 1.0))
                    for e in range(args.num_envs)
                ]
                print(f'self-test {targets} pair {pair}: single {np.asarray(single).tolist()} joint(B muted) {joint}', flush=True)
                assert np.allclose(np.asarray(single), joint, atol=1e-3)
            lower = block_coordinate(dp, sub, after, args.rounds)
            optimum = np.asarray([
                float(exact(sub.camera_location[e], sub.camera_orientation[e], sub.camera_viewing_angle[e], after[e],
                            sub.obstacle_location[e], sub.obstacle_radius[e], 0.0))
                for e in range(args.num_envs)
            ], np.float64)
            row = {
                'config': args.config, 'targets': targets, 'pair': list(pair), 'episodes': args.num_envs, 'cells': bool(args.cells),
                'num_steps': args.num_steps, 'lower': (lower / scale).tolist(), 'optimum': (optimum / scale).tolist(),
            }
            if args.dual_iterations:
                upper, _ = C.dual_bound(cell_dp or dp, sub, after, lower, args.dual_iterations)
                row['upper'] = np.asarray(upper, np.float64).tolist()
            row['seconds'] = time.perf_counter() - start
            results.append(row)
            lo, op = np.asarray(row['lower']), np.asarray(row['optimum'])
            message = (f'{args.config} {targets} pair {pair}: L {lo.mean():.4f}  OPT {op.mean():.4f}  '
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
