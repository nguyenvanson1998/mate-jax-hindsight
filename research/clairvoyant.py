"""Clairvoyant coverage under the rate limits: how much could a camera team achieve at all?

Target trajectories do not depend on the cameras, so they are recorded first (same keys as
``headroom.py``). Then every camera's pose trajectory is optimized exactly by dynamic programming
(Viterbi) over a pose lattice that respects the rate limits (rotation by 0 or +-5 degrees and zoom by
0 or +-2.5 degrees per step, anchored at the camera's initial pose), given the true positions of all
targets in every future step and the coverage of the other cameras (block-coordinate ascent: the
first round is a sequential pass, later rounds re-optimize each camera given the others). Every lattice
trajectory is feasible, so the result is a lower bound on the clairvoyant optimum; coverage counts
range, field of view and occlusion only (no transmittance), which makes it conservative.

Usage: python3 research/clairvoyant.py --targets greedy waypoint --out results/clairvoyant_4v8.json
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

import mate_jax  # noqa: E402
from mate_jax import agents as A  # noqa: E402
from mate_jax.constants import MAX_CAMERA_VIEWING_ANGLE  # noqa: E402
from mate_jax.foresight import ForesightCameraPolicy  # noqa: E402
from mate_jax.geometry import arctan2_deg, line_of_sight, normalize_angle, safe_norm  # noqa: E402


TARGETS = {'greedy': A.GreedyTargetPolicy, 'waypoint': A.RandomWaypointTargetPolicy}
NEG = -1e9


def record(env, target_policy, keys, record_steps):
    """Initial states and target positions after every step, (E, record_steps, T, 2), with static cameras."""

    def episode(key):
        k_reset, _, k_target, k_loop = jax.random.split(key, 4)
        (camera_obs, target_obs), state = env.reset(k_reset)
        target_carry = target_policy.init(k_target, target_obs)
        idle = jnp.zeros((env.num_cameras, 2), jnp.float32)

        def body(carry, k):
            state, target_obs, target_carry = carry
            _, k_target, k_env = jax.random.split(k, 3)
            target_action, target_carry = target_policy.act(k_target, target_carry, target_obs)
            timestep, state = env.step(k_env, state, idle, target_action)
            return (state, timestep.target_observation, target_carry), state.target_location

        _, after = jax.lax.scan(body, (state, target_obs, target_carry), jax.random.split(k_loop, record_steps))
        return state, after

    return jax.jit(jax.vmap(episode))(keys)


def make_camera_dp(env, num_steps, refine=1, cells=False):
    """Compiled ``dp(location, theta0, view0, targets, weights, obstacles) -> ((num_steps, T) coverage, actions,
    value)``: the best schedule of one camera when covering target j at step t is worth weights[t, j].

    ``cells=False`` (lattice): poses on a lattice anchored at the initial pose with steps rotation / refine and
    zoom / refine; one time step moves by at most ``refine`` lattice steps per axis, so every lattice schedule is
    feasible and its coverage is achievable.

    ``cells=True`` (continuous poses): the pose space is partitioned into half-open cells of the same widths, anchored
    at the initial pose. A target counts as covered in a cell if *some* pose of the cell covers it (an exact test:
    the closest orientation of the cell interval, and a viewing angle that is both wide enough for that offset and
    narrow enough for the distance). Every continuous schedule within the rate limits passes through cells that are
    at most ``refine`` cells apart per step, so the optimum over cell paths bounds the optimum over continuous
    schedules from above (it is not achievable)."""

    cfg = env.config
    rotation, zooming = cfg.camera_rotation_step / refine, cfg.camera_zooming_step / refine
    num_orientations = int(round(360.0 / rotation))
    num_views = int(np.ceil((MAX_CAMERA_VIEWING_ANGLE - cfg.camera_min_viewing_angle) / zooming - 1e-6)) + 2
    if cells:
        # One more cell below: when the initial viewing angle is not on the grid, the lowest cell is cut by the minimum
        # viewing angle, and without it the poses between the minimum and the first grid edge would belong to no cell.
        num_views += 1
    moves_1d = list(range(-refine, refine + 1))

    def dp(location, theta0, view0, targets, weights, obstacle_location, obstacle_radius):
        orientation = theta0 + rotation * jnp.arange(num_orientations)  # (O,) lattice poses or cell lower edges
        j0 = jnp.floor((view0 - cfg.camera_min_viewing_angle) / zooming + 1e-6).astype(jnp.int32) + (1 if cells else 0)
        viewing = view0 + zooming * (jnp.arange(num_views) - j0)  # (V,) lattice poses or cell lower edges
        if cells:
            # half-open cells [viewing, viewing + zooming) cut to the range [min, max]; together they cover the range
            view_lo = jnp.maximum(viewing, cfg.camera_min_viewing_angle)
            view_hi = jnp.minimum(viewing + zooming, MAX_CAMERA_VIEWING_ANGLE)
            valid = (viewing + zooming > cfg.camera_min_viewing_angle + 1e-9) & (viewing <= MAX_CAMERA_VIEWING_ANGLE + 1e-9)
        else:
            valid = (viewing >= cfg.camera_min_viewing_angle - 1e-6) & (viewing <= MAX_CAMERA_VIEWING_ANGLE + 1e-6)
            sight = jnp.sqrt(env.camera_area_product / jnp.maximum(viewing, 1e-3))

        relative = targets - location  # (S, T, 2)
        distance = safe_norm(relative)
        angle = arctan2_deg(relative[..., 1], relative[..., 0])
        visible = jax.vmap(
            lambda d, p: line_of_sight(location[None, None, :], p[None], d[None], obstacle_location, obstacle_radius)[0]
        )(distance, targets)  # (S, T)

        def covers(t):
            if cells:
                # angular distance from the target bearing to the orientation interval [o, o + rotation)
                start = normalize_angle(angle[t][None, :] - orientation[:, None])  # in (-180, 180]
                inside = (start >= 0.0) & (start <= rotation)
                offset = jnp.where(inside, 0.0, jnp.minimum(jnp.abs(start), jnp.abs(normalize_angle(start - rotation))))
                widest = env.camera_area_product / jnp.maximum(jnp.square(distance[t]), 1e-6)  # alpha <= A / d^2
                need = jnp.maximum(view_lo[None, :, None], 2.0 * offset[:, None, :])  # (O, V, T)
                return (need <= jnp.minimum(view_hi[None, :, None], widest[None, None, :]) + 1e-6) & visible[t][None, None, :]
            offset = jnp.abs(normalize_angle(angle[t][None, :] - orientation[:, None]))  # (O, T)
            return (
                (2.0 * offset[:, None, :] <= viewing[None, :, None])
                & (distance[t][None, None, :] <= sight[None, :, None])
                & visible[t][None, None, :]
            )  # (O, V, T)

        moves = [(do, dv) for do in moves_1d for dv in moves_1d]

        def neighbours(value):  # (O, V) -> (M, O, V): value of each predecessor move
            padded = jnp.pad(value, ((0, 0), (refine, refine)), constant_values=NEG)
            return jnp.stack([
                jnp.roll(padded, -do, axis=0)[:, refine + dv: refine + dv + num_views] for do, dv in moves
            ])

        start = jnp.full((num_orientations, num_views), NEG).at[0, j0].set(0.0)

        def forward(value, t):
            options = neighbours(value)
            best, pointer = options.max(0), options.argmax(0).astype(jnp.int16)
            gain = jnp.sum(covers(t) * weights[t][None, None, :], axis=-1)
            value = jnp.where(valid[None, :], best + gain, NEG)
            return value, pointer

        final, pointers = jax.lax.scan(forward, start, jnp.arange(num_steps))
        index = jnp.argmax(final.ravel())
        last = jnp.stack([index // num_views, index % num_views])
        move_array = jnp.array(moves)

        def backward(pose, pointer):
            move = move_array[pointer[pose[0], pose[1]]]
            previous = jnp.stack([(pose[0] + move[0]) % num_orientations, pose[1] + move[1]])
            return previous, pose

        _, path = jax.lax.scan(backward, last, pointers, reverse=True)  # (S, 2) pose after every step

        def coverage(t, pose):
            return covers(t)[pose[0], pose[1]]

        steps = jnp.diff(path[:, 0], prepend=0).astype(jnp.float32)
        steps = jnp.where(steps > num_orientations / 2, steps - num_orientations, steps)
        steps = jnp.where(steps < -num_orientations / 2, steps + num_orientations, steps)
        actions = jnp.stack(  # rate-limited actions that follow the lattice path (meaningless for cells)
            [rotation * steps, zooming * jnp.diff(path[:, 1], prepend=j0).astype(jnp.float32)], axis=-1,
        )
        return jax.vmap(coverage)(jnp.arange(num_steps), path), actions, final.max()  # (S, T), (S, 2), ()

    return jax.jit(jax.vmap(dp))


def replay(env, target_policy, keys, actions):
    """Environment coverage when the cameras execute ``actions`` (E, C, S, 2) on the recorded episodes."""

    def episode(key, camera_actions):
        k_reset, _, k_target, k_loop = jax.random.split(key, 4)
        (_, target_obs), state = env.reset(k_reset)
        target_carry = target_policy.init(k_target, target_obs)

        def body(carry, inputs):
            state, target_obs, target_carry = carry
            k, camera_action = inputs
            _, k_target, k_env = jax.random.split(k, 3)
            target_action, target_carry = target_policy.act(k_target, target_carry, target_obs)
            timestep, state = env.step(k_env, state, camera_action, target_action)
            return (state, timestep.target_observation, target_carry), timestep.info['coverage_rate']

        steps = camera_actions.shape[1]
        record_keys = jax.random.split(k_loop, RECORD_STEPS[0])[:steps]
        _, coverage = jax.lax.scan(body, (state, target_obs, target_carry), (record_keys, camera_actions.swapaxes(0, 1)))
        return coverage.mean()

    return jax.device_get(jax.jit(jax.vmap(episode))(keys, actions))


RECORD_STEPS = [0]


def dual_bound(dp, state, after, lower, iterations):
    """Lagrangian upper bound on the clairvoyant optimum of the lattice coverage model, per episode.

    Coverage of target j at step t is y_tj = min(1, sum_i c_itj). Relaxing y_tj <= sum_i c_itj with
    multipliers mu_tj in [0, 1] gives, for every mu, the upper bound
        U(mu) = sum_tj (1 - mu_tj) + sum_i max_{schedule of camera i} sum_tj mu_tj c_itj,
    whose inner maxima are single-camera dynamic programs. ``lower`` (E,) are feasible coverages (target-steps)
    used by the Polyak step size; the bound is minimized by projected subgradient descent.
    """

    num_envs, num_steps, num_targets = after.shape[0], after.shape[1], after.shape[2]
    num_cameras = state.camera_location.shape[1]
    mu = jnp.full((num_envs, num_steps, num_targets), 0.5, jnp.float32)
    best = np.full(num_envs, np.inf)
    scale, stall = np.ones(num_envs), np.zeros(num_envs, int)
    history = []
    for _ in range(iterations):
        total = jnp.sum(1.0 - mu, axis=(1, 2))
        covered = jnp.zeros_like(mu)
        for camera in range(num_cameras):
            mask, _, value = dp(
                state.camera_location[:, camera], state.camera_orientation[:, camera],
                state.camera_viewing_angle[:, camera], after, mu, state.obstacle_location, state.obstacle_radius,
            )
            total = total + value
            covered = covered + mask
        total = np.asarray(total)
        improved = total < best - 1e-6
        stall = np.where(improved, 0, stall + 1)
        scale = np.where(stall >= 10, scale / 2.0, scale)  # halve the step after 10 iterations without progress
        stall = np.where(stall >= 10, 0, stall)
        best = np.minimum(best, total)
        history.append(float(np.mean(best) / (num_steps * num_targets)))
        gradient = covered - 1.0  # dU/dmu
        norm = jnp.maximum(jnp.sum(gradient**2, axis=(1, 2)), 1e-6)
        step = jnp.asarray(scale) * jnp.maximum(jnp.asarray(total) - jnp.asarray(lower), 0.0) / norm
        mu = jnp.clip(mu - step[:, None, None] * gradient, 0.0, 1.0)
    return best / (num_steps * num_targets), history


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', default='MATE-4v8-9.yaml')
    parser.add_argument('--targets', nargs='+', default=['greedy', 'waypoint'])
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--num-steps', type=int, default=1000)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--key-steps', type=int, default=None,
                        help='episode key length (default num_steps + 51; num_steps pairs with eval_foresight.py)')
    parser.add_argument('--dual-iterations', type=int, default=0, help='Lagrangian upper bound (0: skip)')
    parser.add_argument('--env-json', default='{}', help='environment overrides, e.g. \'{"target": {"step_size": 30.0}}\'')
    parser.add_argument('--refine', type=int, default=1, help='lattice / cell width = rate limit / refine')
    parser.add_argument('--cells', action='store_true',
                        help='upper bound for continuous poses: Lagrangian bound over cells (the lower bound stays on the lattice)')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    env = mate_jax.make(args.config, **json.loads(args.env_json))
    planner = ForesightCameraPolicy.from_env(env)  # same recording length (and keys) as headroom.py
    record_steps = args.key_steps or args.num_steps + int(np.max(planner.horizons)) + 1
    RECORD_STEPS[0] = record_steps
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.num_envs)
    dp = make_camera_dp(env, args.num_steps, refine=args.refine)
    cell_dp = make_camera_dp(env, args.num_steps, refine=args.refine, cells=True) if args.cells else None
    results = []
    for targets in args.targets:
        start = time.perf_counter()
        state, after = record(env, TARGETS[targets].from_env(env), keys, record_steps)
        after = after[:, : args.num_steps]
        num_cameras, num_targets = env.num_cameras, env.num_targets
        masks = jnp.zeros((args.num_envs, num_cameras, args.num_steps, num_targets), bool)
        plans = jnp.zeros((args.num_envs, num_cameras, args.num_steps, 2), jnp.float32)
        history = []
        for round_index in range(args.rounds):
            for camera in range(num_cameras):
                if round_index == 0:  # sequential pass: only the cameras optimized before this one
                    others = jnp.any(masks[:, :camera], axis=1)
                else:
                    others = jnp.any(jnp.delete(masks, camera, axis=1), axis=1)
                mask, action, _ = dp(
                    state.camera_location[:, camera], state.camera_orientation[:, camera],
                    state.camera_viewing_angle[:, camera], after, (~others).astype(jnp.float32),
                    state.obstacle_location, state.obstacle_radius,
                )
                masks = masks.at[:, camera].set(mask)
                plans = plans.at[:, camera].set(action)
            coverage = np.asarray(jnp.any(masks, axis=1).mean((1, 2)))
            history.append(float(coverage.mean()))
        executed = replay(env, TARGETS[targets].from_env(env), keys, plans)
        row = {
            'targets': targets, 'mode': 'clairvoyant-dp', 'coverage': float(coverage.mean()),
            'executed_coverage': float(executed.mean()),
            'coverage_se': float(coverage.std(ddof=1) / np.sqrt(coverage.size)),
            'per_episode_coverage': coverage.tolist(), 'rounds': history, 'seconds': time.perf_counter() - start,
        }
        if args.dual_iterations:
            lower = np.asarray(jnp.any(masks, axis=1).sum((1, 2)), np.float64)
            upper, dual_history = dual_bound(cell_dp or dp, state, after, lower, args.dual_iterations)
            row.update(dual_bound=float(upper.mean()), per_episode_dual_bound=upper.tolist(), dual_history=dual_history,
                       seconds=time.perf_counter() - start, refine=args.refine, cells=bool(args.cells))
        results.append(row)
        print(json.dumps({k: v for k, v in row.items() if not k.startswith('per_episode') and k != 'dual_history'}), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='UTF-8') as file:
            json.dump(results, file, indent=2)


if __name__ == '__main__':
    main()
