"""Penalized information-relaxation bound: an upper bound on the coverage of causal camera teams.

The clairvoyant optimum of clairvoyant.py bounds every causal controller from above, but it may be loose,
because it uses the whole future. Brown, Smith and Sun (Oper. Res. 2010) tighten such bounds with a penalty
that charges the clairvoyant for using information it could not have: for any terms z_t(a_t) with
E[z_t(a_t) | G_{t-1}] = 0 whenever a_t is chosen from the information G_{t-1},

    V_causal <= E[ max over schedules of sum_t ( r_t - z_t ) ].

Here G_{t-1} is the full simulator state before step t (positions, waypoints and internal state of all targets),
which is more than a camera team observes, so the bound holds for every causal camera team. Our penalty is the
martingale increment of a pose-value proxy: for camera i in pose p at step t,

    z_t(p) = sum_h w_h ( E[phi(p, x_{t+h}) | G_t] - E[phi(p, x_{t+h}) | G_{t-1}] ),   h in H,

where phi(p, x) counts the targets that pose p covers at positions x, and E[phi(p, x_t) | G_t] = phi(p, x_t). Both
expectations are estimated without bias by branching the simulator from the recorded states with fresh random
keys (``--samples`` rollouts per state), so E[z_t | G_{t-1}] = 0 holds for the estimates as well. The penalized
problem is bounded from above by the same Lagrangian relaxation as in clairvoyant.py; the penalty only adds a
per-pose term to every single-camera dynamic program. Every choice of weights w_h gives a valid bound (in
expectation over episodes); the weights are chosen on separate tuning episodes (``--tune-seed``).

Usage: python3 research/penalty_bound.py --config MATE-4v8-9.yaml --targets greedy waypoint --num-envs 32 \
           --num-steps 1000 --key-steps 1000 --out results/paper/penalty_4v8.json
"""

import argparse
import itertools
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
import mate_jax  # noqa: E402
from mate_jax.constants import MAX_CAMERA_VIEWING_ANGLE  # noqa: E402
from mate_jax.geometry import arctan2_deg, line_of_sight, normalize_angle, safe_norm  # noqa: E402


NEG = -1e9


# ---------------------------------------------------------------------------------------------------- recording
def record_full(env, target_policy, keys, steps):
    """As ``clairvoyant.record`` (same keys, hence the same trajectories), but also the simulator state before
    every step and after the last one, from which the simulator can be branched."""

    idle = jnp.zeros((env.num_cameras, 2), jnp.float32)

    def episode(key):
        k_reset, _, k_target, k_loop = jax.random.split(key, 4)
        (_, target_obs), state = env.reset(k_reset)
        target_carry = target_policy.init(k_target, target_obs)

        def body(carry, k):
            state, target_obs, target_carry = carry
            _, k_target, k_env = jax.random.split(k, 3)
            target_action, new_carry = target_policy.act(k_target, target_carry, target_obs)
            timestep, new_state = env.step(k_env, state, idle, target_action)
            return (new_state, timestep.target_observation, new_carry), (carry, new_state.target_location)

        last, (carries, after) = jax.lax.scan(body, (state, target_obs, target_carry), jax.random.split(k_loop, steps))
        origins = jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a, b[None]], 0), carries, last)  # (S + 1, ...)
        return state, origins, after

    return jax.jit(jax.vmap(episode))(keys)


def make_rollout(env, target_policy, length):
    """``rollout(carry, key) -> (length, T, 2)`` target positions after each of ``length`` steps from ``carry``."""

    idle = jnp.zeros((env.num_cameras, 2), jnp.float32)

    def rollout(carry, key):
        def body(c, k):
            state, target_obs, target_carry = c
            _, k_target, k_env = jax.random.split(k, 3)
            target_action, target_carry = target_policy.act(k_target, target_carry, target_obs)
            timestep, state = env.step(k_env, state, idle, target_action)
            return (state, timestep.target_observation, target_carry), state.target_location

        _, positions = jax.lax.scan(body, carry, jax.random.split(key, length))
        return positions

    return rollout


# ---------------------------------------------------------------------------------------------------- lattice
def lattice_geometry(env):
    cfg = env.config
    rotation, zooming = cfg.camera_rotation_step, cfg.camera_zooming_step
    num_orientations = int(round(360.0 / rotation))
    num_views = int((MAX_CAMERA_VIEWING_ANGLE - cfg.camera_min_viewing_angle) / zooming) + 2

    def grid(theta0, view0):
        orientation = theta0 + rotation * jnp.arange(num_orientations)
        j0 = jnp.floor((view0 - cfg.camera_min_viewing_angle) / zooming + 1e-6).astype(jnp.int32)
        viewing = view0 + zooming * (jnp.arange(num_views) - j0)
        valid = (viewing >= cfg.camera_min_viewing_angle - 1e-6) & (viewing <= MAX_CAMERA_VIEWING_ANGLE + 1e-6)
        sight = jnp.sqrt(env.camera_area_product / jnp.maximum(viewing, 1e-3))
        return orientation, viewing, sight, valid, j0

    return num_orientations, num_views, grid


def make_pose_counts(env):
    """``counts(location, theta0, view0, positions (N, T, 2), obstacles) -> (N, O, V)``: phi for every pose."""

    _, _, grid = lattice_geometry(env)

    def counts(location, theta0, view0, positions, obstacle_location, obstacle_radius):
        orientation, viewing, sight, _, _ = grid(theta0, view0)
        relative = positions - location
        distance = safe_norm(relative)  # (N, T)
        angle = arctan2_deg(relative[..., 1], relative[..., 0])
        visible = jax.vmap(
            lambda d, p: line_of_sight(location[None, None, :], p[None], d[None], obstacle_location, obstacle_radius)[0]
        )(distance, positions)  # (N, T)
        offset = jnp.abs(normalize_angle(angle[:, None, :] - orientation[None, :, None]))  # (N, O, T)
        mask = (
            (2.0 * offset[:, :, None, :] <= viewing[None, None, :, None])
            & (distance[:, None, None, :] <= sight[None, None, :, None])
            & visible[:, None, None, :]
        )  # (N, O, V, T)
        return mask.sum(-1).astype(jnp.float32)

    return counts


def make_penalty_dp(env, num_steps):
    """``dp(location, theta0, view0, targets, weights, bonus (S, O, V), obstacles) -> (mask (S, T), value)``:
    the best lattice schedule of one camera when covering target j at step t is worth weights[t, j] and being in
    pose p at step t adds bonus[t, p] (minus the penalty)."""

    num_orientations, num_views, grid = lattice_geometry(env)

    def dp(location, theta0, view0, targets, weights, bonus, obstacle_location, obstacle_radius):
        orientation, viewing, sight, valid, j0 = grid(theta0, view0)
        relative = targets - location
        distance = safe_norm(relative)
        angle = arctan2_deg(relative[..., 1], relative[..., 0])
        visible = jax.vmap(
            lambda d, p: line_of_sight(location[None, None, :], p[None], d[None], obstacle_location, obstacle_radius)[0]
        )(distance, targets)

        def covers(t):
            offset = jnp.abs(normalize_angle(angle[t][None, :] - orientation[:, None]))
            return (
                (2.0 * offset[:, None, :] <= viewing[None, :, None])
                & (distance[t][None, None, :] <= sight[None, :, None])
                & visible[t][None, None, :]
            )

        def neighbours(value):
            padded = jnp.pad(value, ((0, 0), (1, 1)), constant_values=NEG)
            return jnp.stack([
                jnp.roll(padded, -do, axis=0)[:, 1 + dv: 1 + dv + num_views] for do in (-1, 0, 1) for dv in (-1, 0, 1)
            ])

        start = jnp.full((num_orientations, num_views), NEG).at[0, j0].set(0.0)

        def forward(value, t):
            options = neighbours(value)
            best, pointer = options.max(0), options.argmax(0).astype(jnp.int8)
            gain = jnp.sum(covers(t) * weights[t][None, None, :], axis=-1) + bonus[t]
            return jnp.where(valid[None, :], best + gain, NEG), pointer

        final, pointers = jax.lax.scan(forward, start, jnp.arange(num_steps))
        index = jnp.argmax(final.ravel())
        last = jnp.stack([index // num_views, index % num_views])
        moves = jnp.array([(do, dv) for do in (-1, 0, 1) for dv in (-1, 0, 1)])

        def backward(pose, pointer):
            move = moves[pointer[pose[0], pose[1]]]
            return jnp.stack([(pose[0] + move[0]) % num_orientations, pose[1] + move[1]]), pose

        _, path = jax.lax.scan(backward, last, pointers, reverse=True)
        mask = jax.vmap(lambda t, pose: covers(t)[pose[0], pose[1]])(jnp.arange(num_steps), path)
        return mask, final.max()

    return jax.jit(jax.vmap(dp))


# ---------------------------------------------------------------------------------------------------- penalty
def penalty_terms(env, rollout, counts, state0, origins, after, episode, camera, horizons, samples, key, chunk):
    """D (|H|, S, O, V) for one episode and camera: D_h[t] = E[phi(., x_{t+h}) | G_t] - E[phi(., x_{t+h}) | G_{t-1}],
    estimated from ``samples`` simulator branches per state (independent keys for every state)."""

    num_steps = after.shape[1]
    lags = sorted({h for h in horizons if h > 0} | {h + 1 for h in horizons})
    length = max(lags)
    lag_index = {lag: k for k, lag in enumerate(lags)}
    location = state0.camera_location[episode, camera]
    theta0 = state0.camera_orientation[episode, camera]
    view0 = state0.camera_viewing_angle[episode, camera]
    obstacles = (state0.obstacle_location[episode], state0.obstacle_radius[episode])
    carries = jax.tree_util.tree_map(lambda a: a[episode], origins)  # (S + 1, ...)
    lag_array = jnp.asarray(lags) - 1

    @jax.jit
    def bank(carry_chunk, keys):  # (U, ...) origins -> (U, |lags|, O, V) mean phi over samples
        def one(carry, k):
            ks = jax.random.split(k, samples)
            positions = jax.vmap(lambda kk: rollout(carry, kk))(ks)  # (K, length, T, 2)
            positions = positions[:, lag_array]  # (K, |lags|, T, 2)
            flat = positions.reshape((-1,) + positions.shape[2:])
            phi = counts(location, theta0, view0, flat, *obstacles)  # (K * |lags|, O, V)
            return phi.reshape((samples, len(lags)) + phi.shape[1:]).mean(0)
        return jax.vmap(one)(carry_chunk, keys)

    total = num_steps + 1
    keys = jax.random.split(key, total)
    parts = []
    for begin in range(0, total, chunk):
        end = min(total, begin + chunk)
        piece = jax.tree_util.tree_map(lambda a: a[begin:end], carries)
        parts.append(bank(piece, keys[begin:end]))
    B = jnp.concatenate(parts, 0)  # (S + 1, |lags|, O, V): B[u, L] = E[phi(., x after step u + L - 1) | state before step u]
    actual = jax.lax.map(lambda x: counts(location, theta0, view0, x[None], *obstacles)[0], after[episode])  # (S, O, V)
    terms = []
    for h in horizons:
        now = actual if h == 0 else B[1: num_steps + 1, lag_index[h]]  # E[phi(x_{t+h}) | G_t]
        before = B[:num_steps, lag_index[h + 1]]  # E[phi(x_{t+h}) | G_{t-1}]
        terms.append(now - before)
    _, _, grid = lattice_geometry(env)
    valid = grid(theta0, view0)[3]
    return jnp.where(valid[None, None, :], jnp.stack(terms), 0.0)


def make_step(env, target_policy):
    """``step(carry, key) -> carry``: one target step from a recorded simulator state (cameras idle)."""

    idle = jnp.zeros((env.num_cameras, 2), jnp.float32)

    def step(carry, key):
        state, target_obs, target_carry = carry
        _, k_target, k_env = jax.random.split(key, 3)
        target_action, target_carry = target_policy.act(k_target, target_carry, target_obs)
        timestep, state = env.step(k_env, state, idle, target_action)
        return (state, timestep.target_observation, target_carry)

    return step


def path_terms(env, step, rollout, counts, state0, origins, after, episode, camera, horizon, children, samples, key, chunk):
    """(2, S, O, V) penalty terms of the path-value proxy for one episode and camera:

    [0] reward surprise  phi(p, x_t) - E[phi(p, x_t) | G_{t-1}];
    [1] value increment  W(p, G_t) - E[W(p, G_t) | G_{t-1}],  where W(p, G) is the best expected number of covered
        targets over the next ``horizon`` steps along a lattice path that starts in pose p, with the expectations
        estimated from ``samples`` branches of the state G. E[. | G_{t-1}] averages ``children`` one-step branches
        of G_{t-1}; each child is valued exactly like the recorded G_t, so the difference has mean zero."""

    num_orientations, num_views, grid = lattice_geometry(env)
    num_steps = after.shape[1]
    location = state0.camera_location[episode, camera]
    theta0 = state0.camera_orientation[episode, camera]
    view0 = state0.camera_viewing_angle[episode, camera]
    obstacles = (state0.obstacle_location[episode], state0.obstacle_radius[episode])
    _, _, _, valid, _ = grid(theta0, view0)
    carries = jax.tree_util.tree_map(lambda a: a[episode], origins)  # (S + 1, ...)

    def best_next(value):  # best value reachable in one lattice move, (O, V) -> (O, V)
        value = jnp.where(valid[None, :], value, NEG)
        padded = jnp.pad(value, ((0, 0), (1, 1)), constant_values=NEG)
        return jnp.max(jnp.stack([
            jnp.roll(padded, -do, axis=0)[:, 1 + dv: 1 + dv + num_views] for do in (-1, 0, 1) for dv in (-1, 0, 1)
        ]), 0)

    def value_of(carry, k):  # W(., G) for the state ``carry``
        ks = jax.random.split(k, samples)
        positions = jax.vmap(lambda kk: rollout(carry, kk))(ks)  # (K2, H, T, 2)
        phi = counts(location, theta0, view0, positions.reshape((-1,) + positions.shape[2:]), *obstacles)
        phi = phi.reshape((samples, horizon) + phi.shape[1:]).mean(0)  # (H, O, V)

        def back(future, current):
            return current + best_next(future), None

        first, _ = jax.lax.scan(back, phi[-1], phi[:-1], reverse=True)  # value of being in a pose one step ahead
        return best_next(first)

    @jax.jit
    def realized(carry_chunk, keys):
        return jax.vmap(value_of)(carry_chunk, keys)

    @jax.jit
    def expected(carry_chunk, child_keys, value_keys):
        def one(carry, k_children, k_value):
            kids = jax.vmap(lambda kk: step(carry, kk))(jax.random.split(k_children, children))
            # Common random numbers: every child is valued with the same branch keys as the recorded G_t, so the
            # sampling noise of W largely cancels in the difference (which keeps its zero mean, since the keys are
            # independent of the children and of G_t).
            values = jax.vmap(lambda kid: value_of(kid, k_value))(kids).mean(0)  # (O, V)
            phi_now = counts(location, theta0, view0, kids[0].target_location, *obstacles).mean(0)  # E[phi(x_t) | G_{t-1}]
            return values, phi_now
        return jax.vmap(one)(carry_chunk, child_keys, value_keys)

    k_value, k_child = jax.random.split(key)
    value_keys = jax.random.split(k_value, num_steps)  # value_keys[t]: branches of G_t and of the children of G_{t-1}
    child_keys = jax.random.split(k_child, num_steps)
    R, E, P = [], [], []
    for begin in range(0, num_steps, chunk):
        end = min(num_steps, begin + chunk)
        R.append(realized(jax.tree_util.tree_map(lambda a: a[begin + 1: end + 1], carries), value_keys[begin:end]))
        e, ph = expected(jax.tree_util.tree_map(lambda a: a[begin:end], carries), child_keys[begin:end], value_keys[begin:end])
        E.append(e)
        P.append(ph)
    R, E, P = jnp.concatenate(R), jnp.concatenate(E), jnp.concatenate(P)
    actual = jax.lax.map(lambda x: counts(location, theta0, view0, x[None], *obstacles)[0], after[episode])
    # Poses outside the viewing-angle range are never chosen; zero their terms so that smoothing cannot mix them in.
    return jnp.where(valid[None, None, None, :], jnp.stack([actual - P, R - E]), 0.0)


# ---------------------------------------------------------------------------------------------------- bounds
def penalized_dual(dp, state, after, bonus, iterations):
    """Lagrangian upper bound on max_s sum_t (coverage_t - sum_i z_t(s_i)), per episode (in target-steps)."""

    num_envs, num_steps, num_targets = after.shape[0], after.shape[1], after.shape[2]
    num_cameras = state.camera_location.shape[1]
    mu = jnp.full((num_envs, num_steps, num_targets), 0.5, jnp.float32)
    best = np.full(num_envs, np.inf)
    feasible = np.full(num_envs, -np.inf)
    scale, stall = np.ones(num_envs), np.zeros(num_envs, int)
    history = []
    for _ in range(iterations):
        total = jnp.sum(1.0 - mu, axis=(1, 2))
        covered = jnp.zeros_like(mu)
        penalty_sum = jnp.zeros(num_envs)
        for camera in range(num_cameras):
            mask, value = dp(
                state.camera_location[:, camera], state.camera_orientation[:, camera], state.camera_viewing_angle[:, camera],
                after, mu, bonus[:, camera], state.obstacle_location, state.obstacle_radius,
            )
            total = total + value
            covered = covered + mask
            penalty_sum = penalty_sum + value - jnp.sum(mu * mask, axis=(1, 2))  # bonus along the chosen schedule
        total = np.asarray(total)
        # The chosen schedules are feasible: their penalized value is a lower estimate of the penalized optimum.
        feasible = np.maximum(feasible, np.asarray(jnp.sum(jnp.minimum(covered, 1.0), axis=(1, 2)) + penalty_sum))
        improved = total < best - 1e-6
        stall = np.where(improved, 0, stall + 1)
        scale = np.where(stall >= 10, scale / 2.0, scale)
        stall = np.where(stall >= 10, 0, stall)
        best = np.minimum(best, total)
        history.append(float(np.mean(best) / (num_steps * num_targets)))
        gradient = covered - 1.0
        norm = jnp.maximum(jnp.sum(gradient ** 2, axis=(1, 2)), 1e-6)
        step = jnp.asarray(scale) * jnp.maximum(jnp.asarray(total) - jnp.asarray(feasible), 0.0) / norm
        mu = jnp.clip(mu - step[:, None, None] * gradient, 0.0, 1.0)
    return best / (num_steps * num_targets), feasible / (num_steps * num_targets), history


def smooth_terms(terms, orientation_radius, view_radius):
    """Average the terms over neighbouring lattice poses (circular in orientation, clipped in viewing angle).

    For every fixed pose q the term z_t(q) has mean zero given G_{t-1}, so any fixed average over poses has mean zero
    as well; the average keeps the signal, which varies slowly over the poses, and removes much of the sampling noise
    that the maximization over schedules would otherwise exploit."""

    if orientation_radius == 0 and view_radius == 0:
        return terms
    out = terms
    if orientation_radius:
        out = sum(np.roll(out, k, axis=-2) for k in range(-orientation_radius, orientation_radius + 1)) / (2 * orientation_radius + 1)
    if view_radius:
        padded = np.concatenate([np.repeat(out[..., :1], view_radius, -1), out, np.repeat(out[..., -1:], view_radius, -1)], -1)
        out = sum(padded[..., k: k + out.shape[-1]] for k in range(2 * view_radius + 1)) / (2 * view_radius + 1)
    return out.astype(np.float32)


def all_terms(env, target_policy, keys, args, horizons, samples, seed):
    """Record the episodes and compute the penalty terms D (E, C, |H|, S, O, V) on the host."""

    state0, origins, after = record_full(env, target_policy, keys, args.key_steps)
    after = after[:, : args.num_steps]
    counts = make_pose_counts(env)
    if args.penalty == 'path':
        rollout = make_rollout(env, target_policy, args.path_horizon)
        step = make_step(env, target_policy)
    else:
        rollout = make_rollout(env, target_policy, max({h for h in horizons if h > 0} | {h + 1 for h in horizons}))
    key = jax.random.PRNGKey(10_000 + seed)
    terms = []
    for episode in range(keys.shape[0]):
        per_camera = []
        for camera in range(env.num_cameras):
            key, sub = jax.random.split(key)
            if args.penalty == 'path':
                per_camera.append(np.asarray(path_terms(env, step, rollout, counts, state0, origins, after, episode, camera,
                                                        args.path_horizon, args.children, samples, sub, args.chunk)))
            else:
                per_camera.append(np.asarray(penalty_terms(env, rollout, counts, state0, origins, after, episode, camera,
                                                           horizons, samples, sub, args.chunk)))
        terms.append(smooth_terms(np.stack(per_camera), args.smooth_orientation, args.smooth_view))
        print(f'    penalty terms of episode {episode + 1}/{keys.shape[0]} done', flush=True)
    return state0, after, np.stack(terms)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', default='MATE-4v8-9.yaml')
    parser.add_argument('--targets', nargs='+', default=['greedy', 'waypoint'])
    parser.add_argument('--num-envs', type=int, default=32)
    parser.add_argument('--num-steps', type=int, default=1000)
    parser.add_argument('--key-steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--horizons', type=int, nargs='+', default=[0, 1, 3, 7, 15, 31], help='h of the penalty terms')
    parser.add_argument('--samples', type=int, default=16, help='simulator branches per state and expectation')
    parser.add_argument('--chunk', type=int, default=50, help='states per batch of branches')
    parser.add_argument('--dual-iterations', type=int, default=150)
    parser.add_argument('--tune-seed', type=int, default=1, help='episodes on which the penalty weights are chosen')
    parser.add_argument('--tune-envs', type=int, default=8)
    parser.add_argument('--tune-samples', type=int, default=8)
    parser.add_argument('--tune-iterations', type=int, default=60)
    parser.add_argument('--lambdas', type=float, nargs='+', default=[0.25, 0.5, 1.0])
    parser.add_argument('--profiles', nargs='+', default=['0', '0,1,3,7', '0,1,3,7,15,31'],
                        help='subsets of --horizons that get weight lambda (the others get 0)')
    parser.add_argument('--penalty', default='path', choices=['path', 'lags'],
                        help='path: value of the best path from each pose; lags: future coverage of the pose itself')
    parser.add_argument('--path-horizon', type=int, default=20, help='steps of the path value W')
    parser.add_argument('--children', type=int, default=4, help='one-step branches for E[W(G_t) | G_{t-1}]')
    parser.add_argument('--path-grid', nargs='+', default=['0,0.25', '0,0.5', '0,1', '0.1,0.5', '0.1,1'],
                        help='weights (reward surprise, value increment) tried on the tuning episodes; n0.1 means -0.1')
    parser.add_argument('--smooth-orientation', type=int, default=0, help='average the penalty over +-k orientations')
    parser.add_argument('--smooth-view', type=int, default=0, help='average the penalty over +-k viewing angles')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    env = mate_jax.make(args.config)
    dp = make_penalty_dp(env, args.num_steps)
    horizons = list(args.horizons)
    results = []
    for targets in args.targets:
        target_policy = C.TARGETS[targets].from_env(env)
        start = time.perf_counter()

        def weights_of(profile, lam):
            chosen = {int(h) for h in profile.split(',')}
            return np.asarray([lam if h in chosen else 0.0 for h in horizons], np.float32)

        def bound(state0, after, terms, weights, iterations):
            bonus = -np.einsum('h,echsov->ecsov', weights, terms)  # minus the penalty
            upper, feasible, history = penalized_dual(dp, state0, after, jnp.asarray(bonus), iterations)
            return upper, feasible, history

        # 1. Choose the weights on separate tuning episodes.
        tune_keys = jax.random.split(jax.random.PRNGKey(args.tune_seed), args.tune_envs)
        print(f'{targets}: tuning on {args.tune_envs} episodes (seed {args.tune_seed})', flush=True)
        t_state, t_after, t_terms = all_terms(env, target_policy, tune_keys, args, horizons, args.tune_samples, args.tune_seed)
        zero = np.zeros(2 if args.penalty == 'path' else len(horizons), np.float32)
        base, _, _ = bound(t_state, t_after, t_terms, zero, args.tune_iterations)
        tuning = [{'profile': 'none', 'weights': zero.tolist(), 'upper': float(base.mean())}]
        print(f'  no penalty: {base.mean():.4f}', flush=True)
        if args.penalty == 'path':
            candidates = [(g, np.asarray([float(x.replace('n', '-')) for x in g.split(',')], np.float32)) for g in args.path_grid]
        else:
            candidates = [(f'{profile} x {lam}', weights_of(profile, lam))
                          for profile, lam in itertools.product(args.profiles, args.lambdas)]
        for name, w in candidates:
            upper, _, _ = bound(t_state, t_after, t_terms, w, args.tune_iterations)
            tuning.append({'profile': name, 'weights': w.tolist(), 'upper': float(upper.mean())})
            print(f'  weights {name}: {upper.mean():.4f}', flush=True)
        best = min(tuning, key=lambda r: r['upper'])
        weights = np.asarray(best.get('weights', zero), np.float32) if best['profile'] != 'none' else zero
        del t_terms

        # 2. Evaluate the chosen penalty on the episodes of the paper.
        keys = jax.random.split(jax.random.PRNGKey(args.seed), args.num_envs)
        print(f'{targets}: evaluation on {args.num_envs} episodes (seed {args.seed}), weights {weights.tolist()}', flush=True)
        state0, after, terms = all_terms(env, target_policy, keys, args, horizons, args.samples, args.seed)
        plain, _, plain_history = bound(state0, after, terms, zero, args.dual_iterations)
        upper, feasible, history = bound(state0, after, terms, weights, args.dual_iterations)
        row = {
            'targets': targets, 'config': args.config, 'episodes': args.num_envs, 'samples': args.samples,
            'penalty': args.penalty, 'path_horizon': args.path_horizon, 'children': args.children,
            'smooth': [args.smooth_orientation, args.smooth_view],
            'horizons': horizons, 'weights': weights.tolist(), 'tuning': tuning,
            'lagrangian': float(plain.mean()), 'per_episode_lagrangian': plain.tolist(),
            'penalized': float(upper.mean()), 'penalized_se': float(upper.std(ddof=1) / np.sqrt(upper.size)),
            'per_episode_penalized': upper.tolist(), 'per_episode_penalized_feasible': feasible.tolist(),
            'history': history, 'plain_history': plain_history, 'seconds': time.perf_counter() - start,
        }
        results.append(row)
        print(json.dumps({k: v for k, v in row.items() if not k.startswith('per_episode') and 'history' not in k}), flush=True)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, 'w', encoding='UTF-8') as file:
                json.dump(results, file, indent=2)


if __name__ == '__main__':
    main()
