"""Ceiling of a learned cost-to-go for the trajectory planner: plan with the *hindsight* value of every pose.

The hindsight teacher's value tables give, for every lattice pose and step, the coverage the team can still
collect from that pose until the end of the episode, knowing the future. Adding ``weight`` times this value
at the planner's last horizon tells how much a perfect terminal value (cost-to-go) would add to the
receding-horizon planner; a learned value function can at best approach it.

Usage: python3 research/terminal_probe.py --targets greedy --weights 0 0.05 0.2 --out results/terminal_probe.json
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
from mate_jax.foresight.options import NUM_ORIENTATIONS  # noqa: E402
from mate_jax.geometry import normalize_angle  # noqa: E402
from mate_jax.hindsight import HindsightTeacher, record_targets, trajectory_planner  # noqa: E402


TARGETS = {'greedy': A.GreedyTargetPolicy, 'waypoint': A.RandomWaypointTargetPolicy}


def option_to_lattice(teacher, planner, lattice):
    """Teacher lattice indices (o, j), each (C, L, O), of the planner's options for one episode."""

    orientations = jnp.asarray(planner.options.orientations)  # (O,)
    viewing = jnp.asarray(planner.options.viewing_angles)  # (L,)
    o = jnp.mod(jnp.round(normalize_angle(orientations[None, :] - lattice.theta0[:, None]) / teacher.rotation), teacher.num_orientations)
    j = lattice.j0[:, None] + jnp.round((viewing[None, :] - lattice.view0[:, None]) / teacher.zooming)
    j = jnp.clip(j, 0, teacher.num_views - 1)
    shape = (lattice.theta0.shape[0], viewing.size, NUM_ORIENTATIONS)
    return jnp.broadcast_to(o[:, None, :], shape).astype(jnp.int32), jnp.broadcast_to(j[:, :, None], shape).astype(jnp.int32)


def make_runner(env, target_policy, planner, teacher, num_steps, key_steps, weight):
    lookahead = int(np.max(planner.horizons))
    cameras = np.arange(env.num_cameras)[:, None, None]

    def episode(key, plan):
        o_index, j_index = option_to_lattice(teacher, planner, plan.lattice)
        k_reset, k_camera, k_target, k_loop = jax.random.split(key, 4)
        (camera_obs, target_obs), state = env.reset(k_reset)
        camera_carry = planner.init(k_camera, camera_obs)
        camera_carry['terminal_values'] = jnp.zeros((env.num_cameras,) + o_index.shape[1:], jnp.float32)
        target_carry = target_policy.init(k_target, target_obs)

        def body(carry, inputs):
            k, t = inputs
            state, camera_obs, target_obs, camera_carry, target_carry = carry
            k_camera, k_target, k_env = jax.random.split(k, 3)
            values = jax.lax.dynamic_index_in_dim(plan.values, jnp.minimum(t + lookahead, num_steps - 1), axis=1, keepdims=False)
            camera_carry['terminal_values'] = weight * values[cameras, o_index, j_index].astype(jnp.float32)
            camera_action, camera_carry = planner.act(k_camera, camera_carry, camera_obs)
            target_action, target_carry = target_policy.act(k_target, target_carry, target_obs)
            timestep, state = env.step(k_env, state, camera_action, target_action)
            carry = (state, timestep.camera_observation, timestep.target_observation, camera_carry, target_carry)
            return carry, timestep.info['coverage_rate']

        carry = (state, camera_obs, target_obs, camera_carry, target_carry)
        inputs = (jax.random.split(k_loop, key_steps)[:num_steps], jnp.arange(num_steps))
        _, coverage = jax.lax.scan(body, carry, inputs)
        return coverage.mean()

    return jax.jit(jax.vmap(episode))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', default='MATE-4v8-9.yaml')
    parser.add_argument('--targets', default='greedy', choices=sorted(TARGETS))
    parser.add_argument('--weights', type=float, nargs='+', default=[0.0, 0.05, 0.2])
    parser.add_argument('--rounds', type=int, default=2, help='coordination rounds of the planner')
    parser.add_argument('--num-envs', type=int, default=8)
    parser.add_argument('--num-steps', type=int, default=1000)
    parser.add_argument('--key-steps', type=int, default=None, help='default num_steps + 51 (headroom.py episodes)')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    env = mate_jax.make(args.config)
    target_policy = TARGETS[args.targets].from_env(env)
    planner = trajectory_planner(env, coordination_rounds=args.rounds)
    key_steps = args.key_steps or args.num_steps + 51
    keys = jax.random.split(jax.random.PRNGKey(0), args.num_envs)
    teacher = HindsightTeacher(env, args.num_steps, rounds=3, keep_values=True)
    state0, targets = jax.jit(lambda k: record_targets(env, target_policy, k, args.num_steps, key_steps))(keys)
    plan = teacher.solve(state0, targets)
    bound = np.asarray(jnp.any(plan.masks, axis=1).mean((1, 2)))
    results, base = [], None
    for weight in args.weights:
        start = time.perf_counter()
        run = make_runner(env, target_policy, planner, teacher, args.num_steps, key_steps, float(weight))
        coverage = np.asarray(run(keys, plan))
        base = coverage if base is None else base
        delta = coverage - base
        row = {
            'targets': args.targets, 'rounds': args.rounds, 'weight': weight, 'coverage': float(coverage.mean()),
            'delta': float(delta.mean()), 'delta_se': float(delta.std(ddof=1) / np.sqrt(delta.size)),
            'bound': float(bound.mean()), 'seconds': time.perf_counter() - start,
        }
        results.append(row)
        print(json.dumps(row), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='UTF-8') as file:
            json.dump(results, file, indent=2)


if __name__ == '__main__':
    main()
