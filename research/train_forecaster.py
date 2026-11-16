"""Train the hindsight heat-map forecaster and evaluate the trajectory planner that uses it.

1. Roll out the belief-space trajectory planner (team memory) on training episodes and record its
   belief state at every few steps, with the cells the targets truly occupy at every planning horizon.
2. Train :class:`ForecastNet` (residual on the analytic forecast) by cross-entropy.
3. Evaluate the planner with learned forecasts on the paired episodes of ``headroom.py`` (seed 0).

Usage: python3 research/train_forecaster.py --targets greedy --out results/forecaster_greedy.json
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
import optax  # noqa: E402
from flax import serialization  # noqa: E402

import mate_jax  # noqa: E402
from mate_jax import agents as A  # noqa: E402
from mate_jax.foresight.belief import BeliefState  # noqa: E402
from mate_jax.hindsight.forecaster import ForecastNet, LearnedForecastPlanner, forecast_inputs, learned_forecast  # noqa: E402
from mate_jax.hindsight.student import trajectory_planner  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from headroom import RECORD_EXTRA, make_runner  # noqa: E402


TARGETS = {'greedy': A.GreedyTargetPolicy, 'waypoint': A.RandomWaypointTargetPolicy}


def collect(env, target_policy, planner, keys, num_steps, every):
    """Belief states every ``every`` steps and the targets' cells at every horizon."""

    horizons = np.asarray(planner.horizons)
    record_steps = num_steps + int(horizons.max()) + 1
    grid = planner.grid

    def episode(key):
        k_reset, k_camera, k_target, k_loop = jax.random.split(key, 4)
        (camera_obs, target_obs), state = env.reset(k_reset)
        camera_carry = planner.init(k_camera, camera_obs)
        target_carry = target_policy.init(k_target, target_obs)

        def body(carry, k):
            state, camera_obs, target_obs, camera_carry, target_carry = carry
            k_camera, k_target, k_env = jax.random.split(k, 3)
            location = state.target_location
            camera_action, camera_carry = planner.act(k_camera, camera_carry, camera_obs)
            target_action, target_carry = target_policy.act(k_target, target_carry, target_obs)
            timestep, state = env.step(k_env, state, camera_action, target_action)
            b = camera_carry['belief']
            record = (b.belief[0].astype(jnp.float16), b.last_position[0], b.last_seen[0], b.velocity[0], b.velocity_known[0], b.step)
            carry = (state, timestep.camera_observation, timestep.target_observation, camera_carry, target_carry)
            return carry, (record, location)

        carry = (state, camera_obs, target_obs, camera_carry, target_carry)
        _, (records, location) = jax.lax.scan(body, carry, jax.random.split(k_loop, record_steps))
        index = np.arange(0, num_steps, every)
        cells = jnp.clip(jnp.floor((location + 1000.0) / grid.cell).astype(jnp.int32), 0, grid.size - 1)
        flat = cells[..., 0] * grid.size + cells[..., 1]  # (S, T)
        labels = flat[index[:, None] + horizons[None, :]]  # (n, H, T)
        records = jax.tree_util.tree_map(lambda x: x[index], records)
        return records, labels, camera_carry['free']

    return jax.jit(jax.vmap(episode))(keys)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', default='MATE-4v8-9.yaml')
    parser.add_argument('--targets', default='greedy', choices=sorted(TARGETS))
    parser.add_argument('--train-envs', type=int, default=32)
    parser.add_argument('--train-steps', type=int, default=1000)
    parser.add_argument('--every', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--features', type=int, default=64)
    parser.add_argument('--eval-envs', type=int, default=16)
    parser.add_argument('--eval-steps', type=int, default=1000)
    parser.add_argument('--eval-rounds', type=int, default=1, help='coordination rounds of the evaluated planners')
    parser.add_argument('--load', default=None, help='skip training: evaluate this saved forecaster.msgpack')
    parser.add_argument('--out', default=None)
    parser.add_argument('--save-dir', default=None)
    args = parser.parse_args()

    env = mate_jax.make(args.config)
    target_policy = TARGETS[args.targets].from_env(env)
    planner = trajectory_planner(env)
    horizons = np.asarray(planner.horizons)
    start = time.perf_counter()
    train_envs = 2 if args.load else args.train_envs  # with --load, only the parameter shapes are needed
    keys = jax.random.split(jax.random.PRNGKey(7), train_envs)
    records, labels, free = jax.device_get(
        collect(env, target_policy, planner, keys, 8 if args.load else args.train_steps, args.every)
    )
    num_envs, per_env = labels.shape[:2]
    print(f'collected {num_envs * per_env} belief states in {time.perf_counter() - start:.0f}s', flush=True)

    # Hold out the last 4 episodes.
    split = num_envs - max(1, num_envs // 8)
    net = ForecastNet(num_horizons=horizons.size, features=args.features)

    def make_belief(record):
        belief, last_position, last_seen, velocity, velocity_known, step = record
        return BeliefState(
            belief=belief[None].astype(jnp.float32), last_position=last_position[None], last_seen=last_seen[None],
            velocity=velocity[None], velocity_known=velocity_known[None], step=step,
        )

    def sample_losses(params, record, label, free_map):
        belief = make_belief(record)
        inputs, analytic = forecast_inputs(planner, belief, free_map, horizons)
        forecast = learned_forecast(net, params, inputs, analytic, free_map)[0]  # (H, T, G, G)
        flat = forecast.reshape(forecast.shape[:2] + (-1,))
        analytic_flat = analytic[0].reshape(flat.shape)
        pick = lambda p: jnp.take_along_axis(p, label[..., None], axis=-1)[..., 0]  # noqa: E731
        learned = -jnp.log(jnp.maximum(pick(flat), 1e-9)).mean()
        baseline = -jnp.log(jnp.maximum(pick(analytic_flat), 1e-9)).mean()
        return learned, baseline

    def batch_losses(params, records, labels, free_maps):
        learned, baseline = jax.vmap(sample_losses, in_axes=(None, 0, 0, 0))(params, records, labels, free_maps)
        return learned.mean(), baseline.mean()

    dummy_inputs, _ = forecast_inputs(planner, make_belief(jax.tree_util.tree_map(lambda x: x[0, 0], records)), free[0], horizons)
    params = net.init(jax.random.PRNGKey(0), dummy_inputs)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.learning_rate))
    opt_state = optimizer.init(params)

    @jax.jit
    def update(params, opt_state, records, labels, free_maps):
        (loss, baseline), grads = jax.value_and_grad(batch_losses, has_aux=True)(params, records, labels, free_maps)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, baseline

    evaluate_losses = jax.jit(batch_losses)

    def flatten(x, lo, hi):
        return x[lo:hi].reshape((-1,) + x.shape[2:])

    train = jax.tree_util.tree_map(lambda x: flatten(x, 0, split), records), flatten(labels, 0, split)
    train_free = np.repeat(free[:split], per_env, axis=0)
    held = jax.tree_util.tree_map(lambda x: flatten(x, split, num_envs), records), flatten(labels, split, num_envs)
    held_free = np.repeat(free[split:], per_env, axis=0)
    num_train = train[1].shape[0]
    rng = np.random.default_rng(0)

    def held_out(params):
        losses = []
        for lo in range(0, held[1].shape[0], args.batch_size):
            sl = slice(lo, lo + args.batch_size)
            batch = jax.tree_util.tree_map(lambda x: x[sl], held[0])
            losses.append(jax.device_get(evaluate_losses(params, batch, held[1][sl], held_free[sl])))
        return np.mean(losses, axis=0)

    history = []
    if args.load:
        with open(args.load, 'rb') as file:
            params = serialization.from_bytes(params, file.read())
        epochs = 0
    else:
        learned, baseline = held_out(params)
        print(json.dumps({'epoch': 0, 'held_out_nll': float(learned), 'analytic_nll': float(baseline)}), flush=True)
        epochs = args.epochs
    best = (np.inf, params)  # early stopping on the held-out likelihood
    for epoch in range(1, epochs + 1):
        order = rng.permutation(num_train)
        for lo in range(0, num_train - args.batch_size + 1, args.batch_size):
            index = order[lo : lo + args.batch_size]
            batch = jax.tree_util.tree_map(lambda x: x[index], train[0])
            params, opt_state, _, _ = update(params, opt_state, batch, train[1][index], train_free[index])
        learned, baseline = held_out(params)
        history.append({'epoch': epoch, 'held_out_nll': float(learned), 'analytic_nll': float(baseline)})
        print(json.dumps(history[-1]), flush=True)
        if learned < best[0]:
            best = (learned, params)
    if epochs:
        params = best[1]

    # Closed loop on the paired evaluation episodes.
    eval_keys = jax.random.split(jax.random.PRNGKey(0), args.eval_envs)
    record_steps = args.eval_steps + RECORD_EXTRA
    dummy = jnp.zeros((args.eval_envs, record_steps, env.num_targets, 2), jnp.float32)
    results = {'history': history, 'eval_rounds': args.eval_rounds}
    base = planner.replace(coordination_rounds=args.eval_rounds)
    for name, policy in (
        ('trajectory', base),
        ('trajectory+learned-forecast', LearnedForecastPlanner.from_env(
            env, levels=planner.options.levels, objective='trajectory', forecaster_params=params,
            forecaster_features=args.features, coordination_rounds=args.eval_rounds,
        )),
    ):
        tick = time.perf_counter()
        run = make_runner(env, target_policy, policy, 'foresight', args.eval_steps, record_steps)
        coverage = np.asarray(jax.device_get(run(eval_keys, dummy))[0])
        results[name] = {'coverage': float(coverage.mean()), 'per_episode': coverage.tolist()}
        print(json.dumps({'policy': name, 'coverage': float(coverage.mean()), 'seconds': time.perf_counter() - tick}), flush=True)
    delta = np.asarray(results['trajectory+learned-forecast']['per_episode']) - np.asarray(results['trajectory']['per_episode'])
    results['paired_delta'] = {'mean': float(delta.mean()), 'se': float(delta.std(ddof=1) / np.sqrt(delta.size))}
    print(json.dumps({'paired_delta': results['paired_delta']}), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='UTF-8') as file:
            json.dump(results, file, indent=2)
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, 'forecaster.msgpack'), 'wb') as file:
            file.write(serialization.to_bytes(jax.device_get(params)))


if __name__ == '__main__':
    main()
