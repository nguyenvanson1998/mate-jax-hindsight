"""Throughput benchmark: original NumPy MATE vs. MATE-JAX (jit + vmap + sharding).

    python3 -m mate_jax.benchmark --config MATE-4v8-9.yaml --batch-sizes 1 256 4096
"""

import argparse
import sys
import time
import warnings


def benchmark_numpy(config, num_steps):
    warnings.filterwarnings('ignore')
    try:
        import gym  # pylint: disable=import-outside-toplevel

        gym.logger.set_level(40)
        import mate  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None

    env = mate.MultiAgentTracking(config)
    env.reset(seed=0)
    start = time.perf_counter()
    for _ in range(num_steps):
        _, _, done, _ = env.step(env.action_space.sample())
        if done:
            env.reset()
    return num_steps / (time.perf_counter() - start)


def benchmark_jax(config, batch_size, num_steps, repeats=3):
    # pylint: disable=import-outside-toplevel
    import jax
    from jax.sharding import PartitionSpec as P

    import mate_jax
    from mate_jax.sharding import DATA_AXIS, create_mesh, shard_map
    from mate_jax.wrappers import AutoResetEnv

    env = AutoResetEnv(mate_jax.make(config))
    mesh = create_mesh()
    num_devices = mesh.size if batch_size % mesh.size == 0 else 1
    if num_devices == 1:
        mesh = create_mesh(1)
    local = batch_size // num_devices

    def rollout_local(key):
        key = jax.random.fold_in(key, jax.lax.axis_index(DATA_AXIS))
        key, k_reset = jax.random.split(key)
        _, states = jax.vmap(env.reset)(jax.random.split(k_reset, local))

        def body(states, k):
            k_act, k_step = jax.random.split(k)
            actions = jax.vmap(env.sample_actions)(jax.random.split(k_act, local), states)
            timestep, states = jax.vmap(env.step)(jax.random.split(k_step, local), states, *actions)
            return states, timestep.camera_reward.mean()

        _, rewards = jax.lax.scan(body, states, jax.random.split(key, num_steps))
        return jax.lax.pmean(rewards.mean(), DATA_AXIS)

    fn = jax.jit(shard_map(rollout_local, mesh, in_specs=P(), out_specs=P()))
    start = time.perf_counter()
    jax.block_until_ready(fn(jax.random.PRNGKey(0)))
    compile_time = time.perf_counter() - start
    best = float('inf')
    for r in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn(jax.random.PRNGKey(r + 1)))
        best = min(best, time.perf_counter() - start)
    # reset cost is included (once per call); it is negligible for long rollouts
    return batch_size * num_steps / best, compile_time, num_devices


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', default='MATE-4v8-9.yaml')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 64, 256, 1024, 4096])
    parser.add_argument('--num-steps', type=int, default=200)
    parser.add_argument('--numpy-steps', type=int, default=2000)
    parser.add_argument('--prng-impl', choices=('threefry2x32', 'rbg', 'unsafe_rbg'), default=None)
    args = parser.parse_args(argv)

    if args.prng_impl is not None:
        import jax  # pylint: disable=import-outside-toplevel

        jax.config.update('jax_default_prng_impl', args.prng_impl)

    from mate_jax.sharding import describe_devices  # pylint: disable=import-outside-toplevel

    print(describe_devices())
    numpy_sps = benchmark_numpy(args.config, args.numpy_steps)
    if numpy_sps is not None:
        print(f'{"numpy mate (1 env)":>28s}: {numpy_sps:12,.0f} steps/s')
    for batch_size in args.batch_sizes:
        sps, compile_time, devices = benchmark_jax(args.config, batch_size, args.num_steps)
        speedup = f'  x{sps / numpy_sps:,.0f} vs numpy' if numpy_sps else ''
        print(
            f'{f"mate_jax (B={batch_size}, {devices} dev)":>28s}: {sps:12,.0f} steps/s'
            f'  (compile {compile_time:.1f}s){speedup}',
            flush=True,
        )


if __name__ == '__main__':
    sys.exit(main())
