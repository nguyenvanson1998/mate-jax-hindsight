"""Train MAPPO on MATE with JAX (single host, multi-device, or a TPU pod slice).

Examples:

    # camera team vs. greedy targets (coverage reward), all local devices
    python3 -m mate_jax.train --team camera --config MATE-4v8-9.yaml --num-envs 1024

    # target team on TPU: bfloat16 matmuls + hardware PRNG
    python3 -m mate_jax.train --team target --reward raw --dtype bfloat16 --prng-impl rbg

    # TPU pod slice: run the same command on every host
    python3 -m mate_jax.train --multi-host --num-envs 8192 ...
"""

import argparse
import json
import math
import os
import sys
import time


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog='python3 -m mate_jax.train', description=__doc__.split('\n')[0])
    env = parser.add_argument_group('environment')
    env.add_argument('--config', default='MATE-4v8-9.yaml', help='built-in or custom YAML/JSON config')
    env.add_argument('--team', choices=('camera', 'target'), default='camera')
    env.add_argument('--opponent', choices=('greedy', 'random', 'static', 'waypoint'), default='greedy')
    env.add_argument('--reward', choices=('coverage', 'raw', 'normalized'), default=None)
    env.add_argument('--reward-scale', type=float, default=None)
    env.add_argument('--frame-skip', type=int, default=5)
    env.add_argument('--horizon', type=int, default=500)
    env.add_argument('--transmittance', type=float, default=None, help='override obstacle transmittance')

    ppo = parser.add_argument_group('MAPPO')
    ppo.add_argument('--num-envs', type=int, default=256)
    ppo.add_argument('--num-steps', type=int, default=25)
    ppo.add_argument('--total-timesteps', type=float, default=5e6, help='agent steps over all envs')
    ppo.add_argument('--update-epochs', type=int, default=4)
    ppo.add_argument('--num-minibatches', type=int, default=4)
    ppo.add_argument('--lr', type=float, default=5e-4)
    ppo.add_argument('--gamma', type=float, default=0.99)
    ppo.add_argument('--gae-lambda', type=float, default=0.95)
    ppo.add_argument('--clip-eps', type=float, default=0.3)
    ppo.add_argument('--ent-coef', type=float, default=0.01)
    ppo.add_argument('--vf-coef', type=float, default=0.5)
    ppo.add_argument('--max-grad-norm', type=float, default=0.5)
    ppo.add_argument('--no-anneal-lr', action='store_true')
    ppo.add_argument('--action-type', choices=('discrete', 'continuous'), default='discrete')
    ppo.add_argument('--discrete-levels', type=int, default=5)
    ppo.add_argument('--hiddens', type=int, nargs='+', default=[512, 256])
    ppo.add_argument('--activation', default='tanh')
    ppo.add_argument('--layer-norm', action='store_true')
    ppo.add_argument('--critic', choices=('centralized', 'local'), default='centralized', help='MAPPO / IPPO')
    ppo.add_argument('--advisor', choices=('none', 'foresight'), default='none', help='append FORESIGHT advice')
    ppo.add_argument('--hierarchical', action='store_true', help='FORESIGHT-HRL: (option, transmit) actions')
    ppo.add_argument('--student', action='store_true', help='Hindsight student: lattice moves on FORESIGHT features')
    ppo.add_argument('--comm', action='store_true', help='communication control over FORESIGHT-T (published intents)')
    ppo.add_argument('--no-comm-hindsight', action='store_true', help='critic without the exogenous future')
    ppo.add_argument('--packet-cost', type=float, default=0.05)
    ppo.add_argument('--intent-cost', type=float, default=0.04)
    ppo.add_argument('--init-actor', default=None, help='msgpack of initial actor parameters (e.g. distilled student)')
    ppo.add_argument('--actor-warmup-updates', type=int, default=0, help='critic-only updates at the start')
    ppo.add_argument('--kl-coef', type=float, default=0.0, help='KL(initial actor || actor) weight')
    ppo.add_argument('--no-learn-transmission', action='store_true', help='FORESIGHT-HRL: always transmit')
    ppo.add_argument('--comm-cost', type=float, default=0.1)
    ppo.add_argument('--actuation-cost', type=float, default=0.05)
    ppo.add_argument('--num-candidates', type=int, default=8)
    ppo.add_argument('--channel-slots', type=int, default=4)
    ppo.add_argument('--channel-loss', type=float, default=0.0)

    system = parser.add_argument_group('system')
    system.add_argument('--dtype', choices=('float32', 'bfloat16'), default='float32', help='matmul dtype')
    system.add_argument(
        '--prng-impl',
        choices=('threefry2x32', 'rbg', 'unsafe_rbg'),
        default=None,
        help='PRNG implementation (rbg is much faster on TPU)',
    )
    system.add_argument('--compilation-cache', default=None, help='persistent XLA compilation cache dir')
    system.add_argument('--num-devices', type=int, default=None)
    system.add_argument('--multi-host', action='store_true', help='call jax.distributed.initialize()')
    system.add_argument('--updates-per-call', type=int, default=10)
    system.add_argument('--seed', type=int, default=0)
    system.add_argument('--eval-interval', type=int, default=0, help='evaluate every N updates (0: off)')
    system.add_argument('--eval-envs', type=int, default=16)
    system.add_argument('--eval-env-steps', type=int, default=1000)
    system.add_argument('--no-final-eval', action='store_true', help='skip the final evaluation (saves a compile)')
    system.add_argument('--save-dir', default=None)
    system.add_argument('--log-file', default=None, help='JSON-lines metrics file')
    return parser.parse_args(argv)


def main(argv=None):  # pylint: disable=too-many-locals
    args = parse_args(argv)

    # pylint: disable=import-outside-toplevel
    import jax

    if args.multi_host:
        jax.distributed.initialize()  # must run before any computation
    if args.prng_impl is not None:
        jax.config.update('jax_default_prng_impl', args.prng_impl)
    if args.compilation_cache is not None:
        jax.config.update('jax_compilation_cache_dir', args.compilation_cache)
        jax.config.update('jax_persistent_cache_min_compile_time_secs', 1.0)

    from mate_jax.mappo import MAPPOConfig, MAPPOTrainer

    reward = args.reward or ('coverage' if args.team == 'camera' else 'raw')
    overrides = {}
    if args.transmittance is not None:
        overrides['obstacle'] = {'transmittance': args.transmittance}
    config = MAPPOConfig(
        env_config=args.config,
        env_overrides=overrides,
        team=args.team,
        opponent=args.opponent,
        frame_skip=args.frame_skip,
        horizon=args.horizon,
        reward=reward,
        reward_scale=args.reward_scale,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        total_timesteps=int(args.total_timesteps),
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        actor_lr=args.lr,
        critic_lr=args.lr,
        anneal_lr=not args.no_anneal_lr,
        action_type=args.action_type,
        discrete_levels=args.discrete_levels,
        actor_hiddens=tuple(args.hiddens),
        critic_hiddens=tuple(args.hiddens),
        activation=args.activation,
        layer_norm=args.layer_norm,
        critic=args.critic,
        advisor=args.advisor,
        hierarchical=args.hierarchical,
        student=args.student,
        comm=args.comm,
        comm_hindsight=not args.no_comm_hindsight,
        packet_cost=args.packet_cost,
        intent_cost=args.intent_cost,
        init_actor=args.init_actor,
        actor_warmup_updates=args.actor_warmup_updates,
        kl_coef=args.kl_coef,
        learn_transmission=not args.no_learn_transmission,
        comm_cost=args.comm_cost,
        actuation_cost=args.actuation_cost,
        num_candidates=args.num_candidates,
        channel_slots=args.channel_slots,
        channel_loss=args.channel_loss,
        compute_dtype=args.dtype,
        seed=args.seed,
        num_devices=args.num_devices,
        updates_per_call=args.updates_per_call,
        eval_interval=args.eval_interval,
        eval_envs=args.eval_envs,
        eval_env_steps=args.eval_env_steps,
    )
    trainer = MAPPOTrainer(config)
    is_main = jax.process_index() == 0
    if is_main:
        print(trainer.describe(), flush=True)

    start = time.perf_counter()
    runner = trainer.init()
    log_file = open(args.log_file, 'a', encoding='UTF-8') if (args.log_file and is_main) else None
    try:
        for metrics, runner in trainer.iterate(runner):
            if not is_main:
                continue
            fmt = lambda v: 'nan' if math.isnan(v) else f'{v:.4g}'  # noqa: E731
            print(
                f'[update {metrics["update"]:>5d}/{config.num_updates}] '
                f'agent_steps={metrics["agent_steps"]:.3g} '
                f'reward={fmt(metrics["reward"])} coverage={fmt(metrics["coverage_rate"])} '
                f'ep_return={fmt(metrics["episode_return"])} ep_coverage={fmt(metrics["episode_coverage"])} '
                f'entropy={fmt(metrics["entropy"])} kl={fmt(metrics["approx_kl"])} '
                f'vf_loss={fmt(metrics["vf_loss"])} SPS={metrics["sps"]:.3g}'
                + f' tx={metrics["tx_rate"]:.3f}'
                + (f' eval_coverage={metrics["eval_coverage"]:.4f} eval_tx={metrics["eval_tx_rate"]:.3f}' if 'eval_coverage' in metrics else ''),
                flush=True,
            )
            if log_file is not None:
                log_file.write(json.dumps(metrics) + '\n')
                log_file.flush()
    finally:
        if log_file is not None:
            log_file.close()

    if is_main:
        print(f'Training finished in {time.perf_counter() - start:.1f}s', flush=True)
        if not args.no_final_eval:
            evaluation = trainer.evaluate(runner.actor_state.params)
            print(f'Evaluation (deterministic, {trainer.team_env.horizon} steps): {evaluation}', flush=True)
        if args.save_dir:
            path = trainer.save(runner, args.save_dir)
            print(f'Saved parameters to {os.path.abspath(path)}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
