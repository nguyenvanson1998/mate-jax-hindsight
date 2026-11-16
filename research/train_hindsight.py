"""Hindsight -> Foresight: distill the clairvoyant teacher into decentralized cameras (BC + DAgger).

Every iteration solves the teacher on fresh episodes, rolls out the DAgger mixture (the teacher acts
with probability beta, decaying to 0), labels every visited state with the teacher's move, aggregates
the data and trains the student. Evaluation uses the episodes of ``headroom.py`` / ``clairvoyant.py``
(seed 0, 16 environments, 1000 steps), so the coverage is paired with FORESIGHT and the teacher bound.

Usage: python3 research/train_hindsight.py --targets greedy --iterations 10 --log results/hindsight/greedy.jsonl
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
from flax import serialization  # noqa: E402

import mate_jax  # noqa: E402
from mate_jax import agents as A  # noqa: E402
from mate_jax.foresight import ForesightCameraPolicy  # noqa: E402
from mate_jax.hindsight import HindsightDistiller  # noqa: E402


TARGETS = {'greedy': A.GreedyTargetPolicy, 'waypoint': A.RandomWaypointTargetPolicy}
EVAL_KEY_STEPS_EXTRA = 51  # headroom.py records num_steps + 51 steps


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', default='MATE-4v8-9.yaml')
    parser.add_argument('--targets', default='greedy', choices=sorted(TARGETS))
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--envs', type=int, default=16, help='episodes per DAgger iteration')
    parser.add_argument('--steps', type=int, default=500, help='steps per training episode')
    parser.add_argument('--betas', default='1,0.5,0.3,0.2,0.1,0', help='teacher probability per iteration (last repeats)')
    parser.add_argument('--subsample', type=int, default=2, help='keep every k-th step of a rollout')
    parser.add_argument('--max-samples', type=int, default=200000, help='aggregated dataset size (most recent)')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--hiddens', type=int, nargs='+', default=[512, 256])
    parser.add_argument('--target', default='value', choices=['value', 'waypoint', 'move', 'planner', 'privileged'])
    parser.add_argument('--layer-norm', action='store_true', help='privileged: LayerNorm in the student')
    parser.add_argument('--motion', action='store_true', help="privileged: teammates' recent pose changes as features")
    parser.add_argument('--prior', action='store_true', help="privileged: residual on the student's own planner move")
    parser.add_argument('--planner-rounds', type=int, default=2, help='planner target: best-response rounds of the teacher')
    parser.add_argument('--lookahead', type=int, default=3, help='waypoint: teacher pose this many steps ahead')
    parser.add_argument('--no-trajectory', action='store_true', help='no trajectory-planner waypoint in the features')
    parser.add_argument('--eval-envs', type=int, default=16)
    parser.add_argument('--eval-steps', type=int, default=1000)
    parser.add_argument('--eval-every', type=int, default=1)
    parser.add_argument('--planner-json', default='{}', help='FORESIGHT keyword arguments for the student features')
    parser.add_argument('--student-planner-json', default=None,
                        help="privileged: FORESIGHT keyword arguments of the student's own planner (default: --planner-json)")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval-seed', type=int, default=0,
                        help='episodes of the training-time evaluation (1: validation episodes disjoint from the test seed 0)')
    parser.add_argument('--lr-decay', action='store_true', help='privileged: learning rate decays linearly to zero over the iterations')
    parser.add_argument('--ema', type=float, default=0.0, help='privileged: evaluate and keep an exponential moving average of the parameters')
    parser.add_argument('--log', default=None)
    parser.add_argument('--save-dir', default=None)
    args = parser.parse_args()

    env = mate_jax.make(args.config)
    target_policy = TARGETS[args.targets].from_env(env)
    planner = ForesightCameraPolicy.from_env(env, **json.loads(args.planner_json))
    common = {
        'hiddens': args.hiddens, 'learning_rate': args.learning_rate, 'target': args.target, 'lookahead': args.lookahead,
        'use_trajectory': not args.no_trajectory, 'planner_rounds': args.planner_rounds,
    }
    if args.target == 'privileged':  # communicating teacher, communication-free student
        from mate_jax.hindsight.privileged import PrivilegedDistiller  # pylint: disable=import-outside-toplevel

        lite = json.loads(args.planner_json)  # lighter planners for teacher and student (training speed)
        own = json.loads(args.student_planner_json) if args.student_planner_json is not None else lite
        kwargs = {'hiddens': args.hiddens, 'learning_rate': args.learning_rate, 'teacher_kwargs': lite, 'student_kwargs': own,
                  'layer_norm': args.layer_norm, 'motion': args.motion, 'prior': args.prior}
        distiller = PrivilegedDistiller(env, target_policy, num_steps=args.steps, **kwargs)
        evaluator = PrivilegedDistiller(env, target_policy, num_steps=args.eval_steps,
                                        key_steps=args.eval_steps + EVAL_KEY_STEPS_EXTRA, **kwargs)
    else:
        distiller = HindsightDistiller(env, target_policy, planner, num_steps=args.steps, **common)
        evaluator = HindsightDistiller(
            env, target_policy, planner, num_steps=args.eval_steps, key_steps=args.eval_steps + EVAL_KEY_STEPS_EXTRA, **common
        )
    betas = [float(b) for b in args.betas.split(',')]
    params, opt_state = distiller.init(jax.random.PRNGKey(args.seed))
    print(f'features: {distiller.features.dim}, parameters: '
          f'{sum(x.size for x in jax.tree_util.tree_leaves(params)):,}', flush=True)

    eval_keys = jax.random.split(jax.random.PRNGKey(args.eval_seed), args.eval_envs)
    start = time.perf_counter()
    eval_plan, eval_teacher = evaluator.solve(eval_keys)
    _, _, teacher_closed_loop, _ = evaluator.rollout(params, eval_keys, eval_plan, beta=1.0)
    _, _, initial_student, _ = evaluator.rollout(params, eval_keys, eval_plan, beta=0.0)
    reference = {
        'teacher_bound': float(eval_teacher.mean()),
        'teacher_closed_loop': float(np.mean(teacher_closed_loop)),
        'initial_student': float(np.mean(initial_student)),  # = the trajectory planner with the prior actor
        'seconds': time.perf_counter() - start,
    }
    print(json.dumps({'reference': reference}), flush=True)
    if args.log:
        os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    log = open(args.log, 'a', encoding='UTF-8') if args.log else None

    features_buffer, labels_buffer = [], []
    best = (-1.0, params)  # checkpoint with the highest training-time evaluation (early stopping on validation)
    key = jax.random.PRNGKey(args.seed + 1)
    for iteration in range(args.iterations):
        tick = time.perf_counter()
        beta = betas[min(iteration, len(betas) - 1)]
        keys = jax.random.split(jax.random.PRNGKey(1000 + 100 * args.seed + iteration), args.envs)
        plan, teacher_coverage = distiller.solve(keys)
        features, labels, coverage, agreement = distiller.rollout(params, keys, plan, beta, seed=iteration)
        features = np.asarray(features[:, :: args.subsample]).reshape((-1,) + features.shape[2:])
        labels = np.asarray(labels[:, :: args.subsample]).reshape((-1,) + labels.shape[2:])
        features_buffer.append(features)
        labels_buffer.append(labels)
        data_x, data_y = np.concatenate(features_buffer), np.concatenate(labels_buffer)
        if data_x.shape[0] > args.max_samples:
            data_x, data_y = data_x[-args.max_samples :], data_y[-args.max_samples :]
            features_buffer, labels_buffer = [data_x], [data_y]
        if args.lr_decay:
            opt_state[1].hyperparams['learning_rate'] = jnp.asarray(args.learning_rate * (1.0 - iteration / args.iterations), jnp.float32)
        key, k = jax.random.split(key)
        params, opt_state, stats = distiller.train(
            params, opt_state, data_x, data_y, k, epochs=args.epochs, batch_size=args.batch_size
        )
        if args.ema:  # policy that is evaluated and saved; the rollouts use the current parameters
            averaged = params if iteration == 0 else jax.tree_util.tree_map(lambda a, p: args.ema * a + (1.0 - args.ema) * p, averaged, params)
        policy = averaged if args.ema else params
        row = {
            'iteration': iteration, 'beta': beta, 'samples': int(data_x.shape[0]),
            'rollout_coverage': float(np.mean(coverage)), 'teacher_coverage': float(np.mean(teacher_coverage)),
            'agreement': float(np.mean(agreement)), **stats,
        }
        if (iteration + 1) % args.eval_every == 0 or iteration == args.iterations - 1:
            _, _, eval_coverage, eval_agreement = evaluator.rollout(policy, eval_keys, eval_plan, beta=0.0)
            row.update({
                'eval_coverage': float(np.mean(eval_coverage)),
                'eval_coverage_se': float(np.std(eval_coverage, ddof=1) / np.sqrt(len(eval_coverage))),
                'eval_agreement': float(np.mean(eval_agreement)),
                'per_episode_eval_coverage': np.asarray(eval_coverage).tolist(),
            })
            if args.save_dir:  # every evaluated checkpoint, for checkpoint ensembles chosen on validation
                os.makedirs(os.path.join(args.save_dir, 'checkpoints'), exist_ok=True)
                with open(os.path.join(args.save_dir, 'checkpoints', f'{iteration:03d}.msgpack'), 'wb') as file:
                    file.write(serialization.to_bytes(jax.device_get(policy)))
            if row['eval_coverage'] > best[0]:
                best = (row['eval_coverage'], jax.device_get(policy))
        row['seconds'] = time.perf_counter() - tick
        print(json.dumps({k: v for k, v in row.items() if k != 'per_episode_eval_coverage'}), flush=True)
        if log:
            log.write(json.dumps({**row, **({'reference': reference} if iteration == 0 else {})}) + '\n')
            log.flush()
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, 'params.msgpack'), 'wb') as file:  # final policy
            file.write(serialization.to_bytes(jax.device_get(params if not args.ema else averaged)))
        with open(os.path.join(args.save_dir, 'best.msgpack'), 'wb') as file:  # best training-time evaluation
            file.write(serialization.to_bytes(best[1]))
        with open(os.path.join(args.save_dir, 'config.json'), 'w', encoding='UTF-8') as file:
            json.dump(vars(args), file, indent=2)


if __name__ == '__main__':
    main()
