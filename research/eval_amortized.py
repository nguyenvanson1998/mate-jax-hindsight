"""Paired evaluation and decision latency of learned FORESIGHT (students that imitate the trajectory planner).

Coverage and network metrics: the paired episodes of eval_foresight.py (seed 0, same keys), for the student and for
its teacher executed through the same waypoint controller. Latency: one team decision of the student (option-level
features + MLP) against the trajectory planner it imitates, for a single environment. All runs share one config.
Usage: python3 research/eval_amortized.py --runs results/amortized/*_s? --out results/paper/amortized_4v8.json
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from flax import serialization  # noqa: E402

import mate_jax  # noqa: E402
from mate_jax import agents as A  # noqa: E402
from mate_jax.agents import _camera_self  # noqa: E402
from mate_jax.evaluation import make_coverage_evaluator  # noqa: E402
from mate_jax.foresight import ForesightCameraPolicy  # noqa: E402
from mate_jax.foresight.planner import OPTION_INTENT_BYTES  # noqa: E402
from mate_jax.hindsight.student import HindsightDistiller  # noqa: E402

TARGETS = {'greedy': A.GreedyTargetPolicy, 'waypoint': A.RandomWaypointTargetPolicy}
METRICS = ('camera_coverage', 'camera_unique_coverage', 'camera_rotation', 'camera_zoom', 'camera_bytes')


class WaypointPolicy:
    """The student (or its teacher) as a team policy for :func:`make_coverage_evaluator`."""

    def __init__(self, distiller, params, teacher=False):
        self.distiller, self.params, self.teacher = distiller, params, teacher

    def init(self, key, observation):
        return self.distiller.planner.init(key, observation)

    def choose(self, carry, observation):
        d = self.distiller
        if self.teacher:
            return d.features.suggestion(carry, observation)
        logits, _ = d.actor.apply(self.params, d.actor_input(d.features(carry, observation)))
        return jnp.argmax(logits, axis=-1)

    def act(self, key, carry, observation):
        d = self.distiller
        carry = d.planner.observe(key, carry, observation)
        me = _camera_self(observation)
        action = d.waypoints.action(me['orientation'], me['viewing_angle'], self.choose(carry, observation))
        if not self.teacher:  # the option-level features need the teammates' option indices
            carry = {**carry, 'traffic': carry['traffic'] + OPTION_INTENT_BYTES}
        return action.astype(observation.dtype), carry


def latency(fn, *args, repeats=50):
    jax.block_until_ready(fn(*args))
    start = time.perf_counter()
    for _ in range(repeats):
        jax.block_until_ready(fn(*args))
    return (time.perf_counter() - start) / repeats * 1e3


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--num-envs', type=int, default=32)
    parser.add_argument('--num-steps', type=int, default=1000)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    rows, measured = [], None

    def save():
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, 'w', encoding='UTF-8') as file:
                json.dump(rows, file, indent=2)

    for directory in args.runs:
        with open(os.path.join(directory, 'config.json'), encoding='UTF-8') as file:
            config = json.load(file)
        env = mate_jax.make(config['config'])
        targets = config['targets']
        target_policy = TARGETS[targets].from_env(env)
        distiller = HindsightDistiller(
            env, target_policy, ForesightCameraPolicy.from_env(env, **json.loads(config['planner_json'])),
            num_steps=1, hiddens=config['hiddens'], target='planner', lookahead=config['lookahead'],
            planner_rounds=config.get('planner_rounds', 2),
        )
        params, _ = distiller.init(jax.random.PRNGKey(0))
        with open(os.path.join(directory, 'params.msgpack'), 'rb') as file:
            params = serialization.from_bytes(params, file.read())
        for mode, teacher in (('student', False), ('teacher', True)):
            start = time.perf_counter()
            metrics = make_coverage_evaluator(env, WaypointPolicy(distiller, params, teacher), target_policy,
                                              args.num_envs, args.num_steps)(0)
            coverage = np.asarray(metrics['coverage_rate'])
            row = {
                'config': config['config'], 'targets': targets, 'variant': f'amortized-{mode}', 'seed': config['seed'],
                'run': os.path.basename(os.path.normpath(directory)), 'coverage': float(coverage.mean()),
                'coverage_se': float(coverage.std(ddof=1) / np.sqrt(coverage.size)), 'per_episode_coverage': coverage.tolist(),
                'redundancy': float(np.mean(metrics['redundancy'])),
                'per_camera': {key: np.asarray(metrics[key]).mean(0).tolist() for key in METRICS if key in metrics},
                'seconds': time.perf_counter() - start,
            }
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k not in ('per_episode_coverage', 'per_camera')}), flush=True)
            save()
        if measured is None:  # latency with trained weights, on one state
            (camera_obs, _), _ = env.reset(jax.random.PRNGKey(1))
            carry = distiller.planner.init(jax.random.PRNGKey(2), camera_obs)
            student = jax.jit(WaypointPolicy(distiller, params).choose)
            planner = jax.jit(distiller.features.suggestion)
            measured = {'config': config['config'], 'variant': 'latency', 'student_ms': latency(student, carry, camera_obs),
                        'planner_ms': latency(planner, carry, camera_obs)}
            rows.append(measured)
            print(json.dumps(measured), flush=True)
            save()


if __name__ == '__main__':
    main()
