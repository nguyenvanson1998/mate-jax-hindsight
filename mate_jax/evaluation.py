"""Evaluate a camera team policy against a target team policy (both pure JAX)."""

from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np


__all__ = ['evaluate_coverage', 'make_coverage_evaluator']


def evaluate_coverage(
    env, camera_policy, target_policy, seed: int = 0, num_envs: int = 32, num_steps: int = 1000
) -> Dict[str, np.ndarray]:
    """One-shot evaluation (compiles a new program; use :func:`make_coverage_evaluator` to reuse it)."""

    return make_coverage_evaluator(env, camera_policy, target_policy, num_envs, num_steps)(seed)


def make_coverage_evaluator(env, camera_policy, target_policy, num_envs: int = 32, num_steps: int = 1000):
    """Compile once and return ``evaluate(seed)``, which runs ``num_envs`` parallel episodes of
    ``num_steps`` steps (no reset) and returns per-episode means of the environment metrics (``coverage_rate``, ``real_coverage_rate``, ...) and of the
    per-camera metrics, shaped (num_envs, C):

    * ``camera_coverage``: fraction of the targets the camera sees;
    * ``camera_unique_coverage``: fraction of the targets only this camera sees;
    * ``camera_rotation`` / ``camera_zoom``: mean |rotation| / |zoom| per step (actuation effort);

    plus ``redundancy``, the fraction of targets seen by two or more cameras."""

    def run(key):
        k_reset, k_camera, k_target, k_loop = jax.random.split(key, 4)
        (camera_obs, target_obs), state = env.reset(k_reset)
        camera_carry = camera_policy.init(k_camera, camera_obs)
        target_carry = target_policy.init(k_target, target_obs)

        def body(carry, k):
            state, camera_obs, target_obs, camera_carry, target_carry = carry
            k_camera, k_target, k_env = jax.random.split(k, 3)
            camera_action, camera_carry = camera_policy.act(k_camera, camera_carry, camera_obs)
            target_action, target_carry = target_policy.act(k_target, target_carry, target_obs)
            timestep, state = env.step(k_env, state, camera_action, target_action)
            mask = state.camera_target_view_mask.astype(jnp.float32)  # (C, T)
            viewers = mask.sum(0)
            metrics = {
                'coverage_rate': timestep.info['coverage_rate'],
                'real_coverage_rate': timestep.info['real_coverage_rate'],
                'camera_reward': timestep.camera_reward,
                'camera_coverage': mask.mean(-1),
                'camera_unique_coverage': (mask * (viewers == 1)[None]).mean(-1),
                'camera_rotation': jnp.abs(camera_action[:, 0]),
                'camera_zoom': jnp.abs(camera_action[:, 1]),
                'redundancy': (viewers >= 2).astype(jnp.float32).mean(),
            }
            if isinstance(camera_carry, dict) and 'traffic' in camera_carry:  # bytes sent per camera
                metrics['camera_bytes'] = camera_carry['traffic']
            carry = (state, timestep.camera_observation, timestep.target_observation, camera_carry, target_carry)
            return carry, metrics

        carry = (state, camera_obs, target_obs, camera_carry, target_carry)
        _, metrics = jax.lax.scan(body, carry, jax.random.split(k_loop, num_steps))
        return jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), metrics)

    compiled = jax.jit(jax.vmap(run))

    def evaluate(seed: int = 0) -> Dict[str, np.ndarray]:
        return jax.device_get(compiled(jax.random.split(jax.random.PRNGKey(seed), num_envs)))

    return evaluate


class FrameSkipPolicy:
    """Re-decide every ``frame_skip`` environment steps and repeat the action in between.

    Learned controllers are trained with a frame skip; this wrapper lets :func:`make_coverage_evaluator`
    evaluate them step by step on the same paired episodes as the planners. ``decide(key, observation)``
    returns the (N, 2) environment action.
    """

    def __init__(self, decide, num_agents: int, frame_skip: int):
        self.decide = decide
        self.num_agents = int(num_agents)
        self.frame_skip = int(frame_skip)

    def init(self, key, observation):  # pylint: disable=unused-argument
        return (jnp.zeros((), jnp.int32), jnp.zeros((self.num_agents, 2), observation.dtype))

    def act(self, key, carry, observation):
        step, held = carry
        action = jax.lax.cond(
            step % self.frame_skip == 0, lambda: self.decide(key, observation).astype(held.dtype), lambda: held
        )
        return action, (step + 1, action)
