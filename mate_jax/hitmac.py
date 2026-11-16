"""HiT-MAC (Xu, Zhong and Wang, NeurIPS 2020) re-implemented for MATE-JAX.

A centralized coordinator acts every ``period`` steps: it encodes every (camera, target) pair, applies
self-attention over all pairs and outputs, per pair, the probability that the camera should track the target
(a factorized Bernoulli goal map). Executors then track their assigned targets for ``period`` steps. As in the
paper's coordinator training, the executor is scripted: it turns and zooms so that its sector spans the assigned
targets at their last known positions (it scans when nothing is assigned). The critic is a sum of per-pair
contributions of the attended pair embeddings (a state-value form of the approximate marginal contribution).
The coordinator observes the team's last known target positions, i.e. it assumes perfect communication.
Training is PPO on the coordinator's decisions (the original uses A3C).
"""

from typing import Any, Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct

from mate_jax import constants as consts
from mate_jax.agents import _camera_self
from mate_jax.constants import MAX_CAMERA_VIEWING_ANGLE
from mate_jax.geometry import arctan2_deg, normalize_angle, safe_norm


__all__ = ['HitMacNet', 'HitMacPolicy', 'HitMacTrainer']

P = consts.PRESERVED_DIM
PAIR_FEATURES = 14


class HitMacNet(nn.Module):
    """Pair encoder, self-attention over all pairs, Bernoulli logits and per-pair value contributions."""

    embed: int = 64
    heads: int = 4

    @nn.compact
    def __call__(self, pairs):  # pairs: (..., C * T, F)
        h = nn.relu(nn.Dense(self.embed)(pairs))
        h = nn.relu(nn.Dense(self.embed)(h))
        attended = nn.MultiHeadDotProductAttention(num_heads=self.heads, qkv_features=self.embed)(h, h)
        h = nn.LayerNorm()(h + attended)
        logits = nn.Dense(1)(nn.relu(nn.Dense(self.embed)(h)))[..., 0]
        contribution = nn.Dense(1)(nn.relu(nn.Dense(self.embed)(h)))[..., 0]
        return logits, contribution.sum(-1)


@struct.dataclass
class HitMacPolicy:
    """Camera-team policy (``init`` / ``act``) of the coordinator plus scripted executors."""

    num_cameras: int = struct.field(pytree_node=False)
    num_targets: int = struct.field(pytree_node=False)
    area: float = struct.field(pytree_node=False)
    rotation_step: float = struct.field(pytree_node=False)
    zooming_step: float = struct.field(pytree_node=False)
    period: int = struct.field(pytree_node=False, default=10)
    deterministic: bool = struct.field(pytree_node=False, default=True)
    params: Any = None

    @classmethod
    def from_env(cls, env, params=None, **kwargs):
        cfg = env.config
        return cls(num_cameras=env.num_cameras, num_targets=env.num_targets, area=float(env.camera_area_product),
                   rotation_step=float(cfg.camera_rotation_step), zooming_step=float(cfg.camera_zooming_step),
                   params=params, **kwargs)

    @property
    def net(self):
        return HitMacNet()

    # Team memory and pair features ------------------------------------------------------------- #

    def _targets(self, observation):
        offset = P + consts.CAMERA_STATE_DIM_PRIVATE
        size = consts.TARGET_STATE_DIM_PUBLIC + 1
        block = observation[..., offset : offset + self.num_targets * size]
        block = block.reshape(block.shape[:-1] + (self.num_targets, size))
        return block[..., 0:2], block[..., -1] > 0.5  # (C, T, 2), (C, T)

    def memory(self, carry, observation):
        positions, seen = self._targets(observation)
        any_seen = jnp.any(seen, axis=0)
        first = jnp.argmax(seen, axis=0)
        now = jnp.take_along_axis(positions, first[None, :, None], axis=0)[0]
        return {
            **carry,
            'position': jnp.where(any_seen[:, None], now, carry['position']),
            'age': jnp.where(any_seen, 0, carry['age'] + 1),
            'known': carry['known'] | any_seen,
            'seen': seen,
        }

    def pair_features(self, carry, observation):
        me = _camera_self(observation)
        location, orientation = me['location'], me['orientation']
        relative = carry['position'][None, :, :] - location[:, None, :]  # (C, T, 2)
        distance = safe_norm(relative)
        bearing = normalize_angle(arctan2_deg(relative[..., 1], relative[..., 0]) - orientation[:, None])
        rad = jnp.deg2rad(bearing)
        theta = jnp.deg2rad(orientation)
        C, T = self.num_cameras, self.num_targets
        features = jnp.concatenate([
            jnp.broadcast_to(location[:, None, :] / 1000.0, (C, T, 2)),
            jnp.broadcast_to(jnp.stack([jnp.sin(theta), jnp.cos(theta)], -1)[:, None, :], (C, T, 2)),
            jnp.broadcast_to((me['viewing_angle'] / 180.0)[:, None, None], (C, T, 1)),
            relative / 1000.0, (distance / 1000.0)[..., None],
            jnp.stack([jnp.sin(rad), jnp.cos(rad)], -1),
            carry['seen'].astype(jnp.float32)[..., None],
            jnp.broadcast_to(jnp.any(carry['seen'], 0)[None, :, None], (C, T, 1)).astype(jnp.float32),
            jnp.broadcast_to((jnp.minimum(carry['age'], 100) / 100.0)[None, :, None], (C, T, 1)),
            jnp.broadcast_to(carry['known'][None, :, None], (C, T, 1)).astype(jnp.float32),
        ], axis=-1)
        return features.reshape(C * T, PAIR_FEATURES)

    # Scripted executor ------------------------------------------------------------------------- #

    def executor(self, carry, observation):
        me = _camera_self(observation)
        location, orientation, viewing = me['location'], me['orientation'], me['viewing_angle']
        relative = carry['position'][None, :, :] - location[:, None, :]
        distance = safe_norm(relative)
        bearing = normalize_angle(arctan2_deg(relative[..., 1], relative[..., 0]) - orientation[:, None])
        mine = carry['assign'] & carry['known'][None, :]  # (C, T)
        any_mine = jnp.any(mine, axis=1)
        low = jnp.min(jnp.where(mine, bearing, 180.0), axis=1)
        high = jnp.max(jnp.where(mine, bearing, -180.0), axis=1)
        far = jnp.max(jnp.where(mine, distance, 0.0), axis=1)
        goal_orientation = normalize_angle(orientation + 0.5 * (low + high))
        # Widest viewing angle whose range still reaches the farthest assigned target (as MATE's greedy cameras),
        # so that the assigned targets stay in view while they move.
        reach = self.area / jnp.maximum(far, 1.0) ** 2
        goal_viewing = jnp.clip(reach, me['min_viewing_angle'], MAX_CAMERA_VIEWING_ANGLE)
        rotate = jnp.clip(normalize_angle(goal_orientation - orientation), -self.rotation_step, self.rotation_step)
        zoom = jnp.clip(goal_viewing - viewing, -self.zooming_step, self.zooming_step)
        rotate = jnp.where(any_mine, rotate, self.rotation_step)  # scan when nothing is assigned
        zoom = jnp.where(any_mine, zoom, 0.0)
        return jnp.stack([rotate, zoom], axis=-1)

    # Policy API ------------------------------------------------------------------------------- #

    def init(self, key, observation):  # pylint: disable=unused-argument
        carry = {
            'position': jnp.zeros((self.num_targets, 2), jnp.float32),
            'age': jnp.full((self.num_targets,), 100, jnp.int32),
            'known': jnp.zeros((self.num_targets,), bool),
            'seen': jnp.zeros((self.num_cameras, self.num_targets), bool),
            'assign': jnp.zeros((self.num_cameras, self.num_targets), bool),
            'step': jnp.zeros((), jnp.int32),
        }
        return self.memory(carry, observation)

    def assign(self, key, carry, observation):
        logits, _ = self.net.apply(self.params, self.pair_features(carry, observation))
        if self.deterministic:
            choice = logits > 0.0
        else:
            choice = jax.random.bernoulli(key, jax.nn.sigmoid(logits))
        return choice.reshape(self.num_cameras, self.num_targets)

    def act(self, key, carry, observation):
        carry = self.memory(carry, observation)
        decide = jnp.mod(carry['step'], self.period) == 0
        carry = {**carry, 'assign': jnp.where(decide, self.assign(key, carry, observation), carry['assign'])}
        action = self.executor(carry, observation)
        return action.astype(observation.dtype), {**carry, 'step': carry['step'] + 1}


class HitMacTrainer:
    """PPO on the coordinator's decisions (factorized per-pair ratios, shared advantage)."""

    def __init__(self, env, target_policy, num_envs=128, decisions=16, horizon=100, gamma=0.95, lam=0.95,
                 lr=3e-4, epochs=4, minibatches=4, clip=0.2, ent_coef=0.01, vf_coef=0.5, seed=0, anneal_updates=0):
        self.env, self.target_policy = env, target_policy
        self.policy = HitMacPolicy.from_env(env, deterministic=False)
        self.num_envs, self.decisions, self.horizon = num_envs, decisions, horizon
        self.gamma, self.lam, self.epochs, self.minibatches = gamma, lam, epochs, minibatches
        self.clip, self.ent_coef, self.vf_coef = clip, ent_coef, vf_coef
        # With anneal_updates > 0 the learning rate and the entropy bonus decay linearly to zero over that many
        # updates, so that the policy converges instead of drifting (the step size of PPO stays constant otherwise).
        self.anneal_updates = int(anneal_updates)
        steps = max(1, self.anneal_updates * epochs * minibatches)
        schedule = optax.linear_schedule(lr, 0.0, steps) if self.anneal_updates else lr
        self.optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(schedule))
        self.seed = seed
        self._rollout = jax.jit(self.rollout)
        self._update = jax.jit(self.update)

    def init(self):
        key = jax.random.PRNGKey(self.seed)
        k_net, k_env = jax.random.split(key)
        params = self.policy.net.init(k_net, jnp.zeros((self.policy.num_cameras * self.policy.num_targets, PAIR_FEATURES)))
        envs = jax.vmap(self._reset)(jax.random.split(k_env, self.num_envs))
        return params, self.optimizer.init(params), envs

    def _reset(self, key):
        k_reset, k_camera, k_target = jax.random.split(key, 3)
        (camera_obs, target_obs), state = self.env.reset(k_reset)
        policy = self.policy
        return {'state': state, 'camera_obs': camera_obs, 'target_obs': target_obs,
                'carry': policy.init(k_camera, camera_obs), 'target_carry': self.target_policy.init(k_target, target_obs),
                't': jnp.zeros((), jnp.int32)}

    def _decision(self, params, env_state, key):
        """One coordinator decision and ``period`` executor steps in one environment."""

        policy = self.policy.replace(params=params)
        k_assign, k_steps, k_reset = jax.random.split(key, 3)
        carry, obs = env_state['carry'], env_state['camera_obs']
        pairs = policy.pair_features(carry, obs)
        logits, value = policy.net.apply(params, pairs)
        choice = jax.random.bernoulli(k_assign, jax.nn.sigmoid(logits))
        logp = choice * jax.nn.log_sigmoid(logits) + (1 - choice) * jax.nn.log_sigmoid(-logits)
        carry = {**carry, 'assign': choice.reshape(policy.num_cameras, policy.num_targets)}

        def step(s, k):
            state, camera_obs, target_obs, carry, target_carry = s
            k_target, k_env = jax.random.split(k)
            carry = policy.memory(carry, camera_obs)
            action = policy.executor(carry, camera_obs)
            target_action, target_carry = self.target_policy.act(k_target, target_carry, target_obs)
            timestep, state = self.env.step(k_env, state, action.astype(camera_obs.dtype), target_action)
            return (state, timestep.camera_observation, timestep.target_observation, carry, target_carry), timestep.info['coverage_rate']

        s = (env_state['state'], obs, env_state['target_obs'], carry, env_state['target_carry'])
        s, coverage = jax.lax.scan(step, s, jax.random.split(k_steps, policy.period))
        t = env_state['t'] + 1
        done = t >= self.horizon
        new = {'state': s[0], 'camera_obs': s[1], 'target_obs': s[2], 'carry': s[3], 'target_carry': s[4], 't': t}
        fresh = self._reset(k_reset)
        new = jax.tree_util.tree_map(lambda a, b: jnp.where(done, a, b), fresh, new)
        return new, {'pairs': pairs, 'choice': choice, 'logp': logp, 'value': value,
                     'reward': coverage.mean(), 'done': done}

    def rollout(self, params, envs, key):
        def body(envs, k):
            envs, out = jax.vmap(lambda e, kk: self._decision(params, e, kk))(envs, jax.random.split(k, self.num_envs))
            return envs, out

        envs, batch = jax.lax.scan(body, envs, jax.random.split(key, self.decisions))  # (D, E, ...)
        last_value = jax.vmap(lambda e: self.policy.net.apply(params, self.policy.pair_features(e['carry'], e['camera_obs']))[1])(envs)
        return envs, batch, last_value

    def advantages(self, batch, last_value):
        def body(carry, x):
            gae, next_value = carry
            reward, value, done = x
            mask = 1.0 - done.astype(jnp.float32)
            delta = reward + self.gamma * next_value * mask - value
            gae = delta + self.gamma * self.lam * mask * gae
            return (gae, value), gae

        _, adv = jax.lax.scan(body, (jnp.zeros_like(last_value), last_value),
                              (batch['reward'], batch['value'], batch['done']), reverse=True)
        return adv, adv + batch['value']

    def update(self, params, opt_state, batch, last_value, key, progress=0.0):
        adv, returns = self.advantages(batch, last_value)
        ent_coef = self.ent_coef * (1.0 - progress) if self.anneal_updates else self.ent_coef
        flat = lambda x: x.reshape((-1,) + x.shape[2:])  # noqa: E731
        data = {'pairs': flat(batch['pairs']), 'choice': flat(batch['choice']), 'logp': flat(batch['logp']),
                'adv': flat(adv), 'ret': flat(returns)}
        data['adv'] = (data['adv'] - data['adv'].mean()) / (data['adv'].std() + 1e-8)
        n = data['adv'].shape[0]
        size = n // self.minibatches

        def loss_fn(params, mb):
            logits, value = jax.vmap(lambda p: self.policy.net.apply(params, p))(mb['pairs'])
            logp = mb['choice'] * jax.nn.log_sigmoid(logits) + (1 - mb['choice']) * jax.nn.log_sigmoid(-logits)
            ratio = jnp.exp(logp - mb['logp'])  # per pair
            a = mb['adv'][:, None]
            pg = -jnp.minimum(ratio * a, jnp.clip(ratio, 1 - self.clip, 1 + self.clip) * a).mean()
            p = jax.nn.sigmoid(logits)
            entropy = -(p * jnp.log(p + 1e-8) + (1 - p) * jnp.log(1 - p + 1e-8)).mean()
            vf = jnp.square(value - mb['ret']).mean()
            return pg + self.vf_coef * vf - ent_coef * entropy, (pg, vf, entropy)

        def epoch(carry, k):
            params, opt_state = carry
            order = jax.random.permutation(k, n)

            def minibatch(carry, idx):
                params, opt_state = carry
                mb = jax.tree_util.tree_map(lambda x: x[idx], data)
                (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, mb)
                updates, opt_state = self.optimizer.update(grads, opt_state, params)
                return (optax.apply_updates(params, updates), opt_state), (loss,) + aux

            return jax.lax.scan(minibatch, (params, opt_state), order[: size * self.minibatches].reshape(self.minibatches, size))

        (params, opt_state), stats = jax.lax.scan(epoch, (params, opt_state), jax.random.split(key, self.epochs))
        return params, opt_state, jax.tree_util.tree_map(jnp.mean, stats)

    def env_steps_per_update(self):
        return self.num_envs * self.decisions * self.policy.period

    def summary(self, batch, stats) -> Dict[str, float]:
        loss, pg, vf, entropy = (float(x) for x in stats)
        return {'coverage_rate': float(batch['reward'].mean()), 'assign_rate': float(np.mean(batch['choice'])),
                'loss': loss, 'pg_loss': pg, 'vf_loss': vf, 'entropy': entropy}
