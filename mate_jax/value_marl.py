"""Off-policy value-factorization MARL (VDN, QMIX, QPLEX, ...) for the camera team, fully on device.

Agents share one Q-network over their local observations (with one-hot ids); a mixing network combines the
chosen per-agent utilities into the joint value under the IGM principle. Everything (vectorized
environments, epsilon-greedy collection, a transition replay buffer, double-Q targets, target networks)
runs inside jitted ``lax.scan`` loops, as in :mod:`mate_jax.mappo`. The environment, the 5 x 5 grid of
pan/zoom commands, the frame skip and the evaluation protocol are those of the MAPPO baselines, so that
all learned controllers are compared on equal terms.

Mixers (``algo``):
* ``vdn``: Q_tot = sum_i Q_i (Sunehag et al., 2018);
* ``qmix``: monotonic mixing network with hypernetworks on the global state (Rashid et al., 2018);
* ``qplex``: duplex dueling with state-dependent positive weights and an attention over (state, action)
  for the advantages (Wang et al., 2021).
"""

import dataclasses
import json
import math
import time
from functools import partial
from typing import Any, Dict, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct

from mate_jax.env import MultiAgentTracking
from mate_jax.mappo import OPPONENTS
from mate_jax.models import MLP, discrete_action_table
from mate_jax.wrappers import SingleTeamEnv


__all__ = ['ValueConfig', 'ValueTrainer']


@dataclasses.dataclass
class ValueConfig:  # pylint: disable=too-many-instance-attributes
    algo: str = 'qmix'  # vdn / qmix / qplex / riskq / qfree / macpf
    env_config: str = 'MATE-4v8-9.yaml'
    opponent: str = 'greedy'
    frame_skip: int = 5
    horizon: int = 500
    discrete_levels: int = 5
    num_envs: int = 128
    total_timesteps: int = 2_000_000  # agent steps (decisions x environments), as for MAPPO
    hiddens: Sequence[int] = (256, 256)
    mixer_embed: int = 64
    buffer_size: int = 200_000  # transitions (one per environment and decision)
    batch_size: int = 1024
    learning_rate: float = 5e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_fraction: float = 0.3  # of the training budget
    target_update: int = 200  # gradient updates between target-network copies
    updates_per_step: int = 1  # gradient updates per vectorized environment step
    learning_starts: int = 10_000  # transitions before the first update
    max_grad_norm: float = 10.0
    steps_per_call: int = 200  # vectorized environment steps per jitted call (logging interval)
    eval_interval: int = 0  # evaluate every this many calls (0: never)
    eval_envs: int = 16
    eval_env_steps: int = 1000
    seed: int = 0

    def to_dict(self):
        result = dataclasses.asdict(self)
        result['hiddens'] = list(self.hiddens)
        return result


# Networks -------------------------------------------------------------------------------------- #


class AgentQ(nn.Module):
    num_actions: int
    hiddens: Sequence[int]

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.num_actions)(MLP(tuple(self.hiddens), 'relu')(x))


class VDNMixer(nn.Module):
    @nn.compact
    def __call__(self, chosen, state, utilities=None, actions=None):  # pylint: disable=unused-argument
        return chosen.sum(-1)


class QMixer(nn.Module):
    """Monotonic mixing: non-negative hypernetwork weights on the global state."""

    num_agents: int
    embed: int = 64

    @nn.compact
    def __call__(self, chosen, state, utilities=None, actions=None):  # pylint: disable=unused-argument
        w1 = jnp.abs(nn.Dense(self.num_agents * self.embed)(nn.relu(nn.Dense(self.embed)(state))))
        w1 = w1.reshape(state.shape[:-1] + (self.num_agents, self.embed))
        b1 = nn.Dense(self.embed)(state)
        hidden = nn.elu(jnp.einsum('...n,...ne->...e', chosen, w1) + b1)
        w2 = jnp.abs(nn.Dense(self.embed)(nn.relu(nn.Dense(self.embed)(state))))
        b2 = nn.Dense(1)(nn.relu(nn.Dense(self.embed)(state)))[..., 0]
        return jnp.sum(hidden * w2, axis=-1) + b2


class QPLEXMixer(nn.Module):
    """Duplex dueling: Q_tot = sum_i [w_i(s) V_i + b_i(s)] + sum_i lambda_i(s, a) w_i(s) A_i, all weights > 0."""

    num_agents: int
    num_actions: int
    embed: int = 64
    heads: int = 4

    @nn.compact
    def __call__(self, chosen, state, utilities=None, actions=None):
        values = utilities.max(-1)  # V_i = max_a Q_i
        advantages = chosen - values  # A_i <= 0
        w = jnp.abs(nn.Dense(self.num_agents)(nn.relu(nn.Dense(self.embed)(state)))) + 1e-10
        b = nn.Dense(self.num_agents)(nn.relu(nn.Dense(self.embed)(state)))
        joint_action = jax.nn.one_hot(actions, self.num_actions).reshape(actions.shape[:-1] + (-1,))
        sa = jnp.concatenate([state, joint_action], axis=-1)
        keys = nn.Dense(self.heads * self.num_agents)(nn.relu(nn.Dense(self.embed)(sa)))
        keys = keys.reshape(sa.shape[:-1] + (self.heads, self.num_agents))
        lam = jnp.abs(keys).sum(-2) + 1e-10  # (..., N) positive importance of every agent's advantage
        v_tot = jnp.sum(w * values + b, axis=-1)
        return v_tot + jnp.sum(lam * w * advantages, axis=-1)


# Replay buffer --------------------------------------------------------------------------------- #


@struct.dataclass
class Buffer:
    obs: jax.Array  # (B, N, D)
    state: jax.Array  # (B, S)
    actions: jax.Array  # (B, N)
    reward: jax.Array  # (B,)
    next_obs: jax.Array
    next_state: jax.Array
    terminal: jax.Array  # (B,) bool (truncation still bootstraps)
    position: jax.Array
    size: jax.Array


@struct.dataclass
class Runner:
    params: Any
    target_params: Any
    opt_state: Any
    env_state: Any
    obs: jax.Array
    state: jax.Array
    buffer: Buffer
    key: jax.Array
    env_steps: jax.Array  # vectorized environment steps so far
    updates: jax.Array


class ValueTrainer:
    def __init__(self, config: ValueConfig):
        self.config = cfg = config
        self.env = MultiAgentTracking(cfg.env_config)
        opponent = OPPONENTS[('camera', cfg.opponent)].from_env(self.env)
        self.team_env = SingleTeamEnv(
            self.env, team='camera', opponent=opponent, frame_skip=cfg.frame_skip, horizon=cfg.horizon,
            reward='coverage', cache_resets=True,
        )
        self.num_agents = self.team_env.num_agents
        self.action_table = discrete_action_table(cfg.discrete_levels)
        self.num_actions = self.action_table.shape[0]
        self.obs_dim = self.team_env.observation_dim + self.num_agents
        self.state_dim = self.team_env.state_dim
        self.agent = AgentQ(self.num_actions, tuple(cfg.hiddens))
        from mate_jax.value_marl_ext import EXTENSIONS  # pylint: disable=import-outside-toplevel

        self.ext = EXTENSIONS[cfg.algo](self) if cfg.algo in EXTENSIONS else None
        if self.ext is not None:
            self.mixer = None
        elif cfg.algo == 'vdn':
            self.mixer = VDNMixer()
        elif cfg.algo == 'qmix':
            self.mixer = QMixer(self.num_agents, cfg.mixer_embed)
        elif cfg.algo == 'qplex':
            self.mixer = QPLEXMixer(self.num_agents, self.num_actions, cfg.mixer_embed)
        else:
            raise ValueError(f'Unknown algorithm {cfg.algo!r}.')
        self.optimizer = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.learning_rate))
        self.total_env_steps = max(1, cfg.total_timesteps // cfg.num_envs)
        self.reward_scale = 1.0 / cfg.frame_skip

    # Helpers ----------------------------------------------------------------------------------- #

    def agent_input(self, obs):
        ids = jnp.broadcast_to(jnp.eye(self.num_agents, dtype=obs.dtype), obs.shape[:-1] + (self.num_agents,))
        return jnp.concatenate([obs, ids], axis=-1)

    def env_action(self, actions):
        return self.team_env.scale_actions(jnp.asarray(self.action_table)[actions])

    def utilities(self, params, obs):
        if self.ext is not None:
            return self.ext.scores(params, obs)
        return self.agent.apply(params['agent'], self.agent_input(obs))  # (..., N, A)

    def mix(self, params, chosen, state, utilities, actions):
        return self.mixer.apply(params['mixer'], chosen, state, utilities, actions)

    def epsilon(self, env_steps):
        cfg = self.config
        fraction = jnp.clip(env_steps / (cfg.epsilon_fraction * self.total_env_steps), 0.0, 1.0)
        return cfg.epsilon_start + fraction * (cfg.epsilon_end - cfg.epsilon_start)

    # Initialization ---------------------------------------------------------------------------- #

    def init(self, seed: Optional[int] = None) -> Runner:
        cfg = self.config
        key = jax.random.PRNGKey(cfg.seed if seed is None else seed)
        key, k_agent, k_mixer, k_env = jax.random.split(key, 4)
        obs, state, env_state = jax.vmap(self.team_env.reset)(jax.random.split(k_env, cfg.num_envs))
        dummy_obs = jnp.zeros((1, self.num_agents, self.team_env.observation_dim))
        agent_params = self.agent.init(k_agent, self.agent_input(dummy_obs))
        dummy_q = jnp.zeros((1, self.num_agents))
        mixer_params = None if self.mixer is None else self.mixer.init(
            k_mixer, dummy_q, jnp.zeros((1, self.state_dim)), jnp.zeros((1, self.num_agents, self.num_actions)),
            jnp.zeros((1, self.num_agents), jnp.int32),
        )
        params = {'agent': agent_params, 'mixer': mixer_params}
        if self.ext is not None:
            params = self.ext.init(k_agent, k_mixer)
        size = cfg.buffer_size
        buffer = Buffer(
            obs=jnp.zeros((size,) + obs.shape[1:], jnp.float32), state=jnp.zeros((size,) + state.shape[1:], jnp.float32),
            actions=jnp.zeros((size, self.num_agents), jnp.int32), reward=jnp.zeros((size,), jnp.float32),
            next_obs=jnp.zeros((size,) + obs.shape[1:], jnp.float32),
            next_state=jnp.zeros((size,) + state.shape[1:], jnp.float32), terminal=jnp.zeros((size,), bool),
            position=jnp.zeros((), jnp.int32), size=jnp.zeros((), jnp.int32),
        )
        return Runner(
            params=params, target_params=jax.tree_util.tree_map(jnp.copy, params), opt_state=self.optimizer.init(params),
            env_state=env_state,
            obs=obs, state=state, buffer=buffer, key=key, env_steps=jnp.zeros((), jnp.int32),
            updates=jnp.zeros((), jnp.int32),
        )

    # Learning ----------------------------------------------------------------------------------- #

    def td_loss(self, params, target_params, batch):
        cfg = self.config
        if self.ext is not None:
            return self.ext.loss(params, target_params, batch)
        q = self.utilities(params, batch['obs'])  # (B, N, A)
        chosen = jnp.take_along_axis(q, batch['actions'][..., None], axis=-1)[..., 0]
        q_tot = self.mix(params, chosen, batch['state'], q, batch['actions'])
        next_online = self.utilities(params, batch['next_obs'])
        next_actions = jnp.argmax(next_online, axis=-1)  # double Q: online argmax, target evaluation
        next_target = self.utilities(target_params, batch['next_obs'])
        next_chosen = jnp.take_along_axis(next_target, next_actions[..., None], axis=-1)[..., 0]
        next_tot = self.mix(target_params, next_chosen, batch['next_state'], next_target, next_actions)
        target = batch['reward'] + cfg.gamma * (1.0 - batch['terminal'].astype(jnp.float32)) * next_tot
        error = q_tot - jax.lax.stop_gradient(target)
        return jnp.mean(error**2), {'td_error': jnp.mean(jnp.abs(error)), 'q_tot': jnp.mean(q_tot)}

    def _sample(self, buffer: Buffer, key):
        index = jax.random.randint(key, (self.config.batch_size,), 0, jnp.maximum(buffer.size, 1))
        return {k: getattr(buffer, k)[index] for k in ('obs', 'state', 'actions', 'reward', 'next_obs', 'next_state', 'terminal')}

    def _add(self, buffer: Buffer, obs, state, actions, reward, next_obs, next_state, terminal):
        num = obs.shape[0]
        index = (buffer.position + jnp.arange(num)) % self.config.buffer_size
        return buffer.replace(
            obs=buffer.obs.at[index].set(obs), state=buffer.state.at[index].set(state),
            actions=buffer.actions.at[index].set(actions), reward=buffer.reward.at[index].set(reward),
            next_obs=buffer.next_obs.at[index].set(next_obs), next_state=buffer.next_state.at[index].set(next_state),
            terminal=buffer.terminal.at[index].set(terminal),
            position=(buffer.position + num) % self.config.buffer_size,
            size=jnp.minimum(buffer.size + num, self.config.buffer_size),
        )

    def _step(self, runner: Runner, _):
        cfg = self.config
        key, k_explore, k_random, k_env, k_sample = jax.random.split(runner.key, 5)
        q = self.utilities(runner.params, runner.obs)
        greedy = jnp.argmax(q, axis=-1)
        random_actions = jax.random.randint(k_random, greedy.shape, 0, self.num_actions)
        explore = jax.random.uniform(k_explore, greedy.shape) < self.epsilon(runner.env_steps)
        actions = jnp.where(explore, random_actions, greedy)
        if self.ext is not None and hasattr(self.ext, 'collect'):
            actions = self.ext.collect(runner.params, runner.obs, k_explore, self.epsilon(runner.env_steps))
        next_obs, next_state, rewards, done, info, env_state = jax.vmap(self.team_env.step)(
            jax.random.split(k_env, cfg.num_envs), runner.env_state, self.env_action(actions)
        )
        reward = rewards[:, 0] * self.reward_scale
        # Bootstrap through truncations; the stored next observation of a finished episode is the reset one,
        # which only matters for terminations (masked).
        buffer = self._add(runner.buffer, runner.obs, runner.state, actions, reward, next_obs, next_state, info['terminated'])

        def learn(carry, k):
            params, opt_state, updates = carry
            batch = self._sample(buffer, k)
            batch['key'] = jax.random.fold_in(k, 3)
            batch['progress'] = runner.env_steps / (self.config.epsilon_fraction * self.total_env_steps)
            (loss, aux), grads = jax.value_and_grad(self.td_loss, has_aux=True)(params, runner.target_params, batch)
            changes, opt_state = self.optimizer.update(grads, opt_state, params)
            return (optax.apply_updates(params, changes), opt_state, updates + 1), (loss, aux['q_tot'])

        ready = buffer.size >= cfg.learning_starts
        (params, opt_state, updates), (loss, q_tot) = jax.lax.cond(
            ready,
            lambda: jax.lax.scan(learn, (runner.params, runner.opt_state, runner.updates),
                                 jax.random.split(k_sample, cfg.updates_per_step)),
            lambda: ((runner.params, runner.opt_state, runner.updates),
                     (jnp.zeros(cfg.updates_per_step), jnp.zeros(cfg.updates_per_step))),
        )
        refresh = (updates // cfg.target_update) > (runner.updates // cfg.target_update)
        target_params = jax.tree_util.tree_map(lambda t, p: jnp.where(refresh, p, t), runner.target_params, params)
        runner = runner.replace(
            params=params, target_params=target_params, opt_state=opt_state, env_state=env_state, obs=next_obs,
            state=next_state, buffer=buffer, key=key, env_steps=runner.env_steps + 1, updates=updates,
        )
        metrics = {
            'coverage_rate': info['coverage_rate'].mean(), 'loss': loss.mean(), 'q_tot': q_tot.mean(),
            'episode_done': info['returned_episode'].astype(jnp.float32),
            'episode_coverage': info['returned_episode_coverage'],
        }
        return runner, metrics

    @partial(jax.jit, static_argnums=0, donate_argnums=1)
    def _call(self, runner: Runner):
        runner = runner.replace(env_state=jax.vmap(self.team_env.refresh_resets)(
            jax.random.split(jax.random.fold_in(runner.key, 7), self.config.num_envs), runner.env_state))
        runner, metrics = jax.lax.scan(self._step, runner, None, length=self.config.steps_per_call)
        done = metrics.pop('episode_done')
        episode = metrics.pop('episode_coverage')
        summary = {k: v.mean() for k, v in metrics.items()}
        summary['episode_coverage'] = jnp.where(done.sum() > 0, (episode * done).sum() / jnp.maximum(done.sum(), 1), jnp.nan)
        summary['epsilon'] = self.epsilon(runner.env_steps)
        return runner, summary

    # Evaluation ------------------------------------------------------------------------------- #

    def evaluate(self, params, num_envs=None, seed=1, num_steps=None):
        cfg = self.config
        num_envs = num_envs or cfg.eval_envs
        num_steps = int(num_steps or -(-cfg.eval_env_steps // cfg.frame_skip))
        if not hasattr(self, '_eval'):
            def run(params, key):
                k_reset, k_loop = jax.random.split(key)
                obs, _, state = jax.vmap(self.team_env.reset)(jax.random.split(k_reset, num_envs))

                def body(carry, k):
                    obs, state = carry
                    actions = jnp.argmax(self.utilities(params, obs), axis=-1)
                    obs, _, _, _, info, state = jax.vmap(self.team_env.step)(
                        jax.random.split(k, num_envs), state, self.env_action(actions))
                    return (obs, state), info['coverage_rate']

                _, coverage = jax.lax.scan(body, (obs, state), jax.random.split(k_loop, num_steps))
                return coverage.mean()

            self._eval = jax.jit(run)
        return float(self._eval(params, jax.random.PRNGKey(seed)))

    # Training loop ------------------------------------------------------------------------------ #

    def train(self, log_file=None, seed=None):
        cfg = self.config
        runner = self.init(seed)
        calls = math.ceil(self.total_env_steps / cfg.steps_per_call)
        log = open(log_file, 'a', encoding='UTF-8') if log_file else None
        start = time.perf_counter()
        try:
            for call in range(calls):
                tick = time.perf_counter()
                runner, summary = self._call(runner)
                summary = {k: float(v) for k, v in jax.device_get(summary).items()}
                env_steps = int(runner.env_steps)
                row = {
                    'update': call + 1, 'agent_steps': env_steps * cfg.num_envs,
                    'env_steps': env_steps * cfg.num_envs * cfg.frame_skip,
                    'sps': cfg.steps_per_call * cfg.num_envs / (time.perf_counter() - tick), **summary,
                }
                if cfg.eval_interval and ((call + 1) % cfg.eval_interval == 0 or call == calls - 1):
                    row['eval_coverage'] = self.evaluate(runner.params)
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
                if log:
                    log.write(json.dumps(row) + '\n')
                    log.flush()
        finally:
            if log:
                log.close()
        print(f'Training finished in {time.perf_counter() - start:.1f}s', flush=True)
        return runner
