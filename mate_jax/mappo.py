"""MAPPO for MATE in JAX + Flax + Optax, data-parallel over TPU cores.

The whole training iteration (environment rollout -> GAE -> PPO epochs) is a single XLA
program. It is written per device with ``shard_map``: each core owns ``num_envs / num_devices``
environments, computes gradients on its local rollout, and gradients are averaged with
``lax.pmean`` over the ``'data'`` axis. Parameters and optimizer states stay replicated.

Typical use::

    trainer = MAPPOTrainer(MAPPOConfig(team='camera', num_envs=1024))
    runner_state = trainer.init()
    for metrics, runner_state in trainer.iterate(runner_state):
        print(metrics)
"""

# pylint: disable=too-many-instance-attributes,too-many-locals,too-many-statements

import dataclasses
import json
import os
import time
import warnings
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization, struct
from flax.training.train_state import TrainState
from jax.sharding import PartitionSpec as P

from mate_jax import agents as agents_lib
from mate_jax.env import MultiAgentTracking
from mate_jax.models import Actor, Categorical, Critic, DiagGaussian, discrete_action_table
from mate_jax.sharding import DATA_AXIS, create_mesh, describe_devices, replicate, shard_map
from mate_jax.wrappers import SingleTeamEnv


__all__ = ['MAPPOConfig', 'MAPPOTrainer', 'RunnerState', 'LearnedPolicy']

OPPONENTS = {
    ('camera', 'greedy'): agents_lib.GreedyTargetPolicy,
    ('camera', 'random'): agents_lib.RandomTargetPolicy,
    ('camera', 'static'): agents_lib.StaticTargetPolicy,
    ('camera', 'waypoint'): agents_lib.RandomWaypointTargetPolicy,
    ('target', 'greedy'): agents_lib.GreedyCameraPolicy,
    ('target', 'random'): agents_lib.RandomCameraPolicy,
    ('target', 'static'): agents_lib.StaticCameraPolicy,
}


@dataclasses.dataclass
class MAPPOConfig:  # pylint: disable=too-many-instance-attributes
    """Hyper-parameters (defaults follow ``examples/mappo/camera/config.py`` where applicable)."""

    # Environment
    env_config: str = 'MATE-4v8-9.yaml'
    env_overrides: Dict[str, Any] = dataclasses.field(default_factory=dict)
    team: str = 'camera'  # the learning team
    opponent: str = 'greedy'  # greedy / random / static / waypoint (camera team only)
    frame_skip: int = 5
    horizon: int = 500  # agent steps per episode (truncation)
    reward: str = 'coverage'  # raw / normalized / coverage (cameras only)
    reward_scale: Optional[float] = None  # default: 1 / frame_skip (coverage), 0.01 (raw)
    num_reset_candidates: int = 32

    # Rollout
    num_envs: int = 256  # global number of parallel environments (divisible by #devices)
    num_steps: int = 25  # rollout fragment length
    total_timesteps: int = 5_000_000  # agent steps summed over all environments

    # PPO
    update_epochs: int = 4
    num_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.3
    vf_clip: Optional[float] = None
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    actor_lr: float = 5e-4
    critic_lr: float = 5e-4
    anneal_lr: bool = True
    normalize_advantages: bool = True

    # Model
    action_type: str = 'discrete'  # discrete / continuous
    discrete_levels: int = 5
    actor_hiddens: Sequence[int] = (512, 256)
    critic_hiddens: Sequence[int] = (512, 256)
    activation: str = 'tanh'
    layer_norm: bool = False
    agent_id: bool = True  # append one-hot agent ids to the inputs
    critic: str = 'centralized'  # centralized: global state (MAPPO); local: own observation (IPPO)
    advisor: str = 'none'  # none / foresight: append FORESIGHT advice to the camera observations
    hierarchical: bool = False  # FORESIGHT-HRL: actions are (top-K option, transmit) pairs
    student: bool = False  # Hindsight student: lattice moves on FORESIGHT features (mate_jax.hindsight)
    comm: bool = False  # communication control over FORESIGHT-T with published intents (mate_jax.hindsight.comm_env)
    comm_hindsight: bool = True  # the critic sees the exogenous future target positions (input-driven baseline)
    packet_cost: float = 0.05  # reward cost per detection packet and camera-step
    intent_cost: float = 0.04  # reward cost per published intent and camera-step
    comm_lite: bool = True  # lighter planner and re-planning once per decision (training speed)
    init_actor: Optional[str] = None  # msgpack of actor parameters to start from (e.g. a distilled student)
    actor_warmup_updates: int = 0  # first updates train only the critic (the initial actor is kept)
    kl_coef: float = 0.0  # weight of KL(initial actor || actor), keeps the policy near its initialization
    learn_transmission: bool = True
    comm_cost: float = 0.1  # reward penalty per transmission rate (FORESIGHT-HRL)
    actuation_cost: float = 0.05  # reward penalty per normalized actuation effort (FORESIGHT-HRL)
    num_candidates: int = 8
    channel_slots: int = 4
    channel_loss: float = 0.0
    advisor_kwargs: Dict[str, Any] = dataclasses.field(
        default_factory=lambda: {'samples_per_axis': 1, 'horizon': 20, 'window': 8, 'stride': 4}
    )
    compute_dtype: str = 'float32'  # 'bfloat16' recommended on TPU

    # System
    seed: int = 0
    num_devices: Optional[int] = None  # default: all devices
    eval_interval: int = 0  # deterministic evaluation every this many updates (0: off)
    eval_envs: int = 16
    eval_env_steps: int = 1000  # environment steps per evaluation episode
    updates_per_call: int = 10  # updates fused into one jitted call (logging interval)

    @property
    def num_updates(self) -> int:
        return max(1, self.total_timesteps // (self.num_envs * self.num_steps))

    def to_dict(self) -> Dict[str, Any]:
        result = dataclasses.asdict(self)
        result['actor_hiddens'] = list(self.actor_hiddens)
        result['critic_hiddens'] = list(self.critic_hiddens)
        return result


@struct.dataclass
class RunnerState:
    actor_state: TrainState
    critic_state: TrainState
    env_state: Any  # TeamState, batched over (num_envs,)
    observation: jax.Array  # (num_envs, N, D)
    global_state: jax.Array  # (num_envs, S)
    rng: jax.Array  # replicated PRNG key
    update: jax.Array  # () int32


class Transition(struct.PyTreeNode):
    observation: jax.Array
    global_state: jax.Array
    action: jax.Array
    log_prob: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array
    episode_done: jax.Array
    episode_return: jax.Array
    episode_length: jax.Array
    episode_coverage: jax.Array
    episode_delivered: jax.Array
    coverage_rate: jax.Array
    tx_rate: jax.Array
    actuation: jax.Array


class MAPPOTrainer:
    """Builds the environment, networks, optimizers and the sharded training step."""

    def __init__(self, config: MAPPOConfig):
        self.config = cfg = config
        self.mesh = create_mesh(cfg.num_devices)
        self.num_devices = self.mesh.size
        if cfg.num_envs % self.num_devices != 0:
            raise ValueError(
                f'num_envs ({cfg.num_envs}) must be divisible by the number of devices ({self.num_devices}).'
            )
        self.local_envs = cfg.num_envs // self.num_devices
        local_batch = self.local_envs * cfg.num_steps
        if local_batch % cfg.num_minibatches != 0:
            raise ValueError('num_envs / num_devices * num_steps must be divisible by num_minibatches.')
        self.minibatch_size = local_batch // cfg.num_minibatches

        self.env = MultiAgentTracking(
            cfg.env_config, num_reset_candidates=cfg.num_reset_candidates, **cfg.env_overrides
        )
        opponent = OPPONENTS[(cfg.team, cfg.opponent)].from_env(self.env)
        advisor = None
        if cfg.advisor == 'foresight':
            from mate_jax.foresight import ForesightCameraPolicy  # pylint: disable=import-outside-toplevel

            advisor = ForesightCameraPolicy.from_env(self.env, **cfg.advisor_kwargs)
        if cfg.hierarchical:
            # pylint: disable-next=import-outside-toplevel
            from mate_jax.foresight import ForesightCameraPolicy
            from mate_jax.foresight.hrl import ForesightHRLEnv  # pylint: disable=import-outside-toplevel

            planner = ForesightCameraPolicy.from_env(
                self.env,
                communication='hybrid',
                num_candidates=cfg.num_candidates,
                channel_slots=cfg.channel_slots,
                message_drop=cfg.channel_loss,
                **cfg.advisor_kwargs,
            )
            self.hrl_env = ForesightHRLEnv(
                self.env,
                opponent=opponent,
                planner=planner,
                frame_skip=cfg.frame_skip,
                horizon=cfg.horizon,
                learn_transmission=cfg.learn_transmission,
                comm_cost=cfg.comm_cost,
                actuation_cost=cfg.actuation_cost,
            )
        self.team_env = SingleTeamEnv(
            self.env,
            team=cfg.team,
            opponent=opponent,
            frame_skip=cfg.frame_skip,
            horizon=cfg.horizon,
            reward=cfg.reward,
            cache_resets=True,  # one reset sample per env and rollout fragment
            advisor=advisor,
        )
        if cfg.hierarchical:
            self.team_env = self.hrl_env
        if cfg.comm:
            from mate_jax.hindsight.comm_env import CommEnv  # pylint: disable=import-outside-toplevel

            self.team_env = CommEnv(
                self.env, opponent=opponent, frame_skip=cfg.frame_skip, horizon=cfg.horizon,
                packet_cost=cfg.packet_cost, intent_cost=cfg.intent_cost, hindsight=cfg.comm_hindsight,
                replan_every=cfg.frame_skip if cfg.comm_lite else 1, lite=cfg.comm_lite,
            )
        if cfg.student:
            from mate_jax.foresight import ForesightCameraPolicy  # pylint: disable=import-outside-toplevel
            from mate_jax.hindsight.env import StudentEnv  # pylint: disable=import-outside-toplevel

            self.team_env = StudentEnv(
                self.env, opponent=opponent, planner=ForesightCameraPolicy.from_env(self.env), horizon=cfg.horizon
            )
        if cfg.horizon < cfg.num_steps:
            warnings.warn(
                'horizon < num_steps: an environment may finish several episodes in one rollout '
                'fragment and then restart from the same cached initial state.',
                stacklevel=2,
            )
        self.num_agents = self.team_env.num_agents
        if cfg.reward_scale is not None:
            self.reward_scale = cfg.reward_scale
        elif cfg.reward == 'coverage':
            self.reward_scale = 1.0 / cfg.frame_skip  # mean coverage rate per agent step
        elif cfg.reward == 'raw':
            self.reward_scale = 0.01  # tracking penalties are -1 per target and env step
        else:
            self.reward_scale = 1.0

        dtype = {'float32': jnp.float32, 'bfloat16': jnp.bfloat16}[cfg.compute_dtype]
        self.discrete = cfg.action_type == 'discrete' or cfg.hierarchical or cfg.student or cfg.comm
        if cfg.hierarchical or cfg.student or cfg.comm:
            self.action_table = None
            self.distribution = Categorical
            num_outputs = self.team_env.num_actions
        elif self.discrete:
            self.action_table = discrete_action_table(cfg.discrete_levels)
            self.distribution = Categorical
            num_outputs = self.action_table.shape[0]
        else:
            self.action_table = None
            self.distribution = DiagGaussian
            num_outputs = self.team_env.action_dim
        self.actor = Actor(
            num_outputs=num_outputs,
            discrete=self.discrete,
            hidden_sizes=tuple(cfg.actor_hiddens),
            activation=cfg.activation,
            dtype=dtype,
            layer_norm=cfg.layer_norm,
        )
        if cfg.student:  # same policy class as the distilled student (planner prior on the waypoint)
            from mate_jax.hindsight.student import PriorActor  # pylint: disable=import-outside-toplevel

            self.actor = PriorActor(
                num_outputs=num_outputs, hidden_sizes=tuple(cfg.actor_hiddens),
                prior_start=self.team_env.features.suggestion_start,
            )
        self.critic = Critic(
            hidden_sizes=tuple(cfg.critic_hiddens),
            activation=cfg.activation,
            dtype=dtype,
            layer_norm=cfg.layer_norm,
        )
        self._update_fn = None

    # Inputs ------------------------------------------------------------------------------------ #

    def actor_input(self, observation):
        if not self.config.agent_id:
            return observation
        ids = jnp.broadcast_to(
            jnp.eye(self.num_agents, dtype=observation.dtype), observation.shape[:-1] + (self.num_agents,)
        )
        return jnp.concatenate([observation, ids], axis=-1)

    def critic_input(self, observation, global_state):
        if self.config.critic == 'local':  # IPPO: independent critics (shared parameters)
            return self.actor_input(observation)
        state = jnp.broadcast_to(
            global_state[..., None, :], global_state.shape[:-1] + (self.num_agents, global_state.shape[-1])
        )
        return self.actor_input(state) if self.config.agent_id else state

    def env_action(self, action):
        if self.config.hierarchical or self.config.student or self.config.comm:  # discrete indices
            return action
        if self.discrete:
            normalized = jnp.asarray(self.action_table)[action]
        else:
            normalized = action
        return self.team_env.scale_actions(normalized)

    # Initialization ---------------------------------------------------------------------------- #

    def _optimizer(self, lr):
        cfg = self.config
        num_steps = cfg.num_updates * cfg.update_epochs * cfg.num_minibatches
        schedule = optax.linear_schedule(lr, 0.0, num_steps) if cfg.anneal_lr else lr
        return optax.chain(
            optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(schedule, eps=1e-5)
        )

    def init(self, seed: Optional[int] = None) -> RunnerState:
        cfg = self.config
        key = jax.random.PRNGKey(cfg.seed if seed is None else seed)
        key, k_actor, k_critic, k_env = jax.random.split(key, 4)

        obs_dim = self.team_env.observation_dim + (self.num_agents if cfg.agent_id else 0)
        state_dim = self.team_env.state_dim + (self.num_agents if cfg.agent_id else 0)
        if cfg.critic == 'local':
            state_dim = obs_dim
        actor_params = self.actor.init(k_actor, jnp.zeros((1, obs_dim)))
        if cfg.init_actor:
            with open(cfg.init_actor, 'rb') as file:
                actor_params = serialization.from_bytes(actor_params, file.read())
        self._reference_params = actor_params  # the initial policy (target of the KL regularization)
        critic_params = self.critic.init(k_critic, jnp.zeros((1, state_dim)))
        actor_state = TrainState.create(
            apply_fn=self.actor.apply, params=actor_params, tx=self._optimizer(cfg.actor_lr)
        )
        critic_state = TrainState.create(
            apply_fn=self.critic.apply, params=critic_params, tx=self._optimizer(cfg.critic_lr)
        )

        def reset_local(key):
            key = jax.random.fold_in(key, jax.lax.axis_index(DATA_AXIS))
            return jax.vmap(self.team_env.reset)(jax.random.split(key, self.local_envs))

        reset = jax.jit(
            shard_map(reset_local, self.mesh, in_specs=P(), out_specs=P(DATA_AXIS))
        )
        observation, global_state, env_state = reset(replicate(k_env, self.mesh))

        actor_state, critic_state, key, update = replicate(
            (actor_state, critic_state, key, np.zeros((), np.int32)), self.mesh
        )
        return RunnerState(
            actor_state=actor_state,
            critic_state=critic_state,
            env_state=env_state,
            observation=observation,
            global_state=global_state,
            rng=key,
            update=update,
        )

    # One PPO iteration (per device) ------------------------------------------------------------ #

    def _rollout(self, runner: RunnerState):
        cfg = self.config
        dist = self.distribution
        device = jax.lax.axis_index(DATA_AXIS)

        def env_step(carry, _):
            env_state, obs, gstate, rng = carry
            rng, k_act, k_env = jax.random.split(rng, 3)
            k_act = jax.random.fold_in(k_act, device)
            k_env = jax.random.fold_in(k_env, device)
            out, log_std = runner.actor_state.apply_fn(runner.actor_state.params, self.actor_input(obs))
            action = dist.sample(k_act, out, log_std)
            log_prob = dist.log_prob(action, out, log_std)
            value = runner.critic_state.apply_fn(runner.critic_state.params, self.critic_input(obs, gstate))
            next_obs, next_gstate, reward, done, info, env_state = jax.vmap(self.team_env.step)(
                jax.random.split(k_env, self.local_envs), env_state, self.env_action(action)
            )
            transition = Transition(
                observation=obs,
                global_state=gstate,
                action=action,
                log_prob=log_prob,
                value=value,
                reward=reward * self.reward_scale,
                done=done,
                episode_done=info['returned_episode'],
                episode_return=info['returned_episode_return'],
                episode_length=info['returned_episode_length'],
                episode_coverage=info['returned_episode_coverage'],
                episode_delivered=info['returned_episode_delivered'],
                coverage_rate=info['coverage_rate'],
                tx_rate=info['tx_rate'],
                actuation=info['actuation'],
            )
            return (env_state, next_obs, next_gstate, rng), transition

        rng, k_refresh = jax.random.split(runner.rng)
        k_refresh = jax.random.fold_in(k_refresh, device)
        env_state = jax.vmap(self.team_env.refresh_resets)(
            jax.random.split(k_refresh, self.local_envs), runner.env_state
        )
        carry = (env_state, runner.observation, runner.global_state, rng)
        (env_state, obs, gstate, rng), traj = jax.lax.scan(env_step, carry, None, cfg.num_steps)
        return traj, env_state, obs, gstate, rng

    def _gae(self, traj: Transition, last_value):
        cfg = self.config

        def step(carry, transition):
            gae, next_value = carry
            not_done = 1.0 - transition.done.astype(jnp.float32)[..., None]
            delta = transition.reward + cfg.gamma * next_value * not_done - transition.value
            gae = delta + cfg.gamma * cfg.gae_lambda * not_done * gae
            return (gae, transition.value), gae

        _, advantages = jax.lax.scan(
            step, (jnp.zeros_like(last_value), last_value), traj, reverse=True, unroll=8
        )
        return advantages, advantages + traj.value

    def _losses(self, actor_params, critic_params, actor_state, critic_state, batch):
        cfg = self.config
        dist = self.distribution
        traj, advantages, targets = batch

        out, log_std = actor_state.apply_fn(actor_params, self.actor_input(traj.observation))
        log_prob = dist.log_prob(traj.action, out, log_std)
        ratio = jnp.exp(log_prob - traj.log_prob)
        if cfg.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        pg_loss = -jnp.minimum(
            ratio * advantages, jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantages
        ).mean()
        entropy = dist.entropy(out, log_std).mean()

        value = critic_state.apply_fn(critic_params, self.critic_input(traj.observation, traj.global_state))
        if cfg.vf_clip is not None:
            clipped = traj.value + jnp.clip(value - traj.value, -cfg.vf_clip, cfg.vf_clip)
            vf_loss = 0.5 * jnp.maximum(jnp.square(value - targets), jnp.square(clipped - targets)).mean()
        else:
            vf_loss = 0.5 * jnp.square(value - targets).mean()

        loss = pg_loss + cfg.vf_coef * vf_loss - cfg.ent_coef * entropy
        if cfg.kl_coef > 0.0:  # categorical policies only
            reference, _ = actor_state.apply_fn(self._reference_params, self.actor_input(traj.observation))
            log_reference = jax.nn.log_softmax(reference)
            kl = jnp.sum(jnp.exp(log_reference) * (log_reference - jax.nn.log_softmax(out)), axis=-1).mean()
            loss = loss + cfg.kl_coef * kl
        aux = {
            'pg_loss': pg_loss,
            'vf_loss': vf_loss,
            'entropy': entropy,
            'approx_kl': ((ratio - 1.0) - jnp.log(ratio)).mean(),
            'clip_frac': (jnp.abs(ratio - 1.0) > cfg.clip_eps).astype(jnp.float32).mean(),
        }
        return loss, aux

    def _update(self, runner: RunnerState, traj, advantages, targets, rng):
        cfg = self.config
        device = jax.lax.axis_index(DATA_AXIS)
        grad_fn = jax.value_and_grad(self._losses, argnums=(0, 1), has_aux=True)

        def flatten(x):  # (T, B, ...) -> (T * B, ...)
            return x.reshape((-1,) + x.shape[2:])

        batch = jax.tree_util.tree_map(flatten, (traj, advantages, targets))

        def epoch(carry, k):
            actor_state, critic_state = carry
            permutation = jax.random.permutation(jax.random.fold_in(k, device), self.local_envs * cfg.num_steps)
            shuffled = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
            minibatches = jax.tree_util.tree_map(
                lambda x: x.reshape((cfg.num_minibatches, self.minibatch_size) + x.shape[1:]), shuffled
            )

            def minibatch(carry, mb):
                actor_state, critic_state = carry
                (_, aux), (actor_grads, critic_grads) = grad_fn(
                    actor_state.params, critic_state.params, actor_state, critic_state, mb
                )
                # Data-parallel all-reduce over TPU cores.
                actor_grads, critic_grads, aux = jax.lax.pmean(
                    (actor_grads, critic_grads, aux), axis_name=DATA_AXIS
                )
                if cfg.actor_warmup_updates > 0:  # critic-only updates first
                    learn = runner.update >= cfg.actor_warmup_updates
                    actor_grads = jax.tree_util.tree_map(lambda g: jnp.where(learn, g, 0.0), actor_grads)
                actor_state = actor_state.apply_gradients(grads=actor_grads)
                critic_state = critic_state.apply_gradients(grads=critic_grads)
                return (actor_state, critic_state), aux

            return jax.lax.scan(minibatch, (actor_state, critic_state), minibatches)

        (actor_state, critic_state), aux = jax.lax.scan(
            epoch, (runner.actor_state, runner.critic_state), jax.random.split(rng, cfg.update_epochs)
        )
        return actor_state, critic_state, jax.tree_util.tree_map(jnp.mean, aux)

    def _iteration(self, runner: RunnerState, _):
        traj, env_state, obs, gstate, rng = self._rollout(runner)
        last_value = runner.critic_state.apply_fn(runner.critic_state.params, self.critic_input(obs, gstate))
        advantages, targets = self._gae(traj, last_value)
        rng, k_update = jax.random.split(rng)
        actor_state, critic_state, losses = self._update(runner, traj, advantages, targets, k_update)

        def masked_mean(values, mask):
            mask = mask.astype(jnp.float32)
            total = jax.lax.psum(jnp.sum(values.astype(jnp.float32) * mask), DATA_AXIS)
            count = jax.lax.psum(jnp.sum(mask), DATA_AXIS)
            return jnp.where(count > 0, total / jnp.maximum(count, 1.0), jnp.nan)

        metrics = {
            **losses,
            'reward': jax.lax.pmean(traj.reward.mean() / self.reward_scale, DATA_AXIS),
            'coverage_rate': jax.lax.pmean(traj.coverage_rate.mean(), DATA_AXIS),
            'tx_rate': jax.lax.pmean(traj.tx_rate.mean(), DATA_AXIS),
            'actuation': jax.lax.pmean(traj.actuation.mean(), DATA_AXIS),
            'episode_return': masked_mean(traj.episode_return, traj.episode_done),
            'episode_length': masked_mean(traj.episode_length, traj.episode_done),
            'episode_coverage': masked_mean(traj.episode_coverage, traj.episode_done),
            'episode_delivered': masked_mean(traj.episode_delivered, traj.episode_done),
            'num_episodes': jax.lax.psum(traj.episode_done.sum(), DATA_AXIS),
        }
        runner = RunnerState(
            actor_state=actor_state,
            critic_state=critic_state,
            env_state=env_state,
            observation=obs,
            global_state=gstate,
            rng=rng,
            update=runner.update + 1,
        )
        return runner, metrics

    def _chunk(self, runner: RunnerState, num_updates: int):
        return jax.lax.scan(self._iteration, runner, None, num_updates)

    def update_fn(self, num_updates: Optional[int] = None):
        """The jitted, sharded function running ``num_updates`` PPO iterations."""

        num_updates = num_updates or self.config.updates_per_call
        specs = RunnerState(
            actor_state=P(),
            critic_state=P(),
            env_state=P(DATA_AXIS),
            observation=P(DATA_AXIS),
            global_state=P(DATA_AXIS),
            rng=P(),
            update=P(),
        )
        fn = shard_map(
            lambda runner: self._chunk(runner, num_updates),
            self.mesh,
            in_specs=(specs,),
            out_specs=(specs, P()),
        )
        return jax.jit(fn, donate_argnums=0)

    # Driver ------------------------------------------------------------------------------------ #

    def iterate(self, runner: RunnerState) -> Iterator[Tuple[Dict[str, float], RunnerState]]:
        """Run the full training; yields host-side metrics every ``updates_per_call`` updates.

        The runner state buffers are donated to XLA on every call, so only keep the most
        recently yielded ``RunnerState`` (earlier ones are deleted).
        """

        cfg = self.config
        steps_per_update = cfg.num_envs * cfg.num_steps
        done_updates = int(runner.update)
        cache = {}
        while done_updates < cfg.num_updates:
            chunk = min(cfg.updates_per_call, cfg.num_updates - done_updates)
            if chunk not in cache:
                cache[chunk] = self.update_fn(chunk)
            start = time.perf_counter()
            runner, metrics = cache[chunk](runner)
            metrics = jax.device_get(metrics)
            elapsed = time.perf_counter() - start
            done_updates += chunk
            summary = {key: float(np.nanmean(value)) if np.isfinite(value).any() else float('nan') for key, value in metrics.items()}
            summary['num_episodes'] = float(np.sum(metrics['num_episodes']))
            summary.update(
                update=done_updates,
                agent_steps=done_updates * steps_per_update,
                env_steps=done_updates * steps_per_update * cfg.frame_skip,
                sps=chunk * steps_per_update * cfg.frame_skip / elapsed,
                seconds=elapsed,
            )
            if cfg.eval_interval and (
                done_updates // cfg.eval_interval != (done_updates - chunk) // cfg.eval_interval
                or done_updates >= cfg.num_updates
            ):
                evaluation = self.evaluate(
                    runner.actor_state.params,
                    num_envs=cfg.eval_envs,
                    num_steps=max(1, cfg.eval_env_steps // cfg.frame_skip),
                )
                summary.update(
                    eval_coverage=evaluation['coverage_rate'],
                    eval_return=evaluation['episode_return'],
                    eval_tx_rate=evaluation['tx_rate'],
                    eval_actuation=evaluation['actuation'],
                )
            yield summary, runner

    def describe(self) -> str:
        cfg = self.config
        return (
            f'{describe_devices()}\n'
            f'{self.env} | team={cfg.team} vs {cfg.opponent} | {self.num_agents} agents | '
            f'critic={cfg.critic} advisor={cfg.advisor} hierarchical={cfg.hierarchical}\n'
            f'{cfg.num_envs} envs ({self.local_envs}/device) x {cfg.num_steps} steps, '
            f'frame_skip={cfg.frame_skip}, {cfg.num_updates} updates, '
            f'compute_dtype={cfg.compute_dtype}'
        )

    # Checkpointing / evaluation ---------------------------------------------------------------- #

    def save(self, runner: RunnerState, directory: str) -> str:
        os.makedirs(directory, exist_ok=True)
        params = jax.device_get(
            {'actor': runner.actor_state.params, 'critic': runner.critic_state.params}
        )
        path = os.path.join(directory, 'params.msgpack')
        with open(path, 'wb') as file:
            file.write(serialization.to_bytes(params))
        with open(os.path.join(directory, 'config.json'), 'w', encoding='UTF-8') as file:
            json.dump(self.config.to_dict(), file, indent=2)
        return path

    def load_params(self, directory: str) -> Dict[str, Any]:
        runner = self.init()
        template = {'actor': runner.actor_state.params, 'critic': runner.critic_state.params}
        with open(os.path.join(directory, 'params.msgpack'), 'rb') as file:
            return serialization.from_bytes(jax.device_get(template), file.read())

    def policy(self, actor_params, deterministic: bool = True) -> 'LearnedPolicy':
        return LearnedPolicy(trainer=self, params=actor_params, deterministic=deterministic)

    def evaluate(
        self,
        actor_params,
        num_envs: int = 64,
        seed: int = 1,
        deterministic: bool = True,
        num_steps: Optional[int] = None,
    ):
        """Run ``num_steps`` agent steps (default: one horizon) in ``num_envs`` environments on one device."""

        team_env = self.team_env
        dist = self.distribution
        num_steps = int(num_steps or team_env.horizon)
        actor_params = jax.device_get(actor_params)  # local copy (works on any host of a pod)
        cache_key = (num_envs, num_steps, deterministic)
        if not hasattr(self, '_eval_cache'):
            self._eval_cache = {}  # pylint: disable=attribute-defined-outside-init

        if cache_key not in self._eval_cache:

            def run(params, key):
                k_reset, k_loop = jax.random.split(key)
                obs, _, state = jax.vmap(team_env.reset)(jax.random.split(k_reset, num_envs))

                def body(carry, k):
                    obs, state = carry
                    k_act, k_env = jax.random.split(k)
                    out, log_std = self.actor.apply(params, self.actor_input(obs))
                    action = dist.mode(out, log_std) if deterministic else dist.sample(k_act, out, log_std)
                    obs, _, reward, done, info, state = jax.vmap(team_env.step)(
                        jax.random.split(k_env, num_envs), state, self.env_action(action)
                    )
                    return (obs, state), (reward[:, 0], info['coverage_rate'], info['tx_rate'], info['actuation'])

                _, (reward, coverage, tx_rate, actuation) = jax.lax.scan(
                    body, (obs, state), jax.random.split(k_loop, num_steps)
                )
                return reward.sum(0).mean(), coverage.mean(), tx_rate.mean(), actuation.mean()

            self._eval_cache[cache_key] = jax.jit(run)

        episode_return, coverage, tx_rate, actuation = self._eval_cache[cache_key](actor_params, jax.random.PRNGKey(seed))
        return {
            'episode_return': float(episode_return),
            'coverage_rate': float(coverage),
            'tx_rate': float(tx_rate),
            'actuation': float(actuation),
        }


@struct.dataclass
class LearnedPolicy:
    """Use a trained MAPPO actor as a :mod:`mate_jax.agents`-style team policy (e.g. as an opponent)."""

    trainer: MAPPOTrainer = struct.field(pytree_node=False)
    params: Any
    deterministic: bool = struct.field(pytree_node=False, default=True)

    def init(self, key, observation):  # pylint: disable=unused-argument
        return ()

    def act(self, key, carry, observation):
        trainer = self.trainer
        obs = trainer.team_env.transform(observation)
        out, log_std = trainer.actor.apply(self.params, trainer.actor_input(obs))
        dist = trainer.distribution
        action = dist.mode(out, log_std) if self.deterministic else dist.sample(key, out, log_std)
        return trainer.env_action(action), carry
