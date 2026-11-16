"""FORESIGHT-HRL: a learned high-level camera policy over reachability-aware options and transmissions.

Every agent step (``frame_skip`` environment steps) each camera chooses

* one of the top-``K`` pose options proposed by the FORESIGHT planner for it (option 0 is the
  sequential-greedy choice), and
* whether to broadcast its packet (pose + detections) during the next ``frame_skip`` steps.

The low level moves the camera towards the chosen pose at the rotation / zoom rate limits.
Packets contend for a shared channel (collisions) and update the shared team memory, while
every camera keeps its own private memory (hybrid memory). The team reward is

    coverage rate - comm_cost * transmission rate - actuation_cost * actuation effort

averaged over the environment steps, so the learner trades coverage against radio and motor
energy. The interface matches :class:`mate_jax.wrappers.SingleTeamEnv`, so the MAPPO / IPPO
trainer of :mod:`mate_jax.mappo` trains it unchanged.
"""

# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-locals

from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
from flax import struct

from mate_jax.agents import _camera_self
from mate_jax.env import float_dtype
from mate_jax.foresight.planner import ForesightCameraPolicy
from mate_jax.wrappers import SingleTeamEnv, _select


__all__ = ['HRLState', 'ForesightHRLEnv', 'TRANSMIT_RULES', 'evaluate_actions', 'evaluate_rule']

LITE_PLANNER = {'samples_per_axis': 1, 'horizon': 20, 'window': 8, 'stride': 4}


@struct.dataclass
class HRLState:
    env_state: Any
    opponent_carry: Any
    opponent_observation: jax.Array
    team_observation: jax.Array
    planner_carry: Any
    goals: Any  # (orientation, viewing angle), each (C, K)
    features: jax.Array  # (C, K, F)
    comm: jax.Array  # (C, 4)
    step: jax.Array
    episode_return: jax.Array
    episode_coverage: jax.Array
    returned_episode_return: jax.Array
    returned_episode_length: jax.Array
    returned_episode_coverage: jax.Array
    returned_episode_delivered: jax.Array
    next_reset: Any = None


class ForesightHRLEnv:
    """Camera-team training environment whose actions are (option, transmit) pairs."""

    def __init__(
        self,
        env,
        opponent: Optional[Any] = None,
        planner: Optional[ForesightCameraPolicy] = None,
        frame_skip: int = 5,
        horizon: Optional[int] = 500,
        learn_transmission: bool = True,
        comm_cost: float = 0.1,
        actuation_cost: float = 0.05,
        cache_resets: bool = True,
    ):
        self.base = SingleTeamEnv(
            env, team='camera', opponent=opponent, frame_skip=frame_skip, horizon=horizon, reward='coverage',
            cache_resets=cache_resets,
        )
        self.env = env
        self.opponent = self.base.opponent
        if planner is None:
            planner = ForesightCameraPolicy.from_env(env, communication='hybrid', **LITE_PLANNER)
        self.planner = planner
        self.frame_skip = self.base.frame_skip
        self.horizon = self.base.horizon
        self.cache_resets = bool(cache_resets)
        self.learn_transmission = bool(learn_transmission)
        self.comm_cost = float(comm_cost)
        self.actuation_cost = float(actuation_cost)

        self.num_candidates = K = planner.num_candidates
        self.num_agents = env.num_cameras
        self.num_actions = K * (2 if self.learn_transmission else 1)
        self.observation_dim = (
            self.base.transform.dim + K * planner.CANDIDATE_FEATURES + planner.COMM_FEATURES
        )
        self.state_dim = env.state_dim
        self.action_dim = 2
        self.action_scale = self.base.action_scale

    # Helpers ----------------------------------------------------------------------------------- #

    def global_state(self, env_state):
        return self.base.global_state(env_state)

    def _observation(self, team_observation, features, comm):
        observation = self.base.transform(team_observation)
        return jnp.concatenate(
            [observation, features.reshape(features.shape[0], -1).astype(observation.dtype), comm.astype(observation.dtype)],
            axis=-1,
        )

    def observe(self, state: HRLState):
        return self._observation(state.team_observation, state.features, state.comm)

    def decode(self, actions):
        """(C,) action indices -> option indices (C,) and transmit flags (C,)."""

        actions = actions.astype(jnp.int32)
        option = actions % self.num_candidates
        if self.learn_transmission:
            transmit = actions // self.num_candidates > 0
        else:
            transmit = jnp.ones(actions.shape, bool)
        return option, transmit

    def encode(self, option, transmit):
        return option.astype(jnp.int32) + self.num_candidates * transmit.astype(jnp.int32) * int(self.learn_transmission)

    def _sample_start(self, key):
        key_env, key_opponent, key_planner = jax.random.split(key, 3)
        (camera_obs, target_obs), env_state = self.env.reset(key_env)
        planner_carry = self.planner.init(key_planner, camera_obs)
        goals, features, comm = self.planner.candidates(planner_carry, camera_obs)
        opponent_carry = self.opponent.init(key_opponent, target_obs)
        return env_state, camera_obs, target_obs, opponent_carry, planner_carry, goals, features, comm

    def refresh_resets(self, key, state: HRLState) -> HRLState:
        if not self.cache_resets:
            return state
        return state.replace(next_reset=self._sample_start(key))

    def scale_actions(self, actions):  # actions are discrete indices
        return actions

    # API ---------------------------------------------------------------------------------------- #

    def reset(self, key):
        key_start, key_cache = jax.random.split(key)
        env_state, team_obs, opponent_obs, opponent_carry, planner_carry, goals, features, comm = self._sample_start(
            key_start
        )
        fdt = float_dtype()
        state = HRLState(
            env_state=env_state,
            opponent_carry=opponent_carry,
            opponent_observation=opponent_obs,
            team_observation=team_obs,
            planner_carry=planner_carry,
            goals=goals,
            features=features,
            comm=comm,
            step=jnp.zeros((), jnp.int32),
            episode_return=jnp.zeros((), fdt),
            episode_coverage=jnp.zeros((), fdt),
            returned_episode_return=jnp.zeros((), fdt),
            returned_episode_length=jnp.zeros((), jnp.int32),
            returned_episode_coverage=jnp.zeros((), fdt),
            returned_episode_delivered=jnp.zeros((), jnp.int32),
        )
        state = self.refresh_resets(key_cache, state)
        return self._observation(team_obs, features, comm), self.global_state(env_state), state

    def step(self, key, state: HRLState, actions):
        fdt = float_dtype()
        key_reset, key_skip = jax.random.split(key)
        option, transmit = self.decode(actions)
        goal_orientation = jnp.take_along_axis(state.goals[0], option[:, None], axis=1)[:, 0]
        goal_viewing = jnp.take_along_axis(state.goals[1], option[:, None], axis=1)[:, 0]
        options = self.planner.options
        num_cameras = self.num_agents

        def substep(carry, k):
            env_state, opponent_carry, opponent_obs, team_obs, planner_carry, finished, totals = carry
            k_opponent, k_env, k_planner = jax.random.split(k, 3)
            me = _camera_self(team_obs)
            camera_action = options.action_towards(me['orientation'], me['viewing_angle'], goal_orientation, goal_viewing)
            camera_action = camera_action.astype(team_obs.dtype)
            target_action, new_opponent_carry = self.opponent.act(k_opponent, opponent_carry, opponent_obs)
            timestep, new_env_state = self.env.step(k_env, env_state, camera_action, target_action)
            new_planner_carry = self.planner.observe(k_planner, planner_carry, timestep.camera_observation, transmit)
            sent, delivered, collided = new_planner_carry['channel']
            effort = 0.5 * (
                jnp.abs(camera_action[:, 0]) / options.rotation_step + jnp.abs(camera_action[:, 1]) / options.zooming_step
            )
            active = jnp.logical_not(finished)
            mask = active.astype(fdt)
            tx_rate = jnp.mean(sent.astype(fdt))
            actuation = jnp.mean(effort.astype(fdt))
            reward = timestep.info['coverage_rate'] - self.comm_cost * tx_rate - self.actuation_cost * actuation
            totals = {
                'reward': totals['reward'] + mask * reward,
                'coverage_rate': totals['coverage_rate'] + mask * timestep.info['coverage_rate'],
                'real_coverage_rate': totals['real_coverage_rate'] + mask * timestep.info['real_coverage_rate'],
                'sent': totals['sent'] + mask * jnp.sum(sent.astype(fdt)),
                'delivered': totals['delivered'] + mask * jnp.sum(delivered.astype(fdt)),
                'collided': totals['collided'] + mask * jnp.sum(collided.astype(fdt)),
                'actuation': totals['actuation'] + mask * actuation,
                'num_substeps': totals['num_substeps'] + active.astype(jnp.int32),
            }
            env_state = _select(active, new_env_state, env_state)
            opponent_carry = _select(active, new_opponent_carry, opponent_carry)
            planner_carry = _select(active, new_planner_carry, planner_carry)
            opponent_obs = jnp.where(active, timestep.target_observation, opponent_obs)
            team_obs = jnp.where(active, timestep.camera_observation, team_obs)
            finished = finished | (active & timestep.done)
            return (env_state, opponent_carry, opponent_obs, team_obs, planner_carry, finished, totals), None

        zero = jnp.zeros((), fdt)
        totals = {
            'reward': zero, 'coverage_rate': zero, 'real_coverage_rate': zero, 'sent': zero, 'delivered': zero,
            'collided': zero, 'actuation': zero, 'num_substeps': jnp.zeros((), jnp.int32),
        }  # fmt: skip
        carry = (
            state.env_state, state.opponent_carry, state.opponent_observation, state.team_observation,
            state.planner_carry, jnp.zeros((), bool), totals,
        )  # fmt: skip
        carry, _ = jax.lax.scan(substep, carry, jax.random.split(key_skip, self.frame_skip))
        env_state, opponent_carry, opponent_obs, team_obs, planner_carry, terminated, totals = carry
        goals, features, comm = self.planner.candidates(planner_carry, team_obs)

        substeps = jnp.maximum(totals['num_substeps'], 1).astype(fdt)
        step = state.step + 1
        truncated = step >= self.horizon
        done = terminated | truncated
        coverage = totals['coverage_rate'] / substeps
        episode_return = state.episode_return + totals['reward']
        episode_coverage = state.episode_coverage + coverage

        if self.cache_resets:
            start = state.next_reset
        else:
            start = self._sample_start(key_reset)
        (r_env_state, r_team_obs, r_opponent_obs, r_opponent_carry, r_planner_carry, r_goals, r_features, r_comm) = start

        new_state = HRLState(
            env_state=_select(done, r_env_state, env_state),
            opponent_carry=_select(done, r_opponent_carry, opponent_carry),
            opponent_observation=jnp.where(done, r_opponent_obs, opponent_obs),
            team_observation=jnp.where(done, r_team_obs, team_obs),
            planner_carry=_select(done, r_planner_carry, planner_carry),
            goals=_select(done, r_goals, goals),
            features=jnp.where(done, r_features, features),
            comm=jnp.where(done, r_comm, comm),
            step=jnp.where(done, 0, step),
            episode_return=jnp.where(done, 0.0, episode_return),
            episode_coverage=jnp.where(done, 0.0, episode_coverage),
            returned_episode_return=jnp.where(done, episode_return, state.returned_episode_return),
            returned_episode_length=jnp.where(done, step, state.returned_episode_length),
            returned_episode_coverage=jnp.where(done, episode_coverage / step.astype(fdt), state.returned_episode_coverage),
            returned_episode_delivered=state.returned_episode_delivered,
            next_reset=state.next_reset,
        )
        sent = totals['sent']
        info = {
            'coverage_rate': coverage,
            'real_coverage_rate': totals['real_coverage_rate'] / substeps,
            'num_delivered_cargoes': jnp.zeros((), jnp.int32),
            'tx_rate': sent / (substeps * num_cameras),
            'delivery_ratio': jnp.where(sent > 0, totals['delivered'] / jnp.maximum(sent, 1.0), 1.0),
            'collision_rate': jnp.where(sent > 0, totals['collided'] / jnp.maximum(sent, 1.0), 0.0),
            'actuation': totals['actuation'] / substeps,
            'terminated': terminated,
            'truncated': truncated,
            'returned_episode': done,
            'returned_episode_return': new_state.returned_episode_return,
            'returned_episode_length': new_state.returned_episode_length,
            'returned_episode_coverage': new_state.returned_episode_coverage,
            'returned_episode_delivered': new_state.returned_episode_delivered,
        }
        rewards = jnp.full((num_cameras,), totals['reward'], fdt)
        obs = self._observation(new_state.team_observation, new_state.features, new_state.comm)
        return obs, self.global_state(new_state.env_state), rewards, done, info, new_state


# Rule-based high levels (baselines) -------------------------------------------------------------- #


def _rule_actions(hrl: ForesightHRLEnv, rule: str, state: HRLState):
    """Option 0 (sequential greedy) and a transmission rule."""

    num = hrl.num_agents
    option = jnp.zeros(num, jnp.int32)
    if rule == 'always':
        transmit = jnp.ones(num, bool)
    elif rule == 'never':
        transmit = jnp.zeros(num, bool)
    elif rule.startswith('periodic'):  # staggered (TDMA-like) every k agent steps
        period = int(rule.split('-')[1]) if '-' in rule else num
        transmit = (state.step + jnp.arange(num)) % period == 0
    elif rule == 'event':  # own detections missing from the team memory, or a 50-step heartbeat
        transmit = (state.comm[:, 0] > 0.0) | (state.comm[:, 2] >= 1.0)
    else:
        raise ValueError(f'Unknown transmission rule {rule!r}.')
    return option, transmit


TRANSMIT_RULES = ('always', 'event', 'periodic-4', 'never')


def evaluate_actions(
    hrl: ForesightHRLEnv, select, num_envs: int = 16, num_steps: int = 200, seed: int = 1
) -> Dict[str, float]:
    """Evaluate a high level ``select(key, observation, state) -> actions`` (batched over environments).

    The environment keys do not depend on ``select``, so every high level sees the same target trajectories.
    ``hrl`` may also be a flat team environment (:class:`SingleTeamEnv`), which has no collisions.
    """

    def run(key):
        k_reset, k_loop = jax.random.split(key)
        obs, _, state = jax.vmap(hrl.reset)(jax.random.split(k_reset, num_envs))

        def body(carry, k):
            obs, state = carry
            actions = select(jax.random.fold_in(k, 1), obs, state)
            obs, _, reward, _, info, state = jax.vmap(hrl.step)(jax.random.split(k, num_envs), state, actions)
            collision = info.get('collision_rate', jnp.zeros_like(info['coverage_rate']))
            return (obs, state), (reward[:, 0], info['coverage_rate'], info['tx_rate'], collision, info['actuation'])

        _, out = jax.lax.scan(body, (obs, state), jax.random.split(k_loop, num_steps))
        return [x.mean() for x in out]

    reward, coverage, tx_rate, collision, actuation = jax.jit(run)(jax.random.PRNGKey(seed))
    return {
        'reward': float(reward) / hrl.frame_skip,
        'coverage_rate': float(coverage),
        'tx_rate': float(tx_rate),
        'collision_rate': float(collision),
        'actuation': float(actuation),
    }


def evaluate_rule(hrl: ForesightHRLEnv, rule: str, num_envs: int = 16, num_steps: int = 200, seed: int = 1) -> Dict[str, float]:
    """Evaluate a rule-based high level (greedy option + transmission rule) like the trainer does."""

    def select(key, obs, state):  # pylint: disable=unused-argument
        option, transmit = jax.vmap(lambda s: _rule_actions(hrl, rule, s))(state)
        return jax.vmap(hrl.encode)(option, transmit)

    return evaluate_actions(hrl, select, num_envs=num_envs, num_steps=num_steps, seed=seed)
