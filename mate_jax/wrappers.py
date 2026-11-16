"""Functional wrappers: auto-reset and single-team training environments."""

# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-locals

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from mate_jax import agents as agents_lib
from mate_jax import constants as consts
from mate_jax.env import EnvState, MultiAgentTracking, float_dtype
from mate_jax.transforms import ObservationTransform


__all__ = ['AutoResetEnv', 'TeamState', 'SingleTeamEnv']


def _select(condition, on_true, on_false):
    """``where`` over pytrees (vmap friendly alternative to ``lax.cond``)."""

    return jax.tree_util.tree_map(
        lambda x, y: jnp.where(jnp.reshape(condition, condition.shape + (1,) * (x.ndim - condition.ndim)), x, y),
        on_true,
        on_false,
    )


class AutoResetEnv:
    """Reset the environment automatically when an episode ends.

    The returned timestep keeps the reward and ``done`` flag of the final transition, while the
    observations and the state belong to the new episode.
    """

    def __init__(self, env: MultiAgentTracking):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, key):
        return self.env.reset(key)

    def step(self, key, state: EnvState, camera_joint_action, target_joint_action):
        key_step, key_reset = jax.random.split(key)
        timestep, state = self.env.step(key_step, state, camera_joint_action, target_joint_action)
        (camera_obs, target_obs), reset_state = self.env.reset(key_reset)
        state = _select(timestep.done, reset_state, state)
        timestep = timestep._replace(
            camera_observation=jnp.where(timestep.done, camera_obs, timestep.camera_observation),
            target_observation=jnp.where(timestep.done, target_obs, timestep.target_observation),
        )
        return timestep, state


@struct.dataclass
class TeamState:
    env_state: EnvState
    opponent_carry: Any
    opponent_observation: jax.Array
    team_observation: jax.Array  # raw (untransformed) joint observation of the learning team
    step: jax.Array  # agent steps in the current episode
    episode_return: jax.Array
    episode_coverage: jax.Array
    returned_episode_return: jax.Array
    returned_episode_length: jax.Array
    returned_episode_coverage: jax.Array
    returned_episode_delivered: jax.Array
    next_reset: Any = None  # pre-sampled episode start (only with `cache_resets=True`)
    advisor_carry: Any = None  # state of the advisor policy (only with an advisor)
    advice: Any = None  # (N, F) advice features appended to the observations


class SingleTeamEnv:
    """One learning team versus a fixed JAX opponent policy (``mate.MultiCamera`` / ``mate.MultiTarget``).

    Includes the preprocessing used by ``examples/mappo``: relative coordinates, rescaled
    observations, shared team rewards, frame skipping and a training horizon.

    Args:
        env: the base :class:`MultiAgentTracking` environment.
        team: the learning team, ``'camera'`` or ``'target'``.
        opponent: a policy from :mod:`mate_jax.agents` (greedy opponent by default).
        frame_skip: repeat each team action for this many environment steps (rewards are summed).
        horizon: truncate episodes after this many (frame-skipped) agent steps.
        reward: ``'raw'`` (team reward), ``'normalized'`` or ``'coverage'`` (camera team only).
        cache_resets: keep a pre-sampled episode start in the state instead of sampling a new
            one on every step for the auto-reset. Call :meth:`refresh_resets` periodically (e.g.
            once per rollout fragment); resets are ~3x more expensive than a step otherwise.
        advisor: an optional model-based camera team policy with ``init``, ``observe`` (every
            environment step) and ``decide`` (once per agent step, returning ``(actions, advice)``),
            e.g. :class:`mate_jax.foresight.ForesightCameraPolicy`. Its advice features are
            appended to every agent's observation; the learner decides how to use them.
    """

    def __init__(
        self,
        env: MultiAgentTracking,
        team: str = 'camera',
        opponent: Optional[Any] = None,
        frame_skip: int = 1,
        horizon: Optional[int] = None,
        reward: str = 'raw',
        relative_coordinates: bool = True,
        rescale_observation: bool = True,
        cache_resets: bool = False,
        advisor: Optional[Any] = None,
    ):
        assert team in ('camera', 'target')
        assert advisor is None or team == 'camera', 'advisors are implemented for the camera team only'
        self.advisor = advisor
        assert reward in ('raw', 'normalized', 'coverage')
        assert reward != 'coverage' or team == 'camera', '`coverage` reward is for cameras only.'
        assert frame_skip >= 1
        self.env = env
        self.team = team
        self.opponent_team = 'target' if team == 'camera' else 'camera'
        if opponent is None:
            opponent, _ = agents_lib.default_opponent(team, env)
        self.opponent = opponent
        self.frame_skip = int(frame_skip)
        max_steps = -(-(env.max_episode_steps + 1) // self.frame_skip)
        self.horizon = int(min(horizon, max_steps) if horizon is not None else max_steps)
        self.reward_type = reward
        self.cache_resets = bool(cache_resets)
        self.transform = ObservationTransform(env, team, relative_coordinates, rescale_observation)

        self.num_agents = env.num_cameras if team == 'camera' else env.num_targets
        self.observation_dim = self.transform.dim + (advisor.ADVICE_DIM if advisor is not None else 0)
        self.state_dim = env.state_dim
        self.action_dim = 2
        cfg = env.config
        if team == 'camera':
            self.action_scale = np.array([cfg.camera_rotation_step, cfg.camera_zooming_step])
        else:
            self.action_scale = np.array([cfg.target_step_size, cfg.target_step_size])
        self._state_shift, self._state_scale = self._state_normalizer()

    # Global state normalization (for centralized critics) ------------------------------------ #

    def _state_normalizer(self):
        env, cfg = self.env, self.env.config
        C, T, O = env.num_cameras, env.num_targets, env.num_obstacles
        low = np.concatenate(
            [consts.PRESERVED_LOW]
            + [consts.CAMERA_PRIVATE_LOW] * C
            + [consts.TARGET_PRIVATE_LOW] * T
            + [consts.OBSTACLE_LOW] * O
            + [np.zeros(2 * T + consts.NUM_WAREHOUSES**2)]
        )
        high = np.concatenate(
            [consts.PRESERVED_HIGH]
            + [consts.CAMERA_PRIVATE_HIGH] * C
            + [consts.TARGET_PRIVATE_HIGH] * T
            + [consts.OBSTACLE_HIGH] * O
            + [np.full(2 * T, 4.0 * env.reward_scale)]
            + [np.full(consts.NUM_WAREHOUSES**2, 2.0 * max(1.0, cfg.num_cargoes_per_target * T / 4.0))]
        )
        shift = np.where(np.isfinite(low), low, 0.0)
        finite = np.isfinite(low) & np.isfinite(high) & (high > low)
        scale = np.where(finite, 2.0 / np.where(finite, high - low, 1.0), 1.0)
        shift = shift + np.where(finite, (high - low) / 2.0, 0.0)  # map to [-1, +1]
        return shift, scale

    def global_state(self, env_state: EnvState) -> jax.Array:
        state = self.env.state_vector(env_state)
        return (state - jnp.asarray(self._state_shift, state.dtype)) * jnp.asarray(
            self._state_scale, state.dtype
        )

    # Helpers ----------------------------------------------------------------------------------- #

    def _split(self, camera_obs, target_obs):
        if self.team == 'camera':
            return camera_obs, target_obs
        return target_obs, camera_obs

    def _merge(self, team_action, opponent_action):
        if self.team == 'camera':
            return team_action, opponent_action
        return opponent_action, team_action

    def _observation(self, team_observation, advice):
        observation = self.transform(team_observation)
        if self.advisor is not None:
            observation = jnp.concatenate([observation, advice.astype(observation.dtype)], axis=-1)
        return observation

    def observe(self, state: TeamState) -> jax.Array:
        """Transformed joint observation (N, D) of the learning team (with advice, if any)."""

        return self._observation(state.team_observation, state.advice)

    # API ---------------------------------------------------------------------------------------- #

    def _sample_start(self, key):
        """A fresh episode start: (env_state, team_obs, opponent_obs, opponent_carry, advisor_carry, advice)."""

        key_env, key_opponent, key_advisor = jax.random.split(key, 3)
        (camera_obs, target_obs), env_state = self.env.reset(key_env)
        team_obs, opponent_obs = self._split(camera_obs, target_obs)
        advisor_carry = advice = None
        if self.advisor is not None:
            advisor_carry = self.advisor.init(key_advisor, team_obs)
            _, advice = self.advisor.decide(advisor_carry, team_obs)
        opponent_carry = self.opponent.init(key_opponent, opponent_obs)
        return env_state, team_obs, opponent_obs, opponent_carry, advisor_carry, advice

    def refresh_resets(self, key, state: TeamState) -> TeamState:
        """Re-sample the cached episode start (no-op without ``cache_resets``)."""

        if not self.cache_resets:
            return state
        return state.replace(next_reset=self._sample_start(key))

    def reset(self, key) -> Tuple[jax.Array, jax.Array, TeamState]:
        key_start, key_cache = jax.random.split(key)
        env_state, team_obs, opponent_obs, opponent_carry, advisor_carry, advice = self._sample_start(key_start)
        fdt = float_dtype()
        state = TeamState(
            env_state=env_state,
            opponent_carry=opponent_carry,
            opponent_observation=opponent_obs,
            team_observation=team_obs,
            step=jnp.zeros((), jnp.int32),
            episode_return=jnp.zeros((), fdt),
            episode_coverage=jnp.zeros((), fdt),
            returned_episode_return=jnp.zeros((), fdt),
            returned_episode_length=jnp.zeros((), jnp.int32),
            returned_episode_coverage=jnp.zeros((), fdt),
            returned_episode_delivered=jnp.zeros((), jnp.int32),
            advisor_carry=advisor_carry,
            advice=advice,
        )
        state = self.refresh_resets(key_cache, state)
        return self._observation(team_obs, advice), self.global_state(env_state), state

    def scale_actions(self, normalized_actions):
        """Map actions from ``[-1, +1]`` to the team's action bounds."""

        return jnp.clip(normalized_actions, -1.0, 1.0) * jnp.asarray(
            self.action_scale, normalized_actions.dtype
        )

    def step(self, key, state: TeamState, actions) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, Dict[str, jax.Array], TeamState]:
        """Step with team actions in environment units (see :meth:`scale_actions`).

        Returns ``(obs (N, D), global_state (S,), reward (N,), done (), info, state)``; the
        environment is reset automatically when ``done`` is True.
        """

        fdt = float_dtype()
        key_reset, key_skip = jax.random.split(key)

        def substep(carry, k):
            env_state, opponent_carry, opponent_obs, team_obs, finished, totals, advisor_carry = carry
            k_opponent, k_env, k_advisor = jax.random.split(k, 3)
            opponent_action, new_carry = self.opponent.act(k_opponent, opponent_carry, opponent_obs)
            camera_action, target_action = self._merge(actions, opponent_action)
            timestep, new_env_state = self.env.step(k_env, env_state, camera_action, target_action)
            new_team_obs, new_opponent_obs = self._split(
                timestep.camera_observation, timestep.target_observation
            )
            if self.reward_type == 'coverage':
                reward = timestep.info['coverage_rate']
            else:
                reward = timestep.camera_reward if self.team == 'camera' else timestep.target_reward
                if self.reward_type == 'normalized':
                    reward = reward / self.env.max_target_team_episode_reward
            active = jnp.logical_not(finished)
            new_totals = {
                'reward': totals['reward'] + jnp.where(active, reward, 0.0),
                'coverage_rate': totals['coverage_rate'] + jnp.where(active, timestep.info['coverage_rate'], 0.0),
                'real_coverage_rate': totals['real_coverage_rate']
                + jnp.where(active, timestep.info['real_coverage_rate'], 0.0),
                'num_substeps': totals['num_substeps'] + active.astype(jnp.int32),
                'num_delivered_cargoes': jnp.where(
                    active, timestep.info['num_delivered_cargoes'], totals['num_delivered_cargoes']
                ),
            }
            env_state = _select(active, new_env_state, env_state)
            opponent_carry = _select(active, new_carry, opponent_carry)
            opponent_obs = jnp.where(active, new_opponent_obs, opponent_obs)
            team_obs = jnp.where(active, new_team_obs, team_obs)
            if self.advisor is not None:  # the advisor's beliefs follow every environment step
                advisor_carry = _select(
                    active, self.advisor.observe(k_advisor, advisor_carry, new_team_obs), advisor_carry
                )
            finished = finished | (active & timestep.done)
            return (env_state, opponent_carry, opponent_obs, team_obs, finished, new_totals, advisor_carry), None

        totals = {
            'reward': jnp.zeros((), fdt),
            'coverage_rate': jnp.zeros((), fdt),
            'real_coverage_rate': jnp.zeros((), fdt),
            'num_substeps': jnp.zeros((), jnp.int32),
            'num_delivered_cargoes': jnp.zeros((), jnp.int32),
        }
        carry = (
            state.env_state,
            state.opponent_carry,
            state.opponent_observation,
            state.team_observation,
            jnp.zeros((), bool),
            totals,
            state.advisor_carry,
        )
        carry, _ = jax.lax.scan(substep, carry, jax.random.split(key_skip, self.frame_skip))
        env_state, opponent_carry, opponent_obs, team_obs, terminated, totals, advisor_carry = carry
        advice = state.advice
        if self.advisor is not None:  # plan once per agent step
            _, advice = self.advisor.decide(advisor_carry, team_obs)

        step = state.step + 1
        truncated = step >= self.horizon
        done = terminated | truncated
        reward = totals['reward']
        coverage = totals['coverage_rate'] / jnp.maximum(totals['num_substeps'], 1)
        episode_return = state.episode_return + reward
        episode_coverage = state.episode_coverage + coverage

        # Auto-reset.
        if self.cache_resets:
            start = state.next_reset
        else:
            start = self._sample_start(key_reset)
        reset_env_state, reset_team_obs, reset_opponent_obs, reset_carry, reset_advisor_carry, reset_advice = start

        new_state = TeamState(
            env_state=_select(done, reset_env_state, env_state),
            opponent_carry=_select(done, reset_carry, opponent_carry),
            opponent_observation=jnp.where(done, reset_opponent_obs, opponent_obs),
            team_observation=jnp.where(done, reset_team_obs, team_obs),
            step=jnp.where(done, 0, step),
            episode_return=jnp.where(done, 0.0, episode_return),
            episode_coverage=jnp.where(done, 0.0, episode_coverage),
            returned_episode_return=jnp.where(done, episode_return, state.returned_episode_return),
            returned_episode_length=jnp.where(done, step, state.returned_episode_length),
            returned_episode_coverage=jnp.where(
                done, episode_coverage / step.astype(fdt), state.returned_episode_coverage
            ),
            returned_episode_delivered=jnp.where(
                done, totals['num_delivered_cargoes'], state.returned_episode_delivered
            ),
            next_reset=state.next_reset,
            advisor_carry=_select(done, reset_advisor_carry, advisor_carry),
            advice=_select(done, reset_advice, advice),
        )
        normalized = jnp.abs(actions) / jnp.asarray(self.action_scale, fdt)
        info = {
            'coverage_rate': coverage,
            'tx_rate': jnp.zeros((), fdt),  # no explicit communication
            'actuation': jnp.mean(normalized),
            'real_coverage_rate': totals['real_coverage_rate'] / jnp.maximum(totals['num_substeps'], 1),
            'num_delivered_cargoes': totals['num_delivered_cargoes'],
            'terminated': terminated,
            'truncated': truncated,
            'returned_episode': done,
            'returned_episode_return': new_state.returned_episode_return,
            'returned_episode_length': new_state.returned_episode_length,
            'returned_episode_coverage': new_state.returned_episode_coverage,
            'returned_episode_delivered': new_state.returned_episode_delivered,
        }
        rewards = jnp.full((self.num_agents,), reward, fdt)  # shared team reward
        obs = self._observation(new_state.team_observation, new_state.advice)
        return obs, self.global_state(new_state.env_state), rewards, done, info, new_state
