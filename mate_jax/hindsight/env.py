"""Training environment of the student cameras for PPO fine-tuning (MAPPO / IPPO).

Actions are local waypoints (:class:`LocalWaypoints`), re-decided every environment step and executed at
the rate limits; observations are the :class:`StudentFeatures`; the team reward is the coverage rate
minus an optional actuation cost. The
interface matches :class:`mate_jax.wrappers.SingleTeamEnv`, so :class:`mate_jax.mappo.MAPPOTrainer`
can train it directly.
"""

from typing import Any, Optional

import jax
import jax.numpy as jnp
from flax import struct

from mate_jax.env import float_dtype
from mate_jax.foresight import ForesightCameraPolicy
from mate_jax.hindsight.student import LocalWaypoints, StudentFeatures, trajectory_planner
from mate_jax.wrappers import SingleTeamEnv, _select


__all__ = ['StudentState', 'StudentEnv']


@struct.dataclass
class StudentState:
    env_state: Any
    opponent_carry: Any
    opponent_observation: jax.Array
    team_observation: jax.Array
    planner_carry: Any
    features: jax.Array  # (C, F)
    step: jax.Array
    episode_return: jax.Array
    episode_coverage: jax.Array
    returned_episode_return: jax.Array
    returned_episode_length: jax.Array
    returned_episode_coverage: jax.Array
    next_reset: Any = None


class StudentEnv:
    """Camera-team environment whose actions are local waypoints and whose observations are student features."""

    def __init__(
        self,
        env,
        opponent: Optional[Any] = None,
        planner: Optional[ForesightCameraPolicy] = None,
        horizon: int = 500,
        actuation_cost: float = 0.0,
        cache_resets: bool = True,
        use_trajectory: bool = True,
    ):
        self.base = SingleTeamEnv(env, team='camera', opponent=opponent, frame_skip=1, horizon=horizon, cache_resets=cache_resets)
        self.env = env
        self.opponent = self.base.opponent
        self.planner = planner if planner is not None else ForesightCameraPolicy.from_env(env)
        self.waypoints = LocalWaypoints(env)
        trajectory = trajectory_planner(env) if use_trajectory else None
        self.features = StudentFeatures(env, self.planner, self.waypoints, trajectory)
        self.transform = self.base.transform
        self.frame_skip = 1
        self.horizon = self.base.horizon
        self.actuation_cost = float(actuation_cost)
        self.cache_resets = bool(cache_resets)
        self.num_agents = env.num_cameras
        self.num_actions = self.waypoints.num_classes
        self.observation_dim = self.features.dim
        self.state_dim = env.state_dim
        self.action_dim = 2

    def global_state(self, env_state):
        return self.base.global_state(env_state)

    def scale_actions(self, actions):  # discrete waypoint classes
        return actions

    def _sample_start(self, key):
        key_env, key_opponent, key_planner = jax.random.split(key, 3)
        (camera_obs, target_obs), env_state = self.env.reset(key_env)
        planner_carry = self.planner.init(key_planner, camera_obs)
        features = self.features(planner_carry, camera_obs)
        opponent_carry = self.opponent.init(key_opponent, target_obs)
        return env_state, camera_obs, target_obs, opponent_carry, planner_carry, features

    def refresh_resets(self, key, state: StudentState) -> StudentState:
        if not self.cache_resets:
            return state
        return state.replace(next_reset=self._sample_start(key))

    def reset(self, key):
        key_start, key_cache = jax.random.split(key)
        env_state, team_obs, opponent_obs, opponent_carry, planner_carry, features = self._sample_start(key_start)
        fdt = float_dtype()
        zero = jnp.zeros((), fdt)
        state = StudentState(
            env_state=env_state, opponent_carry=opponent_carry, opponent_observation=opponent_obs,
            team_observation=team_obs, planner_carry=planner_carry, features=features,
            step=jnp.zeros((), jnp.int32), episode_return=zero, episode_coverage=zero,
            returned_episode_return=zero, returned_episode_length=jnp.zeros((), jnp.int32),
            returned_episode_coverage=zero,
        )  # fmt: skip
        state = self.refresh_resets(key_cache, state)
        return features, self.global_state(env_state), state

    def step(self, key, state: StudentState, actions):
        fdt = float_dtype()
        key_reset, key_opponent, key_env, key_planner = jax.random.split(key, 4)
        camera_action = self.waypoints.action(
            state.env_state.camera_orientation, state.env_state.camera_viewing_angle, actions.astype(jnp.int32)
        ).astype(state.team_observation.dtype)
        target_action, opponent_carry = self.opponent.act(key_opponent, state.opponent_carry, state.opponent_observation)
        timestep, env_state = self.env.step(key_env, state.env_state, camera_action, target_action)
        planner_carry = self.planner.observe(key_planner, state.planner_carry, timestep.camera_observation)
        features = self.features(planner_carry, timestep.camera_observation)

        options = self.waypoints.controller
        actuation = jnp.mean(
            0.5 * (jnp.abs(camera_action[:, 0]) / options.rotation_step + jnp.abs(camera_action[:, 1]) / options.zooming_step)
        ).astype(fdt)
        coverage = timestep.info['coverage_rate'].astype(fdt)
        reward = coverage - self.actuation_cost * actuation
        step = state.step + 1
        terminated = timestep.done
        truncated = step >= self.horizon
        done = terminated | truncated
        episode_return = state.episode_return + reward
        episode_coverage = state.episode_coverage + coverage

        if self.cache_resets:
            start = state.next_reset
        else:
            start = self._sample_start(key_reset)
        r_env_state, r_team_obs, r_opponent_obs, r_opponent_carry, r_planner_carry, r_features = start
        new_state = StudentState(
            env_state=_select(done, r_env_state, env_state),
            opponent_carry=_select(done, r_opponent_carry, opponent_carry),
            opponent_observation=jnp.where(done, r_opponent_obs, timestep.target_observation),
            team_observation=jnp.where(done, r_team_obs, timestep.camera_observation),
            planner_carry=_select(done, r_planner_carry, planner_carry),
            features=jnp.where(done, r_features, features),
            step=jnp.where(done, 0, step),
            episode_return=jnp.where(done, 0.0, episode_return),
            episode_coverage=jnp.where(done, 0.0, episode_coverage),
            returned_episode_return=jnp.where(done, episode_return, state.returned_episode_return),
            returned_episode_length=jnp.where(done, step, state.returned_episode_length),
            returned_episode_coverage=jnp.where(
                done, episode_coverage / step.astype(fdt), state.returned_episode_coverage
            ),
            next_reset=state.next_reset,
        )
        info = {
            'coverage_rate': coverage,
            'real_coverage_rate': timestep.info['real_coverage_rate'],
            'tx_rate': jnp.zeros((), fdt),
            'actuation': actuation,
            'terminated': terminated,
            'truncated': truncated,
            'returned_episode': done,
            'returned_episode_return': new_state.returned_episode_return,
            'returned_episode_length': new_state.returned_episode_length,
            'returned_episode_coverage': new_state.returned_episode_coverage,
            'returned_episode_delivered': jnp.zeros((), jnp.int32),
        }
        rewards = jnp.full((self.num_agents,), reward, fdt)
        return new_state.features, self.global_state(new_state.env_state), rewards, done, info, new_state
