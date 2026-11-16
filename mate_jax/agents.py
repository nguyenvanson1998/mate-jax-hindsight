"""Rule-based team policies in pure JAX (used as opponents during training).

Every policy acts on the *joint* observation of one team and keeps a pytree carry::

    carry = policy.init(key, joint_observation)              # after env.reset()
    actions, carry = policy.act(key, carry, joint_observation)  # before every env.step()

These are vectorized re-implementations of :mod:`mate.agents.random` and
:mod:`mate.agents.greedy`. Intra-team messages of the original agents are modelled as
instantaneous team-wide sharing (the original greedy agents exchange the same information
through ``send_responses`` / ``receive_responses`` with random delays).
"""

from typing import Any, Tuple

import jax
import jax.numpy as jnp
from flax import struct

from mate_jax import constants as consts
from mate_jax.geometry import arctan2_deg, normalize_angle, safe_norm, sin_deg


__all__ = [
    'RandomCameraPolicy',
    'RandomTargetPolicy',
    'RandomWaypointTargetPolicy',
    'GreedyCameraPolicy',
    'GreedyTargetPolicy',
    'StaticCameraPolicy',
    'StaticTargetPolicy',
]

P = consts.PRESERVED_DIM
W = consts.NUM_WAREHOUSES


def _camera_self(obs):
    """Parse the private camera state from (..., Dc) camera observations."""

    s = obs[..., P : P + consts.CAMERA_STATE_DIM_PRIVATE]
    sight = safe_norm(s[..., 3:5])
    max_sight = s[..., 6]
    return {
        'location': s[..., 0:2],
        'sight_range': sight,
        'orientation': arctan2_deg(s[..., 4], s[..., 3]),
        'viewing_angle': s[..., 5],
        'max_sight_range': max_sight,
        'min_viewing_angle': s[..., 5] * jnp.square(sight / max_sight),
        'rotation_step': s[..., 7],
        'zooming_step': s[..., 8],
    }


def _target_self(obs):
    """Parse the private target state from (..., Dt) target observations."""

    s = obs[..., P : P + consts.TARGET_STATE_DIM_PRIVATE]
    return {
        'location': s[..., 0:2],
        'step_size': s[..., 4],
        'goal_bits': s[..., 6 : 6 + W],
        'empty_bits': s[..., 6 + W : 6 + 2 * W] > 0.5,
    }


@struct.dataclass
class _Base:
    num_cameras: int = struct.field(pytree_node=False)
    num_targets: int = struct.field(pytree_node=False)
    num_obstacles: int = struct.field(pytree_node=False)

    @classmethod
    def from_env(cls, env, **kwargs):
        return cls(
            num_cameras=env.num_cameras,
            num_targets=env.num_targets,
            num_obstacles=env.num_obstacles,
            **kwargs,
        )

    def init(self, key, observation) -> Any:  # pylint: disable=unused-argument
        return ()


@struct.dataclass
class StaticCameraPolicy(_Base):
    """Cameras that never move."""

    def act(self, key, carry, observation):  # pylint: disable=unused-argument
        return jnp.zeros(observation.shape[:-1] + (2,), observation.dtype), carry


@struct.dataclass
class StaticTargetPolicy(StaticCameraPolicy):
    """Targets that never move."""


@struct.dataclass
class RandomCameraPolicy(_Base):
    """Uniformly random camera actions (``mate.RandomCameraAgent``)."""

    def act(self, key, carry, observation):
        me = _camera_self(observation)
        high = jnp.stack([me['rotation_step'], me['zooming_step']], axis=-1)
        u = jax.random.uniform(key, high.shape, observation.dtype, minval=-1.0, maxval=1.0)
        return u * high, carry


@struct.dataclass
class RandomTargetPolicy(_Base):
    """Uniformly random target actions (``mate.RandomTargetAgent``)."""

    def act(self, key, carry, observation):
        me = _target_self(observation)
        u = jax.random.uniform(key, me['location'].shape, observation.dtype, minval=-1.0, maxval=1.0)
        return u * me['step_size'][..., None], carry


@struct.dataclass
class RandomWaypointTargetPolicy(_Base):
    """Random Waypoint mobility: move at a known constant speed towards a uniformly random
    waypoint, then draw a new one (the classic sensor-network mobility model)."""

    speed_ratio: float = struct.field(pytree_node=False, default=1.0)  # fraction of step size
    margin: float = struct.field(pytree_node=False, default=100.0)

    def _waypoints(self, key, shape, dtype):
        bound = consts.TERRAIN_SIZE - self.margin
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)

    def init(self, key, observation):
        me = _target_self(observation)
        return {'waypoint': self._waypoints(key, me['location'].shape, observation.dtype)}

    def act(self, key, carry, observation):
        me = _target_self(observation)
        speed = self.speed_ratio * me['step_size']
        direction = carry['waypoint'] - me['location']
        distance = safe_norm(direction)
        arrived = distance <= speed
        waypoint = jnp.where(
            arrived[..., None],
            self._waypoints(key, direction.shape, observation.dtype),
            carry['waypoint'],
        )
        direction = waypoint - me['location']
        distance = safe_norm(direction)
        action = direction * (jnp.minimum(speed, distance) / jnp.maximum(distance, 1e-6))[..., None]
        return action, {'waypoint': waypoint}


@struct.dataclass
class GreedyTargetPolicy(_Base):
    """Run towards the destination warehouse with sticky noise (``mate.GreedyTargetAgent``)."""

    noise_scale: float = struct.field(pytree_node=False, default=0.5)

    def init(self, key, observation):
        me = _target_self(observation)
        u = jax.random.uniform(key, me['location'].shape, observation.dtype, minval=-1.0, maxval=1.0)
        return {
            'goal_bits': me['goal_bits'],
            'prev_location': me['location'],
            'prev_noise': 0.5 * u * me['step_size'][..., None],
            'non_empty': jnp.ones(me['goal_bits'].shape, dtype=bool),
        }

    def act(self, key, carry, observation):
        me = _target_self(observation)
        k_goal, k_switch, k_noise = jax.random.split(key, 3)
        step_size = me['step_size']

        # Team-wide knowledge of the empty warehouses (broadcast messages).
        non_empty = jnp.logical_and(carry['non_empty'], jnp.logical_not(me['empty_bits']))
        non_empty = jnp.broadcast_to(jnp.all(non_empty, axis=-2, keepdims=True), non_empty.shape)

        has_cargo = jnp.any(me['goal_bits'] != 0, axis=-1)
        goal_bits = jnp.where(has_cargo[..., None], me['goal_bits'], carry['goal_bits'])
        has_goal = jnp.any(goal_bits != 0, axis=-1)
        goal = jnp.argmax(goal_bits != 0, axis=-1)
        goal_non_empty = jnp.take_along_axis(non_empty, goal[..., None], axis=-1)[..., 0]
        need_new = jnp.logical_not(has_goal) | (jnp.logical_not(has_cargo) & jnp.logical_not(goal_non_empty))
        new_goal = jax.random.categorical(k_goal, jnp.where(non_empty, 0.0, -jnp.inf), axis=-1)
        can_pick = jnp.any(non_empty, axis=-1)
        new_bits = jnp.where(can_pick[..., None], jax.nn.one_hot(new_goal, W, dtype=goal_bits.dtype), 0.0)
        goal_bits = jnp.where(need_new[..., None], new_bits, goal_bits)
        has_goal = jnp.any(goal_bits != 0, axis=-1)
        goal = jnp.argmax(goal_bits != 0, axis=-1)

        warehouses = jnp.asarray(consts.WAREHOUSES, observation.dtype)
        action = jnp.where(has_goal[..., None], warehouses[goal] - me['location'], 0.0)
        norm = safe_norm(action)
        action = action * jnp.where(norm > step_size, step_size / jnp.maximum(norm, 1e-12), 1.0)[..., None]

        prev_actual = safe_norm(me['location'] - carry['prev_location'])
        prob = jnp.where(prev_actual > 0.2 * step_size, 0.05, 0.75)
        switch = jax.random.bernoulli(k_switch, prob)
        u = jax.random.uniform(k_noise, action.shape, action.dtype, minval=-1.0, maxval=1.0)
        noise = jnp.where(switch[..., None], self.noise_scale * u * step_size[..., None], carry['prev_noise'])

        action = jnp.clip(action + noise, -step_size[..., None], step_size[..., None])
        carry = {
            'goal_bits': goal_bits,
            'prev_location': me['location'],
            'prev_noise': noise,
            'non_empty': non_empty,
        }
        return action, carry


@struct.dataclass
class GreedyCameraPolicy(_Base):
    """Track the nearest remembered target (``mate.GreedyCameraAgent``).

    The cameras share their observed target states with all teammates every step.
    """

    memory_period: int = struct.field(pytree_node=False, default=25)
    filterout_unloaded: bool = struct.field(pytree_node=False, default=False)
    filterout_beyond_range: bool = struct.field(pytree_node=False, default=True)
    share_observations: bool = struct.field(pytree_node=False, default=True)
    range_factor: float = struct.field(pytree_node=False, default=1.1)

    def _target_view(self, observation):
        offset = P + consts.CAMERA_STATE_DIM_PRIVATE
        size = self.num_targets * (consts.TARGET_STATE_DIM_PUBLIC + 1)
        view = observation[..., offset : offset + size]
        view = view.reshape(view.shape[:-1] + (self.num_targets, consts.TARGET_STATE_DIM_PUBLIC + 1))
        return view[..., :-1], view[..., -1] > 0.5  # states (..., C, T, 4), flags (..., C, T)

    def _memorize(self, carry, observation):
        states, seen = self._target_view(observation)
        if self.share_observations:
            # Every visible target state is identical for all observers: merge team-wide.
            team_seen = jnp.any(seen, axis=-2, keepdims=True)
            first = jnp.argmax(seen, axis=-2)[..., None, :, None]
            team_states = jnp.take_along_axis(states, first, axis=-3)
            seen = jnp.broadcast_to(team_seen, seen.shape)
            states = jnp.broadcast_to(team_states, states.shape)
        memory = jnp.where(seen[..., None], states, carry['memory'])
        time2forget = jnp.where(seen, self.memory_period, carry['time2forget'])
        never_loaded = carry['never_loaded'] & jnp.logical_not(seen & (states[..., 3] > 0.5))
        return {**carry, 'memory': memory, 'time2forget': time2forget, 'never_loaded': never_loaded}

    def init(self, key, observation):
        states, seen = self._target_view(observation)
        carry = {
            'memory': states,
            'time2forget': jnp.zeros(seen.shape, jnp.int32),
            'never_loaded': jnp.ones(seen.shape, dtype=bool),
            'prev_action': jnp.zeros(observation.shape[:-1] + (2,), observation.dtype),
        }
        return self._memorize(carry, observation)

    def _best_viewing_angle(self, me, distance):
        min_va, max_sight = me['min_viewing_angle'], me['max_sight_range']
        area = me['viewing_angle'] * jnp.square(me['sight_range'])
        best = jnp.full_like(distance, min(180.0, consts.MAX_CAMERA_VIEWING_ANGLE))
        for _ in range(20):
            sight = distance * (1.0 + sin_deg(jnp.minimum(best / 2.0, 90.0)))
            best = area / jnp.square(jnp.maximum(sight, 1e-6))
        best = jnp.clip(best, min_va, consts.MAX_CAMERA_VIEWING_ANGLE)
        best = jnp.where(distance <= jnp.sqrt(area / 180.0) / 2.0, consts.MAX_CAMERA_VIEWING_ANGLE, best)
        return jnp.where(distance * (1.0 + sin_deg(min_va / 2.0)) >= max_sight, min_va, best)

    def act(self, key, carry, observation):
        k_explore, k_random = jax.random.split(key)
        carry = {**carry, 'time2forget': jnp.maximum(carry['time2forget'] - 1, 0)}
        carry = self._memorize(carry, observation)
        me = _camera_self(observation)

        locations = carry['memory'][..., 0:2]
        distance = safe_norm(locations - me['location'][..., None, :])  # (..., C, T)
        candidate = carry['time2forget'] > 0
        if self.filterout_beyond_range:
            candidate &= distance < self.range_factor * me['max_sight_range'][..., None]
        if self.filterout_unloaded:
            candidate &= (carry['memory'][..., 3] > 0.5) | carry['never_loaded']
        has_candidate = jnp.any(candidate, axis=-1)
        nearest = jnp.argmin(jnp.where(candidate, distance, jnp.inf), axis=-1)

        relative = jnp.take_along_axis(locations, nearest[..., None, None], axis=-2)[..., 0, :]
        relative = relative - me['location']
        target_distance = safe_norm(relative)
        best_orientation = arctan2_deg(relative[..., 1], relative[..., 0])
        high = jnp.stack([me['rotation_step'], me['zooming_step']], axis=-1)
        tracking = jnp.stack(
            [
                normalize_angle(best_orientation - me['orientation']),
                self._best_viewing_angle(me, target_distance) - me['viewing_angle'],
            ],
            axis=-1,
        )
        tracking = jnp.clip(tracking, -high, high)

        explore = jax.random.bernoulli(k_explore, 0.1, has_candidate.shape)
        random = jax.random.uniform(k_random, high.shape, high.dtype, minval=-1.0, maxval=1.0) * high
        idle = jnp.where(explore[..., None], random, carry['prev_action'])
        action = jnp.where(has_candidate[..., None], tracking, idle)
        return action, {**carry, 'prev_action': action}


def default_opponent(team: str, env) -> Tuple[Any, str]:
    """The default opponent of the given (learning) team, as in ``examples/mappo``."""

    if team == 'camera':
        return GreedyTargetPolicy.from_env(env), 'target'
    return GreedyCameraPolicy.from_env(env), 'camera'
