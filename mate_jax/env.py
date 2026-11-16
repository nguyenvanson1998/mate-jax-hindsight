"""The Multi-Agent Tracking Environment, re-implemented in pure JAX.

Every method of :class:`MultiAgentTracking` is a pure function of ``(key, state, action)``
with static shapes, so the environment can be ``jax.jit``-compiled, ``jax.vmap``-ed over
thousands of parallel instances, and sharded across TPU cores. The dynamics follow
:class:`mate.MultiAgentTracking` step by step; see ``docs/jax_tpu.md`` for the (small)
documented differences.
"""

# pylint: disable=too-many-locals,too-many-statements

from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from mate_jax import constants as consts
from mate_jax.config import EnvConfig, load_config
from mate_jax.geometry import (
    arctan2_deg,
    line_of_sight,
    normalize_angle,
    obstruct_step,
    polar2cartesian,
    safe_norm,
)


__all__ = ['EnvState', 'TimeStep', 'MultiAgentTracking', 'make']

W = consts.NUM_WAREHOUSES
DEFAULT_NUM_RESET_CANDIDATES = 32


def float_dtype():
    """The default floating point dtype (float32, or float64 if ``jax_enable_x64`` is set)."""

    return jax.dtypes.canonicalize_dtype(jnp.float64)


@struct.dataclass
class EnvState:  # pylint: disable=too-many-instance-attributes
    """The full (hidden) state of one environment instance."""

    camera_location: jax.Array  # (C, 2)
    camera_orientation: jax.Array  # (C,) degrees in [-180, 180)
    camera_viewing_angle: jax.Array  # (C,) degrees
    camera_sight_range: jax.Array  # (C,)

    target_location: jax.Array  # (T, 2)
    target_orientation: jax.Array  # (T,) rendering only
    target_capacity: jax.Array  # (T,) int32 in {1, 2}
    target_goal_bits: jax.Array  # (T, W) int32 cargo weights
    target_empty_bits: jax.Array  # (T, W) bool
    target_goals: jax.Array  # (T,) int32, -1 for unloaded
    target_dones: jax.Array  # (T,) bool
    target_is_colliding: jax.Array  # (T,) bool
    target_steps: jax.Array  # (T,) int32
    tracked_steps: jax.Array  # (T,) int32
    freights: jax.Array  # (T,) int32
    bounties: jax.Array  # (T,) int32

    obstacle_location: jax.Array  # (O, 2)
    obstacle_radius: jax.Array  # (O,)

    camera_target_view_mask: jax.Array  # (C, T) bool
    camera_obstacle_view_mask: jax.Array  # (C, O) bool
    camera_camera_view_mask: jax.Array  # (C, C) bool
    target_camera_view_mask: jax.Array  # (T, C) bool
    target_obstacle_view_mask: jax.Array  # (T, O) bool
    target_target_view_mask: jax.Array  # (T, T) bool
    tracked_bits: jax.Array  # (T,) bool

    remaining_cargoes: jax.Array  # (W, W) int32
    awaiting_cargo_counts: jax.Array  # (W,) int32
    num_delivered_cargoes: jax.Array  # () int32
    target_team_episode_reward: jax.Array  # ()
    delayed_target_team_episode_reward: jax.Array  # ()
    episode_step: jax.Array  # () int32

    camera_slots: jax.Array  # (C,) int32 config slot of each camera (after shuffling)
    target_slots: jax.Array  # (T,) int32
    obstacle_slots: jax.Array  # (O,) int32


class TimeStep(NamedTuple):
    """Outputs of :meth:`MultiAgentTracking.step` (besides the new state)."""

    camera_observation: jax.Array  # (C, Dc)
    target_observation: jax.Array  # (T, Dt)
    camera_reward: jax.Array  # () team reward
    target_reward: jax.Array  # () team reward
    done: jax.Array  # () bool
    info: Dict[str, jax.Array]


def _ints(x):
    return jnp.asarray(x, dtype=jnp.int32)


class MultiAgentTracking:  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """Functional JAX implementation of the Multi-Agent Tracking Environment.

    The environment object only holds the static configuration. All dynamic data lives in an
    :class:`EnvState` pytree, so the methods below can be transformed freely::

        env = MultiAgentTracking('MATE-4v8-9.yaml')
        (camera_obs, target_obs), state = jax.jit(env.reset)(key)
        step = jax.jit(env.step)
        timestep, state = step(key, state, camera_actions, target_actions)

        # thousands of environments in parallel
        keys = jax.random.split(key, 4096)
        obs, states = jax.vmap(env.reset)(keys)
    """

    def __init__(
        self,
        config: Optional[Union[EnvConfig, Dict[str, Any], str]] = None,
        num_reset_candidates: int = DEFAULT_NUM_RESET_CANDIDATES,
        unroll_loops: bool = False,
        **overrides,
    ) -> None:
        self.config = cfg = load_config(config, **overrides)
        self.num_reset_candidates = int(num_reset_candidates)
        # Unroll the small fixed loops (collisions, cargo hand-over): faster on TPUs, but larger
        # programs that take much longer to compile on GPUs.
        self.unroll_loops = bool(unroll_loops)

        self.num_cameras = C = cfg.num_cameras
        self.num_targets = T = cfg.num_targets
        self.num_obstacles = O = cfg.num_obstacles
        self.num_warehouses = W
        self.max_episode_steps = cfg.max_episode_steps

        self.num_high_capacity_targets = cfg.num_high_capacity_targets
        self.freight_scale = float(np.ceil(consts.TERRAIN_WIDTH / cfg.target_step_size))
        self.bounty_scale = float(np.ceil(self.freight_scale * cfg.bounty_factor))
        self.reward_scale = self.freight_scale + self.bounty_scale
        self.max_target_team_episode_reward = (
            self.reward_scale * cfg.num_cargoes_per_target * self.num_targets
        )
        self.camera_area_product = cfg.camera_min_viewing_angle * cfg.camera_max_sight_range**2

        self.camera_observation_dim = consts.camera_observation_dim(C, T, O)
        self.target_observation_dim = consts.target_observation_dim(C, T, O)
        self.state_dim = consts.state_dim(C, T, O)

        self.camera_action_low = np.array([-cfg.camera_rotation_step, -cfg.camera_zooming_step])
        self.camera_action_high = -self.camera_action_low
        if C == 0:
            self.camera_action_low = self.camera_action_high = np.zeros(consts.CAMERA_ACTION_DIM)
        # Same as `mate`: the space is built before capacities are assigned (each target still
        # clips its own step to `step_size / capacity` in `step()`).
        self.target_action_low = np.array([-cfg.target_step_size, -cfg.target_step_size])
        self.target_action_high = -self.target_action_low

        self.preserved_data = np.concatenate(
            [[C, T, O, 0], consts.WAREHOUSES.ravel(), [consts.WAREHOUSE_RADIUS]]
        ).astype(np.float64)

    # ----------------------------------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------------------------------- #

    def reset(self, key: jax.Array) -> Tuple[Tuple[jax.Array, jax.Array], EnvState]:
        """Sample a new episode. Returns ``((camera_obs, target_obs), state)``."""

        cfg = self.config
        C, T, O = self.num_cameras, self.num_targets, self.num_obstacles
        (
            key_perm_c,
            key_perm_t,
            key_perm_o,
            key_capacity,
            key_place,
            key_view,
            key_cargo,
            key_goals,
            key_start,
            key_orient,
        ) = jax.random.split(key, 10)

        if cfg.shuffle_entities:
            camera_slots = jax.random.permutation(key_perm_c, C)
            target_slots = jax.random.permutation(key_perm_t, T)
            obstacle_slots = jax.random.permutation(key_perm_o, O)
        else:
            camera_slots, target_slots, obstacle_slots = jnp.arange(C), jnp.arange(T), jnp.arange(O)

        capacity = jnp.ones(T, dtype=jnp.int32)
        if self.num_high_capacity_targets > 0:
            if cfg.shuffle_entities:
                high = jax.random.choice(
                    key_capacity, T, shape=(self.num_high_capacity_targets,), replace=False
                )
            else:
                high = jnp.arange(self.num_high_capacity_targets)
            capacity = capacity.at[high].set(2)

        placement = self._place_entities(key_place, camera_slots, target_slots, obstacle_slots)
        (camera_location, camera_orientation, camera_viewing_angle, camera_sight_range) = placement[:4]
        obstacle_location, obstacle_radius, target_location = placement[4:]

        if C > 0 and O > 0:
            camera_obstacle_distance = safe_norm(
                obstacle_location[None, :, :] - camera_location[:, None, :]
            )
            camera_obstacle_view_mask = (
                camera_obstacle_distance < cfg.camera_max_sight_range + obstacle_radius[None, :]
            )
        else:
            camera_obstacle_view_mask = jnp.zeros((C, O), dtype=bool)

        fdt = float_dtype()
        state = EnvState(
            camera_location=camera_location,
            camera_orientation=camera_orientation,
            camera_viewing_angle=camera_viewing_angle,
            camera_sight_range=camera_sight_range,
            target_location=target_location,
            target_orientation=jnp.zeros(T, dtype=fdt),
            target_capacity=capacity,
            target_goal_bits=jnp.zeros((T, W), dtype=jnp.int32),
            target_empty_bits=jnp.zeros((T, W), dtype=bool),
            target_goals=jnp.full(T, -1, dtype=jnp.int32),
            target_dones=jnp.zeros(T, dtype=bool),
            target_is_colliding=jnp.zeros(T, dtype=bool),
            target_steps=jnp.zeros(T, dtype=jnp.int32),
            tracked_steps=jnp.zeros(T, dtype=jnp.int32),
            freights=jnp.zeros(T, dtype=jnp.int32),
            bounties=jnp.zeros(T, dtype=jnp.int32),
            obstacle_location=obstacle_location,
            obstacle_radius=obstacle_radius,
            camera_target_view_mask=jnp.zeros((C, T), dtype=bool),
            camera_obstacle_view_mask=camera_obstacle_view_mask,
            camera_camera_view_mask=jnp.zeros((C, C), dtype=bool),
            target_camera_view_mask=jnp.zeros((T, C), dtype=bool),
            target_obstacle_view_mask=jnp.zeros((T, O), dtype=bool),
            target_target_view_mask=jnp.zeros((T, T), dtype=bool),
            tracked_bits=jnp.zeros(T, dtype=bool),
            remaining_cargoes=jnp.zeros((W, W), dtype=jnp.int32),
            awaiting_cargo_counts=jnp.zeros(W, dtype=jnp.int32),
            num_delivered_cargoes=_ints(0),
            target_team_episode_reward=jnp.zeros((), dtype=fdt),
            delayed_target_team_episode_reward=jnp.zeros((), dtype=fdt),
            episode_step=_ints(0),
            camera_slots=_ints(camera_slots),
            target_slots=_ints(target_slots),
            obstacle_slots=_ints(obstacle_slots),
        )
        state = self._update_view(key_view, state)

        remaining, awaiting = self._generate_cargoes(key_cargo)
        state = state.replace(remaining_cargoes=remaining, awaiting_cargo_counts=awaiting)
        state, _, _ = self._assign_goals(key_goals, state)
        state = state.replace(
            target_dones=jnp.zeros(T, dtype=bool),
            num_delivered_cargoes=_ints(0),
            target_team_episode_reward=jnp.zeros((), dtype=fdt),
            delayed_target_team_episode_reward=jnp.zeros((), dtype=fdt),
        )
        if cfg.targets_start_with_cargoes:
            state = self._load_initial_cargoes(key_start, state)

        loaded = state.target_goals >= 0
        goal_location = jnp.asarray(consts.WAREHOUSES, dtype=fdt)[jnp.maximum(state.target_goals, 0)]
        towards_goal = goal_location - state.target_location
        random_orientation = normalize_angle(360.0 * jax.random.uniform(key_orient, (T,), dtype=fdt))
        target_orientation = jnp.where(
            loaded, arctan2_deg(towards_goal[:, 1], towards_goal[:, 0]), random_orientation
        )
        state = state.replace(target_orientation=target_orientation)

        return self.joint_observation(state), state

    def step(
        self,
        key: jax.Array,
        state: EnvState,
        camera_joint_action: jax.Array,
        target_joint_action: jax.Array,
    ) -> Tuple[TimeStep, EnvState]:
        """Run one timestep of the environment's dynamics (no auto-reset)."""

        fdt = float_dtype()
        camera_joint_action = jnp.asarray(camera_joint_action, dtype=fdt).reshape(
            self.num_cameras, consts.CAMERA_ACTION_DIM
        )
        target_joint_action = jnp.asarray(target_joint_action, dtype=fdt).reshape(
            self.num_targets, consts.TARGET_ACTION_DIM
        )
        key_view, key_goals = jax.random.split(key)

        state = self._simulate(key_view, state, camera_joint_action, target_joint_action)
        state, target_team_reward, delayed_target_team_reward = self._assign_goals(key_goals, state)
        state = state.replace(
            target_team_episode_reward=state.target_team_episode_reward + target_team_reward,
            delayed_target_team_episode_reward=(
                state.delayed_target_team_episode_reward + delayed_target_team_reward
            ),
        )
        camera_observation, target_observation = self.joint_observation(state)
        metrics = self.metrics(state)

        if self.config.reward_type == 'sparse':
            target_team_reward = delayed_target_team_reward
        camera_team_reward = -target_team_reward

        state = state.replace(
            target_steps=state.target_steps + 1,
            tracked_steps=state.tracked_steps + state.tracked_bits.astype(jnp.int32),
            episode_step=state.episode_step + 1,
        )
        done = jnp.logical_not(
            jnp.logical_and(
                state.episode_step <= self.max_episode_steps,
                jnp.any(state.awaiting_cargo_counts != 0),
            )
        )

        normalized = target_team_reward / self.max_target_team_episode_reward
        info = {
            **metrics,
            'camera_normalized_raw_reward': -normalized,
            'target_normalized_raw_reward': normalized,
            'target_dones': state.target_dones,
            'episode_step': state.episode_step,
        }
        timestep = TimeStep(
            camera_observation=camera_observation,
            target_observation=target_observation,
            camera_reward=camera_team_reward,
            target_reward=target_team_reward,
            done=done,
            info=info,
        )
        return timestep, state

    def joint_observation(self, state: EnvState) -> Tuple[jax.Array, jax.Array]:
        """Joint observations ``(camera_obs (C, Dc), target_obs (T, Dt))`` of both teams."""

        fdt = float_dtype()
        C, T, O = self.num_cameras, self.num_targets, self.num_obstacles

        camera_public = self._camera_states(state, private=False)
        camera_private = self._camera_states(state, private=True)
        camera_public_flagged = jnp.concatenate([camera_public, jnp.ones((C, 1), fdt)], axis=-1)
        target_public = self._target_states(state, private=False)
        target_private = self._target_states(state, private=True)
        target_public_flagged = jnp.concatenate([target_public, jnp.ones((T, 1), fdt)], axis=-1)
        obstacle_flagged = jnp.concatenate(
            [self._obstacle_states(state), jnp.ones((O, 1), fdt)], axis=-1
        )

        def masked(mask, states):
            # mask: (N, M), states: (M, D) -> (N, M * D)
            values = jnp.where(mask[:, :, None], states[None, :, :], 0.0)
            return values.reshape(mask.shape[0], mask.shape[1] * states.shape[1])

        preserved = jnp.asarray(self.preserved_data, dtype=fdt)
        camera_preserved = jnp.tile(preserved, (C, 1)).at[:, 3].set(jnp.arange(C, dtype=fdt))
        target_preserved = jnp.tile(preserved, (T, 1)).at[:, 3].set(jnp.arange(T, dtype=fdt))

        camera_observation = jnp.concatenate(
            [
                camera_preserved,
                camera_private,
                masked(state.camera_target_view_mask, target_public_flagged),
                masked(state.camera_obstacle_view_mask, obstacle_flagged),
                masked(state.camera_camera_view_mask, camera_public_flagged),
            ],
            axis=-1,
        )
        target_observation = jnp.concatenate(
            [
                target_preserved,
                target_private,
                masked(state.target_camera_view_mask, camera_public_flagged),
                masked(state.target_obstacle_view_mask, obstacle_flagged),
                masked(state.target_target_view_mask, target_public_flagged),
            ],
            axis=-1,
        )
        return camera_observation, target_observation

    def state_vector(self, state: EnvState) -> jax.Array:
        """The global state of the environment (same layout as ``mate.MultiAgentTracking.state()``)."""

        fdt = float_dtype()
        return jnp.concatenate(
            [
                jnp.asarray(self.preserved_data, dtype=fdt),
                self._camera_states(state, private=True).ravel(),
                self._target_states(state, private=True).ravel(),
                self._obstacle_states(state).ravel(),
                state.freights.astype(fdt),
                state.bounties.astype(fdt),
                state.remaining_cargoes.ravel().astype(fdt),
            ]
        )

    def metrics(self, state: EnvState) -> Dict[str, jax.Array]:
        """Coverage and transport metrics (the ``info`` fields of the original environment)."""

        fdt = float_dtype()
        tracked = state.tracked_bits.astype(fdt)
        with_bounty = (state.bounties > 0).astype(fdt)
        num_with_bounty = with_bounty.sum()
        delivered = state.num_delivered_cargoes.astype(fdt)
        return {
            'coverage_rate': tracked.sum() / self.num_targets,
            'real_coverage_rate': jnp.where(
                num_with_bounty > 0, (tracked * with_bounty).sum() / jnp.maximum(num_with_bounty, 1.0), 0.0
            ),
            'mean_transport_rate': jnp.where(
                delivered > 0,
                state.delayed_target_team_episode_reward
                / (self.reward_scale * jnp.maximum(delivered, 1.0)),
                0.0,
            ),
            'num_delivered_cargoes': state.num_delivered_cargoes,
        }

    # Action helpers -------------------------------------------------------------------------- #

    def sample_actions(self, key: jax.Array, state: EnvState) -> Tuple[jax.Array, jax.Array]:
        """Uniformly random joint actions (respecting each target's own step size)."""

        fdt = float_dtype()
        key_c, key_t = jax.random.split(key)
        camera = jax.random.uniform(
            key_c,
            (self.num_cameras, consts.CAMERA_ACTION_DIM),
            dtype=fdt,
            minval=self.camera_action_low,
            maxval=self.camera_action_high,
        )
        step_size = self.target_step_sizes(state)[:, None]
        target = jax.random.uniform(
            key_t, (self.num_targets, consts.TARGET_ACTION_DIM), dtype=fdt, minval=-1.0, maxval=1.0
        )
        return camera, target * step_size

    def target_step_sizes(self, state: EnvState) -> jax.Array:
        return self.config.target_step_size / state.target_capacity.astype(float_dtype())

    def camera_observation_bounds(self):
        return consts.camera_observation_bounds(self.num_cameras, self.num_targets, self.num_obstacles)

    def target_observation_bounds(self):
        return consts.target_observation_bounds(self.num_cameras, self.num_targets, self.num_obstacles)

    def __repr__(self) -> str:
        return (
            f'mate_jax.MultiAgentTracking({self.num_cameras} cameras, '
            f'{self.num_targets} targets, {self.num_obstacles} obstacles)'
        )

    # ----------------------------------------------------------------------------------------- #
    # Entity states
    # ----------------------------------------------------------------------------------------- #

    def _camera_states(self, state: EnvState, private: bool) -> jax.Array:
        cfg = self.config
        fdt = float_dtype()
        C = self.num_cameras
        parts = [
            state.camera_location,
            jnp.full((C, 1), cfg.camera_radius, fdt),
            polar2cartesian(state.camera_sight_range, state.camera_orientation),
            state.camera_viewing_angle[:, None],
        ]
        if private:
            parts.append(
                jnp.tile(
                    jnp.asarray(
                        [cfg.camera_max_sight_range, cfg.camera_rotation_step, cfg.camera_zooming_step],
                        dtype=fdt,
                    ),
                    (C, 1),
                )
            )
        return jnp.concatenate(parts, axis=-1).astype(fdt)

    def _target_states(self, state: EnvState, private: bool) -> jax.Array:
        fdt = float_dtype()
        T = self.num_targets
        is_loaded = jnp.any(state.target_goal_bits != 0, axis=-1)
        parts = [
            state.target_location,
            jnp.full((T, 1), self.config.target_sight_range, fdt),
            is_loaded[:, None].astype(fdt),
        ]
        if private:
            parts.extend(
                [
                    self.target_step_sizes(state)[:, None],
                    state.target_capacity[:, None].astype(fdt),
                    state.target_goal_bits.astype(fdt),
                    state.target_empty_bits.astype(fdt),
                ]
            )
        return jnp.concatenate(parts, axis=-1).astype(fdt)

    def _obstacle_states(self, state: EnvState) -> jax.Array:
        return jnp.concatenate([state.obstacle_location, state.obstacle_radius[:, None]], axis=-1)

    # ----------------------------------------------------------------------------------------- #
    # Reset helpers
    # ----------------------------------------------------------------------------------------- #

    def _place_entities(self, key, camera_slots, target_slots, obstacle_slots):
        """Rejection-sample non-overlapping entity locations (``reset()`` of the original env).

        The original places entities one by one (cameras, obstacles, then targets, in shuffled
        order), retrying each up to 500 times. Here ``num_reset_candidates`` candidates are drawn
        at once per entity and the first valid one is taken, which has the same distribution.
        Targets (radius 0) never block each other, so all targets are placed in parallel.
        """

        cfg = self.config
        fdt = float_dtype()
        C, T, O = self.num_cameras, self.num_targets, self.num_obstacles
        M = W + C + O  # entities that can block later placements
        K = self.num_reset_candidates
        k_camera, k_obstacle, k_target = jax.random.split(key, 3)
        terrain = jnp.asarray(consts.TERRAIN_HIGH, fdt)

        def sample_location(u, low, high, radius):
            location = low + u * (high - low)
            margin = 1.2 * radius[..., None]
            return jnp.clip(location, -terrain + margin, terrain - margin)

        def first_valid(overlap, placed):
            # overlap: (..., K, M) -> index of the first valid candidate (or the last one)
            valid = jnp.logical_not(jnp.any(overlap & placed, axis=-1))
            any_valid = jnp.any(valid, axis=-1)
            return jnp.where(any_valid, jnp.argmax(valid, axis=-1), K - 1), any_valid

        location = jnp.concatenate([jnp.asarray(consts.WAREHOUSES, fdt), jnp.zeros((C + O, 2), fdt)])
        radius = jnp.concatenate(
            [jnp.full(W, 0.75 * consts.WAREHOUSE_RADIUS, fdt), jnp.zeros(C + O, fdt)]
        )
        sight = jnp.zeros(M, fdt)
        is_camera = jnp.asarray(np.arange(M) >= W) & jnp.asarray(np.arange(M) < W + C)
        placed = jnp.arange(M) < W

        # Phase 1: cameras (sequential).
        camera_low = jnp.asarray(cfg.camera_location_low, fdt)[camera_slots]
        camera_high = jnp.asarray(cfg.camera_location_high, fdt)[camera_slots]
        camera_radius = jnp.full(K, cfg.camera_radius, fdt)
        num_orientations = max(1, int(360.0 / cfg.camera_rotation_step))  # randint(0, 360 / step)

        def place_camera(c, carry):
            location, radius, sight, placed, orientation, viewing = carry
            u = jax.random.uniform(jax.random.fold_in(k_camera, c), (K, 4), fdt)
            cand_location = sample_location(u[:, :2], camera_low[c], camera_high[c], camera_radius)
            cand_orientation = cfg.camera_rotation_step * jnp.minimum(
                jnp.floor(u[:, 2] * num_orientations), num_orientations - 1
            )
            cand_viewing = cfg.camera_min_viewing_angle + u[:, 3] * (
                consts.MAX_CAMERA_VIEWING_ANGLE - cfg.camera_min_viewing_angle
            )
            cand_sight = jnp.sqrt(self.camera_area_product / cand_viewing)
            distance = safe_norm(cand_location[:, None, :] - location[None, :, :])
            overlap = distance * (1.0 + 1e-6) < (
                cfg.camera_radius + radius[None, :] + cfg.target_step_size
            )
            overlap |= is_camera[None, :] & (
                distance < 0.1 * jnp.minimum(cand_sight[:, None], sight[None, :])
            )
            choice, _ = first_valid(overlap, placed[None, :])
            i = W + c
            return (
                location.at[i].set(cand_location[choice]),
                radius.at[i].set(cfg.camera_radius),
                sight.at[i].set(cand_sight[choice]),
                placed.at[i].set(True),
                orientation.at[c].set(normalize_angle(cand_orientation[choice])),
                viewing.at[c].set(cand_viewing[choice]),
            )

        carry = (location, radius, sight, placed, jnp.zeros(C, fdt), jnp.zeros(C, fdt))
        if C > 0:
            carry = jax.lax.fori_loop(0, C, place_camera, carry)
        location, radius, sight, placed, camera_orientation, camera_viewing_angle = carry

        # Phase 2: obstacles (sequential).
        obstacle_low = jnp.asarray(cfg.obstacle_location_low, fdt)[obstacle_slots]
        obstacle_high = jnp.asarray(cfg.obstacle_location_high, fdt)[obstacle_slots]

        def place_obstacle(o, carry):
            location, radius, placed = carry
            u = jax.random.uniform(jax.random.fold_in(k_obstacle, o), (K, 3), fdt)
            cand_radius = cfg.obstacle_radius_low + u[:, 2] * (
                cfg.obstacle_radius_high - cfg.obstacle_radius_low
            )
            cand_location = sample_location(u[:, :2], obstacle_low[o], obstacle_high[o], cand_radius)
            distance = safe_norm(cand_location[:, None, :] - location[None, :, :])
            overlap = distance * (1.0 + 1e-6) < (
                cand_radius[:, None] + radius[None, :] + cfg.target_step_size
            )
            choice, any_valid = first_valid(overlap, placed[None, :])
            i = W + C + o
            return (
                location.at[i].set(cand_location[choice]),
                radius.at[i].set(jnp.where(any_valid, cand_radius[choice], 0.0)),
                placed.at[i].set(True),
            )

        if O > 0:
            location, radius, placed = jax.lax.fori_loop(
                0, O, place_obstacle, (location, radius, placed)
            )

        # Phase 3: targets (parallel; they only have to avoid cameras and obstacles).
        target_low = jnp.asarray(cfg.target_location_low, fdt)[target_slots]
        target_high = jnp.asarray(cfg.target_location_high, fdt)[target_slots]
        u = jax.random.uniform(k_target, (T, K, 2), fdt)
        cand_location = sample_location(
            u, target_low[:, None, :], target_high[:, None, :], jnp.zeros((T, K), fdt)
        )
        distance = safe_norm(cand_location[:, :, None, :] - location[None, None, :, :])
        overlap = distance * (1.0 + 1e-6) < consts.TARGET_RADIUS + radius[None, None, :]
        choice, _ = first_valid(overlap, placed[None, None, :])
        target_location = jnp.take_along_axis(cand_location, choice[:, None, None], axis=1)[:, 0]

        return (
            location[W : W + C],
            camera_orientation,
            camera_viewing_angle,
            sight[W : W + C],
            location[W + C : M],
            radius[W + C : M],
            target_location,
        )

    def _generate_cargoes(self, key):
        """Random (sender, recipient) cargo pairs until every warehouse has something to send."""

        num_cargoes = self.config.num_cargoes_per_target * self.num_targets

        def body(carry):
            remaining, _, key = carry
            key, k_sender, k_offset = jax.random.split(key, 3)
            sender = jax.random.randint(k_sender, (num_cargoes,), 0, W)
            recipient = (sender + jax.random.randint(k_offset, (num_cargoes,), 1, W)) % W
            remaining = remaining.at[sender, recipient].add(1)
            return remaining, remaining.sum(axis=0), key

        def cond(carry):
            remaining, _, _ = carry
            return jnp.logical_not(jnp.all(jnp.any(remaining > 0, axis=-1)))

        init = (jnp.zeros((W, W), jnp.int32), jnp.zeros(W, jnp.int32), key)
        remaining, awaiting, _ = jax.lax.while_loop(cond, body, body(init))
        return remaining, awaiting

    def _load_initial_cargoes(self, key, state: EnvState) -> EnvState:
        """Assign cargoes to all unloaded targets (``targets_start_with_cargoes``)."""

        keys = jax.random.split(key, self.num_targets)

        def assign(remaining, inputs):
            k, goal, goal_bits, capacity, freight, bounty = inputs
            k_wh, k_goal = jax.random.split(k)
            unloaded = goal < 0
            has_cargo = jnp.any(remaining > 0, axis=-1)
            warehouse = jax.random.categorical(k_wh, jnp.where(has_cargo, 0.0, -jnp.inf))
            row = remaining[warehouse]
            new_goal = jax.random.categorical(k_goal, jnp.where(row > 0, 0.0, -jnp.inf))
            weight = jnp.minimum(capacity, row[new_goal])
            apply = jnp.logical_and(unloaded, jnp.any(has_cargo))
            weight = jnp.where(apply, weight, 0)
            remaining = remaining.at[warehouse, new_goal].add(-weight)
            goal_bits = jnp.where(apply, goal_bits.at[new_goal].set(weight), goal_bits)
            freight = jnp.where(apply, weight * int(self.freight_scale), freight)
            bounty = jnp.where(apply, weight * int(self.bounty_scale), bounty)
            goal = jnp.where(apply, new_goal, goal)
            return remaining, (goal, goal_bits, freight, bounty)

        remaining, (goals, goal_bits, freights, bounties) = jax.lax.scan(
            assign,
            state.remaining_cargoes,
            (
                keys,
                state.target_goals,
                state.target_goal_bits,
                state.target_capacity,
                state.freights,
                state.bounties,
            ),
            unroll=self.unroll_loops,
        )
        return state.replace(
            remaining_cargoes=remaining,
            target_goals=_ints(goals),
            target_goal_bits=_ints(goal_bits),
            freights=_ints(freights),
            bounties=_ints(bounties),
        )

    # ----------------------------------------------------------------------------------------- #
    # Dynamics
    # ----------------------------------------------------------------------------------------- #

    def _simulate(self, key, state: EnvState, camera_action, target_action) -> EnvState:
        cfg = self.config
        fdt = float_dtype()

        # Cameras: rotate and zoom.
        delta = jnp.clip(
            camera_action,
            jnp.asarray(self.camera_action_low, fdt),
            jnp.asarray(self.camera_action_high, fdt),
        )
        orientation = normalize_angle(state.camera_orientation + delta[:, 0])
        viewing = jnp.clip(
            state.camera_viewing_angle + delta[:, 1],
            cfg.camera_min_viewing_angle,
            consts.MAX_CAMERA_VIEWING_ANGLE,
        )
        sight = jnp.sqrt(self.camera_area_product / viewing)

        # Targets: move with collision against obstacles and cameras.
        step_size = self.target_step_sizes(state)
        norm = safe_norm(target_action)
        scale = jnp.where(norm > step_size, step_size / jnp.where(norm > 0.0, norm, 1.0), 1.0)
        step = target_action * scale[:, None]
        origin = state.target_location
        desired = origin + step

        centers = jnp.concatenate([state.obstacle_location, state.camera_location])
        radii = jnp.concatenate(
            [state.obstacle_radius, jnp.full(self.num_cameras, cfg.camera_radius, fdt)]
        )
        if centers.shape[0] > 0:

            def collide(step, obstacle):
                center, radius = obstacle
                return obstruct_step(step, origin, center, radius), None

            step, _ = jax.lax.scan(collide, step, (centers, radii), unroll=self.unroll_loops)

        terrain = jnp.asarray(consts.TERRAIN_HIGH, fdt)
        location = jnp.clip(origin + step, -terrain, terrain)
        is_colliding = jnp.any(jnp.abs(location - desired) > (1e-6 if fdt == jnp.float64 else 1e-3), axis=-1)
        moved = jnp.any(location != origin, axis=-1)
        displacement = location - origin
        target_orientation = jnp.where(
            moved, arctan2_deg(displacement[:, 1], displacement[:, 0]), state.target_orientation
        )

        state = state.replace(
            camera_orientation=orientation,
            camera_viewing_angle=viewing,
            camera_sight_range=sight,
            target_location=location,
            target_orientation=target_orientation,
            target_is_colliding=is_colliding,
        )
        return self._update_view(key, state)

    def _camera_perceive(self, state: EnvState, others, transmit=None):
        """Camera-centric perception (range, field of view and occlusion) of the given points."""

        cfg = self.config
        relative = others[None, :, :] - state.camera_location[:, None, :]  # (C, N, 2)
        distance = safe_norm(relative)
        in_range = distance <= state.camera_sight_range[:, None]
        angle = arctan2_deg(relative[..., 1], relative[..., 0])
        relative_angle = jnp.abs(state.camera_orientation[:, None] - angle)
        relative_angle = jnp.minimum(relative_angle, 360.0 - relative_angle)
        in_view = relative_angle * 2.0 <= state.camera_viewing_angle[:, None]

        if cfg.obstacle_transmittance < 1.0 and self.num_obstacles > 0:
            visible = line_of_sight(
                state.camera_location[:, None, :],
                others[None, :, :],
                distance,
                state.obstacle_location,
                state.obstacle_radius,
            )
        else:
            visible = jnp.ones_like(in_range)
        if transmit is not None:
            visible = jnp.logical_or(visible, transmit)
        return in_range & in_view & visible

    def _update_view(self, key, state: EnvState) -> EnvState:
        cfg = self.config
        C, T = self.num_cameras, self.num_targets

        if cfg.obstacle_transmittance > 0.0:
            transmit = jax.random.bernoulli(key, cfg.obstacle_transmittance, (C, T))
        else:
            transmit = None
        camera_target = self._camera_perceive(state, state.target_location, transmit)
        camera_camera = self._camera_perceive(state, state.camera_location)
        camera_camera = jnp.logical_or(camera_camera, jnp.eye(C, dtype=bool))

        sight = cfg.target_sight_range
        target_camera = (
            safe_norm(state.camera_location[None, :, :] - state.target_location[:, None, :])
            <= sight + cfg.camera_radius
        )
        target_obstacle = (
            safe_norm(state.obstacle_location[None, :, :] - state.target_location[:, None, :])
            <= sight + state.obstacle_radius[None, :]
        )
        target_target = (
            safe_norm(state.target_location[None, :, :] - state.target_location[:, None, :])
            <= sight + consts.TARGET_RADIUS
        )
        target_target = jnp.logical_or(target_target, jnp.eye(T, dtype=bool))

        return state.replace(
            camera_target_view_mask=camera_target,
            camera_camera_view_mask=camera_camera,
            target_camera_view_mask=target_camera,
            target_obstacle_view_mask=target_obstacle,
            target_target_view_mask=target_target,
            tracked_bits=jnp.any(camera_target, axis=0),
        )

    def _assign_goals(self, key, state: EnvState):
        """Cargo delivery and pickup (sequential over targets, as in the original)."""

        fdt = float_dtype()
        freight_scale, bounty_scale = int(self.freight_scale), int(self.bounty_scale)
        old_goals = state.target_goals
        tracked = state.tracked_bits.astype(jnp.int32)

        target_team_reward = -jnp.sum(
            jnp.logical_and(state.tracked_bits, state.bounties > 0).astype(fdt)
        )
        bounties = jnp.maximum(state.bounties - tracked, 0)

        directions = state.target_location[:, None, :] - jnp.asarray(consts.WAREHOUSES, fdt)[None]
        supremum = jnp.max(jnp.abs(directions), axis=-1)  # (T, W)
        in_warehouse = supremum <= consts.WAREHOUSE_RADIUS

        keys = jax.random.split(key, self.num_targets)

        def visit(carry, inputs):
            remaining, awaiting, delivered, reward, delayed = carry
            (k, inside_mask, goal, goal_bits, empty_bits, capacity, freight, bounty, t_steps, tr_steps) = inputs

            inside = jnp.any(inside_mask)
            warehouse = jnp.argmax(inside_mask)
            loaded = goal >= 0
            safe_goal = jnp.maximum(goal, 0)
            deliver = inside & loaded & (goal == warehouse)
            proceed = inside & (jnp.logical_not(loaded) | deliver)

            weight = goal_bits[safe_goal]
            total_bounty = weight * bounty_scale
            gain = (freight + bounty).astype(fdt)
            reward = reward + jnp.where(deliver, gain, 0.0)
            delayed = delayed + jnp.where(deliver, gain - (total_bounty - bounty).astype(fdt), 0.0)
            delivered = delivered + jnp.where(deliver, weight, 0)
            awaiting = awaiting.at[safe_goal].add(jnp.where(deliver, -weight, 0))

            freight = jnp.where(proceed, 0, freight)
            bounty = jnp.where(proceed, 0, bounty)
            t_steps = jnp.where(proceed, 0, t_steps)
            tr_steps = jnp.where(proceed, 0, tr_steps)
            goal_bits = jnp.where(proceed, jnp.zeros_like(goal_bits), goal_bits)
            goal = jnp.where(proceed, -1, goal)

            row = remaining[warehouse]
            pick = proceed & jnp.any(row > 0)
            new_goal = jax.random.categorical(k, jnp.where(row > 0, 0.0, -jnp.inf))
            new_weight = jnp.where(pick, jnp.minimum(capacity, row[new_goal]), 0)
            remaining = remaining.at[warehouse, new_goal].add(-new_weight)
            goal_bits = jnp.where(pick, goal_bits.at[new_goal].set(new_weight), goal_bits)
            freight = jnp.where(pick, new_weight * freight_scale, freight)
            bounty = jnp.where(pick, new_weight * bounty_scale, bounty)
            goal = jnp.where(pick, new_goal, goal)

            empty_bits = jnp.where(
                inside, empty_bits.at[warehouse].set(jnp.logical_not(jnp.any(remaining[warehouse] > 0))), empty_bits
            )
            outputs = (goal, goal_bits, empty_bits, freight, bounty, t_steps, tr_steps)
            return (remaining, awaiting, delivered, reward, delayed), outputs

        carry, outputs = jax.lax.scan(
            visit,
            (
                state.remaining_cargoes,
                state.awaiting_cargo_counts,
                state.num_delivered_cargoes,
                target_team_reward,
                jnp.zeros((), fdt),
            ),
            (
                keys,
                in_warehouse,
                state.target_goals,
                state.target_goal_bits,
                state.target_empty_bits,
                state.target_capacity,
                state.freights,
                bounties,
                state.target_steps,
                state.tracked_steps,
            ),
            unroll=self.unroll_loops,
        )
        remaining, awaiting, delivered, target_team_reward, delayed_target_team_reward = carry
        goals, goal_bits, empty_bits, freights, bounties, target_steps, tracked_steps = outputs

        state = state.replace(
            remaining_cargoes=_ints(remaining),
            awaiting_cargo_counts=_ints(awaiting),
            num_delivered_cargoes=_ints(delivered),
            target_goals=_ints(goals),
            target_goal_bits=_ints(goal_bits),
            target_empty_bits=empty_bits,
            freights=_ints(freights),
            bounties=_ints(bounties),
            target_steps=_ints(target_steps),
            tracked_steps=_ints(tracked_steps),
            target_dones=jnp.logical_and(goals != old_goals, old_goals >= 0),
        )
        return state, target_team_reward, delayed_target_team_reward


def make(config=None, **kwargs) -> MultiAgentTracking:
    """Create a JAX Multi-Agent Tracking Environment (same config files as ``mate.make``)."""

    return MultiAgentTracking(config, **kwargs)
