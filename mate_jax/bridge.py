"""Conversion between the original NumPy environment (:mod:`mate`) and :class:`EnvState`.

Used for parity tests, and for rendering JAX rollouts with the original pyglet viewer::

    np_env = mate.MultiAgentTracking('MATE-4v8-9.yaml'); np_env.reset()
    sync_numpy_env(np_env, jax_state)   # copy one (unbatched) JAX state into the NumPy env
    np_env.render()
"""

import numpy as np
import jax.numpy as jnp

from mate_jax.env import EnvState, float_dtype


__all__ = ['state_from_numpy_env', 'sync_numpy_env']


def _index_in(ordered, entities):
    return np.asarray([ordered.index(entity) for entity in entities], dtype=np.int32)


def state_from_numpy_env(env, np_env) -> EnvState:
    """Build an :class:`EnvState` from a ``mate.MultiAgentTracking`` instance (after reset)."""

    np_env = np_env.unwrapped if hasattr(np_env, 'unwrapped') else np_env
    fdt = float_dtype()
    C, T, O = env.num_cameras, env.num_targets, env.num_obstacles

    def f(array, shape):
        return jnp.asarray(np.asarray(array, dtype=np.float64).reshape(shape), dtype=fdt)

    def i(array, shape):
        return jnp.asarray(np.asarray(array).reshape(shape), dtype=jnp.int32)

    def b(array, shape):
        return jnp.asarray(np.asarray(array, dtype=bool).reshape(shape))

    cameras, targets, obstacles = np_env.cameras, np_env.targets, np_env.obstacles
    return EnvState(
        camera_location=f([c.location for c in cameras], (C, 2)),
        camera_orientation=f([c.orientation for c in cameras], (C,)),
        camera_viewing_angle=f([c.viewing_angle for c in cameras], (C,)),
        camera_sight_range=f([c.sight_range for c in cameras], (C,)),
        target_location=f([t.location for t in targets], (T, 2)),
        target_orientation=f(np_env.target_orientations, (T,)),
        target_capacity=i(np_env.target_capacities, (T,)),
        target_goal_bits=i(np_env.target_goal_bits, (T, 4)),
        target_empty_bits=b([t.empty_bits for t in targets], (T, 4)),
        target_goals=i(np_env.target_goals, (T,)),
        target_dones=b(np_env.target_dones, (T,)),
        target_is_colliding=b([t.is_colliding for t in targets], (T,)),
        target_steps=i(np_env.target_steps, (T,)),
        tracked_steps=i(np_env.tracked_steps, (T,)),
        freights=i(np_env.freights, (T,)),
        bounties=i(np_env.bounties, (T,)),
        obstacle_location=f([o.location for o in obstacles], (O, 2)),
        obstacle_radius=f([o.radius for o in obstacles], (O,)),
        camera_target_view_mask=b(np_env.camera_target_view_mask, (C, T)),
        camera_obstacle_view_mask=b(np_env.camera_obstacle_view_mask, (C, O)),
        camera_camera_view_mask=b(np_env.camera_camera_view_mask, (C, C)),
        target_camera_view_mask=b(np_env.target_camera_view_mask, (T, C)),
        target_obstacle_view_mask=b(np_env.target_obstacle_view_mask, (T, O)),
        target_target_view_mask=b(np_env.target_target_view_mask, (T, T)),
        tracked_bits=b(np_env.tracked_bits, (T,)),
        remaining_cargoes=i(np_env.remaining_cargoes, (4, 4)),
        awaiting_cargo_counts=i(np_env.awaiting_cargo_counts, (4,)),
        num_delivered_cargoes=jnp.asarray(np_env.num_delivered_cargoes, dtype=jnp.int32),
        target_team_episode_reward=jnp.asarray(np_env.target_team_episode_reward, dtype=fdt),
        delayed_target_team_episode_reward=jnp.asarray(
            np_env.delayed_target_team_episode_reward, dtype=fdt
        ),
        episode_step=jnp.asarray(np_env.episode_step, dtype=jnp.int32),
        camera_slots=jnp.asarray(_index_in(np_env.cameras_ordered, cameras)),
        target_slots=jnp.asarray(_index_in(np_env.targets_ordered, targets)),
        obstacle_slots=jnp.asarray(_index_in(np_env.obstacles_ordered, obstacles)),
    )


def sync_numpy_env(np_env, state: EnvState) -> None:
    """Copy an (unbatched) :class:`EnvState` into a ``mate.MultiAgentTracking`` (e.g. to render it)."""

    # pylint: disable-next=import-outside-toplevel
    from mate.entities import Target
    from mate.utils import Vector2D  # pylint: disable=import-outside-toplevel

    np_env = np_env.unwrapped if hasattr(np_env, 'unwrapped') else np_env
    state = type(state)(**{k: np.asarray(v) for k, v in vars(state).items()})

    np_env.cameras = [np_env.cameras_ordered[s] for s in state.camera_slots]
    np_env.targets = [np_env.targets_ordered[s] for s in state.target_slots]
    np_env.obstacles = [np_env.obstacles_ordered[s] for s in state.obstacle_slots]

    for o, obstacle in enumerate(np_env.obstacles):
        obstacle.location = state.obstacle_location[o].astype(np.float64)
        obstacle.radius = float(state.obstacle_radius[o])
    for c, camera in enumerate(np_env.cameras):
        camera.location = state.camera_location[c].astype(np.float64)
        camera.orientation = float(state.camera_orientation[c])
        camera.viewing_angle = float(state.camera_viewing_angle[c])
        camera.sight_range = float(state.camera_sight_range[c])
    for t, target in enumerate(np_env.targets):
        target.location = state.target_location[t].astype(np.float64)
        target.capacity = int(state.target_capacity[t])
        target.goal_bits[:] = state.target_goal_bits[t]
        target.empty_bits[:] = state.target_empty_bits[t]

    # Rebuild the camera sight boundaries (as in `Camera.reset()`) and the collision hashmap.
    angles = np.linspace(-180.0, +180.0, num=360, endpoint=False)
    for camera in np_env.cameras:
        camera.boundary = [
            Vector2D(norm=camera.max_sight_range, angle=a, origin=camera.location) for a in angles
        ]
        camera.boundary_outer = [
            Vector2D(norm=camera.max_sight_range, angle=a, origin=camera.location) for a in angles
        ]
        camera.clear_obstacles()
        camera.add_obstacles(*np_env.obstacles)
    Target.clear_obstacles()
    Target.add_obstacles(*np_env.obstacles, *np_env.cameras)

    np_env.target_capacities = state.target_capacity.astype(np.int64)
    np_env.target_goal_bits = state.target_goal_bits.astype(np.int64)
    np_env.target_goals = state.target_goals.astype(np.int64)
    np_env.target_dones = state.target_dones.astype(bool)
    np_env.target_steps = state.target_steps.astype(np.int64)
    np_env.tracked_steps = state.tracked_steps.astype(np.int64)
    np_env.freights = state.freights.astype(np.int64)
    np_env.bounties = state.bounties.astype(np.int64)
    np_env.target_orientations = state.target_orientation.astype(np.float64)
    np_env.camera_target_view_mask = state.camera_target_view_mask.astype(bool)
    np_env.camera_obstacle_view_mask = state.camera_obstacle_view_mask.astype(bool)
    np_env.camera_camera_view_mask = state.camera_camera_view_mask.astype(bool)
    np_env.target_camera_view_mask = state.target_camera_view_mask.astype(bool)
    np_env.target_obstacle_view_mask = state.target_obstacle_view_mask.astype(bool)
    np_env.target_target_view_mask = state.target_target_view_mask.astype(bool)
    np_env.tracked_bits = state.tracked_bits.astype(bool)
    np_env.remaining_cargoes = state.remaining_cargoes.astype(np.int64)
    np_env.awaiting_cargo_counts = state.awaiting_cargo_counts.astype(np.int64)
    np_env.num_delivered_cargoes = int(state.num_delivered_cargoes)
    np_env.episode_step = int(state.episode_step)
    if np_env.num_obstacles > 0:
        np_env.obstacle_states = np.vstack([o.state() for o in np_env.obstacles])
        np_env.obstacle_states_flagged = np.hstack(
            [np_env.obstacle_states, np.ones((np_env.num_obstacles, 1))]
        )
        np_env.camera_obstacle_observations = np.vstack(
            [
                np.where(
                    np_env.camera_obstacle_view_mask[c, :, None], np_env.obstacle_states_flagged, 0.0
                ).ravel()
                for c in range(np_env.num_cameras)
            ]
        ) if np_env.num_cameras > 0 else np_env.camera_obstacle_observations
    np_env._state = None  # pylint: disable=protected-access
    if np_env.viewer is not None:
        np_env.viewer.geoms.clear()  # rebuild static geometries (obstacle radii may differ)
