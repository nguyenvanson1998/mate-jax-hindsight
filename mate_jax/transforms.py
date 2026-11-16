"""Observation transforms in JAX (``mate.RelativeCoordinates`` / ``mate.RescaledObservation``)."""

import functools

import jax.numpy as jnp
import numpy as np

from mate_jax import constants as consts


__all__ = ['ObservationTransform']


@functools.lru_cache(maxsize=None)
def _relative_tables(team: str, num_cameras: int, num_targets: int, num_obstacles: int):
    """Static index tables: for each feature, which axis of the origin it is relative to."""

    C, T, O = num_cameras, num_targets, num_obstacles
    if team == 'camera':
        mask = consts.camera_coordinate_mask(C, T, O)
        blocks = [
            (consts.TARGET_STATE_DIM_PUBLIC + 1, T),
            (consts.OBSTACLE_STATE_DIM + 1, O),
            (consts.CAMERA_STATE_DIM_PUBLIC + 1, C),
        ]
        self_dim = consts.CAMERA_STATE_DIM_PRIVATE
    else:
        mask = consts.target_coordinate_mask(C, T, O)
        blocks = [
            (consts.CAMERA_STATE_DIM_PUBLIC + 1, C),
            (consts.OBSTACLE_STATE_DIM + 1, O),
            (consts.TARGET_STATE_DIM_PUBLIC + 1, T),
        ]
        self_dim = consts.TARGET_STATE_DIM_PRIVATE

    dim = mask.shape[0]
    axis = np.full(dim, -1, dtype=np.int32)
    flag = np.full(dim, -1, dtype=np.int32)  # index of the view flag (-1: always converted)
    coordinates = np.flatnonzero(mask[: consts.PRESERVED_DIM + self_dim])
    axis[coordinates] = np.arange(coordinates.size) % 2
    offset = consts.PRESERVED_DIM + self_dim
    for size, count in blocks:
        for n in range(count):
            start = offset + n * size
            axis[start : start + 2] = [0, 1]
            flag[start : start + 2] = start + size - 1
        offset += size * count
    assert offset == dim
    return axis, flag


class ObservationTransform:
    """Composable observation preprocessing for one team.

    Args:
        env: a :class:`mate_jax.MultiAgentTracking` instance.
        team: ``'camera'`` or ``'target'``.
        relative_coordinates: convert other entity locations to the agent's own frame.
        rescale: rescale all features to roughly ``[-1, +1]`` with the observation-space bounds.
    """

    def __init__(self, env, team: str, relative_coordinates: bool = True, rescale: bool = True):
        assert team in ('camera', 'target')
        self.team = team
        self.relative_coordinates = relative_coordinates
        self.rescale = rescale
        counts = (env.num_cameras, env.num_targets, env.num_obstacles)

        if team == 'camera':
            low, high = consts.camera_observation_bounds(*counts)
        else:
            low, high = consts.target_observation_bounds(*counts)
        self.dim = low.shape[0]
        self.axis, self.flag = _relative_tables(team, *counts)

        # `mate.RescaledObservation` uses the same bounds with or without relative coordinates.
        bounded_below = np.isfinite(low)
        bounded_both = bounded_below & np.isfinite(high)
        scaled = bounded_both & (high > low)
        self.shift = np.where(bounded_below, low, 0.0)
        self.scale = np.where(scaled, 2.0 / np.where(scaled, high - low, 1.0), 1.0)
        self.offset = np.where(scaled, -1.0, 0.0)

    def __call__(self, observation):
        observation = jnp.asarray(observation)
        if self.relative_coordinates:
            origin = observation[..., consts.PRESERVED_DIM : consts.PRESERVED_DIM + 2]
            axis = jnp.asarray(np.maximum(self.axis, 0))
            delta = jnp.take(origin, axis, axis=-1)
            visible = jnp.where(
                jnp.asarray(self.flag >= 0),
                jnp.take(observation, jnp.asarray(np.maximum(self.flag, 0)), axis=-1) > 0.5,
                True,
            )
            convert = jnp.asarray(self.axis >= 0) & visible
            observation = observation - jnp.where(convert, delta, 0.0)
        if self.rescale:
            shift = jnp.asarray(self.shift, observation.dtype)
            scale = jnp.asarray(self.scale, observation.dtype)
            offset = jnp.asarray(self.offset, observation.dtype)
            observation = (observation - shift) * scale + offset
        return observation
