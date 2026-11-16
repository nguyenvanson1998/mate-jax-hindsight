"""Static per-episode visibility atlas of each camera over the belief grid.

Cameras never translate and obstacles are static, so the set of cells a camera can ever see (line
of sight and within its maximum sight range) is computed once per episode. Each cell is sampled
at ``samples_per_axis ** 2`` points so that partially visible cells get fractional coverage.
"""

import jax.numpy as jnp
import numpy as np
from flax import struct

from mate_jax.foresight.grid import Grid
from mate_jax.geometry import arctan2_deg, line_of_sight, normalize_angle, safe_norm


__all__ = ['Atlas', 'build_atlas', 'free_space', 'sample_points']


def sample_points(grid: Grid, samples_per_axis: int) -> np.ndarray:
    """(G*G, S, 2) sample points inside every cell."""

    offsets = (np.arange(samples_per_axis) + 0.5) / samples_per_axis - 0.5
    dx, dy = np.meshgrid(offsets, offsets, indexing='ij')
    offsets = np.stack([dx.ravel(), dy.ravel()], axis=-1) * grid.cell  # (S, 2)
    return grid.cell_xy[:, None, :] + offsets[None, :, :]


@struct.dataclass
class Atlas:
    distance: jnp.ndarray  # (C, N, S) camera-to-sample distances
    angle: jnp.ndarray  # (C, N, S) bearing of every sample from every camera (degrees)
    visible: jnp.ndarray  # (C, N, S) line of sight and within the maximum sight range

    def field_of_view(self, orientation, viewing_angle, sight_range):
        """(C, N) fraction of each cell inside each camera's current field of view."""

        inside = (
            self.visible
            & (self.distance <= sight_range[:, None, None])
            & (2.0 * jnp.abs(normalize_angle(self.angle - orientation[:, None, None])) <= viewing_angle[:, None, None])
        )
        return inside.astype(jnp.float32).mean(-1)

    def coverage_of_pose(self, camera, orientation, viewing_angle, sight_range):
        """(N,) fraction of each cell covered by camera ``camera`` at the given pose."""

        inside = (
            self.visible[camera]
            & (self.distance[camera] <= sight_range)
            & (2.0 * jnp.abs(normalize_angle(self.angle[camera] - orientation)) <= viewing_angle)
        )
        return inside.astype(jnp.float32).mean(-1)


def free_space(grid: Grid, obstacle_location, obstacle_radius, camera_location, camera_radius):
    """(G, G) mask of cells whose center lies outside all obstacles and camera bodies."""

    xy = jnp.asarray(grid.cell_xy, jnp.float32)
    centers = jnp.concatenate([obstacle_location, camera_location])
    radii = jnp.concatenate([obstacle_radius, jnp.full(camera_location.shape[0], camera_radius)])
    if centers.shape[0] == 0:
        return jnp.ones((grid.size, grid.size), jnp.float32)
    blocked = jnp.any(safe_norm(xy[:, None, :] - centers[None]) < radii[None], axis=-1)
    return jnp.logical_not(blocked).astype(jnp.float32).reshape(grid.size, grid.size)


def build_atlas(grid: Grid, camera_location, max_sight_range, obstacle_location, obstacle_radius, samples_per_axis=2):
    points = jnp.asarray(sample_points(grid, samples_per_axis), jnp.float32)  # (N, S, 2)
    num_cells, num_samples = points.shape[:2]
    flat = points.reshape(-1, 2)
    relative = flat[None] - camera_location[:, None]  # (C, N*S, 2)
    distance = safe_norm(relative)
    angle = arctan2_deg(relative[..., 1], relative[..., 0])
    visible = distance <= max_sight_range[:, None]
    if obstacle_location.shape[0] > 0:
        visible &= line_of_sight(
            camera_location[:, None], flat[None], distance, obstacle_location, obstacle_radius
        )
    shape = (camera_location.shape[0], num_cells, num_samples)
    return Atlas(distance=distance.reshape(shape), angle=angle.reshape(shape), visible=visible.reshape(shape))
