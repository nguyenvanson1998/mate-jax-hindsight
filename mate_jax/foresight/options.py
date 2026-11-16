"""Discrete camera pose options, their reach times, and fast coverage scoring.

A pose option is ``(orientation, viewing angle)`` on a 5-degree orientation grid and a few zoom
levels (the sight range follows from the constant sector area). Its *reach time* is the number of
steps the rate-limited camera needs to get there, which is what makes forecasts useful.

Scoring every option against every cell would cost ``O(options x cells)``. Instead, the weighted
cells of each camera are binned by bearing (2.5-degree bins) once per zoom level, and the mass
inside each sector is read off a circular prefix sum: ``O(cells + options)`` per camera.
"""

import dataclasses
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from mate_jax import constants as consts
from mate_jax.geometry import normalize_angle


__all__ = ['PoseOptions']

BIN_WIDTH = 2.5
NUM_BINS = int(360 / BIN_WIDTH)
ORIENTATION_STEP = 5.0
NUM_ORIENTATIONS = int(360 / ORIENTATION_STEP)


@dataclasses.dataclass(frozen=True)
class PoseOptions:
    rotation_step: float
    zooming_step: float
    min_viewing_angle: float
    area_product: float  # viewing_angle * sight_range ** 2
    levels: Sequence[float] = (30.0, 45.0, 60.0, 90.0, 120.0, 180.0)

    @property
    def viewing_angles(self) -> np.ndarray:
        levels = np.clip(np.asarray(self.levels, np.float64), self.min_viewing_angle, consts.MAX_CAMERA_VIEWING_ANGLE)
        levels = np.unique(np.round(levels / ORIENTATION_STEP) * ORIENTATION_STEP)  # sector edges on bins
        return np.maximum(levels, ORIENTATION_STEP)

    @property
    def sight_ranges(self) -> np.ndarray:
        return np.sqrt(self.area_product / self.viewing_angles)

    @property
    def orientations(self) -> np.ndarray:
        return -180.0 + ORIENTATION_STEP * np.arange(NUM_ORIENTATIONS)

    @property
    def num_levels(self) -> int:
        return int(self.viewing_angles.size)

    def _window(self):
        half_bins = (self.viewing_angles / 2.0 / BIN_WIDTH).astype(np.int64)  # (L,)
        start = (np.arange(NUM_ORIENTATIONS) * ORIENTATION_STEP / BIN_WIDTH).astype(np.int64)  # (O,)
        lo = NUM_BINS + start[None, :] - half_bins[:, None]
        hi = NUM_BINS + start[None, :] + half_bins[:, None]
        return lo, hi  # (L, O) indices into the prefix sum of the tripled histogram

    def histograms(self, weights, distance, angle, visible, method='auto'):
        """Bearing histograms of visible weighted mass for every zoom level.

        Args:
            weights: (H, N) expected target counts per cell (for H horizons).
            distance, angle, visible: (N, S) sample geometry of one camera.
            method: ``'scatter'`` (scatter-add, best on CPU), ``'matmul'`` (one-hot matrix product:
                thousands of samples fall into 144 bins, which serializes atomic scatter-adds on
                GPUs, while a dense product runs on the tensor cores / MXU), or ``'auto'``.

        Returns:
            (H, L, NUM_BINS) histograms.
        """

        num_samples = distance.shape[-1]
        bins = jnp.floor(jnp.mod(angle + 180.0, 360.0) / BIN_WIDTH).astype(jnp.int32)
        bins = jnp.clip(bins, 0, NUM_BINS - 1).reshape(-1)
        radii = jnp.asarray(self.sight_ranges, jnp.float32)
        within = (visible[None] & (distance[None] <= radii[:, None, None])).astype(jnp.float32)  # (L, N, S)
        contribution = weights[:, None, :, None] * within[None] / num_samples  # (H, L, N, S)
        contribution = contribution.reshape(contribution.shape[:2] + (-1,))
        if method == 'auto':
            method = 'scatter' if jax.default_backend() == 'cpu' else 'matmul'
        if method == 'matmul':
            one_hot = jax.nn.one_hot(bins, NUM_BINS, dtype=jnp.float32)  # (N * S, NUM_BINS)
            return jnp.einsum('hlk,kb->hlb', contribution, one_hot, precision=jax.lax.Precision.HIGHEST)
        histogram = jnp.zeros(contribution.shape[:2] + (NUM_BINS,), jnp.float32)
        return histogram.at[..., bins].add(contribution)

    def sector_mass(self, histograms):
        """(H, L, NUM_BINS) histograms -> (H, L, O) mass inside every pose's sector."""

        tripled = jnp.concatenate([histograms] * 3, axis=-1)
        prefix = jnp.concatenate([jnp.zeros(tripled.shape[:-1] + (1,)), jnp.cumsum(tripled, axis=-1)], axis=-1)
        lo, hi = self._window()
        level = np.arange(self.num_levels)[:, None]
        return prefix[..., level, hi] - prefix[..., level, lo]

    def reach_steps(self, orientation, viewing_angle):
        """(L, O) steps needed to reach every pose from the current one."""

        turn = jnp.abs(normalize_angle(jnp.asarray(self.orientations)[None, :] - orientation)) / self.rotation_step
        zoom = jnp.abs(jnp.asarray(self.viewing_angles)[:, None] - viewing_angle) / self.zooming_step
        return jnp.ceil(jnp.maximum(turn, zoom) - 1e-6).astype(jnp.int32)

    def path_indices(self, orientation, viewing_angle, steps):
        """Option indices visited along the rate-limited straight path to every option.

        Args:
            orientation, viewing_angle: the current pose.
            steps: (H,) time offsets.

        Returns:
            (H, L, O) level and orientation indices of the pose ``steps`` after starting to move
            towards each option (the goal itself once it is reached).
        """

        goal_orientation = jnp.asarray(self.orientations)[None, :]  # (1, O)
        goal_viewing = jnp.asarray(self.viewing_angles)[:, None]  # (L, 1)
        steps = jnp.asarray(steps, jnp.float32)[:, None, None]
        turn = normalize_angle(goal_orientation - orientation)[None]
        theta = orientation + jnp.sign(turn) * jnp.minimum(steps * self.rotation_step, jnp.abs(turn))
        orient_index = jnp.mod(jnp.round((theta + 180.0) / ORIENTATION_STEP), NUM_ORIENTATIONS).astype(jnp.int32)
        zoom = (goal_viewing - viewing_angle)[None]
        alpha = viewing_angle + jnp.sign(zoom) * jnp.minimum(steps * self.zooming_step, jnp.abs(zoom))
        levels = jnp.asarray(self.viewing_angles)
        level_index = jnp.argmin(jnp.abs(alpha[..., None] - levels), axis=-1).astype(jnp.int32)
        shape = (steps.shape[0], self.num_levels, NUM_ORIENTATIONS)
        return jnp.broadcast_to(level_index, shape), jnp.broadcast_to(orient_index, shape)

    def trajectory(self, mass, orientation, viewing_angle, stride):
        """Best rate-limited pose trajectory over the horizons (exact dynamic programming, Viterbi).

        Between two horizons (``stride`` steps apart) the camera can turn by ``stride`` rotation steps
        and move between zoom levels whose viewing angles differ by at most ``stride`` zoom steps; the
        trajectory starts at the option nearest to the current pose.

        Args:
            mass: (H, L, O) value of every option at every horizon (already discounted).

        Returns:
            (H,) level and orientation indices of the trajectory and its total value.
        """

        num_levels = self.num_levels
        turn = int(np.floor(stride * self.rotation_step / ORIENTATION_STEP + 1e-6))
        shifts = np.arange(-turn, turn + 1)
        views = self.viewing_angles
        allowed = np.abs(views[:, None] - views[None, :]) <= stride * self.zooming_step + 1e-6  # (from, to)
        neg = jnp.float32(-1e9)

        level0 = jnp.argmin(jnp.abs(jnp.asarray(views) - viewing_angle))
        orient0 = jnp.mod(jnp.round((orientation + 180.0) / ORIENTATION_STEP), NUM_ORIENTATIONS).astype(jnp.int32)
        start = jnp.full((num_levels, NUM_ORIENTATIONS), neg).at[level0, orient0].set(0.0) + mass[0]

        def forward(value, step_mass):
            turned = jnp.stack([jnp.roll(value, int(d), axis=1) for d in shifts])  # [d][l, o] = value[l, o - d]
            by_turn, turn_index = turned.max(0), turned.argmax(0)  # (L, O)
            candidates = jnp.where(allowed[:, :, None], by_turn[:, None, :], neg)  # (from, to, O)
            best, from_level = candidates.max(0), candidates.argmax(0)  # (L, O)
            return best + step_mass, (from_level, turn_index)

        final, (from_levels, turn_indices) = jax.lax.scan(forward, start, mass[1:])
        index = jnp.argmax(final.reshape(-1))
        last = (index // NUM_ORIENTATIONS, index % NUM_ORIENTATIONS)

        def backward(pose, pointers):
            from_level, turn_index = pointers
            level, orient = pose
            previous_level = from_level[level, orient]
            shift = jnp.asarray(shifts)[turn_index[previous_level, orient]]
            previous = (previous_level, jnp.mod(orient - shift, NUM_ORIENTATIONS))
            return previous, pose

        first, (levels, orients) = jax.lax.scan(backward, last, (from_levels, turn_indices), reverse=True)
        levels = jnp.concatenate([first[0][None], levels]).astype(jnp.int32)
        orients = jnp.concatenate([first[1][None], orients]).astype(jnp.int32)
        return levels, orients, final.reshape(-1)[index]

    def pose(self, level, orientation_index):
        """(orientation, viewing angle, sight range) of an option."""

        return (
            jnp.asarray(self.orientations)[orientation_index],
            jnp.asarray(self.viewing_angles)[level],
            jnp.asarray(self.sight_ranges)[level],
        )

    def action_towards(self, orientation, viewing_angle, goal_orientation, goal_viewing_angle):
        """Rate-limited (rotation, zoom) action that moves the camera towards a goal pose."""

        rotate = jnp.clip(normalize_angle(goal_orientation - orientation), -self.rotation_step, self.rotation_step)
        zoom = jnp.clip(goal_viewing_angle - viewing_angle, -self.zooming_step, self.zooming_step)
        return jnp.stack([rotate, zoom], axis=-1)
