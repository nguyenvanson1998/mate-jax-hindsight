"""Occupancy grid over the terrain and separable Gaussian transition operators."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtr

from mate_jax import constants as consts


__all__ = ['Grid']


@dataclasses.dataclass(frozen=True)
class Grid:
    """A ``size x size`` grid over ``[-1000, 1000]^2``; cell ``(ix, iy)`` is flattened to ``ix * size + iy``."""

    size: int = 32

    @property
    def cell(self) -> float:
        return consts.TERRAIN_WIDTH / self.size

    @property
    def centers(self) -> np.ndarray:
        return -consts.TERRAIN_SIZE + (np.arange(self.size) + 0.5) * self.cell

    @property
    def cell_xy(self) -> np.ndarray:
        """(size * size, 2) cell centers in flattened order."""

        x, y = np.meshgrid(self.centers, self.centers, indexing='ij')
        return np.stack([x.ravel(), y.ravel()], axis=-1)

    def _cell_probabilities(self, offset, std):
        """P(cell) for a variable uniform within one cell of width d, plus N(0, std^2) noise.

        ``offset`` is ``destination center - source mean``. This is the second difference of the
        antiderivative of the Gaussian CDF; ``std -> 0`` gives the exact linear split between the
        two overlapping cells (sub-cell shifts are preserved, no numerical diffusion).
        """

        d = self.cell
        std = jnp.maximum(std, 1e-6 * d)

        def tail(z):  # F(z) - relu(z), F = antiderivative of Phi(z / std)
            u = jnp.abs(z) / std
            return std * jnp.exp(-0.5 * jnp.square(u)) / jnp.sqrt(2.0 * jnp.pi) - jnp.abs(z) * ndtr(-u)

        hat = jnp.maximum(d - jnp.abs(offset), 0.0)  # second difference of relu
        p = (hat + tail(offset + d) - 2.0 * tail(offset) + tail(offset - d)) / d
        return jnp.maximum(p, 0.0)

    def gaussian_1d(self, mean, std):
        """(..., size) probabilities over the cells for a position ``mean`` with noise ``std``."""

        centers = jnp.asarray(self.centers, jnp.float32)
        p = self._cell_probabilities(centers - mean[..., None], jnp.asarray(std)[..., None])
        return p / jnp.maximum(p.sum(-1, keepdims=True), 1e-12)

    def point_mass(self, position, std):
        """(..., size, size) belief concentrated around ``position`` (..., 2)."""

        gx = self.gaussian_1d(position[..., 0], std)
        gy = self.gaussian_1d(position[..., 1], std)
        return gx[..., :, None] * gy[..., None, :]

    def transition(self, shift, std):
        """Separable transition matrices ``A[..., dest, src]`` for a mean shift (..., 2) and std (...,).

        Columns are renormalized: mass that would leave the terrain is redistributed over the
        cells inside it (targets are clipped to the terrain).
        """

        # The kernel only depends on dest - src (a Toeplitz matrix): evaluate it for the 2G - 1
        # possible offsets and gather, instead of G * G evaluations of the Gaussian CDF.
        size = self.size
        lags = jnp.asarray((np.arange(2 * size - 1) - (size - 1)) * self.cell, jnp.float32)  # (2G-1,)
        index = jnp.asarray(np.arange(size)[:, None] - np.arange(size)[None, :] + size - 1)  # (dest, src)
        std = jnp.asarray(std)[..., None]

        def axis(delta):
            kernel = self._cell_probabilities(lags - delta[..., None], std)  # (..., 2G-1)
            p = jnp.take(kernel, index, axis=-1)  # (..., dest, src)
            return p / jnp.maximum(p.sum(-2, keepdims=True), 1e-12)

        return axis(shift[..., 0]), axis(shift[..., 1])

    @staticmethod
    def apply(belief, transition):
        """Apply separable transition matrices to beliefs (..., size, size)."""

        ax, ay = transition
        # Full float32 precision (GPU defaults to TF32 and TPU to bfloat16 for matmuls).
        return jnp.einsum('...ik,...kl,...ml->...im', ax, belief, ay, precision=jax.lax.Precision.HIGHEST)

    def uniform_boxes(self, low, high) -> np.ndarray:
        """(size, size) uniform prior over the union of axis-aligned boxes (N, 2)."""

        xy = self.cell_xy
        half = self.cell / 2.0
        inside = np.zeros(xy.shape[0], dtype=bool)
        for lo, hi in zip(np.asarray(low), np.asarray(high)):
            inside |= np.all((xy >= lo - half) & (xy <= hi + half), axis=-1)
        if not inside.any():
            inside[:] = True
        prior = inside.astype(np.float32)
        return (prior / prior.sum()).reshape(self.size, self.size)
