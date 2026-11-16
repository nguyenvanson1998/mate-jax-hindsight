"""Vectorized geometry kernels (angles in degrees, same conventions as :mod:`mate.utils`)."""

import jax.numpy as jnp


__all__ = [
    'normalize_angle',
    'arctan2_deg',
    'sin_deg',
    'cos_deg',
    'polar2cartesian',
    'safe_norm',
    'obstruct_step',
    'line_of_sight',
]


def normalize_angle(angle):
    """Normalize an angle in degrees to :math:`[-180, +180)`."""

    return jnp.mod(angle + 180.0, 360.0) - 180.0


def arctan2_deg(y, x):
    """Element-wise arc tangent of y/x **in degrees**."""

    return jnp.rad2deg(jnp.arctan2(y, x))


def sin_deg(x):
    return jnp.sin(jnp.deg2rad(x))


def cos_deg(x):
    return jnp.cos(jnp.deg2rad(x))


def polar2cartesian(rho, phi):
    """Convert polar coordinates (phi in degrees) to cartesian, stacked on the last axis."""

    phi = jnp.deg2rad(phi)
    return jnp.stack([rho * jnp.cos(phi), rho * jnp.sin(phi)], axis=-1)


def safe_norm(vector, axis=-1):
    """Euclidean norm whose gradient is finite at zero."""

    squared = jnp.sum(jnp.square(vector), axis=axis)
    positive = squared > 0.0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, squared, 1.0)), 0.0)


def _safe_div(numerator, denominator):
    nonzero = denominator != 0.0
    return jnp.where(nonzero, numerator / jnp.where(nonzero, denominator, 1.0), 0.0)


def obstruct_step(step, origin, center, radius):
    """Port of ``Obstacle.obstruct(ray, keep_tangential=True)`` for a batch of rays.

    Args:
        step: (..., 2) ray vectors (the desired displacement of the targets).
        origin: (..., 2) ray origins (the current target locations).
        center: (2,) the obstacle center.
        radius: () the obstacle radius.

    Returns:
        The (possibly) obstructed ray vectors with the same shape as ``step``.
    """

    relative = center - origin
    rel_norm = safe_norm(relative)
    norm = safe_norm(step)

    inside = jnp.logical_or(norm == 0.0, rel_norm < radius)
    far = rel_norm >= norm + radius

    inner = jnp.sum(relative * step, axis=-1)
    cos = jnp.minimum(1.0, _safe_div(inner, rel_norm * norm))
    perpendicular = rel_norm * jnp.sqrt(jnp.maximum(1.0 - jnp.square(cos), 0.0))
    hit = jnp.logical_and(inner >= 0.0, radius > perpendicular)
    half_chord = jnp.sqrt(jnp.maximum(jnp.square(radius) - jnp.square(perpendicular), 0.0))
    new_norm = jnp.maximum(0.0, rel_norm * cos - half_chord)
    shrink = jnp.logical_and(hit, new_norm < norm)

    shrunk = step * _safe_div(new_norm, norm)[..., None]
    radial = origin + shrunk - center
    scale = _safe_div((norm - new_norm) * half_chord, jnp.square(radius))
    tangential = step + radial * scale[..., None]

    result = jnp.where(shrink[..., None], tangential, step)
    result = jnp.where(far[..., None], step, result)
    return jnp.where(inside[..., None], -step, result)


def line_of_sight(source, destination, distance, centers, radii, tolerance=1e-6):
    """Exact occlusion test between sources and destinations through circular obstacles.

    This replaces the sampled + interpolated ``Camera.sight_range_func`` of the original
    implementation with an exact ray-circle intersection, which is fully vectorized and
    has no data-dependent shapes.

    Args:
        source: (S, 1, 2) ray origins (the cameras).
        destination: (1, D, 2) ray endpoints.
        distance: (S, D) distances between sources and destinations.
        centers: (O, 2) obstacle centers.
        radii: (O,) obstacle radii (radius 0 never occludes).

    Returns:
        (S, D) boolean array, True if the destination is not occluded.
    """

    if centers.shape[0] == 0:
        return jnp.ones(distance.shape, dtype=bool)

    direction = destination - source  # (S, D, 2)
    unit = direction / jnp.where(distance > 0.0, distance, 1.0)[..., None]
    relative = centers[None, :, :] - source  # (S, O, 2)
    rel_sq = jnp.sum(jnp.square(relative), axis=-1)  # (S, O)
    # Elementwise (not einsum): matmuls run in TF32 on GPUs and bfloat16 on TPUs by default,
    # which would shift occlusion boundaries by meters at these distances.
    projection = jnp.sum(unit[:, :, None, :] * relative[:, None, :, :], axis=-1)  # (S, D, O)
    perp_sq = rel_sq[:, None, :] - jnp.square(projection)
    radii_sq = jnp.square(radii)[None, None, :]

    inside = (rel_sq < jnp.square(radii)[None, :])[:, None, :]  # source inside an obstacle
    near_hit = projection - jnp.sqrt(jnp.maximum(radii_sq - perp_sq, 0.0))
    crosses = (projection >= 0.0) & (perp_sq < radii_sq) & (near_hit * (1.0 + tolerance) < distance[..., None])
    blocked = jnp.any(inside | crosses, axis=-1)
    return jnp.logical_not(blocked)
