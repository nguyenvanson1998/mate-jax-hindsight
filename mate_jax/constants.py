"""Constants of the JAX Multi-Agent Tracking Environment.

These mirror :mod:`mate.constants` exactly, but are plain NumPy arrays (no ``gym``
dependency) so they can be baked into XLA programs as compile-time constants.
"""

import functools

import numpy as np


__all__ = [
    'TERRAIN_SIZE',
    'TERRAIN_WIDTH',
    'WAREHOUSE_RADIUS',
    'WAREHOUSES',
    'NUM_WAREHOUSES',
    'MAX_CAMERA_VIEWING_ANGLE',
    'TARGET_RADIUS',
    'PRESERVED_DIM',
    'CAMERA_STATE_DIM_PUBLIC',
    'CAMERA_STATE_DIM_PRIVATE',
    'TARGET_STATE_DIM_PUBLIC',
    'TARGET_STATE_DIM_PRIVATE',
    'OBSTACLE_STATE_DIM',
    'CAMERA_ACTION_DIM',
    'TARGET_ACTION_DIM',
    'camera_observation_bounds',
    'target_observation_bounds',
    'camera_observation_dim',
    'target_observation_dim',
    'state_dim',
    'camera_coordinate_mask',
    'target_coordinate_mask',
]

TERRAIN_SIZE = 1000.0
TERRAIN_WIDTH = 2.0 * TERRAIN_SIZE
TERRAIN_LOW = np.array([-TERRAIN_SIZE, -TERRAIN_SIZE])
TERRAIN_HIGH = np.array([+TERRAIN_SIZE, +TERRAIN_SIZE])

WAREHOUSE_RADIUS = 0.075 * TERRAIN_SIZE
WAREHOUSES = (TERRAIN_SIZE - WAREHOUSE_RADIUS) * np.array(
    [[+1.0, +1.0], [-1.0, +1.0], [-1.0, -1.0], [+1.0, -1.0]]
)
NUM_WAREHOUSES = len(WAREHOUSES)

MAX_CAMERA_VIEWING_ANGLE = 180.0
TARGET_RADIUS = 0.0

PRESERVED_DIM = 3 + 1 + 2 * NUM_WAREHOUSES + 1
CAMERA_STATE_DIM_PUBLIC = 6
CAMERA_STATE_DIM_PRIVATE = 9
TARGET_STATE_DIM_PUBLIC = 4
TARGET_STATE_DIM_PRIVATE = 6 + 2 * NUM_WAREHOUSES
OBSTACLE_STATE_DIM = 3

CAMERA_ACTION_DIM = 2
TARGET_ACTION_DIM = 2

_INF = np.inf

PRESERVED_LOW = np.concatenate([[0.0] * 4, 2.0 * np.tile(TERRAIN_LOW, NUM_WAREHOUSES), [0.0]])
PRESERVED_HIGH = np.concatenate(
    [[_INF] * 4, 2.0 * np.tile(TERRAIN_HIGH, NUM_WAREHOUSES), [TERRAIN_SIZE]]
)

CAMERA_PUBLIC_LOW = np.append(2.0 * TERRAIN_LOW, [0.0, -TERRAIN_WIDTH, -TERRAIN_WIDTH, 0.0])
CAMERA_PUBLIC_HIGH = np.append(
    2.0 * TERRAIN_HIGH, [TERRAIN_SIZE, TERRAIN_WIDTH, TERRAIN_WIDTH, MAX_CAMERA_VIEWING_ANGLE]
)
CAMERA_PRIVATE_LOW = np.append(CAMERA_PUBLIC_LOW, [0.0, 0.0, 0.0])
CAMERA_PRIVATE_HIGH = np.append(
    CAMERA_PUBLIC_HIGH, [TERRAIN_WIDTH, MAX_CAMERA_VIEWING_ANGLE, MAX_CAMERA_VIEWING_ANGLE]
)

TARGET_PUBLIC_LOW = np.append(2.0 * TERRAIN_LOW, [0.0, -1.0])
TARGET_PUBLIC_HIGH = np.append(2.0 * TERRAIN_HIGH, [TERRAIN_WIDTH, 1.0])
TARGET_PRIVATE_LOW = np.concatenate(
    [TARGET_PUBLIC_LOW, [0.0, 1.0], [0.0] * NUM_WAREHOUSES, [-1.0] * NUM_WAREHOUSES]
)
TARGET_PRIVATE_HIGH = np.concatenate(
    [TARGET_PUBLIC_HIGH, [TERRAIN_WIDTH, 2.0], [_INF] * NUM_WAREHOUSES, [1.0] * NUM_WAREHOUSES]
)

OBSTACLE_LOW = np.append(2.0 * TERRAIN_LOW, 0.0)
OBSTACLE_HIGH = np.append(2.0 * TERRAIN_HIGH, TERRAIN_SIZE)


def camera_observation_dim(num_cameras: int, num_targets: int, num_obstacles: int) -> int:
    """Dimension of a single camera's observation."""

    return (
        PRESERVED_DIM
        + CAMERA_STATE_DIM_PRIVATE
        + num_targets * (TARGET_STATE_DIM_PUBLIC + 1)
        + num_obstacles * (OBSTACLE_STATE_DIM + 1)
        + num_cameras * (CAMERA_STATE_DIM_PUBLIC + 1)
    )


def target_observation_dim(num_cameras: int, num_targets: int, num_obstacles: int) -> int:
    """Dimension of a single target's observation."""

    return (
        PRESERVED_DIM
        + TARGET_STATE_DIM_PRIVATE
        + num_cameras * (CAMERA_STATE_DIM_PUBLIC + 1)
        + num_obstacles * (OBSTACLE_STATE_DIM + 1)
        + num_targets * (TARGET_STATE_DIM_PUBLIC + 1)
    )


def state_dim(num_cameras: int, num_targets: int, num_obstacles: int) -> int:
    """Dimension of the global state."""

    return (
        PRESERVED_DIM
        + num_cameras * CAMERA_STATE_DIM_PRIVATE
        + num_targets * TARGET_STATE_DIM_PRIVATE
        + num_obstacles * OBSTACLE_STATE_DIM
        + 2 * num_targets
        + NUM_WAREHOUSES * NUM_WAREHOUSES
    )


@functools.lru_cache(maxsize=None)
def camera_observation_bounds(num_cameras: int, num_targets: int, num_obstacles: int):
    """Lower and upper bounds of a single camera's observation (same as ``mate``)."""

    low = np.concatenate(
        [
            PRESERVED_LOW,
            CAMERA_PRIVATE_LOW,
            np.tile(np.append(TARGET_PUBLIC_LOW, -1.0), num_targets),
            np.tile(np.append(OBSTACLE_LOW, -1.0), num_obstacles),
            np.tile(np.append(CAMERA_PUBLIC_LOW, -1.0), num_cameras),
        ]
    )
    high = np.concatenate(
        [
            PRESERVED_HIGH,
            CAMERA_PRIVATE_HIGH,
            np.tile(np.append(TARGET_PUBLIC_HIGH, 1.0), num_targets),
            np.tile(np.append(OBSTACLE_HIGH, 1.0), num_obstacles),
            np.tile(np.append(CAMERA_PUBLIC_HIGH, 1.0), num_cameras),
        ]
    )
    return low.astype(np.float64), high.astype(np.float64)


@functools.lru_cache(maxsize=None)
def target_observation_bounds(num_cameras: int, num_targets: int, num_obstacles: int):
    """Lower and upper bounds of a single target's observation (same as ``mate``)."""

    low = np.concatenate(
        [
            PRESERVED_LOW,
            TARGET_PRIVATE_LOW,
            np.tile(np.append(CAMERA_PUBLIC_LOW, -1.0), num_cameras),
            np.tile(np.append(OBSTACLE_LOW, -1.0), num_obstacles),
            np.tile(np.append(TARGET_PUBLIC_LOW, -1.0), num_targets),
        ]
    )
    high = np.concatenate(
        [
            PRESERVED_HIGH,
            TARGET_PRIVATE_HIGH,
            np.tile(np.append(CAMERA_PUBLIC_HIGH, 1.0), num_cameras),
            np.tile(np.append(OBSTACLE_HIGH, 1.0), num_obstacles),
            np.tile(np.append(TARGET_PUBLIC_HIGH, 1.0), num_targets),
        ]
    )
    return low.astype(np.float64), high.astype(np.float64)


def _coordinate_mask(self_dim, opponent_dim, num_opponents, num_obstacles, teammate_dim, num_teammates):
    preserved = np.zeros(PRESERVED_DIM, dtype=bool)
    preserved[-1 - 2 * NUM_WAREHOUSES : -1] = True  # the warehouse locations

    def tiled(dim, reps):
        mask = np.zeros(dim + 1, dtype=bool)
        mask[:2] = True  # first two elements (x, y)
        return np.tile(mask, reps)

    return np.concatenate(
        [
            preserved,
            np.zeros(self_dim, dtype=bool),
            tiled(opponent_dim, num_opponents),
            tiled(OBSTACLE_STATE_DIM, num_obstacles),
            tiled(teammate_dim, num_teammates),
        ]
    )


@functools.lru_cache(maxsize=None)
def camera_coordinate_mask(num_cameras: int, num_targets: int, num_obstacles: int) -> np.ndarray:
    """Boolean mask of the coordinate entries in a camera observation (``mate`` semantics)."""

    return _coordinate_mask(
        CAMERA_STATE_DIM_PRIVATE,
        TARGET_STATE_DIM_PUBLIC,
        num_targets,
        num_obstacles,
        CAMERA_STATE_DIM_PUBLIC,
        num_cameras,
    )


@functools.lru_cache(maxsize=None)
def target_coordinate_mask(num_cameras: int, num_targets: int, num_obstacles: int) -> np.ndarray:
    """Boolean mask of the coordinate entries in a target observation (``mate`` semantics)."""

    return _coordinate_mask(
        TARGET_STATE_DIM_PRIVATE,
        CAMERA_STATE_DIM_PUBLIC,
        num_cameras,
        num_obstacles,
        TARGET_STATE_DIM_PUBLIC,
        num_targets,
    )
