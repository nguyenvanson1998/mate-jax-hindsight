"""FORESIGHT: forecast-driven, reachability-aware camera allocation (prototype).

* :mod:`~mate_jax.foresight.grid` -- occupancy grid and separable Gaussian transitions
* :mod:`~mate_jax.foresight.belief` -- per-target belief filter with negative information
* :mod:`~mate_jax.foresight.atlas` -- static visibility atlas of every camera
* :mod:`~mate_jax.foresight.options` -- pose options, reach times and fast sector scoring
* :mod:`~mate_jax.foresight.planner` -- sequential-greedy planner (the non-learned baseline)
"""

from mate_jax.foresight.atlas import Atlas, build_atlas, free_space
from mate_jax.foresight.belief import BeliefFilter, BeliefState
from mate_jax.foresight.grid import Grid
from mate_jax.foresight.options import PoseOptions
from mate_jax.foresight.planner import ForesightCameraPolicy


__all__ = [
    'Atlas',
    'BeliefFilter',
    'BeliefState',
    'ForesightCameraPolicy',
    'Grid',
    'PoseOptions',
    'build_atlas',
    'free_space',
]
