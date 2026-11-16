"""The single-camera program over cells (research/clairvoyant.py, cells=True) must bound every continuous schedule.

1. A counterexample of an earlier version: initial viewing angle 31 degrees (off the 2.5-degree grid), a target straight
   ahead at 1490 m. Only viewing angles below about 30.4 degrees see it, and the camera can zoom from 31 to 30 degrees in
   one step, so a continuous schedule covers it; the cells must as well.
2. Random checks: off-grid initial poses, orientations near +-180 degrees, targets near the edges of the sector and of the
   range, and trajectories that push zoom and rotation to their limits. The optimum over cells must not be smaller than
   the coverage of any sampled continuous schedule. Run with: python3 tests/jax/test_cell_bound.py
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'research'))

import jax.numpy as jnp  # noqa: E402

import clairvoyant as C  # noqa: E402
import exact_pair as P  # noqa: E402
import mate_jax  # noqa: E402

ENV = mate_jax.make('MATE-4v8-9.yaml', obstacle={'transmittance': 0.0})
CFG = ENV.config
AREA = float(ENV.camera_area_product)
MIN_VIEW, MAX_VIEW = float(CFG.camera_min_viewing_angle), 180.0
ROT, ZOOM = float(CFG.camera_rotation_step), float(CFG.camera_zooming_step)


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def continuous_coverage(location, theta0, view0, targets, actions):
    """Targets covered by a continuous schedule (the simulator's rule, no obstacles): (S, T) targets, (S, 2) actions."""

    theta, view, total = theta0, view0, 0.0
    for t in range(targets.shape[0]):
        theta = wrap(theta + np.clip(actions[t, 0], -ROT, ROT))
        view = float(np.clip(view + np.clip(actions[t, 1], -ZOOM, ZOOM), MIN_VIEW, MAX_VIEW))
        rel = targets[t] - location
        dist = np.hypot(rel[:, 0], rel[:, 1])
        bearing = np.degrees(np.arctan2(rel[:, 1], rel[:, 0]))
        offset = np.abs(wrap(bearing - theta))
        total += float(np.sum((dist <= np.sqrt(AREA / view)) & (2.0 * offset <= view)))
    return total


def cell_optimum(dp, location, theta0, view0, targets):
    E = 1
    value = dp(jnp.asarray(location[None], jnp.float32), jnp.asarray([theta0], jnp.float32), jnp.asarray([view0], jnp.float32),
               jnp.asarray(targets[None], jnp.float32), jnp.ones((E,) + targets.shape[:2], jnp.float32),
               jnp.zeros((E, 1, 2), jnp.float32) + 5000.0, jnp.zeros((E, 1), jnp.float32))[2]
    return float(value[0])


def test_counterexample():
    dp = C.make_camera_dp(ENV, 1, cells=True)
    location = np.zeros(2)
    target = np.array([[[1490.0, 0.0]]])  # straight ahead (theta0 = 0)
    assert continuous_coverage(location, 0.0, 31.0, target, np.array([[0.0, -1.0]])) == 1.0
    value = cell_optimum(dp, location, 0.0, 31.0, target)
    assert value >= 1.0 - 1e-6, value
    print('counterexample: cells cover the target', value)


def test_random(cases=60, trajectories=300, steps=6, seed=0):
    rng = np.random.default_rng(seed)
    worst = np.inf
    for refine in (1, 2):
        dp = C.make_camera_dp(ENV, steps, refine=refine, cells=True)
        for case in range(cases):
            location = rng.uniform(-100, 100, 2)
            theta0 = float(rng.choice([rng.uniform(-180, 180), 179.0 + rng.uniform(0, 1), -180.0 + rng.uniform(0, 1)]))
            view0 = float(rng.choice([rng.uniform(MIN_VIEW, MAX_VIEW), MIN_VIEW + rng.uniform(0, 3),
                                      MAX_VIEW - rng.uniform(0, 3), 31.0]))
            # targets near the edges of the range and the sector of reachable poses
            T = 8
            dist = rng.uniform(300, 1520, (steps, T))
            bearing = theta0 + rng.uniform(-1, 1, (steps, T)) * (MAX_VIEW / 2 + ROT * steps)
            targets = location + np.stack([dist * np.cos(np.radians(bearing)), dist * np.sin(np.radians(bearing))], -1)
            bound = cell_optimum(dp, location, theta0, view0, targets)
            best = 0.0
            for _ in range(trajectories):
                mode = rng.integers(4)
                if mode == 0:
                    actions = rng.uniform(-1, 1, (steps, 2)) * [ROT, ZOOM]
                elif mode == 1:
                    actions = rng.choice([-1.0, 1.0], (steps, 2)) * [ROT, ZOOM]
                elif mode == 2:  # sweep while zooming in to the minimum viewing angle
                    actions = np.stack([rng.choice([-1.0, 1.0]) * np.full(steps, ROT), np.full(steps, -ZOOM)], -1)
                else:  # partial moves that end off the grid
                    actions = rng.uniform(-1, 1, (steps, 2)) * [ROT, ZOOM] * rng.uniform(0, 1)
                best = max(best, continuous_coverage(location, theta0, view0, targets, actions))
            assert best <= bound + 1e-6, (refine, case, theta0, view0, best, bound)
            worst = min(worst, bound - best)
    print(f'random checks passed; smallest margin of the bound over the best sampled schedule: {worst:.3f}')


def test_pair_matches_single(cases=10, steps=6, seed=1):
    """The joint program over cells with camera B switched off equals the single-camera program over cells."""

    rng = np.random.default_rng(seed)
    single = C.make_camera_dp(ENV, steps, cells=True)
    pair = P.make_pair_dp(ENV, steps, cells=True)
    for _ in range(cases):
        location = rng.uniform(-500, 500, (2, 2))
        theta0, view0 = rng.uniform(-180, 180, 2), rng.uniform(MIN_VIEW, MAX_VIEW, 2)
        targets = rng.uniform(-1200, 1200, (steps, 8, 2))
        joint = float(pair(jnp.asarray(location, jnp.float32), jnp.asarray(theta0, jnp.float32), jnp.asarray(view0, jnp.float32),
                           jnp.asarray(targets, jnp.float32), jnp.zeros((1, 2), jnp.float32) + 5000.0, jnp.zeros(1, jnp.float32), 1.0))
        alone = cell_optimum(single, location[0], float(theta0[0]), float(view0[0]), targets)
        assert abs(joint - alone) < 1e-3, (joint, alone)
    print('pair over cells with one camera off equals the single camera over cells')


if __name__ == '__main__':
    test_counterexample()
    test_random()
    test_pair_matches_single()
