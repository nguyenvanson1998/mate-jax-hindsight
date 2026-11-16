"""Hindsight teacher and student plumbing."""

import jax
import jax.numpy as jnp
import numpy as np

import mate_jax
from mate_jax import agents as A
from mate_jax.foresight import ForesightCameraPolicy
from mate_jax.hindsight import MOVES, HindsightDistiller, HindsightTeacher, LocalWaypoints, record_targets


def _setup(num_steps=12, num_envs=2):
    env = mate_jax.make('MATE-4v4-9.yaml')
    target_policy = A.GreedyTargetPolicy.from_env(env)
    keys = jax.random.split(jax.random.PRNGKey(3), num_envs)
    state0, targets = jax.jit(lambda k: record_targets(env, target_policy, k, num_steps))(keys)
    return env, target_policy, keys, state0, targets


def test_teacher_move_tables_follow_the_optimal_schedule():
    env, _, _, state0, targets = _setup()
    teacher = HindsightTeacher(env, num_steps=12, rounds=2)
    plan = teacher.solve(state0, targets)
    assert plan.moves.shape[:3] == (2, env.num_cameras, 12)
    # Following the move tables from the initial poses visits feasible lattice poses only.
    o = jnp.zeros((2, env.num_cameras), jnp.int32)
    j = plan.lattice.j0
    for t in range(12):
        move = jnp.asarray(MOVES)[plan.moves[jnp.arange(2)[:, None], jnp.arange(env.num_cameras)[None], t, o, j]]
        o, j = jnp.mod(o + move[..., 0], teacher.num_orientations), j + move[..., 1]
        viewing = plan.lattice.view0 + teacher.zooming * (j - plan.lattice.j0)
        assert bool(jnp.all((viewing >= env.config.camera_min_viewing_angle - 1e-4) & (viewing <= 180.0 + 1e-4)))
    # A later round never covers less than the first, sequential one.
    first = HindsightTeacher(env, num_steps=12, rounds=1).solve(state0, targets)
    assert float(jnp.any(plan.masks, 1).mean()) >= float(jnp.any(first.masks, 1).mean()) - 1e-6


def test_teacher_waypoint_matches_label_for_one_step():
    env, _, _, state0, targets = _setup()
    teacher = HindsightTeacher(env, num_steps=12, rounds=1)
    plan = teacher.solve(state0, targets)
    episode = jax.tree_util.tree_map(lambda x: x[0], plan)
    orientation, viewing = state0.camera_orientation[0], state0.camera_viewing_angle[0]
    move = jnp.asarray(MOVES)[teacher.label(episode, 0, orientation, viewing)]
    theta, view = teacher.waypoint(episode, 0, orientation, viewing, 1)
    np.testing.assert_allclose(
        np.asarray(mate_jax.geometry.normalize_angle(theta - orientation)), np.asarray(move[:, 0] * teacher.rotation), atol=1e-3
    )
    np.testing.assert_allclose(np.asarray(view - viewing), np.asarray(move[:, 1] * teacher.zooming), atol=1e-3)


def test_local_waypoints_round_trip():
    env = mate_jax.make('MATE-4v4-9.yaml')
    waypoints = LocalWaypoints(env, 3, 3)
    orientation = jnp.array([10.0, -175.0, 170.0, 0.0])
    viewing = jnp.array([60.0, 90.0, 35.0, 120.0])
    klass = jnp.array([0, 13, 30, waypoints.num_classes - 1])
    goal = waypoints.goal(orientation, viewing, klass)
    again = waypoints.classify(orientation, viewing, *goal)
    inside = (goal[1] > env.config.camera_min_viewing_angle) & (goal[1] < 180.0)
    np.testing.assert_array_equal(np.asarray(again)[np.asarray(inside)], np.asarray(klass)[np.asarray(inside)])
    action = waypoints.action(orientation, viewing, klass)
    assert bool(jnp.all(jnp.abs(action[:, 0]) <= env.config.camera_rotation_step + 1e-6))
    assert bool(jnp.all(jnp.abs(action[:, 1]) <= env.config.camera_zooming_step + 1e-6))


def test_distiller_teacher_closed_loop_matches_its_plan():
    env, target_policy, keys, _, _ = _setup(num_steps=10)
    planner = ForesightCameraPolicy.from_env(env, grid_size=16, samples_per_axis=1, horizon=8, window=4, stride=4)
    distiller = HindsightDistiller(env, target_policy, planner, num_steps=10, rounds=1, hiddens=(16,), use_trajectory=False)
    plan, bound = distiller.solve(keys)
    params, _ = distiller.init(jax.random.PRNGKey(0))
    features, labels, coverage, _ = distiller.rollout(params, keys, plan, beta=1.0)
    assert features.shape == (2, 10, env.num_cameras, distiller.features.dim)
    assert labels.shape == (2, 10, env.num_cameras)
    # Executing the teacher reproduces its planned coverage (the environment adds transmittance only).
    assert np.all(np.asarray(coverage) >= np.asarray(bound) - 1e-6)
