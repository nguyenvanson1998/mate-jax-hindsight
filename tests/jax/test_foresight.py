"""Tests of the FORESIGHT prototype (belief maps, atlas, options, planner)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mate_jax
from mate_jax import agents as A
from mate_jax.evaluation import evaluate_coverage
from mate_jax.foresight import BeliefFilter, ForesightCameraPolicy, Grid, PoseOptions, build_atlas
from mate_jax.geometry import normalize_angle


def test_grid_transition_conserves_mass_and_shifts_mean():
    grid = Grid(32)
    belief = grid.point_mass(jnp.array([0.0, 0.0]), 10.0)
    assert np.isclose(float(belief.sum()), 1.0, atol=1e-5)
    moved = grid.apply(belief, grid.transition(jnp.array([200.0, -100.0]), jnp.array(30.0)))
    assert np.isclose(float(moved.sum()), 1.0, atol=1e-5)
    centers = grid.centers
    mean_x = float((moved.sum(1) * centers).sum())
    mean_y = float((moved.sum(0) * centers).sum())
    assert abs(mean_x - 200.0) < grid.cell / 2 and abs(mean_y + 100.0) < grid.cell / 2


def _filter(**kwargs):
    return BeliefFilter(grid=Grid(32), speed=20.0, **kwargs)


def test_negative_information_and_detection_update():
    belief_filter = _filter(detection_prob=1.0)
    grid = belief_filter.grid
    free = jnp.ones((32, 32))
    state = belief_filter.init(jnp.asarray(grid.uniform_boxes([[-200, -200]], [[200, 200]])), free, 1, 2)
    fov = jnp.zeros((1, 32, 32)).at[0, 16:, :].set(1.0)  # observe x >= 0
    state = belief_filter.update(state, free, fov, jnp.array([[False, True]]), jnp.array([[[0.0, 0.0], [300.0, 50.0]]]))
    belief = np.asarray(state.belief[0])
    assert np.allclose(belief.sum((-2, -1)), 1.0, atol=1e-5)
    assert belief[0, 16:].sum() < 1e-6  # undetected target is not where we looked
    assert belief[1].argmax() == 20 * 32 + 16  # detected target collapses onto (300, 50): cell (20, 16)
    assert int(state.last_seen[0, 1]) == 0 and int(state.last_seen[0, 0]) == -1


def test_velocity_estimate_and_forecast_matches_recursion():
    belief_filter = _filter()
    free = jnp.ones((32, 32))
    state = belief_filter.init(jnp.full((32, 32), 1.0 / 1024), free, 1, 1)
    no_fov = jnp.zeros((1, 32, 32))
    for step in range(2):
        position = jnp.array([[[100.0 + 20.0 * step, -50.0]]])
        if step > 0:
            state = belief_filter.predict(state, free)
        state = belief_filter.update(state, free, no_fov, jnp.array([[True]]), position)
    assert bool(state.velocity_known[0, 0])
    np.testing.assert_allclose(state.velocity[0, 0], [20.0, 0.0], atol=1e-4)

    forecast = belief_filter.forecast(state, free, np.array([0, 1, 5]))
    np.testing.assert_allclose(forecast[0, 0], state.belief[0], atol=1e-5)
    np.testing.assert_allclose(forecast[0, 1], belief_filter.predict(state, free).belief[0], atol=1e-4)
    grid = belief_filter.grid
    mean_x = float((forecast[0, 2, 0].sum(1) * grid.centers).sum())
    assert abs(mean_x - (120.0 + 5 * 20.0)) < grid.cell


def test_atlas_field_of_view_matches_environment_perception():
    env = mate_jax.make('MATE-4v8-9.yaml', obstacle={'transmittance': 0.0})
    _, state = env.reset(jax.random.PRNGKey(4))
    grid = Grid(24)
    atlas = build_atlas(
        grid,
        state.camera_location,
        jnp.full(env.num_cameras, env.config.camera_max_sight_range),
        state.obstacle_location,
        state.obstacle_radius,
        samples_per_axis=1,
    )
    fov = atlas.field_of_view(state.camera_orientation, state.camera_viewing_angle, state.camera_sight_range)
    cells = jnp.asarray(grid.cell_xy, jnp.float32)
    expected = env._camera_perceive(state, cells)  # pylint: disable=protected-access
    assert np.mean(np.asarray(fov > 0.5) == np.asarray(expected)) > 0.999


def test_sector_mass_matches_brute_force():
    options = PoseOptions(rotation_step=5.0, zooming_step=2.5, min_viewing_angle=30.0, area_product=30.0 * 1500.0**2)
    rng = np.random.default_rng(0)
    num_cells, num_samples = 400, 4
    distance = jnp.asarray(rng.uniform(0, 1600, (num_cells, num_samples)), jnp.float32)
    angle = jnp.asarray(rng.uniform(-180, 180, (num_cells, num_samples)), jnp.float32)
    visible = jnp.asarray(rng.random((num_cells, num_samples)) < 0.8)
    weights = jnp.asarray(rng.random((3, num_cells)), jnp.float32)
    mass = options.sector_mass(options.histograms(weights, distance, angle, visible))  # (3, L, O)

    for level in range(options.num_levels):
        for orientation in (0, 17, 40, 71):
            theta = options.orientations[orientation]
            alpha = options.viewing_angles[level]
            radius = options.sight_ranges[level]
            inside = (
                np.asarray(visible)
                & (np.asarray(distance) <= radius)
                & (2.0 * np.abs(np.asarray(normalize_angle(np.asarray(angle) - theta))) <= alpha)
            )
            expected = (np.asarray(weights)[:, :, None] * inside[None]).sum((1, 2)) / num_samples
            np.testing.assert_allclose(mass[:, level, orientation], expected, rtol=1e-4, atol=1e-4)


def test_histogram_methods_agree():
    options = PoseOptions(rotation_step=5.0, zooming_step=2.5, min_viewing_angle=30.0, area_product=30.0 * 1500.0**2)
    rng = np.random.default_rng(1)
    distance = jnp.asarray(rng.uniform(0, 1600, (300, 4)), jnp.float32)
    angle = jnp.asarray(rng.uniform(-180, 180, (300, 4)), jnp.float32)
    visible = jnp.asarray(rng.random((300, 4)) < 0.7)
    weights = jnp.asarray(rng.random((5, 300)), jnp.float32)
    scatter = options.histograms(weights, distance, angle, visible, method='scatter')
    matmul = options.histograms(weights, distance, angle, visible, method='matmul')
    np.testing.assert_allclose(matmul, scatter, rtol=1e-5, atol=1e-5)


def test_reach_steps():
    options = PoseOptions(rotation_step=5.0, zooming_step=2.5, min_viewing_angle=30.0, area_product=1.0)
    steps = np.asarray(options.reach_steps(jnp.float32(0.0), jnp.float32(30.0)))
    level_90 = list(options.viewing_angles).index(90.0)
    assert steps[0, 36] == 0  # same orientation (0 deg) and zoom
    assert steps[0, 0] == 36  # half a turn
    assert steps[level_90, 36] == 24  # 60 deg of zoom at 2.5 deg / step


def test_path_indices_start_at_current_pose_and_end_at_goal():
    options = PoseOptions(rotation_step=5.0, zooming_step=2.5, min_viewing_angle=30.0, area_product=1.0)
    level, orient = options.path_indices(jnp.float32(0.0), jnp.float32(30.0), np.array([0, 4, 100]))
    level, orient = np.asarray(level), np.asarray(orient)
    assert (orient[0] == 36).all() and (level[0] == 0).all()  # still at the current pose
    assert (orient[2] == np.arange(72)[None]).all() and (level[2] == np.arange(options.num_levels)[:, None]).all()
    assert orient[1, 0, 0] == 32  # goal -180 deg: turn negative, 4 steps x 5 deg = -20 deg
    assert orient[1, 0, 71] == 40  # goal +175 deg: turn positive, +20 deg
    assert orient[1, 0, 38] == 38  # goal +10 deg is reached after 2 steps


@pytest.mark.parametrize(
    'kwargs',
    [
        {},
        {'communication': 'none'},
        {'use_forecast': False, 'reach_aware': False},
        {'objective': 'window'},
        {'communication': 'hybrid', 'message_drop': 0.5},
        {'communication': 'team', 'message_drop': 0.5},
        {'refine': True},
    ],
)
def test_planner_runs_and_respects_bounds(kwargs):
    env = mate_jax.make('MATE-4v4-9.yaml')
    policy = ForesightCameraPolicy.from_env(env, grid_size=16, horizon=20, window=6, **kwargs)
    (camera_obs, _), state = env.reset(jax.random.PRNGKey(0))
    carry = jax.jit(policy.init)(jax.random.PRNGKey(1), camera_obs)
    act = jax.jit(policy.act)
    for i in range(3):
        action, carry = act(jax.random.PRNGKey(i), carry, camera_obs)
        assert (np.abs(np.asarray(action)) <= env.camera_action_high + 1e-5).all()
        timestep, state = env.step(jax.random.PRNGKey(10 + i), state, action, jnp.zeros((env.num_targets, 2)))
        camera_obs = timestep.camera_observation
    assert np.allclose(np.asarray(carry['belief'].belief.sum((-2, -1))), 1.0, atol=1e-4)


def test_planner_beats_greedy_cameras():
    env = mate_jax.make('MATE-4v8-9.yaml')
    targets = A.GreedyTargetPolicy.from_env(env)
    greedy = evaluate_coverage(env, A.GreedyCameraPolicy.from_env(env), targets, num_envs=4, num_steps=200)
    foresight = evaluate_coverage(env, ForesightCameraPolicy.from_env(env), targets, num_envs=4, num_steps=200)
    assert foresight['coverage_rate'].mean() > greedy['coverage_rate'].mean()


def _without_detections(policy, observation):
    from mate_jax import constants as consts

    offset = consts.PRESERVED_DIM + consts.CAMERA_STATE_DIM_PRIVATE
    size = policy.num_targets * (consts.TARGET_STATE_DIM_PUBLIC + 1)
    return observation.at[:, offset : offset + size].set(0.0)


def test_hybrid_memory_keeps_own_detections_when_packets_are_lost():
    env = mate_jax.make('MATE-4v8-9.yaml')
    policy = ForesightCameraPolicy.from_env(env, communication='hybrid', message_drop=1.0)
    for seed in range(50):  # an episode start where some camera detects a target
        (camera_obs, _), _ = env.reset(jax.random.PRNGKey(seed))
        _, seen = policy._targets(camera_obs)  # pylint: disable=protected-access
        if np.asarray(seen).any():
            break
    assert np.asarray(seen).any()
    carry = policy.init(jax.random.PRNGKey(0), camera_obs)
    last_seen = np.asarray(carry['belief'].last_seen)  # (1 + C, T)
    assert (last_seen[0] == -1).all()  # nothing reached the team memory
    np.testing.assert_array_equal(last_seen[1:] == 0, np.asarray(seen))  # but every camera remembers its own


def test_refinement_keeps_the_plan_without_detections():
    env = mate_jax.make('MATE-4v8-9.yaml')
    (camera_obs, _), _ = env.reset(jax.random.PRNGKey(5))
    plain = ForesightCameraPolicy.from_env(env)
    refined = ForesightCameraPolicy.from_env(env, refine=True)
    blind = _without_detections(plain, camera_obs)
    a_plain, _ = plain.act(jax.random.PRNGKey(1), plain.init(jax.random.PRNGKey(0), blind), blind)
    a_refined, _ = refined.act(jax.random.PRNGKey(1), refined.init(jax.random.PRNGKey(0), blind), blind)
    np.testing.assert_allclose(a_plain, a_refined)


def test_per_camera_metrics():
    env = mate_jax.make('MATE-4v4-9.yaml')
    metrics = evaluate_coverage(
        env, A.GreedyCameraPolicy.from_env(env), A.GreedyTargetPolicy.from_env(env), num_envs=2, num_steps=50
    )
    assert metrics['camera_coverage'].shape == (2, env.num_cameras)
    assert (metrics['camera_unique_coverage'] <= metrics['camera_coverage'] + 1e-6).all()
    # unique coverage summed over cameras + redundantly covered targets = covered targets
    assert (metrics['camera_unique_coverage'].sum(-1) <= metrics['coverage_rate'] + 1e-6).all()
    assert (metrics['camera_rotation'] <= env.camera_action_high[0] + 1e-6).all()


def test_channel_contention():
    from mate_jax.foresight.channel import contend

    key = jax.random.PRNGKey(0)
    none = jnp.zeros(4, bool)
    delivered, collided = contend(key, none, slots=4)
    assert not np.asarray(delivered).any() and not np.asarray(collided).any()
    single = jnp.array([True, False, False, False])
    delivered, collided = contend(key, single, slots=4)
    assert np.asarray(delivered).tolist() == [True, False, False, False]
    everyone = jnp.ones(4, bool)
    delivered, collided = contend(key, everyone, slots=1)  # one slot: every packet collides
    assert not np.asarray(delivered).any() and np.asarray(collided).all()
    # Monte Carlo: success probability with n transmitters and S slots is (1 - 1/S)^(n - 1)
    keys = jax.random.split(key, 4000)
    delivered, _ = jax.vmap(lambda k: contend(k, everyone, slots=4))(keys)
    np.testing.assert_allclose(np.asarray(delivered).mean(), 0.75**3, atol=0.02)


def test_hrl_env_and_rules():
    from mate_jax.foresight.hrl import ForesightHRLEnv, evaluate_rule

    env = mate_jax.make('MATE-4v4-9.yaml')
    planner = ForesightCameraPolicy.from_env(env, communication='hybrid', grid_size=16, samples_per_axis=1, horizon=12, window=4, stride=4, num_candidates=4)
    hrl = ForesightHRLEnv(env, planner=planner, frame_skip=3, horizon=4)
    obs, gstate, state = jax.jit(hrl.reset)(jax.random.PRNGKey(0))
    assert obs.shape == (env.num_cameras, hrl.observation_dim)
    assert hrl.num_actions == 8
    option, transmit = hrl.decode(jnp.array([0, 3, 4, 7]))
    np.testing.assert_array_equal(option, [0, 3, 0, 3])
    np.testing.assert_array_equal(transmit, [False, False, True, True])
    np.testing.assert_array_equal(hrl.encode(option, transmit), [0, 3, 4, 7])
    step = jax.jit(hrl.step)
    for i in range(4):
        obs, gstate, reward, done, info, state = step(jax.random.PRNGKey(i), state, jnp.array([4, 5, 0, 1]))
        assert np.isfinite(np.asarray(obs)).all()
        np.testing.assert_allclose(info['tx_rate'], 0.5, atol=1e-6)  # two of four cameras transmit
    assert bool(done)  # horizon 4
    silent = evaluate_rule(hrl, 'never', num_envs=2, num_steps=3)
    loud = evaluate_rule(hrl, 'always', num_envs=2, num_steps=3)
    assert silent['tx_rate'] == 0.0 and loud['tx_rate'] == 1.0
    assert loud['reward'] < loud['coverage_rate']  # communication is charged
