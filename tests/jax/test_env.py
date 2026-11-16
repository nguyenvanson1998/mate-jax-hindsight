"""Unit tests of the pure-JAX environment (no dependency on `gym`)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mate_jax
from mate_jax.agents import (
    GreedyCameraPolicy,
    GreedyTargetPolicy,
    RandomCameraPolicy,
    RandomTargetPolicy,
    RandomWaypointTargetPolicy,
)
from mate_jax.wrappers import AutoResetEnv, SingleTeamEnv


CONFIGS = ['MATE-4v2-9.yaml', 'MATE-4v8-0.yaml', 'MATE-8v8-9.yaml', 'MATE-Navigation.yaml']


@pytest.mark.parametrize('config', CONFIGS)
def test_shapes(config):
    env = mate_jax.make(config)
    (camera_obs, target_obs), state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert camera_obs.shape == (env.num_cameras, env.camera_observation_dim)
    assert target_obs.shape == (env.num_targets, env.target_observation_dim)
    assert env.state_vector(state).shape == (env.state_dim,)

    actions = env.sample_actions(jax.random.PRNGKey(1), state)
    timestep, _ = jax.jit(env.step)(jax.random.PRNGKey(2), state, *actions)
    assert timestep.camera_observation.shape == camera_obs.shape
    assert timestep.target_observation.shape == target_obs.shape
    assert np.isfinite(np.asarray(timestep.camera_observation)).all()
    assert np.isfinite(np.asarray(timestep.target_observation)).all()


def test_reset_invariants():
    env = mate_jax.make('MATE-4v8-9.yaml')
    cfg = env.config
    _, states = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(0), 256))
    s = jax.tree_util.tree_map(np.asarray, states)

    # Locations inside the configured ranges (with the terrain margin).
    camera_low = cfg.camera_location_low[s.camera_slots]
    camera_high = cfg.camera_location_high[s.camera_slots]
    assert (s.camera_location >= camera_low - 1e-3).all() and (s.camera_location <= camera_high + 1e-3).all()
    assert (s.obstacle_radius <= cfg.obstacle_radius_high + 1e-4).all()

    # Non-overlapping placement.
    eps = 1e-3
    cam_obs = np.linalg.norm(s.camera_location[:, :, None] - s.obstacle_location[:, None], axis=-1)
    assert (cam_obs + eps >= cfg.camera_radius + s.obstacle_radius[:, None] + cfg.target_step_size).all()
    tgt_obs = np.linalg.norm(s.target_location[:, :, None] - s.obstacle_location[:, None], axis=-1)
    assert (tgt_obs + eps >= s.obstacle_radius[:, None]).all()
    tgt_cam = np.linalg.norm(s.target_location[:, :, None] - s.camera_location[:, None], axis=-1)
    assert (tgt_cam + eps >= cfg.camera_radius).all()

    # Cameras: discrete initial orientations, viewing angles and sight ranges.
    residual = np.mod(s.camera_orientation, cfg.camera_rotation_step)
    assert (np.minimum(residual, cfg.camera_rotation_step - residual) < 1e-3).all()
    assert (s.camera_viewing_angle >= cfg.camera_min_viewing_angle - 1e-4).all()
    assert (s.camera_viewing_angle <= 180.0 + 1e-4).all()
    area = cfg.camera_min_viewing_angle * cfg.camera_max_sight_range**2
    assert np.allclose(s.camera_viewing_angle * s.camera_sight_range**2, area, rtol=1e-4)

    # Capacities and cargoes.
    assert ((s.target_capacity == 2).sum(-1) == env.num_high_capacity_targets).all()
    assert (s.target_goals >= 0).all()  # targets_start_with_cargoes
    assert (s.remaining_cargoes >= 0).all()
    total = s.remaining_cargoes.sum((-2, -1)) + s.target_goal_bits.sum((-2, -1))
    assert (total >= cfg.num_cargoes_per_target * env.num_targets).all()
    assert (s.target_goal_bits.sum(-1) <= s.target_capacity).all()
    assert (s.freights == s.target_goal_bits.sum(-1) * env.freight_scale).all()


def test_shuffle_disabled_is_identity():
    env = mate_jax.make('MATE-4v8-9.yaml', shuffle_entities=False)
    _, state = env.reset(jax.random.PRNGKey(3))
    assert np.array_equal(state.camera_slots, np.arange(env.num_cameras))
    assert np.array_equal(state.target_capacity[: env.num_high_capacity_targets], [2] * env.num_high_capacity_targets)


def test_obstruct_reverses_inside_obstacle():
    from mate_jax.geometry import obstruct_step

    step = jnp.array([[3.0, 4.0]])
    origin = jnp.array([[0.0, 0.0]])
    assert np.allclose(obstruct_step(step, origin, jnp.array([1.0, 0.0]), 5.0), -step)
    # Far obstacle: unchanged.
    assert np.allclose(obstruct_step(step, origin, jnp.array([100.0, 100.0]), 5.0), step)


def test_rollout_autoreset_and_episode_end():
    env = AutoResetEnv(mate_jax.make('MATE-4v2-0.yaml', max_episode_steps=20))

    def run(key):
        key, k_reset = jax.random.split(key)
        _, state = env.reset(k_reset)

        def body(state, k):
            k_act, k_step = jax.random.split(k)
            timestep, state = env.step(k_step, state, *env.sample_actions(k_act, state))
            return state, (timestep.done, state.episode_step)

        return jax.lax.scan(body, state, jax.random.split(key, 50))[1]

    dones, steps = jax.jit(jax.vmap(run))(jax.random.split(jax.random.PRNGKey(0), 8))
    dones, steps = np.asarray(dones), np.asarray(steps)
    # max_episode_steps=20 -> done after 21 steps (same off-by-one as `mate`)
    assert dones[:, 20].all() and not dones[:, :20].any()
    assert (steps[:, 20] == 0).all()


@pytest.mark.parametrize('team', ['camera', 'target'])
def test_single_team_env_with_policies(team):
    env = mate_jax.make('MATE-4v4-9.yaml')
    opponent = (GreedyTargetPolicy if team == 'camera' else GreedyCameraPolicy).from_env(env)
    team_env = SingleTeamEnv(env, team=team, opponent=opponent, frame_skip=3, horizon=7)
    policy = (RandomCameraPolicy if team == 'camera' else RandomTargetPolicy).from_env(env)

    def run(key):
        k_reset, k_loop = jax.random.split(key)
        obs, gstate, state = team_env.reset(k_reset)

        def body(state, k):
            k_act, k_step = jax.random.split(k)
            actions, _ = policy.act(k_act, (), state.team_observation)
            obs, gstate, reward, done, info, state = team_env.step(k_step, state, actions)
            return state, (obs, gstate, reward, done, info['returned_episode_length'])

        return jax.lax.scan(body, state, jax.random.split(k_loop, 15))[1]

    obs, gstate, reward, done, length = jax.jit(jax.vmap(run))(jax.random.split(jax.random.PRNGKey(0), 4))
    assert obs.shape[-2:] == (team_env.num_agents, team_env.observation_dim)
    assert gstate.shape[-1] == team_env.state_dim
    assert np.isfinite(np.asarray(obs)).all() and np.isfinite(np.asarray(gstate)).all()
    assert np.asarray(done)[:, 6].all() and np.asarray(done)[:, 13].all()  # horizon = 7
    assert (np.asarray(length)[:, 6] == 7).all()


def test_greedy_policies_respect_bounds():
    env = mate_jax.make('MATE-4v8-9.yaml')
    (camera_obs, target_obs), state = env.reset(jax.random.PRNGKey(0))
    camera_policy = GreedyCameraPolicy.from_env(env)
    target_policy = GreedyTargetPolicy.from_env(env)
    camera_carry = camera_policy.init(jax.random.PRNGKey(1), camera_obs)
    target_carry = target_policy.init(jax.random.PRNGKey(2), target_obs)
    step = jax.jit(env.step)
    for i in range(30):
        key = jax.random.PRNGKey(10 + i)
        k1, k2, k3 = jax.random.split(key, 3)
        camera_action, camera_carry = camera_policy.act(k1, camera_carry, camera_obs)
        target_action, target_carry = target_policy.act(k2, target_carry, target_obs)
        assert (np.abs(camera_action) <= env.camera_action_high + 1e-5).all()
        step_size = np.asarray(env.target_step_sizes(state))[:, None]
        assert (np.abs(target_action) <= step_size + 1e-5).all()
        timestep, state = step(k3, state, camera_action, target_action)
        camera_obs, target_obs = timestep.camera_observation, timestep.target_observation


def test_greedy_targets_deliver_cargo():
    env = mate_jax.make('MATE-4v2-0.yaml')
    team_env = SingleTeamEnv(env, team='camera', frame_skip=1, horizon=400, reward='raw')
    obs, _, state = team_env.reset(jax.random.PRNGKey(0))

    def body(state, k):
        zeros = jnp.zeros((team_env.num_agents, 2))
        _, _, _, _, info, state = team_env.step(k, state, zeros)
        return state, info['num_delivered_cargoes']

    _, delivered = jax.jit(lambda s: jax.lax.scan(body, s, jax.random.split(jax.random.PRNGKey(1), 400)))(state)
    assert int(np.max(delivered)) > 0


def test_observation_bounds_cover_observations():
    env = mate_jax.make('MATE-4v8-9.yaml')
    obs, _ = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), 32))
    for team_obs, (low, high) in zip(obs, (env.camera_observation_bounds(), env.target_observation_bounds())):
        team_obs = np.asarray(team_obs)
        assert (team_obs >= low - 1e-3).all() and (team_obs <= high + 1e-3).all()


def test_random_waypoint_targets_move_at_constant_speed():
    env = mate_jax.make('MATE-4v8-0.yaml')
    (_, target_obs), state = env.reset(jax.random.PRNGKey(0))
    policy = RandomWaypointTargetPolicy.from_env(env)
    carry = policy.init(jax.random.PRNGKey(1), target_obs)
    step = jax.jit(env.step)
    for i in range(20):
        action, carry = policy.act(jax.random.PRNGKey(2 + i), carry, target_obs)
        speed = np.linalg.norm(np.asarray(action), axis=-1)
        assert np.all(speed <= np.asarray(env.target_step_sizes(state)) + 1e-4)
        timestep, state = step(jax.random.PRNGKey(100 + i), state, jnp.zeros((env.num_cameras, 2)), action)
        target_obs = timestep.target_observation
    assert np.all(np.abs(np.asarray(carry['waypoint'])) <= 900.0 + 1e-3)
