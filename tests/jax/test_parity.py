"""Parity of MATE-JAX with the original NumPy implementation (requires `gym` + `mate`)."""

import warnings

import numpy as np
import pytest


gym = pytest.importorskip('gym')
gym.logger.set_level(40)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import jax  # noqa: E402
import jax.experimental  # noqa: E402

import mate  # noqa: E402
import mate_jax  # noqa: E402
from mate.agents.utils import convert_coordinates, rescale_observation  # noqa: E402
from mate.entities import Obstacle  # noqa: E402
from mate.utils import Team, Vector2D  # noqa: E402
from mate_jax.bridge import state_from_numpy_env, sync_numpy_env  # noqa: E402
from mate_jax.transforms import ObservationTransform  # noqa: E402


NO_TRANSMITTANCE = {'obstacle': {'transmittance': 0.0}}


def test_obstruct_matches_numpy():
    from mate_jax.geometry import obstruct_step

    rng = np.random.default_rng(0)
    with jax.experimental.enable_x64():
        for _ in range(2000):
            origin = rng.uniform(-50, 50, size=2)
            center = rng.uniform(-60, 60, size=2)
            radius = rng.uniform(1, 40)
            step = rng.uniform(-30, 30, size=2)
            obstacle = Obstacle(location=center, radius=radius)
            expected = obstacle.obstruct(Vector2D(vector=step, origin=origin), keep_tangential=True).vector
            actual = obstruct_step(step[None], origin[None], center, radius)[0]
            np.testing.assert_allclose(actual, expected, atol=1e-8)


@pytest.mark.parametrize('config', ['MATE-4v8-9.yaml', 'MATE-8v8-9.yaml', 'MATE-4v2-0.yaml'])
def test_one_step_parity(config):
    """From identical states, one step of both implementations gives the same results."""

    np_env = mate.MultiAgentTracking(config, **NO_TRANSMITTANCE)
    rng = np.random.default_rng(0)
    mask_mismatch = mask_total = compared = 0
    with jax.experimental.enable_x64():
        env = mate_jax.make(config, **NO_TRANSMITTANCE)
        observe = jax.jit(env.joint_observation)
        update_view = jax.jit(env._update_view)  # pylint: disable=protected-access
        step = jax.jit(env.step)
        for episode in range(3):
            np_env.reset(seed=episode)
            for t in range(120):
                state = state_from_numpy_env(env, np_env)

                camera_obs, target_obs = observe(state)
                np.testing.assert_allclose(camera_obs, np_env.joint_observation()[0], atol=1e-9)
                np.testing.assert_allclose(target_obs, np_env.joint_observation()[1], atol=1e-9)
                np.testing.assert_allclose(env.state_vector(state), np_env.state(), atol=1e-9)

                # Perception from the same state (exact occlusion vs. interpolated boundary).
                viewed = update_view(jax.random.PRNGKey(0), state)
                for name in ('target_camera_view_mask', 'target_obstacle_view_mask', 'target_target_view_mask'):
                    assert np.array_equal(getattr(viewed, name), getattr(np_env, name)), name
                mask_mismatch += int((np.asarray(viewed.camera_target_view_mask) != np_env.camera_target_view_mask).sum())
                mask_mismatch += int((np.asarray(viewed.camera_camera_view_mask) != np_env.camera_camera_view_mask).sum())
                mask_total += np_env.camera_target_view_mask.size + np_env.camera_camera_view_mask.size

                camera_action = rng.uniform(-6.0, 6.0, size=(env.num_cameras, 2))
                target_action = rng.uniform(-30.0, 30.0, size=(env.num_targets, 2))
                (np_camera_obs, np_target_obs), (_, np_target_reward), np_done, _ = np_env.step(
                    (camera_action, target_action)
                )
                timestep, new_state = step(jax.random.PRNGKey(t), state, camera_action, target_action)

                np.testing.assert_allclose(
                    new_state.target_location, [x.location for x in np_env.targets], atol=1e-6
                )
                np.testing.assert_allclose(
                    new_state.camera_orientation, [c.orientation for c in np_env.cameras], atol=1e-9
                )
                assert bool(timestep.done) == bool(np_done)
                same_goals = np.array_equal(new_state.target_goals, np_env.target_goals)
                same_masks = np.array_equal(
                    new_state.camera_target_view_mask, np_env.camera_target_view_mask
                ) and np.array_equal(new_state.camera_camera_view_mask, np_env.camera_camera_view_mask)
                if same_goals and same_masks:  # random cargo picks / occlusion edge cases aside
                    compared += 1
                    np.testing.assert_allclose(timestep.camera_observation, np_camera_obs, atol=1e-6)
                    np.testing.assert_allclose(timestep.target_observation, np_target_obs, atol=1e-6)
                    np.testing.assert_allclose(timestep.target_reward, np_target_reward, atol=1e-6)
                if np_done:
                    break

    assert compared > 300
    assert mask_mismatch / mask_total < 1e-3


@pytest.mark.parametrize('team', [Team.CAMERA, Team.TARGET])
def test_observation_transform_matches_numpy(team):
    np_env = mate.MultiAgentTracking('MATE-4v8-9.yaml')
    env = mate_jax.make('MATE-4v8-9.yaml')
    counts = dict(num_cameras=4, num_targets=8, num_obstacles=9)
    with jax.experimental.enable_x64():
        transform = ObservationTransform(env, 'camera' if team is Team.CAMERA else 'target')
        for seed in range(5):
            observations = np_env.reset(seed=seed)
            for _ in range(10):
                observation = observations[team.value]
                expected = rescale_observation(convert_coordinates(observation, team=team, **counts), team=team, **counts)
                np.testing.assert_allclose(transform(observation), expected, atol=1e-9)
                observations, *_ = np_env.step(np_env.action_space.sample())


def test_sync_numpy_env_roundtrip():
    env = mate_jax.make('MATE-4v8-9.yaml')
    np_env = mate.MultiAgentTracking('MATE-4v8-9.yaml')
    np_env.reset(seed=0)
    with jax.experimental.enable_x64():
        _, state = mate_jax.make('MATE-4v8-9.yaml').reset(jax.random.PRNGKey(7))
        sync_numpy_env(np_env, state)
        back = state_from_numpy_env(env, np_env)
        for name in ('camera_location', 'target_location', 'obstacle_radius', 'target_goal_bits', 'remaining_cargoes'):
            np.testing.assert_allclose(getattr(back, name), getattr(state, name), atol=1e-9)
        # The NumPy env continues from the synced state.
        np_env.step(np_env.action_space.sample())
