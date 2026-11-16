"""Smoke tests of the sharded MAPPO trainer (4 virtual CPU devices in a subprocess)."""

import json
import os
import subprocess
import sys
import textwrap

import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCRIPT = textwrap.dedent(
    """
    import json, sys
    import jax
    import numpy as np
    from mate_jax.mappo import MAPPOConfig, MAPPOTrainer

    team, action_type, dtype = sys.argv[1:4]
    config = MAPPOConfig(
        env_config='MATE-4v4-9.yaml', team=team, reward='coverage' if team == 'camera' else 'raw',
        num_envs=8, num_steps=4, total_timesteps=8 * 4 * 4, updates_per_call=2, horizon=6,
        actor_hiddens=(32,), critic_hiddens=(32,), action_type=action_type, compute_dtype=dtype,
    )
    trainer = MAPPOTrainer(config)
    runner = trainer.init()
    history = []
    for metrics, runner in trainer.iterate(runner):  # the previous runner state is donated
        history.append(metrics)
    # parameters stay replicated and identical on every device
    leaves = jax.tree_util.tree_leaves(runner.actor_state.params)
    shards = [np.asarray(s.data) for s in leaves[0].addressable_shards]
    replicated = all(np.array_equal(shards[0], s) for s in shards)
    evaluation = trainer.evaluate(runner.actor_state.params, num_envs=2)
    print(json.dumps({
        'devices': trainer.num_devices, 'updates': history[-1]['update'], 'replicated': replicated,
        'finite': all(np.isfinite(m['pg_loss']) and np.isfinite(m['vf_loss']) for m in history),
        'episodes': sum(m['num_episodes'] for m in history), 'evaluation': evaluation,
    }))
    """
)


@pytest.mark.parametrize(
    'team,action_type,dtype',
    [('camera', 'discrete', 'float32'), ('target', 'continuous', 'bfloat16')],
)
def test_sharded_training(team, action_type, dtype):
    env = dict(os.environ)
    env['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
    env['JAX_PLATFORMS'] = 'cpu'
    env['PYTHONPATH'] = ROOT + os.pathsep + env.get('PYTHONPATH', '')
    result = subprocess.run(
        [sys.executable, '-c', SCRIPT, team, action_type, dtype],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
        timeout=900,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    output = json.loads(result.stdout.strip().splitlines()[-1])
    assert output['devices'] == 4
    assert output['updates'] == 4
    assert output['replicated'] and output['finite']
    assert output['episodes'] > 0


def test_checkpoint_roundtrip_and_learned_opponent(tmp_path):
    import jax
    import numpy as np

    from mate_jax.mappo import MAPPOConfig, MAPPOTrainer
    from mate_jax.wrappers import SingleTeamEnv

    config = MAPPOConfig(
        env_config='MATE-4v2-9.yaml', num_envs=4, num_steps=4, total_timesteps=16, horizon=5,
        actor_hiddens=(16,), critic_hiddens=(16,), updates_per_call=1,
    )
    trainer = MAPPOTrainer(config)
    runner = trainer.init()
    for _, runner in trainer.iterate(runner):
        pass
    trainer.save(runner, str(tmp_path))
    params = trainer.load_params(str(tmp_path))
    for a, b in zip(jax.tree_util.tree_leaves(params['actor']), jax.tree_util.tree_leaves(runner.actor_state.params)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    # The trained camera team becomes the opponent of a target team.
    target_env = SingleTeamEnv(trainer.env, team='target', opponent=trainer.policy(params['actor']), horizon=5)
    obs, _, state = target_env.reset(jax.random.PRNGKey(0))
    obs, _, reward, done, _, state = jax.jit(target_env.step)(
        jax.random.PRNGKey(1), state, np.zeros((target_env.num_agents, 2), np.float32)
    )
    assert obs.shape == (target_env.num_agents, target_env.observation_dim) and np.isfinite(reward).all()


def test_cached_resets_restart_from_cache():
    import jax
    import numpy as np

    import mate_jax
    from mate_jax.wrappers import SingleTeamEnv

    env = mate_jax.make('MATE-4v2-9.yaml')
    team_env = SingleTeamEnv(env, team='camera', horizon=2, cache_resets=True)
    _, _, state = team_env.reset(jax.random.PRNGKey(0))
    cached_obs = np.asarray(state.next_reset[1])
    zeros = np.zeros((team_env.num_agents, 2), np.float32)
    step = jax.jit(team_env.step)
    _, _, _, done, _, state = step(jax.random.PRNGKey(1), state, zeros)
    assert not bool(done)
    _, _, _, done, _, state = step(jax.random.PRNGKey(2), state, zeros)
    assert bool(done)
    np.testing.assert_array_equal(np.asarray(state.team_observation), cached_obs)
    refreshed = team_env.refresh_resets(jax.random.PRNGKey(3), state)
    assert not np.array_equal(np.asarray(refreshed.next_reset[1]), cached_obs)
