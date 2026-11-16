"""Pilot 1: does aiming cameras at *forecast* target positions improve coverage in MATE?

Usage (from the repository root):  python3 research/pilot_aim_ahead.py . MATE-4v8-9.yaml

Greedy cameras (track nearest remembered target) vs. greedy targets, where the camera aims at
  (a) the current/remembered position (lead = 0, original GreedyCameraAgent),
  (b) a constant-velocity extrapolation `p + v * lead`,
  (c) an oracle: the true position `lead` steps in the future (privileged upper bound).
"""

import sys

sys.path.insert(0, sys.argv[1])

import jax
import jax.numpy as jnp
import numpy as np

import mate_jax
from mate_jax import agents as A
from mate_jax.geometry import arctan2_deg, normalize_angle, safe_norm

CONFIG = sys.argv[2] if len(sys.argv) > 2 else 'MATE-4v8-9.yaml'
STEPS = 1000
NUM_ENVS = 64


def make_runner(env, lead, mode):
    cam = A.GreedyCameraPolicy.from_env(env)
    tgt = A.GreedyTargetPolicy.from_env(env)

    def aim(carry, obs, aim_points):
        # GreedyCameraPolicy.act but tracking `aim_points` (C, T, 2) instead of memory locations
        k = carry['key']
        k, k_explore, k_random = jax.random.split(k, 3)
        carry = {**carry, 'time2forget': jnp.maximum(carry['time2forget'] - 1, 0)}
        carry = cam._memorize(carry, obs)
        me = A._camera_self(obs)
        distance = safe_norm(carry['memory'][..., 0:2] - me['location'][..., None, :])
        candidate = (carry['time2forget'] > 0) & (distance < 1.1 * me['max_sight_range'][..., None])
        has = jnp.any(candidate, -1)
        nearest = jnp.argmin(jnp.where(candidate, distance, jnp.inf), -1)
        point = jnp.take_along_axis(aim_points, nearest[..., None, None], axis=-2)[..., 0, :]
        rel = point - me['location']
        high = jnp.stack([me['rotation_step'], me['zooming_step']], -1)
        act = jnp.stack(
            [
                normalize_angle(arctan2_deg(rel[..., 1], rel[..., 0]) - me['orientation']),
                cam._best_viewing_angle(me, safe_norm(rel)) - me['viewing_angle'],
            ],
            -1,
        )
        act = jnp.clip(act, -high, high)
        explore = jax.random.bernoulli(k_explore, 0.1, has.shape)
        rnd = jax.random.uniform(k_random, high.shape, minval=-1.0, maxval=1.0) * high
        act = jnp.where(has[..., None], act, jnp.where(explore[..., None], rnd, carry['prev_action']))
        return act, {**carry, 'prev_action': act, 'key': k}

    def run(key):
        k_reset, k_c, k_t, k_loop = jax.random.split(key, 4)
        (cobs, tobs), state = env.reset(k_reset)
        ccarry = {**cam.init(k_c, cobs), 'key': k_c}
        ccarry['prev_memory'] = ccarry['memory']
        tcarry = tgt.init(k_t, tobs)

        def body(carry, k):
            state, cobs, tobs, ccarry, tcarry, history = carry
            k_t, k_e = jax.random.split(k)
            memory = ccarry['memory'][..., 0:2]
            if mode == 'velocity':
                velocity = memory - ccarry['prev_memory'][..., 0:2]
                points = memory + velocity * lead
            elif mode == 'oracle':
                points = jnp.broadcast_to(history[lead], memory.shape)  # true future positions
            else:
                points = memory
            prev_memory = ccarry['memory']
            ca, ccarry = aim(ccarry, cobs, points)
            ccarry['prev_memory'] = prev_memory
            ta, tcarry = tgt.act(k_t, tcarry, tobs)
            ts, state = env.step(k_e, state, ca, ta)
            return (state, ts.camera_observation, ts.target_observation, ccarry, tcarry, history), ts.info['coverage_rate']

        # Oracle: pre-simulate the target trajectory with the same keys (targets do not depend on
        # cameras except through collisions with camera bodies, which are static).
        history = jnp.zeros((max(lead, 1) + 1, env.num_targets, 2))
        if mode == 'oracle':
            def future(carry, k):
                state, tobs, tcarry = carry
                k_t, k_e = jax.random.split(k)
                ta, tcarry = tgt.act(k_t, tcarry, tobs)
                ts, state = env.step(k_e, state, jnp.zeros((env.num_cameras, 2)), ta)
                return (state, ts.target_observation, tcarry), state.target_location

            keys = jax.random.split(k_loop, STEPS + lead)
            _, traj = jax.lax.scan(future, (state, tobs, tgt.init(k_t, tobs)), keys)
            traj = jnp.concatenate([state.target_location[None], traj], 0)  # (STEPS+lead+1, T, 2)

            def body_oracle(carry, inputs):
                k, window = inputs
                state, cobs, tobs, ccarry, tcarry, _ = carry
                return body((state, cobs, tobs, ccarry, tcarry, window), k)

            idx = jnp.arange(STEPS)[:, None] + jnp.arange(lead + 1)[None, :]
            windows = traj[idx]  # (STEPS, lead+1, T, 2)
            _, cov = jax.lax.scan(body_oracle, (state, cobs, tobs, ccarry, tcarry, windows[0]), (keys[:STEPS], windows))
            return cov.mean()

        _, cov = jax.lax.scan(body, (state, cobs, tobs, ccarry, tcarry, history), jax.random.split(k_loop, STEPS + lead)[:STEPS])
        return cov.mean()

    return jax.jit(jax.vmap(run))


env = mate_jax.make(CONFIG, obstacle={'transmittance': 0.0})
keys = jax.random.split(jax.random.PRNGKey(0), NUM_ENVS)
print(env)
base = None
for mode, leads in [('current', [0]), ('velocity', [5, 10, 20]), ('oracle', [5, 10, 20, 40])]:
    for lead in leads:
        cov = np.asarray(make_runner(env, lead, mode)(keys))
        m, se = cov.mean(), cov.std() / np.sqrt(len(cov))
        base = m if base is None else base
        print(f'{mode:>9s} lead={lead:>3d}: coverage {m:.4f} ± {se:.4f}  ({(m / base - 1) * 100:+.1f}%)', flush=True)
