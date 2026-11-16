"""Pilot 2: value of target forecasts at the *allocation* level (centralized oracle planners).

Every step, cameras greedily (sequential, submodular max-coverage) pick a goal pose
(orientation bin x viewing angle) and move towards it at the rate limits. Variants:
  now        : score poses with current true target positions (full information, no forecast)
  reach      : score each pose with the true positions at t + tau(pose), tau = steps to reach it
  reach+win  : average visibility over [t + tau, t + tau + W]
  cv-reach / cv+win : constant-velocity forecast from true positions (full information)
  mem-static+win / mem-cv+win : team memory of *observed* targets only (partial observability)
  cvp+win / mem-cvp+win : probabilistic version (particles around the constant-velocity path)

Usage (from the repository root):
  python3 research/pilot_allocation.py . MATE-4v8-9.yaml now,reach,reach+win,cv+win,mem-cv+win
"""

import sys

sys.path.insert(0, sys.argv[1])

import jax
import jax.numpy as jnp
import numpy as np

import mate_jax
from mate_jax import agents as A
from mate_jax.geometry import arctan2_deg, line_of_sight, normalize_angle, safe_norm

CONFIG = sys.argv[2] if len(sys.argv) > 2 else 'MATE-4v8-9.yaml'
STEPS, NUM_ENVS, H, WIN = 1000, 32, 40, 10
VIEW_LEVELS = jnp.array([30.0, 45.0, 60.0, 90.0, 120.0, 180.0])

env = mate_jax.make(CONFIG, obstacle={'transmittance': 0.0})
cfg = env.config
THETAS = jnp.arange(-180.0, 180.0, cfg.camera_rotation_step)
POSE_T = jnp.repeat(THETAS, VIEW_LEVELS.size)
POSE_A = jnp.tile(VIEW_LEVELS, THETAS.size)
POSE_R = jnp.sqrt(env.camera_area_product / POSE_A)
import os  # noqa: E402

TARGETS = os.environ.get('PILOT_TARGETS', 'greedy')  # greedy / waypoint
tgt = (A.RandomWaypointTargetPolicy if TARGETS == 'waypoint' else A.GreedyTargetPolicy).from_env(env)


def visible(state, positions):
    """positions (..., T, 2) -> (C, P, ..., T) visibility for every camera pose."""
    flat = positions.reshape(-1, 2)
    rel = flat[None] - state.camera_location[:, None]  # (C, M, 2)
    dist = safe_norm(rel)
    ang = arctan2_deg(rel[..., 1], rel[..., 0])
    los = line_of_sight(state.camera_location[:, None], flat[None], dist, state.obstacle_location, state.obstacle_radius)
    d = jnp.abs(normalize_angle(POSE_T[None, :, None] - ang[:, None, :]))
    vis = (dist[:, None] <= POSE_R[None, :, None]) & (2 * d <= POSE_A[None, :, None]) & los[:, None]
    return vis.reshape((env.num_cameras, POSE_T.size) + positions.shape[:-1])


NUM_PARTICLES, SIGMA = 8, 0.5  # probabilistic forecast: N(cv, (SIGMA * speed * h)^2)


def visible_per_pose(state, positions):
    """positions (C, P, ..., T, 2) (one set per camera pose) -> (C, P, ..., T) visibility."""

    def per_camera(location, positions):
        shape = positions.shape[:-1]
        flat = positions.reshape(-1, 2)
        rel = flat - location
        dist = safe_norm(rel)
        los = line_of_sight(location[None, None], flat[None], dist[None], state.obstacle_location, state.obstacle_radius)[0]
        ang = arctan2_deg(rel[..., 1], rel[..., 0])
        return dist.reshape(shape), ang.reshape(shape), los.reshape(shape)

    dist, ang, los = jax.vmap(per_camera)(state.camera_location, positions)
    extra = (None,) * (dist.ndim - 2)
    d = jnp.abs(normalize_angle(POSE_T[(None, slice(None)) + extra] - ang))
    return (dist <= POSE_R[(None, slice(None)) + extra]) & (2 * d <= POSE_A[(None, slice(None)) + extra]) & los


def plan(state, window, mode):
    tau = jnp.maximum(
        jnp.abs(normalize_angle(POSE_T[None] - state.camera_orientation[:, None])) / cfg.camera_rotation_step,
        jnp.abs(POSE_A[None] - state.camera_viewing_angle[:, None]) / cfg.camera_zooming_step,
    )
    tau = jnp.minimum(jnp.ceil(tau), H).astype(jnp.int32)  # (C, P)
    if mode == 'now':
        v = visible(state, window[0]).astype(jnp.float32)  # (C, P, T)
    elif mode.endswith('cvp+win'):  # window: (H+WIN+1, M, T, 2) particles
        offsets = tau[..., None] + jnp.arange(WIN)[None, None]  # (C, P, WIN)
        positions = window[offsets]  # (C, P, WIN, M, T, 2)
        v = visible_per_pose(state, positions).astype(jnp.float32).mean((2, 3))  # (C, P, T)
    else:
        vis_all = visible(state, window).astype(jnp.float32)  # (C, P, H+WIN+1, T)
        span = 1 if mode in ('reach', 'cv-reach') else WIN
        offsets = tau[..., None] + jnp.arange(span)[None, None]  # (C, P, span)
        v = jnp.take_along_axis(vis_all, offsets[..., None], axis=2).mean(2)  # (C, P, T)
    uncovered = jnp.ones(env.num_targets)
    goals = []
    for c in range(env.num_cameras):  # sequential greedy (submodular max coverage)
        gain = (v[c] * uncovered[None]).sum(-1) - 1e-3 * tau[c]
        p = jnp.argmax(gain)
        uncovered = uncovered * (1.0 - v[c, p])
        goals.append(p)
    goals = jnp.stack(goals)
    action = jnp.stack(
        [normalize_angle(POSE_T[goals] - state.camera_orientation), POSE_A[goals] - state.camera_viewing_angle], -1
    )
    return jnp.clip(action, env.camera_action_low, env.camera_action_high)


def make(mode):
    def run(key):
        k_reset, k_t, k_loop, k_noise = jax.random.split(key, 4)
        (_, tobs), state = env.reset(k_reset)
        keys = jax.random.split(k_loop, STEPS + H + WIN)

        def future(carry, k):  # target trajectories do not depend on camera actions
            state, tobs, tcarry = carry
            k_t, k_e = jax.random.split(k)
            ta, tcarry = tgt.act(k_t, tcarry, tobs)
            ts, state = env.step(k_e, state, jnp.zeros((env.num_cameras, 2)), ta)
            return (state, ts.target_observation, tcarry), state.target_location

        _, traj = jax.lax.scan(future, (state, tobs, tgt.init(k_t, tobs)), keys)
        traj = jnp.concatenate([state.target_location[None], traj], 0)
        windows = traj[jnp.arange(STEPS)[:, None] + jnp.arange(H + WIN + 1)[None]]
        if mode.startswith('cv'):  # constant-velocity forecast from the last two true positions
            prev = traj[jnp.maximum(jnp.arange(STEPS) - 1, 0)]
            velocity = traj[:STEPS] - prev
            h = jnp.arange(H + WIN + 1)[None, :, None, None]
            windows = jnp.clip(traj[:STEPS, None] + velocity[:, None] * h, -1000.0, 1000.0)
            if mode == 'cvp+win':
                noise = jax.random.normal(k_noise, (STEPS, 1, NUM_PARTICLES) + windows.shape[2:])
                speed = safe_norm(velocity)[:, None, None, :, None]
                windows = jnp.clip(
                    windows[:, :, None] + noise * SIGMA * speed * h[..., None], -1000.0, 1000.0
                )

        def body(carry, inputs):
            state, tobs, tcarry, memory = carry
            k, window = inputs
            if mode.startswith('mem'):  # team memory of observed targets + velocity extrapolation
                pos, vel, seen_at, t = memory
                seen = state.tracked_bits
                dt = jnp.maximum(t - seen_at, 1).astype(jnp.float32)[:, None]
                new_vel = jnp.where((seen_at >= 0)[:, None], (state.target_location - pos) / dt, 0.0)
                vel = jnp.where(seen[:, None], new_vel, vel)
                pos = jnp.where(seen[:, None], state.target_location, pos)
                seen_at = jnp.where(seen, t, seen_at)
                age = jnp.where(seen_at >= 0, t - seen_at, 0).astype(jnp.float32)
                h = jnp.arange(H + WIN + 1, dtype=jnp.float32)[:, None, None]
                window = jnp.clip(pos[None] + vel[None] * (age[None, :, None] + h), -1000.0, 1000.0)
                if mode == 'mem-static+win':
                    window = jnp.broadcast_to(pos[None], window.shape)
                if mode == 'mem-cvp+win':
                    k, k_noise = jax.random.split(k)
                    noise = jax.random.normal(k_noise, (H + WIN + 1, NUM_PARTICLES) + window.shape[1:])
                    speed = safe_norm(vel)[None, None, :, None]
                    spread = SIGMA * speed * (age[None, None, :, None] + h[:, None])
                    window = jnp.clip(window[:, None] + noise * spread, -1000.0, 1000.0)
                memory = (pos, vel, seen_at, t + 1)
            k_t, k_e = jax.random.split(k)
            ta, tcarry = tgt.act(k_t, tcarry, tobs)
            ts, state = env.step(k_e, state, plan(state, window, mode), ta)
            return (state, ts.target_observation, tcarry, memory), ts.info['coverage_rate']

        T = env.num_targets
        memory = (jnp.zeros((T, 2)), jnp.zeros((T, 2)), jnp.full(T, -1, jnp.int32), jnp.zeros((), jnp.int32))
        _, cov = jax.lax.scan(body, (state, tobs, tgt.init(k_t, tobs), memory), (keys[:STEPS], windows))
        return cov.mean()

    return jax.jit(jax.vmap(run))


keys = jax.random.split(jax.random.PRNGKey(0), NUM_ENVS)
print(env, '| targets:', TARGETS)
for mode in sys.argv[3].split(',') if len(sys.argv) > 3 else ['now', 'reach', 'reach+win']:
    cov = np.asarray(make(mode)(keys))
    print(f'{mode:>10s}: coverage {cov.mean():.4f} ± {cov.std() / np.sqrt(len(cov)):.4f}', flush=True)
