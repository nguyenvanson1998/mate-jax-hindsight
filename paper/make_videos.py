"""Animated episodes for the README: one recorded episode, three controllers and the clairvoyant hindsight schedule.

    python paper/make_videos.py                      # media/episode_{greedy,waypoint}.{gif,mp4}
    python paper/make_videos.py --targets greedy     # one mobility model

All four panels replay the same target trajectories (same episode key; the targets do not react to the cameras).
The hindsight schedule is computed offline with research/clairvoyant.py and knows every future target position: it is
a reference, not a controller. Runs on CPU in a few minutes; needs matplotlib and ffmpeg.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'research'))
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402

import clairvoyant as CV  # noqa: E402
import mate_jax  # noqa: E402
from mate_jax import agents as A  # noqa: E402
from mate_jax import constants as consts  # noqa: E402
from mate_jax.foresight import ForesightCameraPolicy  # noqa: E402
from mate_jax.hindsight.forecaster import LearnedForecastPlanner, load_forecaster  # noqa: E402

CONFIG = 'MATE-4v8-9.yaml'
TRAJECTORY_LEVELS = tuple(float(v) for v in range(30, 181, 5))
TARGETS = {'greedy': A.GreedyTargetPolicy, 'waypoint': A.RandomWaypointTargetPolicy}
TITLES = {'greedy': 'goal-directed targets', 'waypoint': 'random-waypoint (RWP) targets'}
COLORS = {'rule': '#868E96', 'option': '#74C0FC', 'foresight': '#1C7ED6', 'hindsight': '#E8590C'}
NAMES = {'rule': 'MATE rule-based cameras', 'option': 'Option-level planning', 'foresight': 'FORESIGHT',
         'hindsight': 'Hindsight schedule (offline reference)'}
C_SECTOR, C_SHADOW, C_NODE, C_TARGET, C_OBST = '#4C6EF5', '#DEE2E6', '#3B5BDB', '#E03131', '#868E96'

plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9})


def rollout(env, key, steps, record_steps, target_policy, camera_policy=None, actions=None):
    """One episode with the key convention of research/clairvoyant.py, so that all runs see the same targets."""

    k_reset, k_camera, k_target, k_loop = jax.random.split(key, 4)
    (camera_obs, target_obs), state = env.reset(k_reset)
    target_carry = target_policy.init(k_target, target_obs)
    camera_carry = camera_policy.init(k_camera, camera_obs) if camera_policy is not None else jnp.zeros(())
    planned = actions if actions is not None else jnp.zeros((steps, env.num_cameras, 2), jnp.float32)

    def body(carry, inputs):
        state, camera_obs, target_obs, camera_carry, target_carry = carry
        k, plan = inputs
        k_camera, k_target, k_env = jax.random.split(k, 3)
        if camera_policy is None:
            camera_action = plan
        else:
            camera_action, camera_carry = camera_policy.act(k_camera, camera_carry, camera_obs)
        target_action, target_carry = target_policy.act(k_target, target_carry, target_obs)
        timestep, state = env.step(k_env, state, camera_action, target_action)
        record = {'orientation': state.camera_orientation, 'viewing': state.camera_viewing_angle,
                  'sight': state.camera_sight_range, 'targets': state.target_location,
                  'mask': state.camera_target_view_mask, 'coverage': timestep.info['coverage_rate']}
        return (state, timestep.camera_observation, timestep.target_observation, camera_carry, target_carry), record

    loop_keys = jax.random.split(k_loop, record_steps)[:steps]
    carry = (state, camera_obs, target_obs, camera_carry, target_carry)
    _, records = jax.jit(lambda c, x: jax.lax.scan(body, c, x))(carry, (loop_keys, planned))
    static = {'cameras': state.camera_location, 'obstacles': state.obstacle_location, 'radii': state.obstacle_radius}
    return jax.tree_util.tree_map(np.asarray, static), jax.tree_util.tree_map(np.asarray, records)


def hindsight_actions(env, key, steps, record_steps, target_policy, rounds=3):
    """Block-coordinate ascent of the single-camera dynamic programs, as in research/clairvoyant.py (one episode)."""

    state, after = CV.record(env, target_policy, key[None], record_steps)
    after = after[:, :steps]
    dp = CV.make_camera_dp(env, steps)
    masks = jnp.zeros((1, env.num_cameras, steps, env.num_targets), bool)
    plans = jnp.zeros((1, env.num_cameras, steps, 2), jnp.float32)
    for round_index in range(rounds):
        for camera in range(env.num_cameras):
            if round_index == 0:
                others = jnp.any(masks[:, :camera], axis=1)
            else:
                others = jnp.any(jnp.delete(masks, camera, axis=1), axis=1)
            mask, action, _ = dp(state.camera_location[:, camera], state.camera_orientation[:, camera],
                                 state.camera_viewing_angle[:, camera], after, (~others).astype(jnp.float32),
                                 state.obstacle_location, state.obstacle_radius)
            masks = masks.at[:, camera].set(mask)
            plans = plans.at[:, camera].set(action)
    planned = float(jnp.any(masks, axis=1).mean())
    return plans[0].swapaxes(0, 1), planned  # (steps, C, 2)


def ray_ranges(origin, angles_deg, sight, centers, radii):
    u = np.stack([np.cos(np.radians(angles_deg)), np.sin(np.radians(angles_deg))], -1)
    rel = centers - origin
    proj = u @ rel.T
    perp_sq = np.sum(rel ** 2, -1)[None] - proj ** 2
    hit = (proj > 0) & (perp_sq < radii[None] ** 2)
    near = proj - np.sqrt(np.maximum(radii[None] ** 2 - perp_sq, 0.0))
    dist = np.where(hit, np.maximum(near, 0.0), np.inf).min(-1)
    return np.minimum(dist, sight)


def draw_scene(ax, static, rec, t):
    lim = consts.TERRAIN_SIZE
    frame = Rectangle((-lim, -lim), 2 * lim, 2 * lim, fill=False, ec='#ADB5BD', lw=0.8)
    ax.add_patch(frame)
    for w in consts.WAREHOUSES:
        r = consts.WAREHOUSE_RADIUS
        ax.add_patch(Rectangle(w - r, 2 * r, 2 * r, fc='#FFF3BF', ec='#FAB005', lw=0.5, zorder=0))
    for c, origin in enumerate(static['cameras']):
        o, v, s = rec['orientation'][t, c], rec['viewing'][t, c], rec['sight'][t, c]
        angles = np.linspace(o - v / 2, o + v / 2, 121)
        ranges = ray_ranges(origin, angles, s, static['obstacles'], static['radii'])
        rad = np.stack([np.cos(np.radians(angles)), np.sin(np.radians(angles))], -1)
        outer, seen = origin + s * rad, origin + ranges[:, None] * rad
        shadow = ax.add_patch(Polygon(np.vstack([seen, outer[::-1]]), closed=True, fc=C_SHADOW, ec='none', zorder=1))
        shadow.set_clip_path(frame)
        sensed = ax.add_patch(Polygon(np.vstack([origin, seen]), closed=True, fc=C_SECTOR, alpha=0.30, ec=C_SECTOR,
                                      lw=0.4, zorder=2))
        sensed.set_clip_path(frame)
    for center, radius in zip(static['obstacles'], static['radii']):
        ax.add_patch(Circle(center, radius, fc=C_OBST, ec='#495057', lw=0.5, zorder=3))
    for origin in static['cameras']:
        ax.add_patch(Circle(origin, 45.0, fc=C_NODE, ec='white', lw=0.6, zorder=6))
    covered = rec['mask'][t].any(0)
    pos = rec['targets'][t]
    ax.scatter(pos[covered, 0], pos[covered, 1], s=22, c=C_TARGET, edgecolors='white', linewidths=0.5, zorder=8)
    ax.scatter(pos[~covered, 0], pos[~covered, 1], s=22, facecolors='white', edgecolors=C_TARGET, linewidths=1.1, zorder=8)
    ax.set_xlim(-lim - 30, lim + 30)
    ax.set_ylim(-lim - 30, lim + 30)
    ax.set_aspect('equal')
    ax.axis('off')
    return int(covered.sum())


def render(runs, targets, steps, stride, fps, out_base, num_targets):
    order = ['rule', 'option', 'foresight', 'hindsight']
    running = {k: np.cumsum(runs[k][1]['coverage']) / np.arange(1, steps + 1) for k in order}
    frames = tempfile.mkdtemp()
    try:
        for f, t in enumerate(range(stride - 1, steps, stride)):
            fig = plt.figure(figsize=(10.8, 4.9), dpi=100)
            gs = fig.add_gridspec(2, 4, height_ratios=[3.0, 1.0], hspace=0.30, wspace=0.03,
                                  left=0.07, right=0.98, top=0.86, bottom=0.17)
            for i, k in enumerate(order):
                static, rec = runs[k]
                ax = fig.add_subplot(gs[0, i])
                n = draw_scene(ax, static, rec, t)
                name = 'Hindsight schedule (knows the future)' if k == 'hindsight' else NAMES[k]
                ax.set_title(f'{name}\ncovered now {n} of {num_targets}, mean so far {running[k][t]:.3f}',
                             fontsize=8.5, color='#C2410C' if k == 'hindsight' else 'black')
            ax = fig.add_subplot(gs[1, :])
            for k in order:
                ax.plot(np.arange(t + 1), running[k][:t + 1], color=COLORS[k], lw=1.6, ls='--' if k == 'hindsight' else '-',
                        label=NAMES[k])
            ax.set_xlim(0, steps)
            ax.set_ylim(0.4, 1.0)
            ax.set_ylabel('coverage,\nmean so far', fontsize=8)
            ax.set_xlabel('step', fontsize=8, labelpad=1)
            ax.tick_params(labelsize=7.5)
            ax.grid(alpha=0.25, lw=0.5)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.42), ncol=4, fontsize=7.5, frameon=False)
            fig.suptitle(f'MATE-4v8-9, {TITLES[targets]}, step {t + 1} of {steps}: the same recorded targets in every panel '
                         '(filled dot: covered, open dot: missed)', fontsize=9.5, y=0.985)
            fig.savefig(os.path.join(frames, f'{f:04d}.png'))
            plt.close(fig)
        pattern = os.path.join(frames, '%04d.png')
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', pattern, '-vf',
                        'scale=960:-2:flags=lanczos', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '26',
                        out_base + '.mp4'], check=True)
        palette = os.path.join(frames, 'palette.png')
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', pattern, '-vf',
                        'scale=960:-1:flags=lanczos,palettegen=max_colors=64:stats_mode=diff', palette], check=True)
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', pattern, '-i', palette,
                        '-lavfi', 'scale=960:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=none:diff_mode=rectangle',
                        out_base + '.gif'], check=True)
    finally:
        shutil.rmtree(frames, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--targets', nargs='+', default=['greedy', 'waypoint'])
    parser.add_argument('--episode', type=int, default=3, help='episode index of seed 0 (as Fig. S4 of the paper)')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--stride', type=int, default=5, help='simulation steps per video frame')
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--out', default=os.path.join(ROOT, 'media'))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    env = mate_jax.make(CONFIG)
    planner = ForesightCameraPolicy.from_env(env)
    record_steps = args.steps + int(np.max(planner.horizons)) + 1
    key = jax.random.split(jax.random.PRNGKey(0), args.episode + 1)[args.episode]
    info = {}
    for targets in args.targets:
        target_policy = TARGETS[targets].from_env(env)
        foresight = LearnedForecastPlanner.from_env(env, grid_size=32, objective='trajectory', levels=TRAJECTORY_LEVELS,
                                                    coordination_rounds=2)
        forecaster = os.path.join(ROOT, 'results', f'forecaster_{targets}', 'forecaster.msgpack')
        foresight = foresight.replace(forecaster_params=load_forecaster(forecaster, foresight))
        policies = {'rule': A.GreedyCameraPolicy.from_env(env), 'option': ForesightCameraPolicy.from_env(env, grid_size=32),
                    'foresight': foresight}
        runs = {k: rollout(env, key, args.steps, record_steps, target_policy, camera_policy=p) for k, p in policies.items()}
        actions, planned = hindsight_actions(env, key, args.steps, record_steps, target_policy)
        runs['hindsight'] = rollout(env, key, args.steps, record_steps, target_policy, actions=actions)
        for k, (_, rec) in runs.items():
            assert np.allclose(rec['targets'], runs['rule'][1]['targets']), f'{k}: targets must be identical'
        info[targets] = {k: float(rec['coverage'].mean()) for k, (_, rec) in runs.items()}
        info[targets]['hindsight_planned'] = planned
        print(targets, json.dumps(info[targets]), flush=True)
        render(runs, targets, args.steps, args.stride, args.fps, os.path.join(args.out, f'episode_{targets}'),
               env.num_targets)
    with open(os.path.join(args.out, 'episodes.json'), 'w', encoding='utf-8') as f:
        json.dump({'config': CONFIG, 'episode': args.episode, 'steps': args.steps, 'coverage': info}, f, indent=1)


if __name__ == '__main__':
    main()
