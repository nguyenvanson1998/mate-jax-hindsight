"""Hindsight teacher: exact rate-limited camera schedules computed from the true future.

Target trajectories in MATE do not depend on the cameras, so during training the whole future of an
episode can be recorded first. Every camera then lives on a pose lattice anchored at its initial pose
(orientation steps of the rotation limit, viewing-angle steps of the zoom limit); one environment step
moves it by at most one lattice step per axis (:data:`MOVES`). Given the coverage of the other cameras,
the best schedule of one camera is an exact dynamic program over (time, pose). The backward pass gives,
for every time and every lattice pose, the best move (a *move table*), so the teacher can label any
state a student visits in O(1) (DAgger). Cameras are coordinated by block-coordinate ascent: the first
round is sequential, later rounds re-optimize every camera given the others.

Coverage counts range, field of view and occlusion (obstacle transmittance is ignored).
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from mate_jax.constants import MAX_CAMERA_VIEWING_ANGLE
from mate_jax.geometry import arctan2_deg, line_of_sight, normalize_angle, safe_norm


__all__ = ['MOVES', 'NUM_MOVES', 'Lattice', 'TeacherPlan', 'HindsightTeacher', 'record_targets', 'move_action']

# (orientation, zoom) lattice steps; "stay" first so that ties prefer not to move.
MOVES = np.array([(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)], np.int32)
NUM_MOVES = len(MOVES)
NEG = -1e9
RELATIVE_FLOOR = 1e4  # stored values are clipped below at -RELATIVE_FLOOR (fits float16)
LOCAL_MOVES = np.array([(do, dz) for do in (-1, 0, 1) for dz in (-1, 0, 1)], np.int32)  # LocalWaypoints(1, 1) order


class Lattice(NamedTuple):
    """Per-camera lattice anchors, each (..., C)."""

    theta0: Any  # orientation of lattice index 0 (the initial orientation)
    view0: Any  # viewing angle of zoom index ``j0`` (the initial viewing angle)
    j0: Any  # zoom index of the initial pose


class TeacherPlan(NamedTuple):
    lattice: Lattice
    moves: Any  # (..., C, S, O, V) int8: best move index at every step from every lattice pose
    masks: Any  # (..., C, S, T) bool: targets covered by every camera along the teacher schedule
    values: Any = None  # (..., C, S, O, V) float16: value-to-go of being at a pose at a step, minus its maximum


def record_targets(env, target_policy, keys, num_steps, key_steps=None):
    """Initial states and target positions after every step, (E, num_steps, T, 2).

    Uses the key layout of :func:`mate_jax.evaluation.make_coverage_evaluator` (step keys are the first
    ``num_steps`` of ``split(k_loop, key_steps)``), so that a later rollout with the same keys and any
    camera team sees the same target trajectories.
    """

    def episode(key):
        k_reset, _, k_target, k_loop = jax.random.split(key, 4)
        (_, target_obs), state = env.reset(k_reset)
        target_carry = target_policy.init(k_target, target_obs)
        idle = jnp.zeros((env.num_cameras, 2), jnp.float32)

        def body(carry, k):
            state, target_obs, target_carry = carry
            _, k_target, k_env = jax.random.split(k, 3)
            target_action, target_carry = target_policy.act(k_target, target_carry, target_obs)
            timestep, state = env.step(k_env, state, idle, target_action)
            return (state, timestep.target_observation, target_carry), state.target_location

        step_keys = jax.random.split(k_loop, key_steps or num_steps)[:num_steps]
        _, after = jax.lax.scan(body, (state, target_obs, target_carry), step_keys)
        return state, after

    return jax.vmap(episode)(keys)


def move_action(env, move):
    """Move indices (...) -> (rotation, zoom) environment actions (..., 2) in degrees."""

    cfg = env.config
    step = jnp.asarray(MOVES, jnp.float32) * jnp.asarray([cfg.camera_rotation_step, cfg.camera_zooming_step])
    return step[move]


class HindsightTeacher:
    """Clairvoyant, rate-limited schedules (see the module docstring)."""

    def __init__(self, env, num_steps: int, rounds: int = 3, keep_values: bool = False):
        cfg = env.config
        self.env = env
        self.num_steps = int(num_steps)
        self.rounds = int(rounds)
        self.keep_values = bool(keep_values)
        self.rotation, self.zooming = float(cfg.camera_rotation_step), float(cfg.camera_zooming_step)
        self.min_view = float(cfg.camera_min_viewing_angle)
        self.num_orientations = int(round(360.0 / self.rotation))
        self.num_views = int((MAX_CAMERA_VIEWING_ANGLE - self.min_view) / self.zooming) + 2

    # Lattice ------------------------------------------------------------------------------------ #

    def lattice(self, state) -> Lattice:
        view0 = state.camera_viewing_angle
        j0 = jnp.floor((view0 - self.min_view) / self.zooming + 1e-6).astype(jnp.int32)
        return Lattice(theta0=state.camera_orientation, view0=view0, j0=j0)

    def _axes(self, theta0, view0, j0):
        orientation = theta0 + self.rotation * jnp.arange(self.num_orientations)
        viewing = view0 + self.zooming * (jnp.arange(self.num_views) - j0)
        valid = (viewing >= self.min_view - 1e-6) & (viewing <= MAX_CAMERA_VIEWING_ANGLE + 1e-6)
        return orientation, viewing, valid

    def pose_index(self, lattice: Lattice, orientation, viewing):
        """Nearest lattice indices (o, j) of camera poses; all arguments (..., C)."""

        o = jnp.mod(jnp.round(normalize_angle(orientation - lattice.theta0) / self.rotation), self.num_orientations)
        j = lattice.j0 + jnp.round((viewing - lattice.view0) / self.zooming)
        return o.astype(jnp.int32), jnp.clip(j, 0, self.num_views - 1).astype(jnp.int32)

    def label(self, plan: TeacherPlan, step, orientation, viewing):
        """Teacher moves (C,) at ``step`` for the current camera poses (one episode)."""

        o, j = self.pose_index(plan.lattice, orientation, viewing)
        table = jax.lax.dynamic_index_in_dim(plan.moves, step, axis=1, keepdims=False)  # (C, O, V)
        return table[jnp.arange(o.shape[0]), o, j].astype(jnp.int32)

    def waypoint(self, plan: TeacherPlan, step, orientation, viewing, lookahead: int):
        """Pose (orientation, viewing angle), each (C,), that the teacher reaches ``lookahead`` steps after
        ``step`` when it starts from the current camera poses (one episode)."""

        o, j = self.pose_index(plan.lattice, orientation, viewing)
        cameras = jnp.arange(o.shape[0])
        moves = jnp.asarray(MOVES)
        last = plan.moves.shape[1] - 1

        def advance(i, pose):
            o, j = pose
            table = jax.lax.dynamic_index_in_dim(plan.moves, jnp.minimum(step + i, last), axis=1, keepdims=False)
            move = moves[table[cameras, o, j].astype(jnp.int32)]
            return jnp.mod(o + move[:, 0], self.num_orientations), jnp.clip(j + move[:, 1], 0, self.num_views - 1)

        o, j = jax.lax.fori_loop(0, lookahead, advance, (o, j))
        theta = normalize_angle(plan.lattice.theta0 + self.rotation * o)
        return theta, plan.lattice.view0 + self.zooming * (j - plan.lattice.j0)

    def move_values(self, plan: TeacherPlan, step, orientation, viewing):
        """(C, 9) hindsight values of the one-step moves in :data:`LOCAL_MOVES` order (one episode).

        Values are relative to the best pose at the next step; moves outside the zoom range get the floor.
        """

        o, j = self.pose_index(plan.lattice, orientation, viewing)
        table = jax.lax.dynamic_index_in_dim(plan.values, step, axis=1, keepdims=False)  # (C, O, V)
        moves = jnp.asarray(LOCAL_MOVES)
        oo = jnp.mod(o[:, None] + moves[None, :, 0], self.num_orientations)
        jj = j[:, None] + moves[None, :, 1]
        inside = (jj >= 0) & (jj < self.num_views)
        values = table[jnp.arange(o.shape[0])[:, None], oo, jnp.clip(jj, 0, self.num_views - 1)].astype(jnp.float32)
        return jnp.where(inside, values, -RELATIVE_FLOOR)

    # Dynamic programming --------------------------------------------------------------------- #

    def _gather_moves(self, value):
        """(O, V) -> (M, O, V) with [m][o, j] = value[o + do, j + dv] (NEG outside the zoom range)."""

        padded = jnp.pad(value, ((0, 0), (1, 1)), constant_values=NEG)
        return jnp.stack([
            jnp.roll(padded, -int(do), axis=0)[:, 1 + int(dv) : 1 + int(dv) + self.num_views] for do, dv in MOVES
        ])

    def _camera(self, location, theta0, view0, j0, targets, others, obstacle_location, obstacle_radius):
        """One camera given the others: (move table (S, O, V) int8, coverage mask (S, T) bool, value)."""

        env = self.env
        orientation, viewing, valid = self._axes(theta0, view0, j0)
        sight = jnp.sqrt(env.camera_area_product / jnp.maximum(viewing, 1e-3))
        relative = targets - location  # (S, T, 2)
        distance = safe_norm(relative)
        angle = arctan2_deg(relative[..., 1], relative[..., 0])
        visible = jax.vmap(
            lambda d, p: line_of_sight(location[None, None, :], p[None], d[None], obstacle_location, obstacle_radius)[0]
        )(distance, targets)  # (S, T)

        def covers(t):
            offset = jnp.abs(normalize_angle(angle[t][None, :] - orientation[:, None]))  # (O, T)
            return (
                (2.0 * offset[:, None, :] <= viewing[None, :, None])
                & (distance[t][None, None, :] <= sight[None, :, None])
                & visible[t][None, None, :]
            )  # (O, V, T)

        def backward(next_value, t):
            gain = jnp.sum(covers(t) & ~others[t][None, None, :], axis=-1).astype(jnp.float32)
            value = jnp.where(valid[None, :], gain + self._gather_moves(next_value).max(0), NEG)
            moves = self._gather_moves(value).argmax(0).astype(jnp.int8)  # best move into step t
            relative = jnp.maximum(value - value.max(), -RELATIVE_FLOOR).astype(jnp.float16)
            return value, (moves, relative)

        final = jnp.where(valid[None, :], 0.0, NEG) * jnp.ones((self.num_orientations, 1))
        value0, (moves, values) = jax.lax.scan(backward, final, jnp.arange(self.num_steps), reverse=True)

        def forward(pose, t):
            move = jnp.asarray(MOVES)[moves[t, pose[0], pose[1]]]
            pose = jnp.stack([jnp.mod(pose[0] + move[0], self.num_orientations), pose[1] + move[1]])
            return pose, covers(t)[pose[0], pose[1]]

        start = jnp.stack([jnp.int32(0), j0])
        _, mask = jax.lax.scan(forward, start, jnp.arange(self.num_steps))
        return moves, mask, values

    def solve(self, state0, targets) -> TeacherPlan:
        """Teacher plans for a batch of episodes: ``state0`` (E, ...) initial states, ``targets`` (E, S, T, 2)."""

        camera = jax.jit(jax.vmap(self._camera))
        lattice = self.lattice(state0)
        num_envs, num_cameras = state0.camera_orientation.shape
        masks = jnp.zeros((num_envs, num_cameras, self.num_steps, self.env.num_targets), bool)
        tables, values = [None] * num_cameras, [None] * num_cameras
        for round_index in range(self.rounds):
            for c in range(num_cameras):
                if round_index == 0:
                    others = jnp.any(masks[:, :c], axis=1)
                else:
                    others = jnp.any(jnp.delete(masks, c, axis=1), axis=1)
                tables[c], mask, values[c] = camera(
                    state0.camera_location[:, c], lattice.theta0[:, c], lattice.view0[:, c], lattice.j0[:, c],
                    targets, others, state0.obstacle_location, state0.obstacle_radius,
                )
                masks = masks.at[:, c].set(mask)
        return TeacherPlan(
            lattice=lattice, moves=jnp.stack(tables, axis=1), masks=masks,
            values=jnp.stack(values, axis=1) if self.keep_values else None,
        )
