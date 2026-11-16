"""Decentralized student cameras distilled from the hindsight teacher (behavior cloning + DAgger).

The student never sees the future; the teacher does. Every camera decides, at every step, a *local
waypoint*: the pose it wants to hold a few steps ahead, as an orientation offset (multiples of 5
degrees) and a viewing-angle offset (multiples of 5 degrees) from its current pose, and moves towards
it at the rate limits (:class:`LocalWaypoints`). Its inputs are local:

* its MATE observation (rescaled, relative coordinates);
* *egocentric pose-value maps*: the value of every pose option (zoom level x orientation, rolled so that
  index 0 is the camera's current orientation) at a few forecast horizons, computed from FORESIGHT's
  belief maps after discounting what the cameras earlier in the sequential pass will cover;
* the waypoint of the belief-space trajectory planner (FORESIGHT with ``objective='trajectory'``) as a
  local-waypoint class, and FORESIGHT's communication features.

DAgger rolls out a mixture of teacher and student on fresh episodes, labels every visited state with the
waypoint the teacher would reach from the student's own pose, and aggregates the labeled states.
"""

from typing import Any, Dict, NamedTuple, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from mate_jax.agents import _camera_self
from mate_jax.constants import MAX_CAMERA_VIEWING_ANGLE
from mate_jax.foresight import ForesightCameraPolicy
from mate_jax.foresight.options import NUM_ORIENTATIONS, ORIENTATION_STEP
from mate_jax.geometry import normalize_angle
from mate_jax.hindsight.teacher import NUM_MOVES, HindsightTeacher, TeacherPlan, move_action, record_targets
from mate_jax.models import Actor
from mate_jax.transforms import ObservationTransform


__all__ = ['LocalWaypoints', 'StudentFeatures', 'PriorActor', 'HindsightDistiller', 'trajectory_planner']

ZOOM_STEP = 5.0  # viewing-angle resolution of the trajectory planner
VALUE_CLIP, VALUE_SCALE = 20.0, 5.0  # relative hindsight values are clipped at -20 targets x steps, then scaled


LITE_TRAJECTORY = {'samples_per_axis': 1, 'horizon': 20, 'window': 8, 'stride': 4}  # for training


def trajectory_planner(env, **kwargs) -> ForesightCameraPolicy:
    """FORESIGHT with the belief-space trajectory (Viterbi) objective.

    Zoom levels are spaced by what the camera can zoom between two horizons (stride x zoom limit), 5
    degrees for the default stride of 2."""

    low = env.config.camera_min_viewing_angle
    spacing = kwargs.get('stride', 2) * env.config.camera_zooming_step
    levels = np.arange(low, MAX_CAMERA_VIEWING_ANGLE + 1e-6, spacing)
    return ForesightCameraPolicy.from_env(env, levels=tuple(levels), objective='trajectory', **kwargs)


class LocalWaypoints:
    """Waypoints relative to the current pose: (2 R + 1) orientation offsets x (2 Z + 1) zoom offsets,
    in units of the rotation and zoom limits (so a waypoint ``k`` steps ahead needs R = Z = k)."""

    def __init__(self, env, orientation_range: int = 3, zoom_range: int = 3):
        cfg = env.config
        self.controller = ForesightCameraPolicy.from_env(env).options  # rate-limited action_towards
        self.min_view = float(cfg.camera_min_viewing_angle)
        self.rotation, self.zooming = float(cfg.camera_rotation_step), float(cfg.camera_zooming_step)
        self.r, self.z = int(orientation_range), int(zoom_range)
        self.num_classes = (2 * self.r + 1) * (2 * self.z + 1)

    def classify(self, orientation, viewing, goal_orientation, goal_viewing):
        """Class (C,) of goal poses seen from the current poses (offsets are clipped to the range)."""

        do = jnp.clip(jnp.round(normalize_angle(goal_orientation - orientation) / self.rotation), -self.r, self.r)
        dz = jnp.clip(jnp.round((goal_viewing - viewing) / self.zooming), -self.z, self.z)
        return ((do + self.r) * (2 * self.z + 1) + (dz + self.z)).astype(jnp.int32)

    def goal(self, orientation, viewing, klass):
        do = klass // (2 * self.z + 1) - self.r
        dz = klass % (2 * self.z + 1) - self.z
        goal_orientation = normalize_angle(orientation + self.rotation * do)
        goal_viewing = jnp.clip(viewing + self.zooming * dz, self.min_view, MAX_CAMERA_VIEWING_ANGLE)
        return goal_orientation, goal_viewing

    def action(self, orientation, viewing, klass):
        """Rate-limited (rotation, zoom) action towards the waypoint of every camera."""

        goal_orientation, goal_viewing = self.goal(orientation, viewing, klass)
        return self.controller.action_towards(orientation, viewing, goal_orientation, goal_viewing)


class StudentFeatures:
    """Per-camera feature vectors (C, dim) from the FORESIGHT carry and the camera observations."""

    def __init__(
        self,
        env,
        planner: ForesightCameraPolicy,
        waypoints: LocalWaypoints,
        trajectory: Optional[ForesightCameraPolicy] = None,
        horizon_indices=(0, 5, 10, 20),
        include_suggestion: bool = True,
    ):
        self.env = env
        self.planner = planner
        self.trajectory = trajectory
        self.waypoints = waypoints
        # Without the suggestion in the features, the trajectory planner only provides labels (amortization).
        self.include_suggestion = bool(include_suggestion) and trajectory is not None
        self.transform = ObservationTransform(env, 'camera', True, True)
        self.horizon_indices = np.asarray([i for i in horizon_indices if i < planner.horizons.size])
        num_levels = planner.options.num_levels
        # Position of the trajectory planner's waypoint (one-hot) in the feature vector.
        self.suggestion_start = self.transform.dim + self.horizon_indices.size * num_levels * NUM_ORIENTATIONS
        self.dim = (
            self.transform.dim
            + self.horizon_indices.size * num_levels * NUM_ORIENTATIONS
            + (waypoints.num_classes if self.include_suggestion else 0)
            + planner.COMM_FEATURES
            + 1
        )

    def suggestion(self, carry, observation):
        """The trajectory planner's waypoint as a local-waypoint class (C,)."""

        me = _camera_self(observation)
        levels, orientations, _, _ = self.trajectory.plan(carry, observation)
        goal_orientation, goal_viewing, _ = self.trajectory.options.pose(levels[:, 0], orientations[:, 0])
        return self.waypoints.classify(me['orientation'], me['viewing_angle'], goal_orientation, goal_viewing)

    def __call__(self, carry, observation):
        planner = self.planner
        me = _camera_self(observation)
        _, _, _, _, maps = planner.plan_with_maps(carry, observation)
        maps = maps[:, self.horizon_indices] / planner.num_targets  # (C, h, L, O)
        current = jnp.mod(jnp.round((me['orientation'] + 180.0) / ORIENTATION_STEP), NUM_ORIENTATIONS).astype(jnp.int32)
        egocentric = jax.vmap(lambda m, o: jnp.roll(m, -o, axis=-1))(maps, current)
        parts = [self.transform(observation), egocentric.reshape(egocentric.shape[0], -1)]
        if self.include_suggestion:
            parts.append(jax.nn.one_hot(self.suggestion(carry, observation), self.waypoints.num_classes))
        parts += [planner.comm_features(carry, observation), (me['viewing_angle'] / 180.0)[:, None]]
        return jnp.concatenate(parts, axis=-1).astype(jnp.float32)


class PriorActor(nn.Module):
    """Policy logits = MLP(features) + s * (one-hot waypoint of the trajectory planner), s learnable.

    With the MLP's small initial output the policy starts as the trajectory planner; training learns
    where and how to deviate from it.
    """

    num_outputs: int
    hidden_sizes: Sequence[int]
    prior_start: Optional[int] = None
    prior_scale: float = 4.0
    layer_norm: bool = False  # input and hidden-layer normalization

    @nn.compact
    def __call__(self, observation):
        x = nn.LayerNorm()(observation) if self.layer_norm else observation
        logits, _ = Actor(num_outputs=self.num_outputs, discrete=True, hidden_sizes=tuple(self.hidden_sizes),
                          layer_norm=self.layer_norm)(x)
        if self.prior_start is None:
            return logits, None
        prior = observation[..., self.prior_start : self.prior_start + self.num_outputs]
        scale = self.param('prior_scale', nn.initializers.constant(self.prior_scale), ())
        return logits + scale * prior, None


class Batch(NamedTuple):
    features: Any  # (N, C, F)
    labels: Any  # (N, C)


class HindsightDistiller:
    """Teacher solve, DAgger rollouts, supervised updates and paired evaluation.

    ``target``:

    * ``'value'`` (amortized hindsight optimization): regress the teacher's hindsight value of every
      one-step move (relative to the best one) with a squared error, whose minimizer is the conditional
      expectation of the hindsight value given the student's information, and act greedily on it;
    * ``'waypoint'``: imitate the teacher's pose ``lookahead`` steps ahead (local waypoints);
    * ``'move'``: imitate its next lattice move;
    * ``'planner'`` (amortized FORESIGHT): imitate the next waypoint of the trajectory planner with
      ``planner_rounds`` best-response rounds, from features that do not contain it, so that the student
      replaces the Viterbi recursions and coordination rounds at execution time. The teacher is causal
      (same information as the student), so there is no imitation gap.

    Imitating actions (``'waypoint'``, ``'move'``) suffers from the imitation gap: the teacher's choice
    depends on the future, so its most likely choice given local information can be incoherent.
    """

    def __init__(
        self,
        env,
        target_policy,
        planner: ForesightCameraPolicy,
        num_steps: int = 500,
        key_steps: int = None,
        rounds: int = 3,
        hiddens=(512, 256),
        learning_rate: float = 3e-4,
        target: str = 'waypoint',
        lookahead: int = 3,
        use_trajectory: bool = True,
        follow: str = 'student',
        planner_rounds: int = 2,
    ):
        assert target in ('move', 'waypoint', 'value', 'planner')
        if target == 'value':
            lookahead = 1
        assert follow in ('student', 'labels', 'suggestion')
        self.env = env
        self.target_policy = target_policy
        self.planner = planner
        self.num_steps = int(num_steps)
        # Episodes draw their step keys from split(k_loop, key_steps): equal key_steps give equal targets.
        self.key_steps = int(key_steps or num_steps)
        self.teacher = HindsightTeacher(env, num_steps, rounds, keep_values=target == 'value')
        self.waypoints = LocalWaypoints(env, lookahead, lookahead)
        if target == 'planner':
            trajectory = trajectory_planner(env, coordination_rounds=planner_rounds)
            self.features = StudentFeatures(env, planner, self.waypoints, trajectory, include_suggestion=False)
        else:
            trajectory = trajectory_planner(env) if use_trajectory else None
            self.features = StudentFeatures(env, planner, self.waypoints, trajectory)
        self.num_cameras = env.num_cameras
        self.target = target
        self.lookahead = int(lookahead)
        # Diagnostics: 'labels' executes the teacher labels (the student's ceiling), 'suggestion' the
        # trajectory planner's waypoints (the planner alone through the same controller).
        self.follow = follow
        num_outputs = NUM_MOVES if target == 'move' else self.waypoints.num_classes
        prior = self.features.suggestion_start if (self.features.include_suggestion and target != 'move') else None
        self.actor = PriorActor(num_outputs=num_outputs, hidden_sizes=tuple(hiddens), prior_start=prior)
        self.optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))
        self._rollout = jax.jit(jax.vmap(self._episode, in_axes=(None, 0, 0, None, 0)))
        self._update = jax.jit(self._update_step)

    def actor_input(self, features):
        ids = jnp.broadcast_to(jnp.eye(self.num_cameras, dtype=features.dtype), features.shape[:-1] + (self.num_cameras,))
        return jnp.concatenate([features, ids], axis=-1)

    def init(self, key):
        dummy = jnp.zeros((1, self.num_cameras, self.features.dim), jnp.float32)
        params = self.actor.init(key, self.actor_input(dummy))
        return params, self.optimizer.init(params)

    # Rollouts ----------------------------------------------------------------------------------- #

    def _episode(self, params, key, plan: TeacherPlan, beta, mix_key):
        """One episode with the DAgger mixture: per camera and step the teacher acts with probability ``beta``."""

        env, planner, teacher, waypoints = self.env, self.planner, self.teacher, self.waypoints
        k_reset, k_camera, k_target, k_loop = jax.random.split(key, 4)
        (camera_obs, target_obs), state = env.reset(k_reset)
        camera_carry = planner.init(k_camera, camera_obs)
        target_carry = self.target_policy.init(k_target, target_obs)
        mix = jax.random.uniform(mix_key, (self.num_steps, self.num_cameras))

        def body(carry, inputs):
            k, t, u = inputs
            state, camera_obs, target_obs, camera_carry, target_carry = carry
            k_camera, k_target, k_env = jax.random.split(k, 3)
            camera_carry = planner.observe(k_camera, camera_carry, camera_obs)
            features = self.features(camera_carry, camera_obs)
            orientation, viewing = state.camera_orientation, state.camera_viewing_angle
            logits, _ = self.actor.apply(params, self.actor_input(features))
            choice = jnp.argmax(logits, axis=-1)
            if self.target == 'planner':  # the trajectory planner is the (causal) teacher
                label = self.features.suggestion(camera_carry, camera_obs)
                teacher_action = waypoints.action(orientation, viewing, label)
                student_action = waypoints.action(orientation, viewing, choice)
            else:
                teacher_move = teacher.label(plan, t, orientation, viewing)
                teacher_action = move_action(env, teacher_move)
            if self.target == 'planner':
                pass
            elif self.target == 'move':
                label = teacher_move
                student_action = move_action(env, choice)
            elif self.target == 'value':
                values = teacher.move_values(plan, t, orientation, viewing)
                label = jnp.clip(values - values.max(-1, keepdims=True), -VALUE_CLIP, 0.0) / VALUE_SCALE  # (C, 9)
                executed = jnp.argmax(label, axis=-1) if self.follow == 'labels' else choice
                student_action = waypoints.action(orientation, viewing, executed)
            else:
                goal = teacher.waypoint(plan, t, orientation, viewing, self.lookahead)
                label = waypoints.classify(orientation, viewing, *goal)
                if self.follow == 'labels':
                    executed = label
                elif self.follow == 'suggestion':
                    executed = self.features.suggestion(camera_carry, camera_obs)
                else:
                    executed = choice
                student_action = waypoints.action(orientation, viewing, executed)
            use_teacher = (u < beta)[:, None]
            camera_action = jnp.where(use_teacher, teacher_action, student_action)
            target_action, target_carry = self.target_policy.act(k_target, target_carry, target_obs)
            timestep, state = env.step(k_env, state, camera_action.astype(camera_obs.dtype), target_action)
            carry = (state, timestep.camera_observation, timestep.target_observation, camera_carry, target_carry)
            best = jnp.argmax(label, axis=-1) if self.target == 'value' else label
            return carry, (features, label, timestep.info['coverage_rate'], (choice == best).mean())

        carry = (state, camera_obs, target_obs, camera_carry, target_carry)
        inputs = (jax.random.split(k_loop, self.key_steps)[: self.num_steps], jnp.arange(self.num_steps), mix)
        _, (features, labels, coverage, agreement) = jax.lax.scan(body, carry, inputs)
        return features, labels, coverage.mean(), agreement.mean()

    def solve(self, keys):
        if self.target == 'planner':  # no hindsight plan is needed
            return None, np.full(keys.shape[0], np.nan)
        record = jax.jit(lambda k: record_targets(self.env, self.target_policy, k, self.num_steps, self.key_steps))
        state0, targets = record(keys)
        plan = self.teacher.solve(state0, targets)
        teacher_coverage = np.asarray(jnp.any(plan.masks, axis=1).mean((1, 2)))
        return plan, teacher_coverage

    def rollout(self, params, keys, plan, beta, seed=0):
        mix_keys = jax.random.split(jax.random.PRNGKey(seed), keys.shape[0])
        return self._rollout(params, keys, plan, jnp.float32(beta), mix_keys)

    # Supervised update ------------------------------------------------------------------------- #

    def _update_step(self, params, opt_state, batch: Batch):
        def loss_fn(params):
            logits, _ = self.actor.apply(params, self.actor_input(batch.features))
            if self.target == 'value':  # regression of the relative hindsight values
                loss = jnp.mean(jnp.square(logits - batch.labels))
                accuracy = (jnp.argmax(logits, axis=-1) == jnp.argmax(batch.labels, axis=-1)).mean()
            else:
                loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch.labels).mean()
                accuracy = (jnp.argmax(logits, axis=-1) == batch.labels).mean()
            return loss, accuracy

        (loss, accuracy), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, accuracy

    def train(self, params, opt_state, features, labels, key, epochs=4, batch_size=2048) -> Dict[str, Any]:
        """Supervised epochs over a dataset of (N, C, F) features and (N, C) labels."""

        num = features.shape[0]
        losses, accuracies = [], []
        for _ in range(epochs):
            key, k = jax.random.split(key)
            order = np.asarray(jax.random.permutation(k, num))
            for start in range(0, num - batch_size + 1, batch_size):
                index = order[start : start + batch_size]
                params, opt_state, loss, accuracy = self._update(
                    params, opt_state, Batch(jnp.asarray(features[index]), jnp.asarray(labels[index]))
                )
                losses.append(float(loss))
                accuracies.append(float(accuracy))
        if not losses:
            return params, opt_state, {'loss': float('nan'), 'accuracy': float('nan')}
        return params, opt_state, {'loss': float(np.mean(losses[-20:])), 'accuracy': float(np.mean(accuracies[-20:]))}
