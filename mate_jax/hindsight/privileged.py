"""Communication-free students distilled from a communicating teacher (privileged teacher-student).

The teacher is the option-level planner with the shared team memory: every camera knows what its teammates
observe (through detection packets) and which options they chose. The student runs on every camera without
any packets: its features come from the same planner with ``communication='none'`` (a private belief fed by
its own observations only) and its MATE observation, which includes the teammates' poses. It imitates the
teacher's next pan/zoom step (one of 3 x 3 rate-limited moves) by DAgger, i.e. it learns to coordinate
without talking. With ``motion=True`` the student also sees how its teammates' public poses changed over the last
one and five steps (from its own observations), i.e. where they are turning and zooming. With ``prior=True`` the
student's logits are a residual on the next move of its own communication-free planner (a one-hot feature with a learnable
scale, :class:`PriorActor`): it starts as the planner without communication and learns where to deviate from it. The teacher sees more than the student (the teammates' detections), so this is learning
from a privileged teacher; unlike the clairvoyant teacher of :mod:`mate_jax.hindsight.teacher`, it never sees
the future.
"""

import jax
import jax.numpy as jnp
import optax

from mate_jax import constants as consts

from mate_jax.foresight import ForesightCameraPolicy
from mate_jax.hindsight.student import Batch, HindsightDistiller, LocalWaypoints, PriorActor, StudentFeatures


__all__ = ['PrivilegedDistiller']


class PrivilegedDistiller(HindsightDistiller):
    """DAgger with a communicating teacher and a communication-free student (see the module docstring)."""

    HISTORY = 5

    def __init__(self, env, target_policy, num_steps: int = 500, key_steps: int = None, hiddens=(512, 256),
                 learning_rate: float = 3e-4, teacher_kwargs=None, student_kwargs=None, layer_norm: bool = False,
                 motion: bool = False, prior: bool = False):  # pylint: disable=super-init-not-called
        self.env = env
        self.target_policy = target_policy
        self.teacher = ForesightCameraPolicy.from_env(env, **(teacher_kwargs or {}))
        self.planner = ForesightCameraPolicy.from_env(env, **{'communication': 'none', **(student_kwargs or {})})
        self.num_steps = int(num_steps)
        self.key_steps = int(key_steps or num_steps)
        self.waypoints = LocalWaypoints(env, 1, 1)  # the next rate-limited step: 3 x 3 moves
        self.features = StudentFeatures(env, self.planner, self.waypoints, None)
        self.num_cameras = env.num_cameras
        self.motion = bool(motion)
        size = consts.CAMERA_STATE_DIM_PUBLIC + 1
        self.teammate_slice = (consts.camera_observation_dim(env.num_cameras, env.num_targets, env.num_obstacles)
                               - env.num_cameras * size, env.num_cameras * size)
        self.prior = bool(prior)
        self.prior_start = self.features.dim + (2 * self.teammate_slice[1] if self.motion else 0)
        self.feature_dim = self.prior_start + (self.waypoints.num_classes if self.prior else 0)
        self.target = 'planner'  # cross-entropy on waypoint classes, no hindsight plan
        self.lookahead = 1
        self.follow = 'student'
        self.ensemble = False  # set to True to roll out a stacked ensemble of checkpoints
        self.actor = PriorActor(num_outputs=self.waypoints.num_classes, hidden_sizes=tuple(hiddens), prior_start=self.prior_start if self.prior else None,
                                layer_norm=layer_norm)
        self.optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate))  # rate set per iteration (--lr-decay)
        self._rollout = jax.jit(jax.vmap(self._episode, in_axes=(None, 0, 0, None, 0)))
        self._update = jax.jit(self._update_step)

    def init(self, key):
        dummy = jnp.zeros((1, self.num_cameras, self.feature_dim), jnp.float32)
        params = self.actor.init(key, self.actor_input(dummy))
        return params, self.optimizer.init(params)

    def _teammates(self, observation):
        start, size = self.teammate_slice
        return observation[..., start : start + size]  # (C, C * 7) public poses of all cameras with flags

    def _features(self, student_carry, observation, history, orientation, viewing):
        parts = [self.features(student_carry, observation)]
        if self.motion:
            now = self._teammates(observation)
            parts += [now - history[-1], now - history[0]]
        if self.prior:  # next move of the camera's own planner without communication
            action, _ = self.planner.decide(student_carry, observation)
            move = self.waypoints.classify(orientation, viewing, orientation + action[:, 0], viewing + action[:, 1])
            parts.append(jax.nn.one_hot(move, self.waypoints.num_classes))
        return jnp.concatenate(parts, axis=-1)

    def _episode(self, params, key, plan, beta, mix_key):  # pylint: disable=unused-argument
        env, teacher, student, waypoints = self.env, self.teacher, self.planner, self.waypoints
        k_reset, k_camera, k_target, k_loop = jax.random.split(key, 4)
        (camera_obs, target_obs), state = env.reset(k_reset)
        k_teacher, k_student = jax.random.split(k_camera)
        teacher_carry = teacher.init(k_teacher, camera_obs)
        student_carry = student.init(k_student, camera_obs)
        target_carry = self.target_policy.init(k_target, target_obs)
        mix = jax.random.uniform(mix_key, (self.num_steps, self.num_cameras))
        history = jnp.broadcast_to(self._teammates(camera_obs), (self.HISTORY,) + self._teammates(camera_obs).shape)

        def body(carry, inputs):
            k, u = inputs
            state, camera_obs, target_obs, teacher_carry, student_carry, target_carry, history = carry
            k_teacher, k_student, k_target, k_env = jax.random.split(k, 4)
            teacher_carry = teacher.observe(k_teacher, teacher_carry, camera_obs)
            student_carry = student.observe(k_student, student_carry, camera_obs)
            orientation, viewing = state.camera_orientation, state.camera_viewing_angle
            features = self._features(student_carry, camera_obs, history, orientation, viewing)
            history = jnp.concatenate([history[1:], self._teammates(camera_obs)[None]], axis=0)
            teacher_action, _ = teacher.decide(teacher_carry, camera_obs)
            label = waypoints.classify(orientation, viewing, orientation + teacher_action[:, 0], viewing + teacher_action[:, 1])
            if self.ensemble:  # stacked parameters of several checkpoints: average of their move probabilities
                logits = jnp.log(jax.vmap(lambda p: jax.nn.softmax(self.actor.apply(p, self.actor_input(features))[0]))(params).mean(0))
            else:
                logits, _ = self.actor.apply(params, self.actor_input(features))
            choice = jnp.argmax(logits, axis=-1)
            student_action = waypoints.action(orientation, viewing, choice)
            camera_action = jnp.where((u < beta)[:, None], waypoints.action(orientation, viewing, label), student_action)
            target_action, target_carry = self.target_policy.act(k_target, target_carry, target_obs)
            timestep, state = env.step(k_env, state, camera_action.astype(camera_obs.dtype), target_action)
            carry = (state, timestep.camera_observation, timestep.target_observation, teacher_carry, student_carry, target_carry,
                     history)
            return carry, (features, label, timestep.info['coverage_rate'], (choice == label).mean())

        carry = (state, camera_obs, target_obs, teacher_carry, student_carry, target_carry, history)
        inputs = (jax.random.split(k_loop, self.key_steps)[: self.num_steps], mix)
        _, (features, labels, coverage, agreement) = jax.lax.scan(body, carry, inputs)
        return features, labels, coverage.mean(), agreement.mean()
