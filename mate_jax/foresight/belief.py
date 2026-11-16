"""Per-target grid belief filter with negative information and multi-step forecasts.

Motion model: each target moves with an (estimated) velocity plus Gaussian uncertainty. The
displacement after ``a`` steps without observations is ``v * a + N(0, s(a)^2)`` per axis with

    s(a)^2 = sigma_walk^2 * a + sigma_velocity^2 * a^2,

i.e. a random walk plus velocity uncertainty (``sigma_velocity = speed / sqrt(2)`` when the
velocity is unknown: a uniform heading at the known speed). The recursive prediction uses the
exact Gaussian increments ``s(a + 1)^2 - s(a)^2``, so recursion and direct forecasts agree.

Update: a detection collapses the belief onto the observed position; every cell inside a field of
view in which the target was *not* detected is down-weighted by ``1 - p_detect`` (negative
information), which turns searching for lost targets into ordinary Bayesian filtering.
"""

import jax
import jax.numpy as jnp
from flax import struct

from mate_jax.foresight.grid import Grid


__all__ = ['BeliefState', 'BeliefFilter']


@struct.dataclass
class BeliefState:
    belief: jax.Array  # (K, T, G, G) one belief set per camera (no sharing) or K = 1 (team)
    last_position: jax.Array  # (K, T, 2)
    last_seen: jax.Array  # (K, T) int32, -1 if never seen
    velocity: jax.Array  # (K, T, 2) per step
    velocity_known: jax.Array  # (K, T) bool
    step: jax.Array  # () int32


@struct.dataclass
class BeliefFilter:
    grid: Grid = struct.field(pytree_node=False)
    speed: float = struct.field(pytree_node=False)  # maximum target displacement per step
    sigma_walk_ratio: float = struct.field(pytree_node=False, default=0.25)
    sigma_velocity_known_ratio: float = struct.field(pytree_node=False, default=0.3)
    sigma_velocity_unknown_ratio: float = struct.field(pytree_node=False, default=0.7071)
    detection_prob: float = struct.field(pytree_node=False, default=0.9)
    negative_information: bool = struct.field(pytree_node=False, default=True)
    max_velocity_gap: int = struct.field(pytree_node=False, default=3)
    velocity_smoothing: float = struct.field(pytree_node=False, default=0.5)
    observation_std_ratio: float = struct.field(pytree_node=False, default=0.5)  # of a cell

    # Motion model ------------------------------------------------------------------------------ #

    def _variance(self, age, known):
        sigma_v = jnp.where(known, self.sigma_velocity_known_ratio, self.sigma_velocity_unknown_ratio) * self.speed
        age = age.astype(jnp.float32)
        return jnp.square(self.sigma_walk_ratio * self.speed) * age + jnp.square(sigma_v * age)

    def _age(self, state):
        return jnp.where(state.last_seen >= 0, state.step - state.last_seen, state.step)

    def _mean_velocity(self, state):
        return jnp.where(state.velocity_known[..., None], state.velocity, 0.0)

    # Filter ------------------------------------------------------------------------------------ #

    def init(self, prior, free, num_sets: int, num_targets: int) -> BeliefState:
        belief = prior * free
        belief = belief / jnp.maximum(belief.sum(), 1e-12)
        return BeliefState(
            belief=jnp.broadcast_to(belief, (num_sets, num_targets) + belief.shape).astype(jnp.float32),
            last_position=jnp.zeros((num_sets, num_targets, 2), jnp.float32),
            last_seen=jnp.full((num_sets, num_targets), -1, jnp.int32),
            velocity=jnp.zeros((num_sets, num_targets, 2), jnp.float32),
            velocity_known=jnp.zeros((num_sets, num_targets), bool),
            step=jnp.zeros((), jnp.int32),
        )

    def predict(self, state: BeliefState, free) -> BeliefState:
        """One-step time update (advection + exact Gaussian variance increment, free space only)."""

        age = self._age(state)
        known = state.velocity_known
        increment = self._variance(age + 1, known) - self._variance(age, known)
        transition = self.grid.transition(self._mean_velocity(state), jnp.sqrt(jnp.maximum(increment, 0.0)))
        belief = self.grid.apply(state.belief, transition) * free
        belief = belief / jnp.maximum(belief.sum((-2, -1), keepdims=True), 1e-12)
        return state.replace(belief=belief, step=state.step + 1)

    def update(self, state: BeliefState, free, fov, detected, position) -> BeliefState:
        """Measurement update.

        Args:
            free: (G, G) free-space mask.
            fov: (K, G, G) in [0, 1], fraction of each cell observed by the set's camera(s).
            detected: (K, T) whether the target is currently observed.
            position: (K, T, 2) observed positions (ignored where not detected).
        """

        belief = state.belief
        if self.negative_information:
            belief = belief * (1.0 - self.detection_prob * fov[:, None])
            mass = belief.sum((-2, -1), keepdims=True)
            # The target escaped the model: restart from the unobserved free space.
            fallback = free * (1.0 - fov[:, None])
            fallback = fallback / jnp.maximum(fallback.sum((-2, -1), keepdims=True), 1e-12)
            belief = jnp.where(mass > 1e-9, belief / jnp.maximum(mass, 1e-12), fallback)

        observed = self.grid.point_mass(position, self.observation_std_ratio * self.grid.cell)
        belief = jnp.where(detected[..., None, None], observed, belief)

        # Velocity from consecutive detections.
        gap = state.step - state.last_seen
        recent = detected & (state.last_seen >= 0) & (gap <= self.max_velocity_gap) & (gap > 0)
        measured = (position - state.last_position) / jnp.maximum(gap, 1).astype(jnp.float32)[..., None]
        smoothed = jnp.where(
            state.velocity_known[..., None],
            self.velocity_smoothing * measured + (1.0 - self.velocity_smoothing) * state.velocity,
            measured,
        )
        velocity = jnp.where(recent[..., None], smoothed, jnp.where(detected[..., None], 0.0, state.velocity))
        velocity_known = jnp.where(detected, recent, state.velocity_known)

        return state.replace(
            belief=belief.astype(jnp.float32),
            last_position=jnp.where(detected[..., None], position, state.last_position),
            last_seen=jnp.where(detected, state.step, state.last_seen),
            velocity=velocity,
            velocity_known=velocity_known,
        )

    def forecast(self, state: BeliefState, free, horizons) -> jax.Array:
        """(K, H', T, G, G) beliefs ``h`` steps ahead (no future observations), for ``horizons`` (H',)."""

        horizons = jnp.asarray(horizons, jnp.int32)
        age = self._age(state)[:, None, :]  # (K, 1, T)
        known = state.velocity_known[:, None, :]
        h = horizons[None, :, None]
        variance = self._variance(age + h, known) - self._variance(age, known)
        shift = self._mean_velocity(state)[:, None, :, :] * h[..., None].astype(jnp.float32)
        transition = self.grid.transition(shift, jnp.sqrt(jnp.maximum(variance, 0.0)))
        belief = self.grid.apply(jnp.broadcast_to(state.belief[:, None], shift.shape[:-1] + state.belief.shape[-2:]), transition)
        belief = belief * free
        return belief / jnp.maximum(belief.sum((-2, -1), keepdims=True), 1e-12)

    def expected_counts(self, forecast, weights=None) -> jax.Array:
        """Aggregate per-target forecasts into expected target counts per cell (K, H', G*G)."""

        if weights is not None:
            forecast = forecast * weights[None, None, :, None, None]
        total = forecast.sum(2)
        return total.reshape(total.shape[:2] + (-1,))

    def entropy(self, state: BeliefState) -> jax.Array:
        """(K, T) Shannon entropy of each belief (nats)."""

        p = state.belief
        return -jnp.sum(p * jnp.log(jnp.maximum(p, 1e-12)), axis=(-2, -1))
