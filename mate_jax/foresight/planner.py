"""FORESIGHT planner: forecast-driven, reachability-aware sequential-greedy camera allocation.

The first, non-learned baseline of the method. Each step, for the camera team:

1. belief update: per-target grid beliefs are predicted one step and corrected with the
   observations (detections collapse a belief; empty fields of view remove probability mass).
   Memory layouts (``communication``):

   * ``'team'``: one shared belief fed by every camera's packet (pose + detections) that reaches
     the team (each packet is lost with probability ``message_drop``);
   * ``'none'``: one independent belief per camera, fed by its own observations only;
   * ``'hybrid'``: both. For every target a camera plans with whichever of the shared or its own
     belief holds the more recent detection, so it never loses what it saw itself.

2. forecast: the beliefs are propagated ``h`` steps ahead for ``h`` in ``[0, horizon + window]``;
3. allocation: cameras pick a pose option one after another. Option ``w`` of camera ``i`` is
   worth the expected number of targets it covers, where the reach time ``tau_i(w)`` enters via
   the ``objective``:

   * ``'path'`` (default): discounted coverage over the whole planning horizon along the
     rate-limited straight path to the pose, i.e. the camera keeps scanning while it turns
     (coverage while turning is read from the same option table); with ``reach_aware=False``
     the goal pose is assumed to be reached at once;
   * ``'window'``: ``window`` steps of coverage starting at ``t + tau_i(w)`` (ignores what is
     lost or gained while turning, and over-rates distant poses);
   * ``'cumulative'``: as ``'path'`` but assuming no coverage while turning (too pessimistic).

   The chosen sector is then removed from the expected counts seen by the next cameras
   (probabilistic OR; the tiny intent messages are assumed to always arrive), so the plan is the
   greedy solution of a monotone submodular coverage problem under a partition matroid
   (>= 1/2 of the optimum). Without communication cameras plan independently;
4. control: every camera moves towards its pose at the rotation / zoom rate limits;
5. refinement (``refine=True``, per-agent post-processing): each camera tries small corrections
   of its planned rotation / zoom and keeps the one that covers the most of its currently
   detected targets at their exact, one-step-predicted positions (ties keep the plan; targets
   that the other cameras' planned poses already cover count only ``redundant_weight``). This
   removes the grid discretization error without abandoning the plan.
"""

from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from mate_jax import constants as consts
from mate_jax.agents import _camera_self
from mate_jax.foresight.atlas import build_atlas, free_space
from mate_jax.foresight.belief import BeliefFilter
from mate_jax.foresight.channel import contend
from mate_jax.foresight.grid import Grid
from mate_jax.foresight.options import NUM_ORIENTATIONS, PoseOptions
from mate_jax.geometry import arctan2_deg, line_of_sight, normalize_angle, safe_norm


__all__ = ['ForesightCameraPolicy']

P = consts.PRESERVED_DIM
# Message model (bytes): detection packet = pose + header + 5 B per detection; trajectory intent.
DETECTION_FIXED_BYTES = 8
DETECTION_BYTES = 5
INTENT_BYTES = 17
OPTION_INTENT_BYTES = 2


@struct.dataclass
class ForesightCameraPolicy:  # pylint: disable=too-many-instance-attributes
    num_cameras: int = struct.field(pytree_node=False)
    num_targets: int = struct.field(pytree_node=False)
    num_obstacles: int = struct.field(pytree_node=False)
    camera_radius: float = struct.field(pytree_node=False)
    target_speed: float = struct.field(pytree_node=False)
    options: PoseOptions = struct.field(pytree_node=False)
    prior: Any  # (G, G) prior over the initial target locations (a pytree leaf: arrays are unhashable)

    grid_size: int = struct.field(pytree_node=False, default=32)
    samples_per_axis: int = struct.field(pytree_node=False, default=2)
    horizon: int = struct.field(pytree_node=False, default=40)  # maximum reach time considered
    window: int = struct.field(pytree_node=False, default=10)  # steps of coverage counted per option
    stride: int = struct.field(pytree_node=False, default=2)  # forecast every `stride` steps
    communication: str = struct.field(pytree_node=False, default='team')  # team / none / hybrid
    # With communication='none': every camera runs the sequential pass for the whole team on its *private* belief
    # (teammates' positions and poses are observed, their detections are not) and executes its own option.
    implicit: bool = struct.field(pytree_node=False, default=False)
    message_drop: float = struct.field(pytree_node=False, default=0.0)  # packet loss to the team memory
    channel_slots: int = struct.field(pytree_node=False, default=4)  # contention slots (with transmit decisions)
    num_candidates: int = struct.field(pytree_node=False, default=8)  # top-K options exposed to a learner
    use_forecast: bool = struct.field(pytree_node=False, default=True)
    reach_aware: bool = struct.field(pytree_node=False, default=True)
    negative_information: bool = struct.field(pytree_node=False, default=True)
    detection_prob: float = struct.field(pytree_node=False, default=0.9)
    sigma_walk_ratio: float = struct.field(pytree_node=False, default=0.25)
    sigma_velocity_known_ratio: float = struct.field(pytree_node=False, default=0.3)
    sigma_velocity_unknown_ratio: float = struct.field(pytree_node=False, default=0.7071)
    reach_penalty: float = struct.field(pytree_node=False, default=1e-3)
    objective: str = struct.field(pytree_node=False, default='path')  # path / window / cumulative / trajectory
    discount: float = struct.field(pytree_node=False, default=0.95)  # per step (path / cumulative)
    coordination_rounds: int = struct.field(pytree_node=False, default=1)  # trajectory: rounds of best responses
    # 'synchronous': all trajectories are computed jointly every step (intents are free); 'published': every
    # camera best-responds to the trajectories its teammates last *published* (shifted by their age), and
    # ``carry['publish_mask']`` (default all) decides who publishes its new trajectory (an intent message).
    coordination: str = struct.field(pytree_node=False, default='synchronous')
    # Rule-based publication when no ``publish_mask`` is given: every ``publish_period`` steps (staggered by
    # camera), or, with ``publish_threshold`` > 0, when the new trajectory deviates from the published one by
    # more than this many orientation steps on average.
    publish_period: int = struct.field(pytree_node=False, default=1)
    publish_threshold: float = struct.field(pytree_node=False, default=0.0)
    refine: bool = struct.field(pytree_node=False, default=False)
    redundant_weight: float = struct.field(pytree_node=False, default=0.1)
    # Network models (the defaults reproduce i.i.d. loss, no delay, a packet every step, reliable intents).
    # 'iid': each packet is lost with probability ``message_drop``; 'burst': per-node Gilbert-Elliott link
    # (bad state loses everything) with stationary loss ``message_drop`` and mean burst ``burst_length`` steps.
    loss_model: str = struct.field(pytree_node=False, default='iid')
    burst_length: float = struct.field(pytree_node=False, default=10.0)
    # Non-uniform loss: node i loses with probability p (1 + s (2 i / (C - 1) - 1)), mean p.
    loss_spread: float = struct.field(pytree_node=False, default=0.0)
    message_delay: int = struct.field(pytree_node=False, default=0)  # steps before a packet reaches the team memory
    report_period: int = struct.field(pytree_node=False, default=1)  # detection packet every k steps (staggered)
    intent_delay: int = struct.field(pytree_node=False, default=0)  # published intents arrive k steps late
    intent_loss: bool = struct.field(pytree_node=False, default=False)  # intents share the lossy link

    @classmethod
    def from_env(cls, env, levels: Sequence[float] = (30.0, 45.0, 60.0, 90.0, 120.0, 180.0), **kwargs):
        cfg = env.config
        grid = Grid(kwargs.get('grid_size', 32))
        options = PoseOptions(
            rotation_step=cfg.camera_rotation_step,
            zooming_step=cfg.camera_zooming_step,
            min_viewing_angle=cfg.camera_min_viewing_angle,
            area_product=env.camera_area_product,
            levels=tuple(levels),
        )
        return cls(
            num_cameras=env.num_cameras,
            num_targets=env.num_targets,
            num_obstacles=env.num_obstacles,
            camera_radius=cfg.camera_radius,
            target_speed=cfg.target_step_size,
            prior=grid.uniform_boxes(cfg.target_location_low, cfg.target_location_high),
            options=options,
            **kwargs,
        )

    # Components -------------------------------------------------------------------------------- #

    @property
    def grid(self) -> Grid:
        return Grid(self.grid_size)

    @property
    def filter(self) -> BeliefFilter:
        return BeliefFilter(
            grid=self.grid,
            speed=self.target_speed,
            sigma_walk_ratio=self.sigma_walk_ratio,
            sigma_velocity_known_ratio=self.sigma_velocity_known_ratio,
            sigma_velocity_unknown_ratio=self.sigma_velocity_unknown_ratio,
            detection_prob=self.detection_prob,
            negative_information=self.negative_information,
        )

    @property
    def num_sets(self) -> int:
        """Belief sets: [shared] for team, [private x C] for none, [shared, private x C] for hybrid."""

        return {'team': 1, 'none': self.num_cameras, 'hybrid': 1 + self.num_cameras}[self.communication]

    @property
    def horizons(self) -> np.ndarray:
        return np.arange(0, self.horizon + self.window + 1, self.stride)

    # Observation parsing ----------------------------------------------------------------------- #

    def _targets(self, observation):
        offset = P + consts.CAMERA_STATE_DIM_PRIVATE
        size = consts.TARGET_STATE_DIM_PUBLIC + 1
        block = observation[..., offset : offset + self.num_targets * size]
        block = block.reshape(block.shape[:-1] + (self.num_targets, size))
        return block[..., 0:2], block[..., -1] > 0.5  # (C, T, 2), (C, T)

    def _obstacles(self, observation):
        offset = P + consts.CAMERA_STATE_DIM_PRIVATE + self.num_targets * (consts.TARGET_STATE_DIM_PUBLIC + 1)
        size = consts.OBSTACLE_STATE_DIM + 1
        block = observation[..., offset : offset + self.num_obstacles * size]
        block = block.reshape(block.shape[:-1] + (self.num_obstacles, size))
        seen = block[..., -1] > 0.5  # (C, O)
        first = jnp.argmax(seen, axis=0)  # obstacle states are identical for all observers
        states = jnp.take_along_axis(block, first[None, :, None], axis=0)[0]
        radius = jnp.where(jnp.any(seen, axis=0), states[:, 2], 0.0)
        return states[:, 0:2], radius

    def _default_link(self) -> bool:
        return self.loss_model == 'iid' and self.loss_spread == 0.0

    def _loss_rates(self):
        """(C,) packet loss probability of every node."""

        p = jnp.full((self.num_cameras,), self.message_drop, jnp.float32)
        if self.loss_spread > 0.0 and self.num_cameras > 1:
            p = p * (1.0 + self.loss_spread * (2.0 * jnp.arange(self.num_cameras) / (self.num_cameras - 1) - 1.0))
        return jnp.clip(p, 0.0, 1.0)

    def _link(self, key, carry):
        """(C,) bool whether each node's link delivers this step, and the updated link state."""

        p = self._loss_rates()
        if self.loss_model == 'burst':
            recover = 1.0 / max(self.burst_length, 1.0)
            fail = p * recover / jnp.maximum(1.0 - p, 1e-6)  # stationary loss p
            bad = carry.get('link_bad', jnp.zeros((self.num_cameras,), bool))
            u = jax.random.uniform(key, (self.num_cameras,))
            bad = jnp.where(bad, u >= recover, u < fail)
            return jnp.logical_not(bad), {'link_bad': bad}
        return jnp.logical_not(jax.random.bernoulli(key, p)), {}

    def _measurements(self, key, carry, observation, transmit=None):
        return self._measure(key, carry, observation, transmit)[:4]

    def _measure(self, key, carry, observation, transmit=None):
        """(K, G, G) observed cell fractions, (K, T) detections, (K, T, 2) positions, the
        channel outcome ``(transmit, delivered, collided)`` of every camera's packet and the
        updated network state (link states, delay buffer, bytes sent).

        Without ``transmit`` every camera sends (every ``report_period`` steps) and packets are
        lost according to ``loss_model``; with ``transmit`` (C,) only those cameras send,
        contending for ``channel_slots`` slots (collisions) with the same link losses. With
        ``message_delay`` = d the team memory receives the packets sent d steps ago.
        """

        me = _camera_self(observation)
        fov = carry['atlas'].field_of_view(me['orientation'], me['viewing_angle'], me['sight_range'])  # (C, N)
        positions, seen = self._targets(observation)  # (C, T, 2), (C, T)
        G = self.grid_size
        C = self.num_cameras
        updates = {}

        def pooled(fov, seen, positions, delivered):  # team memory from the packets that arrived
            union = 1.0 - jnp.prod(1.0 - fov * delivered[:, None], axis=0, keepdims=True)
            detections = seen & delivered[:, None]
            first = jnp.argmax(detections, axis=0)
            return (
                union,
                jnp.any(detections, axis=0, keepdims=True),
                jnp.take_along_axis(positions, first[None, :, None], axis=0),
            )

        if transmit is not None:
            transmit = transmit.astype(bool)
            if self._default_link():
                delivered, collided = contend(key, transmit, self.channel_slots, self.message_drop)
                link_ok = jnp.ones((C,), bool)
            else:
                k_slot, k_link = jax.random.split(key)
                delivered, collided = contend(k_slot, transmit, self.channel_slots, 0.0)
                link_ok, state = self._link(k_link, carry)
                delivered &= link_ok
                updates.update(state)
        else:
            transmit = jnp.ones((C,), bool)
            if self.report_period > 1:
                transmit = jnp.mod(carry['belief'].step + jnp.arange(C), self.report_period) == 0
            collided = jnp.zeros((C,), bool)
            if self._default_link():
                link_ok = jnp.ones((C,), bool)
                if self.message_drop > 0.0:
                    link_ok = jnp.logical_not(jax.random.bernoulli(key, self.message_drop, (C,)))
            else:
                link_ok, state = self._link(key, carry)
                updates.update(state)
            delivered = transmit & link_ok
        updates['link_ok'] = link_ok

        shared = (fov, seen, positions, delivered)
        if self.message_delay > 0 and 'delay_buffer' in carry:
            buffer = carry['delay_buffer']
            updates['delay_buffer'] = jax.tree_util.tree_map(
                lambda b, x: jnp.concatenate([b[1:], x[None].astype(b.dtype)]), buffer, shared
            )
            shared = jax.tree_util.tree_map(lambda b: b[0], buffer)
        if self.communication == 'team':
            fov, detected, position = pooled(*shared)
        elif self.communication == 'none':
            detected, position = seen, positions
        else:
            shared_fov, shared_detected, shared_position = pooled(*shared)
            fov = jnp.concatenate([shared_fov, fov])
            detected = jnp.concatenate([shared_detected, seen])
            position = jnp.concatenate([shared_position, positions])
        sent = transmit & (self.communication != 'none')
        updates['traffic'] = jnp.where(
            sent, DETECTION_FIXED_BYTES + DETECTION_BYTES * jnp.sum(seen, axis=-1).astype(jnp.float32), 0.0
        )
        channel = (transmit, delivered, collided)
        return fov.reshape(-1, G, G), detected, position.astype(jnp.float32), channel, updates

    # Policy API -------------------------------------------------------------------------------- #

    def init(self, key, observation):
        me = _camera_self(observation)
        obstacle_location, obstacle_radius = self._obstacles(observation)
        free = free_space(self.grid, obstacle_location, obstacle_radius, me['location'], self.camera_radius)
        atlas = build_atlas(
            self.grid, me['location'], me['max_sight_range'], obstacle_location, obstacle_radius, self.samples_per_axis
        )
        belief = self.filter.init(jnp.asarray(self.prior), free, self.num_sets, self.num_targets)
        carry = {
            'atlas': atlas,
            'free': free,
            'belief': belief,
            'obstacles': (obstacle_location, obstacle_radius),
            'last_delivered': jnp.zeros(self.num_cameras, jnp.int32),
            'channel': (
                jnp.zeros(self.num_cameras, bool),
                jnp.zeros(self.num_cameras, bool),
                jnp.zeros(self.num_cameras, bool),
            ),
        }
        fov, detected, position, _, updates = self._measure(key, carry, observation)
        carry['belief'] = self.filter.update(belief, free, fov, detected, position)
        carry.update(updates)
        if self.message_delay > 0:  # packets in flight to the team memory (none delivered yet)
            d, C = self.message_delay, self.num_cameras
            carry['delay_buffer'] = (
                jnp.zeros((d, C, atlas.distance.shape[1]), jnp.float32),
                jnp.zeros((d, C, self.num_targets), bool),
                jnp.zeros((d, C, self.num_targets, 2), jnp.float32),
                jnp.zeros((d, C), bool),
            )
        if self.coordination == 'published':
            level, orient = self._lattice_pose(me['orientation'], me['viewing_angle'])
            size = self.horizons.size
            carry['published'] = {
                'levels': jnp.broadcast_to(level[:, None], (self.num_cameras, size)).astype(jnp.int32),
                'orients': jnp.broadcast_to(orient[:, None], (self.num_cameras, size)).astype(jnp.int32),
                'age': jnp.zeros(self.num_cameras, jnp.int32),
            }
            carry['planned'] = {'levels': carry['published']['levels'], 'orients': carry['published']['orients']}
            if self.intent_delay > 0:  # intents in flight
                d = self.intent_delay
                carry['intent_queue'] = {
                    'mask': jnp.zeros((d, self.num_cameras), bool),
                    'levels': jnp.zeros((d, self.num_cameras, size), jnp.int32),
                    'orients': jnp.zeros((d, self.num_cameras, size), jnp.int32),
                }
        return carry

    def _lattice_pose(self, orientation, viewing):
        level = jnp.argmin(jnp.abs(jnp.asarray(self.options.viewing_angles)[None, :] - viewing[:, None]), axis=-1)
        orient = jnp.mod(jnp.round((orientation + 180.0) / 5.0), NUM_ORIENTATIONS).astype(jnp.int32)
        return level.astype(jnp.int32), orient

    def plan_published(self, carry, observation):
        """Best response of every camera to the trajectories its teammates last published.

        Returns the new trajectories (levels, orientations), each (C, H), and their values (C,).
        """

        me = _camera_self(observation)
        horizons = self.horizons
        size = horizons.size
        options = self.options
        counts = self._planning_counts(carry)  # (C, H, N)
        weight = jnp.asarray(self.discount ** horizons, jnp.float32)[:, None, None]
        atlas = carry['atlas']
        published = carry['published']
        shift = jnp.minimum(published['age'] // self.stride, size - 1)
        index = jnp.minimum(jnp.arange(size)[None, :] + shift[:, None], size - 1)  # (C, H)
        levels = jnp.take_along_axis(published['levels'], index, axis=1)
        orients = jnp.take_along_axis(published['orients'], index, axis=1)

        def cover(level, orient, distance, angle, visible):
            goal_orientation, goal_viewing, goal_sight = options.pose(level, orient)  # (H,)
            return (
                visible[None]
                & (distance[None] <= goal_sight[:, None, None])
                & (2.0 * jnp.abs(normalize_angle(angle[None] - goal_orientation[:, None, None])) <= goal_viewing[:, None, None])
            ).astype(jnp.float32).mean(-1)  # (H, N)

        covered = jax.vmap(cover)(levels, orients, atlas.distance, atlas.angle, atlas.visible)  # (C, H, N)
        terminal = carry.get(
            'terminal_values', jnp.zeros((self.num_cameras, options.num_levels, NUM_ORIENTATIONS), jnp.float32)
        )

        def respond(index, camera_counts, distance, angle, visible, orientation, viewing, terminal):
            others = jnp.prod(jnp.where((jnp.arange(self.num_cameras) == index)[:, None, None], 1.0, 1.0 - covered), axis=0)
            if self.communication == 'none':
                others = jnp.ones_like(others)
            histograms = options.histograms(camera_counts * others, distance, angle, visible)
            mass = options.sector_mass(histograms) * weight
            mass = mass.at[-1].add(terminal)
            return options.trajectory(mass, orientation, viewing, self.stride)

        return jax.vmap(respond)(
            jnp.arange(self.num_cameras), counts, atlas.distance, atlas.angle, atlas.visible,
            me['orientation'], me['viewing_angle'], terminal,
        )

    def _planning_counts(self, carry):
        """(C, H, N) expected target counts per cell that each camera plans with."""

        belief = carry['belief']
        horizons = self.horizons
        if self.use_forecast:
            forecast = self.filter.forecast(belief, carry['free'], horizons)  # (K, H, T, G, G)
        else:
            forecast = jnp.broadcast_to(
                belief.belief[:, None], (self.num_sets, horizons.size) + belief.belief.shape[1:]
            )
        if self.communication == 'hybrid':
            # Most recent information wins, per target.
            private_newer = belief.last_seen[1:] > belief.last_seen[:1]  # (C, T)
            forecast = jnp.where(private_newer[:, None, :, None, None], forecast[1:], forecast[:1])
        counts = self.filter.expected_counts(forecast)  # (K or C, H, N)
        if self.communication == 'team':
            counts = jnp.broadcast_to(counts, (self.num_cameras,) + counts.shape[1:])
        return counts

    def plan(self, carry, observation):
        """Top-``num_candidates`` options of every camera: (level, orientation, reach, value), each (C, K)."""

        return self.plan_with_maps(carry, observation)[:4]

    def plan_with_maps(self, carry, observation):
        """:meth:`plan` plus every camera's (H, L, O) option values after the earlier cameras' choices.

        Cameras are processed sequentially; the coverage of each camera's best option is
        discounted for the cameras after it (sequential greedy)."""

        me = _camera_self(observation)
        horizons = self.horizons
        counts = self._planning_counts(carry)  # (C, H, N)

        window = max(1, self.window // self.stride)
        if self.objective in ('path', 'cumulative', 'trajectory'):
            max_index = horizons.size - 1
            weight = jnp.asarray(self.discount ** horizons, jnp.float32)[:, None, None]
        else:
            max_index = horizons.size - window
            weight = jnp.ones((horizons.size, 1, 1), jnp.float32)
        atlas = carry['atlas']
        options_shape = (self.options.num_levels, NUM_ORIENTATIONS)
        coordinated = self.communication != 'none' or self.implicit
        top_k = max(1, int(self.num_candidates))
        options = self.options

        def trajectory(mass, distance, angle, visible, orientation, viewing):
            # Exact best rate-limited pose trajectory over the horizons: the camera heads for its
            # next waypoint; ``covered`` (H, N) is what its whole trajectory covers.
            path_levels, path_orients, value = options.trajectory(mass, orientation, viewing, self.stride)
            waypoint = min(1, horizons.size - 1)
            levels = jnp.full((top_k,), path_levels[waypoint])
            orients = jnp.full((top_k,), path_orients[waypoint])
            values = jnp.full((top_k,), value / jnp.sum(weight))
            goal_orientation, goal_viewing, goal_sight = options.pose(path_levels, path_orients)  # (H,)
            covered = (
                visible[None]
                & (distance[None] <= goal_sight[:, None, None])
                & (2.0 * jnp.abs(normalize_angle(angle[None] - goal_orientation[:, None, None])) <= goal_viewing[:, None, None])
            ).astype(jnp.float32).mean(-1)  # (H, N)
            return (levels, orients, jnp.zeros((top_k,), jnp.int32), values, mass), covered

        def trajectory_step(remaining, mass, distance, angle, visible, orientation, viewing):
            # Sequential pass: each trajectory is discounted for the cameras after it.
            outputs, covered = trajectory(mass, distance, angle, visible, orientation, viewing)
            if coordinated:
                remaining = remaining * (1.0 - covered)
            return remaining, outputs + (covered,)

        def camera_step(remaining, inputs):
            # One camera's turn of the sequential pass (a scan keeps the XLA program small).
            camera_counts, distance, angle, visible, orientation, viewing, terminal = inputs
            weights = camera_counts * remaining if coordinated else camera_counts
            histograms = options.histograms(weights, distance, angle, visible)
            mass = options.sector_mass(histograms) * weight  # (H, L, O)
            mass = mass.at[-1].add(terminal)  # value beyond the horizon (zero unless provided)
            prefix = jnp.concatenate([jnp.zeros((1,) + mass.shape[1:]), jnp.cumsum(mass, axis=0)], axis=0)
            if self.objective == 'trajectory':
                return trajectory_step(remaining, mass, distance, angle, visible, orientation, viewing)
            reach = options.reach_steps(orientation, viewing)  # (L, O)
            start =jnp.minimum(-(-reach // self.stride), max_index) if self.reach_aware else jnp.zeros_like(reach)
            level_index = np.arange(mass.shape[1])[:, None]
            orient_index = np.arange(mass.shape[2])[None, :]
            if self.objective == 'path':
                along = mass  # reach_aware=False: as if the goal pose were reached at once
                if self.reach_aware:
                    path_level, path_orient = options.path_indices(orientation, viewing, horizons)
                    along = jnp.take_along_axis(
                        mass.reshape(mass.shape[0], -1),
                        (path_level * mass.shape[2] + path_orient).reshape(mass.shape[0], -1),
                        axis=1,
                    ).reshape(mass.shape)
                gain = along.sum(0) / jnp.sum(weight)
            elif self.objective == 'cumulative':
                gain = (prefix[-1] - prefix[start, level_index, orient_index]) / jnp.sum(weight)
            else:
                end = start + window
                gain = (prefix[end, level_index, orient_index] - prefix[start, level_index, orient_index]) / window
            gain = gain - self.reach_penalty * reach.astype(jnp.float32)
            values, indices = jax.lax.top_k(gain.reshape(-1), top_k)  # indices[0] is the greedy choice
            levels, orients = indices // mass.shape[2], indices % mass.shape[2]
            level, orient = levels[0], orients[0]

            if coordinated:
                goal_orientation, goal_viewing, goal_sight = options.pose(level, orient)
                covered = (
                    visible
                    & (distance <= goal_sight)
                    & (2.0 * jnp.abs(normalize_angle(angle - goal_orientation)) <= goal_viewing)
                ).astype(jnp.float32).mean(-1)  # (N,)
                after = (jnp.arange(horizons.size) >= start[level, orient])[:, None]
                remaining = remaining * (1.0 - jnp.where(after, covered[None, :], 0.0))
            return remaining, (levels, orients, reach[levels, orients], values, mass, jnp.zeros(()))

        # Optional (C, L, O) value of every option beyond the last horizon (e.g. a learned cost-to-go).
        terminal = carry.get('terminal_values', jnp.zeros((counts.shape[0],) + options_shape, jnp.float32))
        inputs = (counts, atlas.distance, atlas.angle, atlas.visible, me['orientation'], me['viewing_angle'], terminal)
        remaining = jnp.ones(counts.shape[1:], jnp.float32)  # (H, N) not yet covered by earlier cameras
        if self.implicit and self.communication == 'none':
            def imagined(viewer_counts):  # the team's sequential pass as seen from one camera's private belief
                return jax.lax.scan(camera_step, remaining, (jnp.broadcast_to(viewer_counts, counts.shape),) + inputs[1:])[1]

            own = jnp.arange(counts.shape[0])
            outputs = jax.tree_util.tree_map(lambda x: x[own, own], jax.vmap(imagined)(counts))
            levels, orientations, reaches, gains, maps, covered = outputs
        else:
            _, (levels, orientations, reaches, gains, maps, covered) = jax.lax.scan(camera_step, remaining, inputs)

        if self.objective == 'trajectory' and coordinated and self.coordination_rounds > 1:
            # Block-coordinate ascent: re-plan every camera against the others' current trajectories.
            def best_response(covered, inputs):
                index, camera_counts, distance, angle, visible, orientation, viewing, terminal = inputs
                others = jnp.prod(jnp.where((jnp.arange(covered.shape[0]) == index)[:, None, None], 1.0, 1.0 - covered), axis=0)
                histograms = options.histograms(camera_counts * others, distance, angle, visible)
                mass = options.sector_mass(histograms) * weight
                mass = mass.at[-1].add(terminal)
                outputs, mine = trajectory(mass, distance, angle, visible, orientation, viewing)
                return covered.at[index].set(mine), outputs

            def round_(covered, _):
                return jax.lax.scan(best_response, covered, (jnp.arange(counts.shape[0]),) + inputs)

            _, (levels, orientations, reaches, gains, maps) = jax.lax.scan(
                lambda c, x: round_(c, x), covered, None, length=self.coordination_rounds - 1
            )
            levels, orientations, reaches, gains, maps = (x[-1] for x in (levels, orientations, reaches, gains, maps))
        # each (C, K); column 0 is the sequential-greedy choice
        # maps: (C, H, L, O) value of every option at every horizon, after the earlier cameras' choices
        return levels, orientations, reaches, gains, maps

    # Per-agent post-processing ------------------------------------------------------------------ #

    def _refine(self, carry, me, action):
        """Correct each camera's planned action against the exact positions of detected targets."""

        belief = carry['belief']
        rotation, zoom = self.options.rotation_step, self.options.zooming_step
        offsets = np.array([[r, z] for r in (-0.5 * rotation, 0.0, 0.5 * rotation) for z in (-zoom, 0.0, zoom)])
        candidates = jnp.clip(
            action[:, None, :] + jnp.asarray(offsets, action.dtype)[None],
            jnp.asarray([-rotation, -zoom], action.dtype),
            jnp.asarray([rotation, zoom], action.dtype),
        )  # (C, A, 2)
        orientation = normalize_angle(me['orientation'][:, None] + candidates[..., 0])
        viewing = jnp.clip(
            me['viewing_angle'][:, None] + candidates[..., 1],
            self.options.min_viewing_angle,
            consts.MAX_CAMERA_VIEWING_ANGLE,
        )
        sight = jnp.sqrt(self.options.area_product / viewing)

        # Exact targets each camera knows about now, one step ahead.
        known = belief.last_seen == belief.step  # (K, T) detected in this step
        predicted = belief.last_position + jnp.where(belief.velocity_known[..., None], belief.velocity, 0.0)
        if self.communication == 'team':
            known = jnp.broadcast_to(known, (self.num_cameras,) + known.shape[1:])
            predicted = jnp.broadcast_to(predicted, (self.num_cameras,) + predicted.shape[1:])
        elif self.communication == 'hybrid':
            own = known[1:]
            predicted = jnp.where(own[..., None], predicted[1:], predicted[:1])
            known = own | known[:1]

        location = me['location']
        relative = predicted - location[:, None, :]  # (C, T, 2)
        distance = safe_norm(relative)
        bearing = arctan2_deg(relative[..., 1], relative[..., 0])
        obstacle_location, obstacle_radius = carry['obstacles']
        if self.num_obstacles > 0:
            visible = jax.vmap(
                lambda src, dst, dist: line_of_sight(src[None, None], dst[None], dist[None], obstacle_location, obstacle_radius)[0]
            )(location, predicted, distance)
        else:
            visible = jnp.ones(distance.shape, bool)

        def covers(theta, alpha, radius):  # (C, A) poses -> (C, A, T)
            inside = distance[:, None, :] <= radius[..., None]
            inside &= 2.0 * jnp.abs(normalize_angle(bearing[:, None, :] - theta[..., None])) <= alpha[..., None]
            return inside & visible[:, None, :] & known[:, None, :]

        cover = covers(orientation, viewing, sight).astype(jnp.float32)  # (C, A, T)
        planned = len(offsets) // 2  # the zero offset
        planned_cover = cover[:, planned, :]  # (C, T)
        others = jnp.sum(planned_cover, axis=0, keepdims=True) - planned_cover  # covered by other cameras
        value = jnp.where(others > 0, self.redundant_weight, 1.0)  # (C, T)
        score = jnp.sum(cover * value[:, None, :], axis=-1)  # (C, A), exact on any backend
        score = score.at[:, planned].add(1e-3)  # ties keep the plan
        best = jnp.argmax(score, axis=-1)
        return jnp.take_along_axis(candidates, best[:, None, None], axis=1)[:, 0]

    # Policy API: observe every environment step, decide when an action is needed ---------------- #

    ADVICE_DIM = 6

    def observe(self, key, carry, observation, transmit=None):
        """Belief time and measurement update (cheap; run it every environment step).

        ``transmit`` (C,) bool optionally decides which cameras send their packet this step.
        """

        belief = self.filter.predict(carry['belief'], carry['free'])
        carry = {**carry, 'belief': belief}
        fov, detected, position, channel, updates = self._measure(key, carry, observation, transmit)
        carry['belief'] = self.filter.update(belief, carry['free'], fov, detected, position)
        carry.update(updates)
        carry['channel'] = channel
        carry['last_delivered'] = jnp.where(channel[1], belief.step, carry['last_delivered'])
        return carry

    def decide(self, carry, observation):
        """Allocation and control for the current beliefs.

        Returns the (C, 2) action and (C, ADVICE_DIM) normalized advice features: planned rotation
        and zoom (in steps), goal orientation offset and goal viewing angle (/180), reach time
        (/36 steps) and the option value (expected covered targets per target).
        """

        me = _camera_self(observation)
        levels, orientations, reaches, gains = (x[:, 0] for x in self.plan(carry, observation))
        goal_orientation, goal_viewing, _ = self.options.pose(levels, orientations)
        action = self.options.action_towards(me['orientation'], me['viewing_angle'], goal_orientation, goal_viewing)
        if self.refine:
            action = self._refine(carry, me, action)
        advice = jnp.stack(
            [
                action[:, 0] / self.options.rotation_step,
                action[:, 1] / self.options.zooming_step,
                normalize_angle(goal_orientation - me['orientation']) / 180.0,
                goal_viewing / 180.0,
                reaches.astype(jnp.float32) / 36.0,
                gains / self.num_targets,
            ],
            axis=-1,
        )
        return action.astype(observation.dtype), advice.astype(observation.dtype)

    COMM_FEATURES = 4
    CANDIDATE_FEATURES = 4

    def candidates(self, carry, observation):
        """Top-K options and communication features for a learned high-level policy.

        Returns ``goals`` = (orientation, viewing angle), each (C, K), ``features`` (C, K, 4)
        (value per target, reach / 36, orientation offset / 180, viewing angle / 180), and
        ``comm`` (C, 4): fraction of targets whose latest detection is missing from the team
        memory (needs ``communication='hybrid'``), own detections per target, steps since the
        last delivered packet (/50, clipped) and the channel load of the last step.
        """

        me = _camera_self(observation)
        levels, orientations, reaches, gains = self.plan(carry, observation)
        goal_orientation, goal_viewing, _ = self.options.pose(levels, orientations)
        offset = normalize_angle(goal_orientation - me['orientation'][:, None])
        features = jnp.stack(
            [gains / self.num_targets, reaches.astype(jnp.float32) / 36.0, offset / 180.0, goal_viewing / 180.0],
            axis=-1,
        )
        return (goal_orientation, goal_viewing), features.astype(observation.dtype), self.comm_features(carry, observation)

    def comm_features(self, carry, observation):
        """(C, COMM_FEATURES) communication features (see :meth:`candidates`)."""

        belief = carry['belief']
        if self.communication == 'hybrid':
            stale = jnp.mean((belief.last_seen[1:] > belief.last_seen[:1]).astype(jnp.float32), axis=-1)
        else:
            stale = jnp.zeros(self.num_cameras, jnp.float32)
        _, seen = self._targets(observation)
        since = jnp.clip((belief.step - carry['last_delivered']).astype(jnp.float32) / 50.0, 0.0, 1.0)
        load = jnp.broadcast_to(jnp.mean(carry['channel'][0].astype(jnp.float32)), (self.num_cameras,))
        comm = jnp.stack([stale, jnp.mean(seen.astype(jnp.float32), axis=-1), since, load], axis=-1)
        return comm.astype(observation.dtype)

    def act(self, key, carry, observation):
        carry = self.observe(key, carry, observation)
        if self.coordination == 'published':
            return self.act_published(carry, observation)
        action, _ = self.decide(carry, observation)
        if self.communication != 'none':  # intents of the synchronous plan (rounds of trajectories, or an option)
            rounds = self.coordination_rounds if self.objective == 'trajectory' else 0
            extra = rounds * INTENT_BYTES if rounds else OPTION_INTENT_BYTES
            carry = {**carry, 'traffic': carry['traffic'] + extra}
        return action, carry

    def publication_deviation(self, published, levels, orients):
        """(C,) mean deviation (orientation steps, zoom levels) of new trajectories from the published ones."""

        size = self.horizons.size
        shift = jnp.minimum(published['age'] // self.stride, size - 1)
        index = jnp.minimum(jnp.arange(size)[None, :] + shift[:, None], size - 1)
        old_levels = jnp.take_along_axis(published['levels'], index, axis=1)
        old_orients = jnp.take_along_axis(published['orients'], index, axis=1)
        turn = jnp.abs(jnp.mod(orients - old_orients + NUM_ORIENTATIONS // 2, NUM_ORIENTATIONS) - NUM_ORIENTATIONS // 2)
        return jnp.mean(turn + jnp.abs(levels - old_levels), axis=-1)

    def act_published(self, carry, observation):
        """Plan against the published trajectories, move to the next waypoint, publish where requested."""

        me = _camera_self(observation)
        levels, orients, _ = self.plan_published(carry, observation)
        waypoint = min(1, self.horizons.size - 1)
        goal_orientation, goal_viewing, _ = self.options.pose(levels[:, waypoint], orients[:, waypoint])
        action = self.options.action_towards(me['orientation'], me['viewing_angle'], goal_orientation, goal_viewing)
        published = carry['published']
        if 'publish_mask' in carry:
            publish = carry['publish_mask']
        elif self.publish_threshold > 0.0:
            publish = self.publication_deviation(published, levels, orients) > self.publish_threshold
        else:
            step = carry['belief'].step
            publish = jnp.mod(step + jnp.arange(self.num_cameras), self.publish_period) == 0
        traffic = carry['traffic'] + jnp.where(publish, float(INTENT_BYTES), 0.0)
        planned = {'levels': levels, 'orients': orients}
        if self.intent_loss:  # intents travel on the same lossy link as the detection packets
            publish = publish & carry['link_ok']
        age = 0
        if self.intent_delay > 0:  # teammates receive the intents published ``intent_delay`` steps ago
            queue = carry['intent_queue']
            arriving = jax.tree_util.tree_map(lambda q: q[0], queue)
            sent = {'mask': publish, 'levels': levels, 'orients': orients}
            carry = {**carry, 'intent_queue': jax.tree_util.tree_map(
                lambda q, x: jnp.concatenate([q[1:], x[None].astype(q.dtype)]), queue, sent
            )}
            publish, levels, orients, age = arriving['mask'], arriving['levels'], arriving['orients'], self.intent_delay
        carry = {
            **carry,
            'traffic': traffic,
            'planned': planned,
            'published': {
                'levels': jnp.where(publish[:, None], levels, published['levels']),
                'orients': jnp.where(publish[:, None], orients, published['orients']),
                'age': jnp.where(publish, age, published['age'] + 1),
            },
        }
        return action.astype(observation.dtype), carry
