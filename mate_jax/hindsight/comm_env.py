"""Communication control for trajectory-level camera coordination: the MARL environment.

The low level is FORESIGHT-T with *published* intents (:attr:`ForesightCameraPolicy.coordination` =
``'published'``) and the hybrid memory: every step each camera best-responds to the trajectories its
teammates last published and moves towards its next waypoint. Every ``frame_skip`` steps (one decision)
each camera chooses two bits for the next decision period:

* send its detection packet in every step (over the slotted-ALOHA channel of :func:`channel.contend`);
* publish its newly planned trajectory in every step (an intent message on the control channel).

The team reward per step is the coverage rate minus the radio cost of the messages,
``coverage - packet_cost * packets / N - intent_cost * intents / N``.

*Hindsight.* Target trajectories are exogenous: they do not depend on the cameras. Every episode therefore
draws its environment and target randomness from one episode key, and the future target trajectory is
simulated once at the start of the episode. :meth:`global_state` can append the true target positions a few
steps ahead, so that a centralized critic is conditioned on the exogenous future (an input-driven baseline,
which keeps the policy gradient unbiased because the future does not depend on the actions).
"""

from typing import Any, Dict, Optional, Sequence

import jax
import jax.numpy as jnp
from flax import struct

from mate_jax.env import float_dtype
from mate_jax.foresight import ForesightCameraPolicy
from mate_jax.wrappers import SingleTeamEnv, _select


__all__ = ['CommState', 'CommEnv', 'COMM_RULES', 'comm_rule_actions']

FUTURE_OFFSETS = (5, 10, 20, 40)


@struct.dataclass
class CommState:
    env_state: Any
    opponent_carry: Any
    opponent_observation: jax.Array
    team_observation: jax.Array
    planner_carry: Any
    episode_key: jax.Array
    future: jax.Array  # (S + max offset, T, 2) target positions before every step of the episode
    step: jax.Array  # environment steps since the reset
    decisions: jax.Array
    episode_return: jax.Array
    episode_coverage: jax.Array
    returned_episode_return: jax.Array
    returned_episode_length: jax.Array
    returned_episode_coverage: jax.Array
    next_reset: Any = None


class CommEnv:
    """MARL environment: (send detections, publish intent) per camera and decision; see the module docstring."""

    def __init__(
        self,
        env,
        opponent: Optional[Any] = None,
        planner: Optional[ForesightCameraPolicy] = None,
        frame_skip: int = 5,
        horizon: int = 100,
        packet_cost: float = 0.05,
        intent_cost: float = 0.04,
        hindsight: bool = True,
        future_offsets: Sequence[int] = FUTURE_OFFSETS,
        cache_resets: bool = True,
        replan_every: Optional[int] = None,
        lite: bool = False,
    ):
        self.base = SingleTeamEnv(env, team='camera', opponent=opponent, frame_skip=1, horizon=None, cache_resets=False)
        self.env = env
        self.opponent = self.base.opponent
        if planner is None:
            from mate_jax.hindsight.student import trajectory_planner  # pylint: disable=import-outside-toplevel

            from mate_jax.hindsight.student import LITE_TRAJECTORY  # pylint: disable=import-outside-toplevel

            planner = trajectory_planner(
                env, coordination='published', communication='hybrid', **(LITE_TRAJECTORY if lite else {})
            )
        assert planner.coordination == 'published' and planner.communication == 'hybrid'
        self.planner = planner
        self.transform = self.base.transform
        self.frame_skip = int(frame_skip)
        # Plan (and publish) every ``replan_every`` steps and follow the planned trajectory in between.
        self.replan_every = int(replan_every or 1)
        self.horizon = int(horizon)
        self.episode_steps = self.horizon * self.frame_skip
        self.packet_cost = float(packet_cost)
        self.intent_cost = float(intent_cost)
        self.hindsight = bool(hindsight)
        self.future_offsets = tuple(int(o) for o in future_offsets)
        self.cache_resets = bool(cache_resets)
        self.num_agents = env.num_cameras
        self.num_actions = 4  # bit 0: send detections, bit 1: publish the trajectory
        self.phases = 4  # length of the slot cycle that cameras can see (a synchronized decision counter)
        self.observation_dim = self.transform.dim + planner.COMM_FEATURES + 4 + self.phases
        self.future_dim = len(self.future_offsets) * env.num_targets * 2
        self.state_dim = env.state_dim + self.future_dim
        self.action_dim = 2
        self.action_scale = self.base.action_scale

    # Helpers ----------------------------------------------------------------------------------- #

    @staticmethod
    def decode(actions):
        actions = actions.astype(jnp.int32)
        return (actions & 1) > 0, (actions & 2) > 0

    @staticmethod
    def encode(transmit, publish):
        return transmit.astype(jnp.int32) + 2 * publish.astype(jnp.int32)

    def scale_actions(self, actions):
        return actions

    def _step_keys(self, episode_key, t):
        k = jax.random.fold_in(episode_key, t)
        k_target, k_env = jax.random.split(k)
        return k_target, k_env

    def _simulate_targets(self, episode_key, env_state, opponent_carry, opponent_obs):
        """Target positions before every step of the episode (plus the largest future offset)."""

        idle = jnp.zeros((self.num_agents, 2), jnp.float32)
        length = self.episode_steps + max(self.future_offsets)

        def body(carry, t):
            env_state, opponent_carry, opponent_obs = carry
            k_target, k_env = self._step_keys(episode_key, t)
            location = env_state.target_location
            target_action, opponent_carry = self.opponent.act(k_target, opponent_carry, opponent_obs)
            timestep, env_state = self.env.step(k_env, env_state, idle, target_action)
            return (env_state, opponent_carry, timestep.target_observation), location

        _, future = jax.lax.scan(body, (env_state, opponent_carry, opponent_obs), jnp.arange(length))
        return future

    def _follow(self, planner_carry, team_obs, since):
        """Rate-limited action towards the planned trajectory ``since`` steps after it was planned."""

        from mate_jax.agents import _camera_self  # pylint: disable=import-outside-toplevel

        planner = self.planner
        me = _camera_self(team_obs)
        planned = planner_carry['planned']
        index = jnp.minimum(since // planner.stride + 1, planned['levels'].shape[1] - 1)
        goal_orientation, goal_viewing, _ = planner.options.pose(planned['levels'][:, index], planned['orients'][:, index])
        action = planner.options.action_towards(me['orientation'], me['viewing_angle'], goal_orientation, goal_viewing)
        return action.astype(team_obs.dtype)

    def _future_features(self, state: CommState):
        offsets = jnp.asarray(self.future_offsets)
        index = jnp.minimum(state.step + offsets, state.future.shape[0] - 1)
        future = state.future[index] / 1000.0  # (F, T, 2), terrain coordinates in [-1, 1]
        features = future.reshape(-1)
        return features if self.hindsight else jnp.zeros_like(features)

    def global_state(self, env_state, state: Optional[CommState] = None):
        base = self.base.global_state(env_state)
        future = jnp.zeros((self.future_dim,), base.dtype) if state is None else self._future_features(state)
        return jnp.concatenate([base, future.astype(base.dtype)])

    def _observation(self, team_observation, planner_carry, last_action, decisions):
        planner = self.planner
        comm = planner.comm_features(planner_carry, team_observation)
        published = planner_carry['published']
        planned = planner_carry['planned']
        deviation = planner.publication_deviation(published, planned['levels'], planned['orients'])
        transmit, publish = self.decode(last_action)
        extra = jnp.stack(
            [
                jnp.minimum(deviation / 10.0, 1.0),
                jnp.minimum(published['age'].astype(jnp.float32) / 50.0, 1.0),
                transmit.astype(jnp.float32),
                publish.astype(jnp.float32),
            ],
            axis=-1,
        )
        # Phase of every camera in the slot cycle (staggered by camera), as in a TDMA schedule.
        phase = jax.nn.one_hot(jnp.mod(decisions + jnp.arange(self.num_agents), self.phases), self.phases)
        observation = self.transform(team_observation)
        return jnp.concatenate(
            [observation, comm.astype(observation.dtype), extra.astype(observation.dtype), phase.astype(observation.dtype)],
            axis=-1,
        )

    def _sample_start(self, key):
        key_env, key_opponent, key_planner, key_episode = jax.random.split(key, 4)
        (camera_obs, target_obs), env_state = self.env.reset(key_env)
        planner_carry = self.planner.init(key_planner, camera_obs)
        opponent_carry = self.opponent.init(key_opponent, target_obs)
        future = self._simulate_targets(key_episode, env_state, opponent_carry, target_obs)
        return env_state, camera_obs, target_obs, opponent_carry, planner_carry, key_episode, future

    def refresh_resets(self, key, state: CommState) -> CommState:
        if not self.cache_resets:
            return state
        return state.replace(next_reset=self._sample_start(key))

    # API ---------------------------------------------------------------------------------------- #

    def reset(self, key):
        key_start, key_cache = jax.random.split(key)
        env_state, team_obs, opponent_obs, opponent_carry, planner_carry, episode_key, future = self._sample_start(key_start)
        fdt = float_dtype()
        zero = jnp.zeros((), fdt)
        state = CommState(
            env_state=env_state, opponent_carry=opponent_carry, opponent_observation=opponent_obs,
            team_observation=team_obs, planner_carry=planner_carry, episode_key=episode_key, future=future,
            step=jnp.zeros((), jnp.int32), decisions=jnp.zeros((), jnp.int32), episode_return=zero,
            episode_coverage=zero, returned_episode_return=zero, returned_episode_length=jnp.zeros((), jnp.int32),
            returned_episode_coverage=zero,
        )  # fmt: skip
        state = self.refresh_resets(key_cache, state)
        last = jnp.full((self.num_agents,), 3, jnp.int32)
        return self._observation(team_obs, planner_carry, last, state.decisions), self.global_state(env_state, state), state

    def step(self, key, state: CommState, actions):
        fdt = float_dtype()
        key_reset, key_skip = jax.random.split(key)
        transmit, publish = self.decode(actions)
        num = self.num_agents

        def substep(carry, inputs):
            k, index = inputs
            env_state, opponent_carry, opponent_obs, team_obs, planner_carry, t, totals = carry
            k_planner, k_channel = jax.random.split(k)
            k_target, k_env = self._step_keys(state.episode_key, t)
            replan = jnp.mod(index, self.replan_every) == 0
            sent_intent = publish & replan

            def plan(planner_carry):
                planner_carry = {**planner_carry, 'publish_mask': sent_intent}
                action, planner_carry = self.planner.act_published(planner_carry, team_obs)
                return action, {key_: value for key_, value in planner_carry.items() if key_ != 'publish_mask'}

            def follow(planner_carry):
                return self._follow(planner_carry, team_obs, jnp.mod(index, self.replan_every)), {
                    **planner_carry,
                    'published': {**planner_carry['published'], 'age': planner_carry['published']['age'] + 1},
                }

            camera_action, planner_carry = jax.lax.cond(replan, plan, follow, planner_carry)
            target_action, opponent_carry = self.opponent.act(k_target, opponent_carry, opponent_obs)
            timestep, env_state = self.env.step(k_env, env_state, camera_action, target_action)
            planner_carry = self.planner.observe(k_channel, planner_carry, timestep.camera_observation, transmit)
            sent, delivered, collided = planner_carry['channel']
            effort = 0.5 * (
                jnp.abs(camera_action[:, 0]) / self.planner.options.rotation_step
                + jnp.abs(camera_action[:, 1]) / self.planner.options.zooming_step
            )
            coverage = timestep.info['coverage_rate'].astype(fdt)
            packets = jnp.mean(sent.astype(fdt))
            intents = jnp.mean(sent_intent.astype(fdt))
            reward = coverage - self.packet_cost * packets - self.intent_cost * intents
            totals = {
                'reward': totals['reward'] + reward,
                'coverage_rate': totals['coverage_rate'] + coverage,
                'real_coverage_rate': totals['real_coverage_rate'] + timestep.info['real_coverage_rate'],
                'sent': totals['sent'] + jnp.sum(sent.astype(fdt)),
                'delivered': totals['delivered'] + jnp.sum(delivered.astype(fdt)),
                'collided': totals['collided'] + jnp.sum(collided.astype(fdt)),
                'intents': totals['intents'] + intents,
                'actuation': totals['actuation'] + jnp.mean(effort.astype(fdt)),
            }
            carry = (env_state, opponent_carry, timestep.target_observation, timestep.camera_observation, planner_carry, t + 1, totals)
            return carry, None

        zero = jnp.zeros((), fdt)
        totals = {
            'reward': zero, 'coverage_rate': zero, 'real_coverage_rate': zero, 'sent': zero, 'delivered': zero,
            'collided': zero, 'intents': zero, 'actuation': zero,
        }  # fmt: skip
        carry = (
            state.env_state, state.opponent_carry, state.opponent_observation, state.team_observation,
            state.planner_carry, state.step, totals,
        )  # fmt: skip
        carry, _ = jax.lax.scan(substep, carry, (jax.random.split(key_skip, self.frame_skip), jnp.arange(self.frame_skip)))
        env_state, opponent_carry, opponent_obs, team_obs, planner_carry, step, totals = carry

        skip = jnp.asarray(self.frame_skip, fdt)
        decisions = state.decisions + 1
        done = decisions >= self.horizon
        coverage = totals['coverage_rate'] / skip
        episode_return = state.episode_return + totals['reward']
        episode_coverage = state.episode_coverage + coverage

        start = state.next_reset if self.cache_resets else self._sample_start(key_reset)
        r_env_state, r_team_obs, r_opponent_obs, r_opponent_carry, r_planner_carry, r_episode_key, r_future = start
        new_state = CommState(
            env_state=_select(done, r_env_state, env_state),
            opponent_carry=_select(done, r_opponent_carry, opponent_carry),
            opponent_observation=jnp.where(done, r_opponent_obs, opponent_obs),
            team_observation=jnp.where(done, r_team_obs, team_obs),
            planner_carry=_select(done, r_planner_carry, planner_carry),
            episode_key=jnp.where(done, r_episode_key, state.episode_key),
            future=jnp.where(done, r_future, state.future),
            step=jnp.where(done, 0, step),
            decisions=jnp.where(done, 0, decisions),
            episode_return=jnp.where(done, 0.0, episode_return),
            episode_coverage=jnp.where(done, 0.0, episode_coverage),
            returned_episode_return=jnp.where(done, episode_return, state.returned_episode_return),
            returned_episode_length=jnp.where(done, decisions, state.returned_episode_length),
            returned_episode_coverage=jnp.where(
                done, episode_coverage / decisions.astype(fdt), state.returned_episode_coverage
            ),
            next_reset=state.next_reset,
        )
        sent = totals['sent']
        info = {
            'coverage_rate': coverage,
            'real_coverage_rate': totals['real_coverage_rate'] / skip,
            'tx_rate': sent / (skip * num),
            'intent_rate': totals['intents'] / skip,
            'delivery_ratio': jnp.where(sent > 0, totals['delivered'] / jnp.maximum(sent, 1.0), 1.0),
            'collision_rate': jnp.where(sent > 0, totals['collided'] / jnp.maximum(sent, 1.0), 0.0),
            'actuation': totals['actuation'] / skip,
            'terminated': jnp.zeros((), bool),
            'truncated': done,
            'returned_episode': done,
            'returned_episode_return': new_state.returned_episode_return,
            'returned_episode_length': new_state.returned_episode_length,
            'returned_episode_coverage': new_state.returned_episode_coverage,
            'returned_episode_delivered': jnp.zeros((), jnp.int32),
        }
        rewards = jnp.full((num,), totals['reward'], fdt)
        last = jnp.where(done, 3, actions.astype(jnp.int32))
        observation = self._observation(new_state.team_observation, new_state.planner_carry, last, new_state.decisions)
        return observation, self.global_state(new_state.env_state, new_state), rewards, done, info, new_state


# Rule-based communication (baselines) ------------------------------------------------------------ #

COMM_RULES = ('always', 'never', 'periodic-4', 'event', 'event-slot')


def comm_rule_actions(env: CommEnv, rule: str, state: CommState):
    """(C,) actions of a fixed rule: always / never / periodic-k (both messages every k-th decision,
    staggered) / event (detections when the team memory lacks the camera's latest detection, intents when
    the plan deviates from the published trajectory by more than one orientation step on average) /
    event-slot (event, but only in the camera's own phase of the 4-decision cycle)."""

    num = env.num_agents
    if rule == 'always':
        transmit = publish = jnp.ones(num, bool)
    elif rule == 'never':
        transmit = publish = jnp.zeros(num, bool)
    elif rule.startswith('periodic'):
        period = int(rule.split('-')[1])
        transmit = publish = jnp.mod(state.decisions + jnp.arange(num), period) == 0
    elif rule in ('event', 'event-slot'):
        planner, carry = env.planner, state.planner_carry
        comm = planner.comm_features(carry, state.team_observation)
        transmit = (comm[:, 0] > 0.0) | (comm[:, 2] >= 1.0)
        planned, published = carry['planned'], carry['published']
        publish = planner.publication_deviation(published, planned['levels'], planned['orients']) > 1.0
        if rule == 'event-slot':  # only in the camera's own phase of the cycle (collision-free, as periodic)
            own = jnp.mod(state.decisions + jnp.arange(num), env.phases) == 0
            transmit, publish = transmit & own, publish & own
    else:
        raise ValueError(f'Unknown communication rule {rule!r}.')
    return env.encode(transmit, publish)


def evaluate_comm(env: CommEnv, select, num_envs: int = 16, num_steps: Optional[int] = None, seed: int = 1) -> Dict[str, float]:
    """Evaluate ``select(key, observation, state) -> actions`` over ``num_steps`` decisions."""

    num_steps = int(num_steps or env.horizon)

    def run(key):
        k_reset, k_loop = jax.random.split(key)
        obs, _, state = jax.vmap(env.reset)(jax.random.split(k_reset, num_envs))

        def body(carry, k):
            obs, state = carry
            actions = select(jax.random.fold_in(k, 1), obs, state)
            obs, _, reward, _, info, state = jax.vmap(env.step)(jax.random.split(k, num_envs), state, actions)
            return (obs, state), (
                reward[:, 0], info['coverage_rate'], info['tx_rate'], info['intent_rate'], info['collision_rate'],
                info['actuation'],
            )

        _, out = jax.lax.scan(body, (obs, state), jax.random.split(k_loop, num_steps))
        return [x.mean() for x in out]

    reward, coverage, tx_rate, intent_rate, collision, actuation = jax.jit(run)(jax.random.PRNGKey(seed))
    return {
        'reward': float(reward) / env.frame_skip, 'coverage_rate': float(coverage), 'tx_rate': float(tx_rate),
        'intent_rate': float(intent_rate), 'collision_rate': float(collision), 'actuation': float(actuation),
    }
