"""RiskQ, QFree and MACPF for :class:`mate_jax.value_marl.ValueTrainer`.

Each algorithm provides ``init(k_agent, k_mixer) -> params``, ``scores(params, obs) -> (..., N, A)`` (greedy
action selection and evaluation), optionally ``collect(params, obs, key, epsilon) -> actions`` and
``loss(params, target_params, batch) -> (loss, aux)``. Agent networks are feed-forward (the observations of
MATE-JAX are not recurrent here) and the replay buffer holds transitions with one-step targets; the original
implementations use GRU agents, episode buffers and TD(lambda). Other details follow the papers and the
official code (see the comments).

* RiskQ (Shen et al., NeurIPS 2023): quantile utilities (32 quantiles, QR-DQN, non-crossing parametrization
  of Zhou et al. 2020 instead of sorting the outputs); joint quantiles
  theta_jt(w) = |w(s)| sum_i lambda_i theta_i(w) + c(s) with attention weights lambda (official mixer);
  greedy actions on the Wang distortion (alpha = 0.75); quantile-regression Huber loss.
* QFree (Wang et al., 2023/2026): V and normalized A heads, the QPLEX transformation
  (w_i(s) > 0, b_i(s)), unconstrained mixers Q_tot = f_v(V) + f_a(A), and the advantage-IGM penalties
  v1 A_tot(s, a*)^2 + v2 max(A_tot(s, a), 0)^2.
* MACPF (Wang, Ye and Lu, ICLR 2023): FOP-style maximum-entropy factorization with dependent policies
  pi_i(a_i | s, a_<i) = softmax(logits_ind + b_i(a_<i)) for data collection and independent policies for
  execution; critics Q_i = Q_ind + c_i(a_<i), mixed by Q_jt = sum_i k_i(s) Q_i + V(s), k_i >= 0.
"""

from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm

from mate_jax.models import MLP


# Shared helpers --------------------------------------------------------------------------------- #


def _state_mlp(x, embed, out):
    return nn.Dense(out)(nn.relu(nn.Dense(embed)(x)))


# RiskQ ----------------------------------------------------------------------------------------- #


class QuantileAgent(nn.Module):
    num_actions: int
    num_quantiles: int
    hiddens: Sequence[int]

    @nn.compact
    def __call__(self, x):
        # Non-crossing quantiles (Zhou et al., 2020): the lowest quantile plus cumulative positive increments,
        # sorted by construction (sorting the outputs of all actions dominated the training time).
        out = nn.Dense(self.num_actions * self.num_quantiles)(MLP(tuple(self.hiddens), 'relu')(x))
        out = out.reshape(x.shape[:-1] + (self.num_actions, self.num_quantiles))
        increments = jnp.concatenate([jnp.zeros_like(out[..., :1]), jax.nn.softplus(out[..., 1:])], axis=-1)
        return out[..., :1] + jnp.cumsum(increments, axis=-1)


class RiskQMixer(nn.Module):
    """theta_jt(w) = |w(s)| sum_i softmax_i(q(s) . k(o_i) / 4) theta_i(w) + c(s) (official RiskQ mixer)."""

    @nn.compact
    def __call__(self, theta, state, obs):  # theta (..., N, K), state (..., S), obs (..., N, D)
        query = _state_mlp(state, 32, 16)  # (..., 16)
        keys = nn.Dense(16)(obs)  # (..., N, 16)
        weights = jax.nn.softmax(jnp.einsum('...d,...nd->...n', query, keys) / 4.0, axis=-1)
        scale = jnp.abs(_state_mlp(state, 32, 1))  # (..., 1)
        bias = _state_mlp(state, 16, 1)
        return scale * jnp.einsum('...n,...nk->...k', weights, theta) + bias


class RiskQ:
    num_quantiles = 32
    wang_alpha = 0.75

    def __init__(self, trainer):
        self.t = trainer
        self.agent = QuantileAgent(trainer.num_actions, self.num_quantiles, tuple(trainer.config.hiddens))
        self.mixer = RiskQMixer()
        taus = (2 * np.arange(self.num_quantiles) + 1) / (2.0 * self.num_quantiles)
        self.taus = jnp.asarray(taus, jnp.float32)
        z = norm.ppf(taus)
        weight = np.clip(norm.pdf(z + self.wang_alpha) / norm.pdf(z), 0.0, 5.0)  # Wang distortion g'(tau)
        self.distortion = jnp.asarray(weight / self.num_quantiles, jnp.float32)

    def init(self, k_agent, k_mixer):
        t = self.t
        obs = jnp.zeros((1, t.num_agents, t.team_env.observation_dim))
        agent = self.agent.init(k_agent, t.agent_input(obs))
        mixer = self.mixer.init(k_mixer, jnp.zeros((1, t.num_agents, self.num_quantiles)), jnp.zeros((1, t.state_dim)),
                                jnp.zeros((1, t.num_agents, t.obs_dim)))
        return {'agent': agent, 'mixer': mixer}

    def quantiles(self, params, obs):
        return self.agent.apply(params['agent'], self.t.agent_input(obs))  # (..., N, A, K)

    def scores(self, params, obs):
        return jnp.einsum('...k,k->...', self.quantiles(params, obs), self.distortion)  # risk-distorted values

    def loss(self, params, target_params, batch):
        t, cfg = self.t, self.t.config
        theta = self.quantiles(params, batch['obs'])
        chosen = jnp.take_along_axis(theta, batch['actions'][..., None, None], axis=-2)[..., 0, :]  # (B, N, K)
        joint = self.mixer.apply(params['mixer'], chosen, batch['state'], t.agent_input(batch['obs']))  # (B, K)
        next_actions = jnp.argmax(self.scores(params, batch['next_obs']), axis=-1)
        next_theta = self.quantiles(target_params, batch['next_obs'])
        next_chosen = jnp.take_along_axis(next_theta, next_actions[..., None, None], axis=-2)[..., 0, :]
        next_joint = self.mixer.apply(target_params['mixer'], next_chosen, batch['next_state'], t.agent_input(batch['next_obs']))
        target = batch['reward'][:, None] + cfg.gamma * (1.0 - batch['terminal'][:, None].astype(jnp.float32)) * next_joint
        target = jax.lax.stop_gradient(target)
        u = target[:, None, :] - joint[:, :, None]  # (B, K_pred, K_target)
        huber = jnp.where(jnp.abs(u) <= 1.0, 0.5 * u**2, jnp.abs(u) - 0.5)
        weight = jnp.abs(self.taus[None, :, None] - (u < 0).astype(jnp.float32))
        loss = (weight * huber).mean(-1).sum(-1).mean()
        return loss, {'q_tot': jnp.mean(joint)}


# QFree ----------------------------------------------------------------------------------------- #


class DuelingAgent(nn.Module):
    num_actions: int
    hiddens: Sequence[int]

    @nn.compact
    def __call__(self, x):
        h = MLP(tuple(self.hiddens), 'relu')(x)
        value = nn.Dense(1)(h)[..., 0]
        advantage = nn.Dense(self.num_actions)(h)
        return value, advantage - advantage.max(-1, keepdims=True)


class QFreeMixer(nn.Module):
    num_agents: int
    embed: int = 64

    @nn.compact
    def __call__(self, values, advantages, state):  # values, advantages: (..., N)
        w = jnp.abs(_state_mlp(state, self.embed, self.num_agents)) + 1e-10  # QPLEX transformation
        b = _state_mlp(state, self.embed, self.num_agents)
        v = w * values + b
        a = w * advantages
        f_v = nn.Dense(1)(nn.relu(nn.Dense(self.embed)(jnp.concatenate([v, state], -1))))[..., 0]
        f_a = nn.Dense(1)(nn.relu(nn.Dense(self.embed)(jnp.concatenate([a, state], -1))))[..., 0]
        return f_v + f_a, f_a


class QFree:
    v1 = 1.0
    v2 = 1.0

    def __init__(self, trainer):
        self.t = trainer
        self.agent = DuelingAgent(trainer.num_actions, tuple(trainer.config.hiddens))
        self.mixer = QFreeMixer(trainer.num_agents, trainer.config.mixer_embed)

    def init(self, k_agent, k_mixer):
        t = self.t
        agent = self.agent.init(k_agent, t.agent_input(jnp.zeros((1, t.num_agents, t.team_env.observation_dim))))
        zeros = jnp.zeros((1, t.num_agents))
        return {'agent': agent, 'mixer': self.mixer.init(k_mixer, zeros, zeros, jnp.zeros((1, t.state_dim)))}

    def heads(self, params, obs):
        return self.agent.apply(params['agent'], self.t.agent_input(obs))

    def scores(self, params, obs):
        value, advantage = self.heads(params, obs)
        return value[..., None] + advantage

    def loss(self, params, target_params, batch):
        cfg = self.t.config
        value, advantage = self.heads(params, batch['obs'])
        chosen = jnp.take_along_axis(advantage, batch['actions'][..., None], axis=-1)[..., 0]
        q_tot, a_tot = self.mixer.apply(params['mixer'], value, chosen, batch['state'])
        _, a_star = self.mixer.apply(params['mixer'], value, jnp.zeros_like(chosen), batch['state'])  # A_i(a*) = 0
        next_actions = jnp.argmax(self.scores(params, batch['next_obs']), axis=-1)
        next_value, next_advantage = self.heads(target_params, batch['next_obs'])
        next_chosen = jnp.take_along_axis(next_advantage, next_actions[..., None], axis=-1)[..., 0]
        next_tot, _ = self.mixer.apply(target_params['mixer'], next_value, next_chosen, batch['next_state'])
        target = batch['reward'] + cfg.gamma * (1.0 - batch['terminal'].astype(jnp.float32)) * next_tot
        td = jnp.mean((q_tot - jax.lax.stop_gradient(target)) ** 2)
        # Advantage-IGM (Theorem 1): A_tot(a*) = 0 and A_tot(a) <= 0 (the sign of the printed max/min is fixed).
        penalty = self.v1 * jnp.mean(a_star**2) + self.v2 * jnp.mean(jnp.maximum(a_tot, 0.0) ** 2)
        return td + penalty, {'q_tot': jnp.mean(q_tot)}


# MACPF ----------------------------------------------------------------------------------------- #


class ConditionalHead(nn.Module):
    """Independent output (from the agent's features) plus a correction from the actions of the earlier agents."""

    num_agents: int
    num_actions: int
    hiddens: Sequence[int]

    @nn.compact
    def __call__(self, x, previous):  # x (..., N, D); previous (..., N, N, A + N): masked one-hot (agent, action)
        h = MLP(tuple(self.hiddens), 'relu')(x)
        independent = nn.Dense(self.num_actions)(h)
        e = nn.elu(nn.Dense(64)(previous))  # (..., N, N, 64): e_j for every earlier agent j
        mask = previous[..., :1] * 0 + (jnp.abs(previous).sum(-1, keepdims=True) > 0)
        d = (e * mask).sum(-2) / self.num_agents  # (..., N, 64)
        correction = nn.Dense(self.num_actions)(jnp.concatenate([jax.lax.stop_gradient(h), d], -1))
        return independent, correction


class LinearMixer(nn.Module):
    num_agents: int
    embed: int = 64

    @nn.compact
    def __call__(self, q, state):
        k = jnp.abs(_state_mlp(state, self.embed, self.num_agents))
        return jnp.sum(k * q, -1) + _state_mlp(state, self.embed, 1)[..., 0], k


class MACPF:
    alpha_start, alpha_end = 0.5, 0.05

    def __init__(self, trainer):
        self.t = trainer
        n, a = trainer.num_agents, trainer.num_actions
        self.policy = ConditionalHead(n, a, tuple(trainer.config.hiddens))
        self.critic = ConditionalHead(n, a, tuple(trainer.config.hiddens))
        self.mixer = LinearMixer(n, trainer.config.mixer_embed)

    def alpha(self, fraction):
        return self.alpha_start + jnp.clip(fraction, 0.0, 1.0) * (self.alpha_end - self.alpha_start)

    def previous(self, actions):
        """(..., N) actions -> (..., N, N, N + A) features of the earlier agents j < i (zero otherwise)."""

        n, a = self.t.num_agents, self.t.num_actions
        feature = jnp.concatenate([jnp.broadcast_to(jnp.eye(n), actions.shape + (n,)), jax.nn.one_hot(actions, a)], -1)
        mask = jnp.tril(jnp.ones((n, n)), k=-1)  # [i, j] = 1 for j < i
        return mask[..., None] * feature[..., None, :, :]

    def init(self, k_agent, k_mixer):
        t = self.t
        x = t.agent_input(jnp.zeros((1, t.num_agents, t.team_env.observation_dim)))
        prev = self.previous(jnp.zeros((1, t.num_agents), jnp.int32))
        k1, k2 = jax.random.split(k_agent)
        return {'policy': self.policy.init(k1, x, prev), 'critic': self.critic.init(k2, x, prev),
                'mixer': self.mixer.init(k_mixer, jnp.zeros((1, t.num_agents)), jnp.zeros((1, t.state_dim)))}

    def scores(self, params, obs):  # independent policy logits (execution)
        x = self.t.agent_input(obs)
        prev = self.previous(jnp.zeros(obs.shape[:-1], jnp.int32))
        independent, _ = self.policy.apply(params['policy'], x, prev)
        return independent

    def sample_dependent(self, params, x, key):
        """Sample a joint action agent by agent from the dependent policy; returns actions and log-probs."""

        n = self.t.num_agents
        actions = jnp.zeros(x.shape[:-1], jnp.int32)
        logps = jnp.zeros(x.shape[:-1])
        for i in range(n):
            key, k = jax.random.split(key)
            independent, correction = self.policy.apply(params['policy'], x, self.previous(actions))
            logits = jax.lax.stop_gradient(independent) + correction
            choice = jax.random.categorical(k, logits[..., i, :])
            actions = actions.at[..., i].set(choice)
            logps = logps.at[..., i].set(jnp.take_along_axis(jax.nn.log_softmax(logits[..., i, :]), choice[..., None], -1)[..., 0])
        return actions, logps

    def collect(self, params, obs, key, epsilon):
        k_dep, k_explore, k_random = jax.random.split(key, 3)
        actions, _ = self.sample_dependent(params, self.t.agent_input(obs), k_dep)
        random_actions = jax.random.randint(k_random, actions.shape, 0, self.t.num_actions)
        return jnp.where(jax.random.uniform(k_explore, actions.shape) < epsilon, random_actions, actions)

    def _joint(self, params, x, state, actions, dependent):
        independent, correction = self.critic.apply(params['critic'], x, self.previous(actions))
        q = independent + (correction if dependent else 0.0)
        chosen = jnp.take_along_axis(q, actions[..., None], -1)[..., 0]
        q_jt, k = self.mixer.apply(params['mixer'], chosen, state)
        return q_jt, q, k

    def loss(self, params, target_params, batch):
        t, cfg = self.t, self.t.config
        alpha = self.alpha(batch['progress'])
        x, nx = t.agent_input(batch['obs']), t.agent_input(batch['next_obs'])
        total, aux = 0.0, {}
        for dependent in (False, True):
            # Next actions from the current policy (dependent or independent) and their log-probabilities.
            if dependent:
                next_actions, next_logp = self.sample_dependent(params, nx, batch['key'])
            else:
                logits = self.scores(params, batch['next_obs'])
                next_actions = jax.random.categorical(batch['key'], logits)
                next_logp = jnp.take_along_axis(jax.nn.log_softmax(logits), next_actions[..., None], -1)[..., 0]
            next_q, _, _ = self._joint(target_params, nx, batch['next_state'], next_actions, dependent)
            soft = next_q - alpha * jnp.sum(jax.lax.stop_gradient(next_logp), -1)
            target = batch['reward'] + cfg.gamma * (1.0 - batch['terminal'].astype(jnp.float32)) * soft
            q_jt, q, k = self._joint(params, x, batch['state'], batch['actions'], dependent)
            critic_loss = jnp.mean((q_jt - jax.lax.stop_gradient(target)) ** 2)
            # Actor: J_i = E[sum_a pi_i(a) (alpha log pi_i(a) - Q_i(a))] with the critic held fixed.
            if dependent:
                actions, _ = self.sample_dependent(params, x, jax.random.fold_in(batch['key'], 1))
                independent, correction = self.policy.apply(params['policy'], x, self.previous(actions))
                logits = jax.lax.stop_gradient(independent) + correction
                ind_c, cor_c = self.critic.apply(params['critic'], x, self.previous(actions))
                q_values = jax.lax.stop_gradient(ind_c + cor_c)
            else:
                logits = self.scores(params, batch['obs'])
                ind_c, _ = self.critic.apply(params['critic'], x, self.previous(batch['actions']))
                q_values = jax.lax.stop_gradient(ind_c)
            logp = jax.nn.log_softmax(logits)
            pi = jnp.exp(logp)
            weight = jax.lax.stop_gradient(k)[..., None]  # credit of every agent in the mixture
            actor_loss = jnp.mean(jnp.sum(pi * (alpha * logp - weight * q_values), -1))
            total = total + critic_loss + actor_loss
            aux['q_tot'] = jnp.mean(q_jt)
        return total, aux


EXTENSIONS = {'riskq': RiskQ, 'qfree': QFree, 'macpf': MACPF}
