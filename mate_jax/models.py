"""Flax actor-critic networks for MAPPO (parameter sharing across teammates)."""

from typing import Any, Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np


__all__ = ['MLP', 'Actor', 'Critic', 'DiagGaussian', 'Categorical', 'discrete_action_table']

ACTIVATIONS = {'tanh': nn.tanh, 'relu': nn.relu, 'gelu': nn.gelu, 'silu': nn.silu}


class MLP(nn.Module):
    hidden_sizes: Sequence[int] = (512, 256)
    activation: str = 'tanh'
    dtype: Any = jnp.float32  # computation dtype (use jnp.bfloat16 on TPU)
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        act = ACTIVATIONS[self.activation]
        x = x.astype(self.dtype)
        for size in self.hidden_sizes:
            x = nn.Dense(
                size,
                dtype=self.dtype,
                param_dtype=jnp.float32,
                kernel_init=nn.initializers.orthogonal(np.sqrt(2.0)),
                bias_init=nn.initializers.zeros,
            )(x)
            if self.layer_norm:
                x = nn.LayerNorm(dtype=self.dtype, param_dtype=jnp.float32)(x)
            x = act(x)
        return x


class Actor(nn.Module):
    """Shared policy. Outputs Gaussian means (continuous) or logits (discrete)."""

    num_outputs: int
    discrete: bool = True
    hidden_sizes: Sequence[int] = (512, 256)
    activation: str = 'tanh'
    dtype: Any = jnp.float32
    layer_norm: bool = False

    @nn.compact
    def __call__(self, observation):
        x = MLP(self.hidden_sizes, self.activation, self.dtype, self.layer_norm)(observation)
        out = nn.Dense(
            self.num_outputs,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.zeros,
        )(x).astype(jnp.float32)
        if self.discrete:
            return out, None
        log_std = self.param('log_std', nn.initializers.constant(-0.5), (self.num_outputs,))
        return out, log_std


class Critic(nn.Module):
    """Centralized value function on (global state, agent id)."""

    hidden_sizes: Sequence[int] = (512, 256)
    activation: str = 'tanh'
    dtype: Any = jnp.float32
    layer_norm: bool = False

    @nn.compact
    def __call__(self, state):
        x = MLP(self.hidden_sizes, self.activation, self.dtype, self.layer_norm)(state)
        value = nn.Dense(
            1,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.zeros,
        )(x)
        return value[..., 0].astype(jnp.float32)


class DiagGaussian:
    """Diagonal Gaussian over unbounded actions (squashed by clipping in the environment)."""

    @staticmethod
    def sample(key, mean, log_std):
        return mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape, mean.dtype)

    @staticmethod
    def log_prob(action, mean, log_std):
        z = (action - mean) * jnp.exp(-log_std)
        return jnp.sum(-0.5 * jnp.square(z) - log_std - 0.5 * jnp.log(2.0 * jnp.pi), axis=-1)

    @staticmethod
    def entropy(mean, log_std):
        return jnp.broadcast_to(jnp.sum(log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)), mean.shape[:-1])

    @staticmethod
    def mode(mean, log_std):  # pylint: disable=unused-argument
        return mean


class Categorical:
    @staticmethod
    def sample(key, logits, _=None):
        return jax.random.categorical(key, logits, axis=-1)

    @staticmethod
    def log_prob(action, logits, _=None):
        log_p = jax.nn.log_softmax(logits, axis=-1)
        return jnp.take_along_axis(log_p, action[..., None].astype(jnp.int32), axis=-1)[..., 0]

    @staticmethod
    def entropy(logits, _=None):
        log_p = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.sum(jnp.exp(log_p) * log_p, axis=-1)

    @staticmethod
    def mode(logits, _=None):
        return jnp.argmax(logits, axis=-1)


def discrete_action_table(levels: int, dim: int = 2) -> np.ndarray:
    """Grid of normalized actions in ``[-1, +1]^dim`` (``mate.DiscreteCamera`` / ``DiscreteTarget``)."""

    grid = np.linspace(-1.0, 1.0, num=levels)
    mesh = np.meshgrid(*([grid] * dim), indexing='ij')
    return np.stack([m.ravel() for m in mesh], axis=-1).astype(np.float32)
