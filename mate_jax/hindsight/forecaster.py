"""Learned heat-map forecasts trained in hindsight, for the belief-space planner.

FORESIGHT forecasts every target with a Gaussian random walk around its last velocity. A convolutional
network corrects these forecasts: from a target's current belief map, its analytic forecasts, the static
free space and its velocity / age it predicts residual log-probabilities over the grid cells for every
planning horizon (initialized at zero, i.e. at the analytic forecast). It is trained by cross-entropy
against the cell the target truly occupies ``h`` steps later, which is known in hindsight because the
simulator can be rolled forward. :class:`LearnedForecastPlanner` plans with the learned forecasts.
"""

from typing import Any, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import struct

from mate_jax.foresight import ForesightCameraPolicy


__all__ = ['ForecastNet', 'forecast_inputs', 'learned_forecast', 'LearnedForecastPlanner', 'load_forecaster']

FORECAST_CHANNELS = 11  # belief, 4 analytic forecasts, free space, 5 scalars (see forecast_inputs)


class ForecastNet(nn.Module):
    """(..., G, G, C_in) -> (..., G, G, H) residual log-probabilities (dilated convolutions)."""

    num_horizons: int
    features: int = 64
    dilations: Sequence[int] = (1, 2, 4, 8, 1)

    @nn.compact
    def __call__(self, x):
        h = nn.Conv(self.features, (3, 3), padding='SAME')(x)
        for dilation in self.dilations:
            h = h + nn.Conv(self.features, (3, 3), padding='SAME', kernel_dilation=dilation)(nn.gelu(h))
        return nn.Conv(self.num_horizons, (1, 1), kernel_init=nn.initializers.zeros)(nn.gelu(h))


def forecast_inputs(planner: ForesightCameraPolicy, belief, free, horizons, input_horizons=(0, 10, 20, 40)):
    """Network inputs (K, T, G, G, C) and analytic forecasts (K, H, T, G, G) for a belief state."""

    analytic = planner.filter.forecast(belief, free, horizons)  # (K, H, T, G, G)
    picks = [int(min(range(len(horizons)), key=lambda i: abs(horizons[i] - h))) for h in input_horizons]
    grid = belief.belief.shape[-1]
    speed = planner.target_speed
    age = jnp.where(belief.last_seen >= 0, belief.step - belief.last_seen, belief.step).astype(jnp.float32)
    scalars = jnp.stack(
        [
            belief.velocity[..., 0] / speed,
            belief.velocity[..., 1] / speed,
            belief.velocity_known.astype(jnp.float32),
            jnp.minimum(age / 50.0, 4.0),
            (belief.last_seen < 0).astype(jnp.float32),
        ],
        axis=-1,
    )  # (K, T, 5)
    num_sets, num_targets = belief.belief.shape[:2]
    channels = [
        belief.belief[..., None] * grid * grid,
        jnp.moveaxis(analytic[:, picks], 1, -1) * grid * grid,
        jnp.broadcast_to(free[None, None, :, :, None], (num_sets, num_targets, grid, grid, 1)).astype(jnp.float32),
        jnp.broadcast_to(scalars[:, :, None, None, :], (num_sets, num_targets, grid, grid, scalars.shape[-1])),
    ]
    return jnp.concatenate(channels, axis=-1), analytic


def learned_forecast(net: ForecastNet, params, inputs, analytic, free):
    """Corrected forecasts (K, H, T, G, G): softmax over free cells of log(analytic) + residual."""

    residual = net.apply(params, inputs)  # (K, T, G, G, H)
    residual = jnp.moveaxis(residual, -1, 1)  # (K, H, T, G, G)
    logits = jnp.log(jnp.maximum(analytic, 1e-6)) + residual
    logits = jnp.where(free > 0, logits, -1e9)
    shape = logits.shape
    probabilities = jax.nn.softmax(logits.reshape(shape[:-2] + (-1,)), axis=-1)
    return probabilities.reshape(shape)


@struct.dataclass
class LearnedForecastPlanner(ForesightCameraPolicy):
    """FORESIGHT whose planning forecasts come from a trained :class:`ForecastNet`."""

    forecaster_params: Any = None
    forecaster_features: int = struct.field(pytree_node=False, default=64)

    def _planning_counts(self, carry):
        if self.forecaster_params is None or self.communication == 'hybrid':
            return super()._planning_counts(carry)
        horizons = self.horizons
        net = ForecastNet(num_horizons=horizons.size, features=self.forecaster_features)
        inputs, analytic = forecast_inputs(self, carry['belief'], carry['free'], horizons)
        forecast = learned_forecast(net, self.forecaster_params, inputs, analytic, carry['free'])
        counts = self.filter.expected_counts(forecast)  # (K, H, N)
        if self.communication == 'team':
            counts = jnp.broadcast_to(counts, (self.num_cameras,) + counts.shape[1:])
        return counts


def load_forecaster(path: str, planner: ForesightCameraPolicy, features: int = 64):
    """Parameters of a :class:`ForecastNet` saved with ``flax.serialization.to_bytes``."""

    from flax import serialization  # pylint: disable=import-outside-toplevel

    grid = planner.grid_size
    net = ForecastNet(num_horizons=planner.horizons.size, features=features)
    template = net.init(jax.random.PRNGKey(0), jnp.zeros((1, planner.num_targets, grid, grid, FORECAST_CHANNELS)))
    with open(path, 'rb') as file:
        return serialization.from_bytes(template, file.read())
