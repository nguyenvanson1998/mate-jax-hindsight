"""MATE-JAX: the Multi-Agent Tracking Environment in pure JAX (jit / vmap / TPU friendly)."""

from mate_jax import agents, constants, geometry, transforms, wrappers
from mate_jax.config import ASSETS_DIR, DEFAULT_CONFIG_FILE, EnvConfig, load_config
from mate_jax.env import EnvState, MultiAgentTracking, TimeStep, make


__all__ = [
    'ASSETS_DIR',
    'DEFAULT_CONFIG_FILE',
    'EnvConfig',
    'EnvState',
    'MultiAgentTracking',
    'TimeStep',
    'agents',
    'constants',
    'geometry',
    'load_config',
    'make',
    'transforms',
    'wrappers',
]
