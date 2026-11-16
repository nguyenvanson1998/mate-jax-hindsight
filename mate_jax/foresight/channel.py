"""Shared wireless channel with contention (slotted random access) and fading losses.

Every environment step, each camera that decides to transmit picks one of ``slots`` slots
uniformly at random; two packets in the same slot collide and are both lost (slotted ALOHA).
Surviving packets are additionally lost with probability ``base_loss`` (fading, interference).
"""

import jax
import jax.numpy as jnp


__all__ = ['contend']


def contend(key, transmit, slots: int = 4, base_loss: float = 0.0):
    """Resolve one step of channel access.

    Args:
        transmit: (C,) bool, the cameras that try to send a packet.

    Returns:
        ``delivered`` (C,) bool and ``collided`` (C,) bool.
    """

    num = transmit.shape[0]
    k_slot, k_loss = jax.random.split(key)
    slot = jax.random.randint(k_slot, (num,), 0, slots)
    same = (slot[:, None] == slot[None, :]) & transmit[:, None] & transmit[None, :]
    same &= jnp.logical_not(jnp.eye(num, dtype=bool))
    collided = transmit & jnp.any(same, axis=1)
    lost = jax.random.bernoulli(k_loss, base_loss, (num,)) if base_loss > 0.0 else jnp.zeros(num, bool)
    delivered = transmit & jnp.logical_not(collided) & jnp.logical_not(lost)
    return delivered, collided
