"""Device mesh helpers for data-parallel training on TPU pods (also works on GPU / CPU).

The training loop is written per device with :func:`shard_map`: every TPU core simulates its
own slice of the parallel environments and runs the PPO update on it; gradients and metrics
are averaged with ``lax.pmean`` over the ``'data'`` mesh axis (ICI all-reduce on TPU).
"""

import functools
import os
from typing import Optional

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec


__all__ = [
    'DATA_AXIS',
    'maybe_initialize_distributed',
    'create_mesh',
    'shard_map',
    'replicated',
    'replicate',
    'data_sharded',
    'describe_devices',
]

DATA_AXIS = 'data'


def maybe_initialize_distributed(force: bool = False) -> None:
    """Initialize multi-host JAX (TPU pod slices) when running on more than one host.

    On Cloud TPU VMs ``jax.distributed.initialize()`` discovers the coordinator automatically.
    Call this before any other JAX operation.
    """

    multi_host_env = any(
        key in os.environ
        for key in ('TPU_WORKER_HOSTNAMES', 'JAX_COORDINATOR_ADDRESS', 'MEGASCALE_COORDINATOR_ADDRESS')
    )
    if force or (multi_host_env and os.environ.get('TPU_WORKER_HOSTNAMES', '').count(',') > 0):
        jax.distributed.initialize()


def create_mesh(num_devices: Optional[int] = None) -> Mesh:
    """A 1-D data-parallel mesh over all (global) devices."""

    devices = jax.devices()
    if num_devices is not None:
        if num_devices > len(devices):
            raise ValueError(f'Requested {num_devices} devices, but only {len(devices)} are available.')
        devices = devices[:num_devices]
    try:
        from jax.experimental import mesh_utils  # pylint: disable=import-outside-toplevel

        device_array = mesh_utils.create_device_mesh((len(devices),), devices=devices)
    except Exception:  # pylint: disable=broad-except
        device_array = np.asarray(devices)
    return Mesh(device_array, (DATA_AXIS,))


def shard_map(fn, mesh: Mesh, in_specs, out_specs, check: bool = False):
    """``jax.shard_map`` with a fallback for older JAX releases.

    Varying-manual-axes checking is off by default: the environment's internal loops start
    from compile-time constants and become device-varying through the PRNG keys, which the
    checker would reject. Replication of parameters is guaranteed by ``lax.pmean`` instead.
    """

    if hasattr(jax, 'shard_map'):
        return jax.shard_map(fn, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=check)
    from jax.experimental.shard_map import shard_map as _shard_map  # pylint: disable=import-outside-toplevel

    return _shard_map(fn, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_rep=check)


def replicated(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, PartitionSpec())


def replicate(tree, mesh: Mesh):
    """Place identical host values on every device of the (possibly multi-host) mesh."""

    sharding = replicated(mesh)

    def put(x):
        x = np.asarray(x)
        return jax.make_array_from_callback(x.shape, sharding, lambda index: x[index])

    return jax.tree_util.tree_map(put, tree)


def data_sharded(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, PartitionSpec(DATA_AXIS))


@functools.lru_cache(maxsize=None)
def describe_devices() -> str:
    devices = jax.devices()
    kinds = sorted({d.device_kind for d in devices})
    return (
        f'{jax.default_backend()} backend: {len(devices)} device(s) '
        f'({", ".join(kinds)}), {jax.process_count()} process(es), '
        f'{jax.local_device_count()} local device(s) on process {jax.process_index()}'
    )
