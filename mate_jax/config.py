"""Configuration loading for the JAX Multi-Agent Tracking Environment.

The YAML/JSON files and their semantics are identical to :func:`mate.read_config`, but this
module does not depend on ``gym``. The result is a frozen :class:`EnvConfig` holding Python
scalars and NumPy arrays only, which become compile-time constants of the jitted environment.
"""

import copy
import dataclasses
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np

from mate_jax import constants as consts


__all__ = ['ASSETS_DIR', 'DEFAULT_CONFIG_FILE', 'EnvConfig', 'load_config']

ASSETS_DIR = Path(__file__).absolute().parent.parent / 'mate' / 'assets'
"""The asset directory shared with the original ``mate`` package."""

DEFAULT_CONFIG_FILE = ASSETS_DIR / 'MATE-4v8-9.yaml'
"""The default configuration file."""

CAMERA_DEFAULTS = {
    'radius': 40.0,
    'min_viewing_angle': 90.0,
    'max_sight_range': 500.0,
    'rotation_step': 5.0,
    'zooming_step': 2.5,
}
TARGET_DEFAULTS = {
    'sight_range': 500.0,
    'step_size': 10.0,
}


@dataclasses.dataclass(frozen=True, eq=False)
class EnvConfig:  # pylint: disable=too-many-instance-attributes
    """Static (compile-time) configuration of the environment."""

    name: str
    max_episode_steps: int
    num_cargoes_per_target: int
    high_capacity_target_split: float
    targets_start_with_cargoes: bool
    bounty_factor: float
    shuffle_entities: bool
    reward_type: str

    camera_location_low: np.ndarray  # (C, 2)
    camera_location_high: np.ndarray  # (C, 2)
    camera_radius: float
    camera_min_viewing_angle: float
    camera_max_sight_range: float
    camera_rotation_step: float
    camera_zooming_step: float

    target_location_low: np.ndarray  # (T, 2)
    target_location_high: np.ndarray  # (T, 2)
    target_step_size: float
    target_sight_range: float

    obstacle_location_low: np.ndarray  # (O, 2)
    obstacle_location_high: np.ndarray  # (O, 2)
    obstacle_radius_low: float
    obstacle_radius_high: float
    obstacle_transmittance: float

    raw: Dict[str, Any] = dataclasses.field(default_factory=dict, repr=False)

    @property
    def num_cameras(self) -> int:
        return int(self.camera_location_low.shape[0])

    @property
    def num_targets(self) -> int:
        return int(self.target_location_low.shape[0])

    @property
    def num_obstacles(self) -> int:
        return int(self.obstacle_location_low.shape[0])

    @property
    def num_high_capacity_targets(self) -> int:
        return int(self.num_targets * self.high_capacity_target_split)


def _resolve_path(path: Union[str, os.PathLike]) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    for candidate in (Path.cwd() / path, ASSETS_DIR / path):
        if candidate.is_file():
            return candidate
    raise ValueError(
        f'Cannot find the configuration file "{path}". '
        f'Available built-in files: {sorted(p.name for p in ASSETS_DIR.glob("*.yaml"))}.'
    )


def _deep_update(dict1: Dict[str, Any], dict2: Mapping[str, Any]) -> Dict[str, Any]:
    dict1 = copy.deepcopy(dict1)
    for key, value in dict2.items():
        if isinstance(dict1.get(key), dict) and isinstance(value, Mapping):
            value = _deep_update(dict1[key], value)
        dict1[key] = copy.deepcopy(value)
    return dict1


def _read_raw(config_or_path: Optional[Union[Mapping[str, Any], str, os.PathLike]]) -> Dict[str, Any]:
    if config_or_path is None:
        return {}
    if isinstance(config_or_path, Mapping):
        return dict(config_or_path)

    path = _resolve_path(config_or_path)
    with open(path, encoding='UTF-8') as file:
        if path.suffix.lower() == '.json':
            return json.load(file)
        import yaml  # pylint: disable=import-outside-toplevel

        return yaml.load(file, yaml.SafeLoader)


def _to_bounds(random_range) -> tuple:
    if isinstance(random_range, Mapping):
        low, high = random_range['low'], random_range['high']
    elif hasattr(random_range, 'low') and hasattr(random_range, 'high'):  # gym.spaces.Box
        low, high = random_range.low, random_range.high
    else:
        low, high = list(random_range)[0::2], list(random_range)[1::2]
    low = np.asarray(low, dtype=np.float64).reshape(-1)
    high = np.asarray(high, dtype=np.float64).reshape(-1)
    return low, high


def _locations(subconfig: Mapping[str, Any]) -> tuple:
    lows, highs = [], []
    for location in subconfig.get('location', []):
        location = np.asarray(location, dtype=np.float64).reshape(2)
        lows.append(location)
        highs.append(location)
    for random_range in subconfig.get('location_random_range', []):
        low, high = _to_bounds(random_range)
        lows.append(low.reshape(2))
        highs.append(high.reshape(2))
    if len(lows) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.stack(lows), np.stack(highs)


def _with_defaults(subconfig: Dict[str, Any], defaults: Mapping[str, float], entity: str):
    for key, default in defaults.items():
        if key not in subconfig:
            warnings.warn(f'Missing key "{entity}/{key}", set to {default}.', stacklevel=3)
            subconfig[key] = default
        if not float(subconfig[key]) > 0.0:
            raise ValueError(f'`{entity}/{key}` must be a positive number. Got {subconfig[key]}.')
    return subconfig


# pylint: disable-next=too-many-branches,too-many-locals
def load_config(
    config: Optional[Union[Mapping[str, Any], str, os.PathLike]] = None, **overrides
) -> EnvConfig:
    """Load configuration from a dictionary mapping or a (built-in) JSON/YAML file.

    Examples:
        >>> load_config('MATE-4v8-9.yaml').num_cameras
        4
        >>> load_config('MATE-4v2-9.yaml', reward_type='sparse').reward_type
        'sparse'
    """

    if isinstance(config, EnvConfig):
        if overrides:
            return load_config(_deep_update(config.raw, overrides))
        return config
    if config is None and len(overrides) == 0:
        config = DEFAULT_CONFIG_FILE

    raw = _deep_update(_read_raw(config), overrides)

    defaults = {
        'name': 'MultiAgentTracking',
        'max_episode_steps': 10000,
        'reward_type': 'dense',
        'high_capacity_target_split': 0.5,
        'targets_start_with_cargoes': True,
        'bounty_factor': 1.0,
        'shuffle_entities': True,
    }
    for key, value in defaults.items():
        raw.setdefault(key, value)

    if int(raw['max_episode_steps']) <= 0:
        raise ValueError('`max_episode_steps` must be a positive integer.')
    if raw['reward_type'] not in ('dense', 'sparse'):
        raise ValueError(f'Invalid reward type {raw["reward_type"]}. Expect one of dense / sparse.')
    if 'num_cargoes_per_target' not in raw:
        raise ValueError('Missing key "num_cargoes_per_target".')
    if int(raw['num_cargoes_per_target']) < consts.NUM_WAREHOUSES:
        raise ValueError(
            f'`num_cargoes_per_target` should be no less than {consts.NUM_WAREHOUSES}. '
            f'Got {raw["num_cargoes_per_target"]}.'
        )
    if not 0.0 <= float(raw['high_capacity_target_split']) <= 1.0:
        raise ValueError('`high_capacity_target_split` must be between 0 and 1.')
    if not float(raw['bounty_factor']) >= 0.0:
        raise ValueError('`bounty_factor` must be a non-negative number.')

    camera = dict(raw.get('camera', {}))
    target = dict(raw.get('target', {}))
    obstacle = dict(raw.get('obstacle', {}))
    if 'camera' in raw:
        _with_defaults(camera, CAMERA_DEFAULTS, 'camera')
    else:
        camera.update(CAMERA_DEFAULTS)
    if 'target' not in raw:
        raise ValueError('Missing key "target". There must be at least one target.')
    _with_defaults(target, TARGET_DEFAULTS, 'target')

    camera_low, camera_high = _locations(camera)
    target_low, target_high = _locations(target)
    obstacle_low, obstacle_high = _locations(obstacle)
    if target_low.shape[0] == 0:
        raise ValueError('There must be at least one target in the environment.')

    if 'radius' in obstacle:
        radius_low = radius_high = float(obstacle['radius'])
    elif 'radius_random_range' in obstacle:
        low, high = _to_bounds(obstacle['radius_random_range'])
        radius_low, radius_high = float(low[0]), float(high[0])
    elif obstacle_low.shape[0] > 0:
        raise ValueError('You should specify either `obstacle/radius` or `obstacle/radius_random_range`.')
    else:
        radius_low = radius_high = 0.0

    if not 0.0 < float(camera['min_viewing_angle']) <= consts.MAX_CAMERA_VIEWING_ANGLE:
        raise ValueError('`camera/min_viewing_angle` must be within (0.0, 180.0].')

    return EnvConfig(
        name=str(raw['name']),
        max_episode_steps=int(raw['max_episode_steps']),
        num_cargoes_per_target=int(raw['num_cargoes_per_target']),
        high_capacity_target_split=float(raw['high_capacity_target_split']),
        targets_start_with_cargoes=bool(raw['targets_start_with_cargoes']),
        bounty_factor=max(0.0, float(raw['bounty_factor'])),
        shuffle_entities=bool(raw['shuffle_entities']),
        reward_type=str(raw['reward_type']),
        camera_location_low=camera_low,
        camera_location_high=camera_high,
        camera_radius=float(camera['radius']),
        camera_min_viewing_angle=float(camera['min_viewing_angle']),
        camera_max_sight_range=float(camera['max_sight_range']),
        camera_rotation_step=float(camera['rotation_step']),
        camera_zooming_step=float(camera['zooming_step']),
        target_location_low=target_low,
        target_location_high=target_high,
        target_step_size=float(target['step_size']),
        target_sight_range=float(target['sight_range']),
        obstacle_location_low=obstacle_low,
        obstacle_location_high=obstacle_high,
        obstacle_radius_low=radius_low,
        obstacle_radius_high=radius_high,
        obstacle_transmittance=min(max(0.0, float(obstacle.get('transmittance', 0.0))), 1.0),
        raw=raw,
    )
