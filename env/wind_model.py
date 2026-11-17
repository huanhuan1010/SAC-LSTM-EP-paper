"""Temporally correlated horizontal wind for UAV energy-stress experiments."""
from __future__ import annotations

import numpy as np


class WindModel:
    """First-order Gauss-Markov wind with a dedicated random stream."""

    def __init__(self, config):
        self.enabled = bool(config.get('wind_enabled', False))
        self.mean_speed = float(config.get('wind_mean_speed', 3.0))
        self.speed_std = float(config.get('wind_speed_std', 1.0))
        self.correlation = float(
            config.get('wind_temporal_correlation', 0.90)
        )
        self.max_speed = float(config.get('wind_max_speed', 8.0))
        direction = config.get('wind_direction_deg', None)
        self.direction_deg = None if direction is None else float(direction)
        self.seed_offset = int(config.get('wind_seed_offset', 7919))

        if self.mean_speed < 0.0:
            raise ValueError('wind_mean_speed must be nonnegative')
        if self.speed_std < 0.0:
            raise ValueError('wind_speed_std must be nonnegative')
        if not 0.0 <= self.correlation < 1.0:
            raise ValueError(
                'wind_temporal_correlation must lie in [0, 1)'
            )
        if self.max_speed <= 0.0:
            raise ValueError('wind_max_speed must be positive')

        self.rng = np.random.default_rng()
        self.mean_vector = np.zeros(3, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)

    def _clip(self, velocity):
        value = np.asarray(velocity, dtype=np.float64).reshape(3).copy()
        value[2] = 0.0
        speed = float(np.linalg.norm(value[:2]))
        if speed > self.max_speed:
            value[:2] *= self.max_speed / speed
        return value

    def reset(self, environment_seed=None):
        """Reset without consuming the main environment random stream."""
        if not self.enabled:
            self.mean_vector.fill(0.0)
            self.velocity.fill(0.0)
            return self.velocity.copy()

        if environment_seed is None:
            self.rng = np.random.default_rng()
        else:
            modulus = np.iinfo(np.uint32).max
            wind_seed = (int(environment_seed) + self.seed_offset) % modulus
            self.rng = np.random.default_rng(wind_seed)

        direction_deg = self.direction_deg
        if direction_deg is None:
            direction_deg = self.rng.uniform(0.0, 360.0)
        direction = np.deg2rad(direction_deg)
        self.mean_vector = np.array([
            self.mean_speed * np.cos(direction),
            self.mean_speed * np.sin(direction),
            0.0,
        ], dtype=np.float64)
        initial_gust = self.rng.normal(0.0, self.speed_std, size=2)
        self.velocity = self._clip(
            self.mean_vector + np.array([*initial_gust, 0.0])
        )
        return self.velocity.copy()

    def advance(self):
        """Advance one slot while preserving stationary gust variance."""
        if not self.enabled:
            self.velocity.fill(0.0)
            return self.velocity.copy()

        deviation = self.velocity[:2] - self.mean_vector[:2]
        innovation_scale = self.speed_std * np.sqrt(
            max(1.0 - self.correlation ** 2, 0.0)
        )
        innovation = self.rng.normal(0.0, innovation_scale, size=2)
        horizontal = (
            self.mean_vector[:2]
            + self.correlation * deviation
            + innovation
        )
        self.velocity = self._clip(np.array([*horizontal, 0.0]))
        return self.velocity.copy()

    @property
    def speed(self):
        return float(np.linalg.norm(self.velocity[:2]))
