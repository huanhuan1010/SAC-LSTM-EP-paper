"""Bounded, temporally correlated UAV energy-consumption uncertainty."""
from __future__ import annotations

import numpy as np


class EnergyUncertaintyModel:
    """Generate a paired multiplicative disturbance for battery-side energy.

    A stationary standard-normal AR(1) latent state is mapped through a
    mean-preserving lognormal transform and then clipped to declared physical
    bounds.  The random stream is independent of task and wind generation, so
    algorithms evaluated with the same environment seed receive the same
    disturbance sequence regardless of their actions.
    """

    def __init__(self, config):
        self.enabled = bool(config.get('energy_uncertainty_enabled', False))
        self.log_std = float(config.get('energy_uncertainty_log_std', 0.10))
        self.correlation = float(config.get(
            'energy_uncertainty_temporal_correlation', 0.85
        ))
        self.min_scale = float(config.get(
            'energy_uncertainty_min_scale', 0.75
        ))
        self.max_scale = float(config.get(
            'energy_uncertainty_max_scale', 1.35
        ))
        self.seed_offset = int(config.get(
            'energy_uncertainty_seed_offset', 15401
        ))
        if self.log_std < 0.0:
            raise ValueError('energy_uncertainty_log_std must be nonnegative')
        if not 0.0 <= self.correlation < 1.0:
            raise ValueError(
                'energy_uncertainty_temporal_correlation must lie in [0, 1)'
            )
        if not 0.0 < self.min_scale <= 1.0 <= self.max_scale:
            raise ValueError(
                'energy uncertainty scales must satisfy '
                '0 < min_scale <= 1 <= max_scale'
            )
        self.rng = np.random.default_rng()
        self.latent = 0.0
        self.scale = 1.0

    def _scale_from_latent(self, latent):
        if not self.enabled:
            return 1.0
        # Without clipping this correction makes E[scale] exactly one for a
        # stationary N(0, 1) latent variable.
        raw = np.exp(self.log_std * float(latent) - 0.5 * self.log_std ** 2)
        return float(np.clip(raw, self.min_scale, self.max_scale))

    def reset(self, environment_seed=None):
        """Reset the dedicated stream and return the first-slot scale."""
        if not self.enabled:
            self.latent = 0.0
            self.scale = 1.0
            return self.scale
        if environment_seed is None:
            self.rng = np.random.default_rng()
        else:
            uncertainty_seed = (
                int(environment_seed) + self.seed_offset
            ) % (2 ** 32)
            self.rng = np.random.default_rng(uncertainty_seed)
        self.latent = float(self.rng.normal())
        self.scale = self._scale_from_latent(self.latent)
        return self.scale

    def advance(self):
        """Advance one slot while retaining unit stationary latent variance."""
        if not self.enabled:
            self.latent = 0.0
            self.scale = 1.0
            return self.scale
        innovation_scale = np.sqrt(max(1.0 - self.correlation ** 2, 0.0))
        self.latent = float(
            self.correlation * self.latent
            + innovation_scale * self.rng.normal()
        )
        self.scale = self._scale_from_latent(self.latent)
        return self.scale

    @property
    def return_energy_scale(self):
        """Worst-case bounded multiplier used by robust return checks."""
        return self.max_scale if self.enabled else 1.0
