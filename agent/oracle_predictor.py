"""Non-causal oracle predictor backed by recorded per-episode energy traces.

This component is deliberately restricted to offline simulation.  A trace must
contain the realized UAV energy for every step of each requested episode seed.
It is never silently approximated from SOC or the current power draw.
"""
from pathlib import Path
import json

import numpy as np


class OracleEnergyTracePredictor:
    def __init__(self, trace_path, horizon, battery_capacity):
        self.trace_path = Path(trace_path)
        self.horizon = int(horizon)
        self.lookback = 0
        self.battery_capacity = float(battery_capacity)
        self.history = []
        self.last_raw_prediction_joule = np.zeros(
            self.horizon, dtype=np.float32
        )
        self.residual_bias_joule = 0.0
        self.residual_abs_ema_joule = 0.0
        self._traces = self._load(self.trace_path)
        self._trace = None
        self._step = 0

    @staticmethod
    def _load(path):
        if not path.is_file():
            raise FileNotFoundError(f'oracle trace file not found: {path}')
        if path.suffix.lower() == '.npz':
            with np.load(path, allow_pickle=False) as archive:
                return {
                    str(key): np.asarray(archive[key], dtype=np.float32)
                    for key in archive.files
                }
        if path.suffix.lower() == '.json':
            with path.open('r', encoding='utf-8') as stream:
                payload = json.load(stream)
            return {
                str(key): np.asarray(value, dtype=np.float32)
                for key, value in payload.items()
            }
        raise ValueError('oracle traces must be .npz or .json')

    @staticmethod
    def _candidate_keys(episode_seed):
        seed = str(int(episode_seed))
        return (seed, f'seed_{seed}', f'episode_{seed}')

    def reset_runtime_state(self, episode_seed=None):
        if episode_seed is None:
            raise ValueError('oracle predictor requires the episode seed')
        for key in self._candidate_keys(episode_seed):
            if key in self._traces:
                trace = np.asarray(self._traces[key], dtype=np.float32).reshape(-1)
                if trace.size == 0 or not np.all(np.isfinite(trace)):
                    raise ValueError(f'invalid oracle energy trace for {key}')
                if np.any(trace < 0.0):
                    raise ValueError(f'oracle energy trace must be non-negative: {key}')
                self._trace = trace
                self._step = 0
                self.history.clear()
                return
        raise KeyError(f'oracle trace missing episode seed {episode_seed}')

    def get_energy_features(self):
        if self._trace is None:
            raise RuntimeError('reset_runtime_state(seed) must be called first')
        values = self._trace[self._step:self._step + self.horizon]
        if values.size < self.horizon:
            values = np.pad(values, (0, self.horizon - values.size))
        self.last_raw_prediction_joule = values.astype(np.float32, copy=True)
        return self.last_raw_prediction_joule.copy()

    def observe_actual_energy(self, actual_energy):
        # The trace is intentionally not corrected by observations.  Advancing
        # after the environment step keeps prediction[t] aligned with E[t].
        self._step += 1

