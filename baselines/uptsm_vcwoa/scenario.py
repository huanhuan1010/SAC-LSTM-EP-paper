from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import PaperConfig


@dataclass(frozen=True)
class Scenario:
    ue_positions_m: np.ndarray       # (K, 2)
    task_bits: np.ndarray            # (N, K)
    uav_initial_positions_m: np.ndarray  # (U, 2)
    uav_final_positions_m: np.ndarray    # (U, 2)
    initial_processing_energy_used_j: np.ndarray  # (U,), zero means full
    seed: int

    def validate(self, config: PaperConfig) -> None:
        expected = {
            "ue_positions_m": (config.num_users, 2),
            "task_bits": (config.num_slots, config.num_users),
            "uav_initial_positions_m": (config.num_uavs, 2),
            "uav_final_positions_m": (config.num_uavs, 2),
            "initial_processing_energy_used_j": (config.num_uavs,),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"{name} expected {shape}, got {value.shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains a non-finite value")


def _default_uav_positions(num_uavs: int, area_size_m: float) -> np.ndarray:
    """Deterministic perimeter placement; the paper does not publish coordinates."""
    if num_uavs == 1:
        return np.array([[area_size_m / 2.0, area_size_m / 2.0]])
    centre = area_size_m / 2.0
    radius = area_size_m * 0.35
    angles = np.linspace(0.0, 2.0 * np.pi, num_uavs, endpoint=False)
    return np.column_stack((
        centre + radius * np.cos(angles),
        centre + radius * np.sin(angles),
    ))


def build_scenario(
    config: PaperConfig,
    seed: Optional[int] = None,
    initial_battery_fraction: Optional[np.ndarray] = None,
) -> Scenario:
    config.validate()
    actual_seed = config.seed if seed is None else int(seed)
    rng = np.random.RandomState(actual_seed)
    ue_positions = rng.uniform(
        0.0, config.area_size_m, size=(config.num_users, 2)
    )
    task_bits = rng.uniform(
        config.task_bits_min,
        config.task_bits_max,
        size=(config.num_slots, config.num_users),
    )
    initial = _default_uav_positions(config.num_uavs, config.area_size_m)
    if initial_battery_fraction is None:
        battery_fraction = np.ones(config.num_uavs, dtype=np.float64)
    else:
        battery_fraction = np.asarray(
            initial_battery_fraction, dtype=np.float64
        ).reshape(-1)
        if battery_fraction.shape != (config.num_uavs,):
            raise ValueError("initial_battery_fraction must have one value per UAV")
        if np.any(battery_fraction < 0.0) or np.any(battery_fraction > 1.0):
            raise ValueError("initial battery fractions must lie in [0, 1]")
    scenario = Scenario(
        ue_positions_m=ue_positions,
        task_bits=task_bits,
        uav_initial_positions_m=initial,
        uav_final_positions_m=initial.copy(),
        initial_processing_energy_used_j=(
            1.0 - battery_fraction
        ) * config.processing_energy_budget_j,
        seed=actual_seed,
    )
    scenario.validate(config)
    return scenario
