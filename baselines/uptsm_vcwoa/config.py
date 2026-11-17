from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class PaperConfig:
    """Parameters stated in Section V plus explicit reproduction controls."""

    num_users: int = 20
    num_uavs: int = 4
    num_slots: int = 50
    slot_seconds: float = 1.0
    area_size_m: float = 200.0
    task_bits_min: float = 100e3
    task_bits_max: float = 200e3

    bandwidth_hz: float = 1e6
    offload_carrier_hz: float = 1e9
    cooperation_carrier_hz: float = 1e9
    los_loss_db: float = 3.0
    nlos_loss_db: float = 23.0
    los_a: float = 11.95
    los_b: float = 0.14
    ue_tx_dbm: float = 10.0
    noise_psd_dbm_hz: float = -174.0

    altitude_m: float = 100.0
    uav_tx_max_dbm: float = 30.0
    cpu_max_hz: float = 2e9
    cycles_per_bit: float = 4000.0
    processing_energy_budget_j: float = 200.0

    energy_weight_depleted: float = 10.0
    energy_weight_full: float = 1.0
    offloaded_bit_price: float = 2e-5
    fairness_subsidy_price: float = 0.1
    time_cost_weight: float = 1.0
    flight_cost_weight: float = 1e-4

    profile_power_w: float = 225.79
    induced_power_w: float = 426.07
    hover_induced_velocity_m_s: float = 4.03
    rotor_tip_speed_m_s: float = 200.0
    fuselage_drag_ratio: float = 0.6
    air_density_kg_m3: float = 1.225
    rotor_solidity: float = 0.05
    rotor_disk_area_m2: float = 0.503
    uav_speed_max_m_s: float = 25.0
    minimum_separation_m: float = 1.0
    effective_capacitance: float = 1e-27

    woa_iterations: int = 50
    population_size: int = 100
    spiral_beta: float = 0.5
    penalty_speed: float = 1e9
    penalty_position: float = 1e9
    penalty_time: float = 1e9
    penalty_energy: float = 1e9
    allow_idle_uavs: bool = True
    seed: int = 2026

    @property
    def ue_tx_w(self) -> float:
        return 10.0 ** ((self.ue_tx_dbm - 30.0) / 10.0)

    @property
    def uav_tx_max_w(self) -> float:
        return 10.0 ** ((self.uav_tx_max_dbm - 30.0) / 10.0)

    @property
    def noise_psd_w_hz(self) -> float:
        return 10.0 ** ((self.noise_psd_dbm_hz - 30.0) / 10.0)

    def validate(self) -> None:
        positive = {
            "num_users": self.num_users,
            "num_uavs": self.num_uavs,
            "num_slots": self.num_slots,
            "slot_seconds": self.slot_seconds,
            "area_size_m": self.area_size_m,
            "bandwidth_hz": self.bandwidth_hz,
            "altitude_m": self.altitude_m,
            "cpu_max_hz": self.cpu_max_hz,
            "population_size": self.population_size,
            "woa_iterations": self.woa_iterations,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"positive parameters required: {invalid}")
        if self.num_uavs > self.num_users:
            raise ValueError("the paper assumes num_uavs <= num_users")
        if self.task_bits_min > self.task_bits_max:
            raise ValueError("task_bits_min must not exceed task_bits_max")
        if self.energy_weight_depleted < self.energy_weight_full:
            raise ValueError("depleted-battery energy weight must be no smaller")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path, **overrides: Any) -> "PaperConfig":
        with Path(path).open("r", encoding="utf-8") as stream:
            values = json.load(stream)
        values.update({key: value for key, value in overrides.items()
                       if value is not None})
        config = cls(**values)
        config.validate()
        return config


def paper_default_path() -> Path:
    return Path(__file__).resolve().with_name("paper_default.json")
