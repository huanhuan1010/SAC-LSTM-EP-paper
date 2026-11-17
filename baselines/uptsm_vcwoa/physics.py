from __future__ import annotations

import numpy as np

from .config import PaperConfig


SPEED_OF_LIGHT_M_S = 299_792_458.0


def offloading_rates(
    ue_positions_m: np.ndarray,
    uav_positions_m: np.ndarray,
    config: PaperConfig,
) -> np.ndarray:
    """Equation (5)-(11), returned as a K x U rate matrix in bit/s."""
    delta = ue_positions_m[:, None, :] - uav_positions_m[None, :, :]
    horizontal = np.linalg.norm(delta, axis=2)
    distance = np.sqrt(horizontal ** 2 + config.altitude_m ** 2)
    elevation_deg = np.degrees(np.arcsin(config.altitude_m / distance))
    p_los = 1.0 / (
        1.0 + config.los_a
        * np.exp(-config.los_b * (elevation_deg - config.los_a))
    )
    free_space = (
        4.0 * np.pi * config.offload_carrier_hz * distance
        / SPEED_OF_LIGHT_M_S
    ) ** 2
    loss_los = free_space * 10.0 ** (config.los_loss_db / 10.0)
    loss_nlos = free_space * 10.0 ** (config.nlos_loss_db / 10.0)
    average_loss = p_los * loss_los + (1.0 - p_los) * loss_nlos
    snr = config.ue_tx_w / (
        config.bandwidth_hz * config.noise_psd_w_hz * average_loss
    )
    return config.bandwidth_hz * np.log2(1.0 + snr)


def cooperation_rate(
    origin: int,
    destination: int,
    positions_m: np.ndarray,
    transmit_power_w: float,
    config: PaperConfig,
) -> float:
    """Equation (13)-(14); self-processing has infinite link rate."""
    if origin == destination:
        return np.inf
    distance = float(np.linalg.norm(
        positions_m[origin] - positions_m[destination]
    ))
    distance = max(distance, 1e-9)
    path_loss = (
        4.0 * np.pi * config.cooperation_carrier_hz * distance
        / SPEED_OF_LIGHT_M_S
    ) ** 2 * 10.0 ** (config.los_loss_db / 10.0)
    snr = max(float(transmit_power_w), 0.0) / (
        config.bandwidth_hz * config.noise_psd_w_hz * path_loss
    )
    return float(config.bandwidth_hz * np.log2(1.0 + snr))


def flight_power(speed_m_s: np.ndarray | float, config: PaperConfig) -> np.ndarray:
    """Rotary-wing propulsion model in equation (24)."""
    speed = np.asarray(speed_m_s, dtype=np.float64)
    profile = config.profile_power_w * (
        1.0 + 3.0 * speed ** 2 / config.rotor_tip_speed_m_s ** 2
    )
    drag = (
        0.5 * config.fuselage_drag_ratio * config.air_density_kg_m3
        * config.rotor_solidity * config.rotor_disk_area_m2 * speed ** 3
    )
    induced_inner = np.sqrt(
        1.0 + speed ** 4
        / (4.0 * config.hover_induced_velocity_m_s ** 4)
    ) - speed ** 2 / (2.0 * config.hover_induced_velocity_m_s ** 2)
    induced = config.induced_power_w * np.sqrt(
        np.maximum(induced_inner, 0.0)
    )
    return profile + drag + induced


def dynamic_energy_weight(
    cumulative_processing_energy_j: np.ndarray,
    config: PaperConfig,
) -> np.ndarray:
    """Equation (20), increasing from w_b at full charge to w_a at depletion."""
    consumed_ratio = np.clip(
        np.asarray(cumulative_processing_energy_j, dtype=np.float64)
        / config.processing_energy_budget_j,
        0.0,
        1.0,
    )
    return (
        config.energy_weight_full
        + (config.energy_weight_depleted - config.energy_weight_full)
        * consumed_ratio
    )


def jain_fairness(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    denominator = values.size * float(np.sum(values ** 2))
    if denominator <= 0.0:
        return 0.0
    return float(np.sum(values) ** 2 / denominator)

