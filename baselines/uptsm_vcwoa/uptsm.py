from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PaperConfig
from .matching import linear_sum_assignment, maximum_weight_uav_matching
from .physics import (
    cooperation_rate,
    dynamic_energy_weight,
    flight_power,
    jain_fairness,
    offloading_rates,
)
from .scenario import Scenario


@dataclass
class ContinuousDecision:
    cpu_hz: np.ndarray       # (N, U)
    transmit_power_w: np.ndarray  # (N, U)
    positions_m: np.ndarray  # (N, U, 2), positions C[1] ... C[N]

    def copy(self) -> "ContinuousDecision":
        return ContinuousDecision(
            self.cpu_hz.copy(),
            self.transmit_power_w.copy(),
            self.positions_m.copy(),
        )

    def validate(self, config: PaperConfig, scenario: Scenario) -> None:
        expected_scalar = (config.num_slots, config.num_uavs)
        if np.asarray(self.cpu_hz).shape != expected_scalar:
            raise ValueError(f"cpu_hz expected shape {expected_scalar}")
        if np.asarray(self.transmit_power_w).shape != expected_scalar:
            raise ValueError(f"transmit_power_w expected shape {expected_scalar}")
        expected_positions = (config.num_slots, config.num_uavs, 2)
        if np.asarray(self.positions_m).shape != expected_positions:
            raise ValueError(f"positions_m expected shape {expected_positions}")
        for value in (self.cpu_hz, self.transmit_power_w, self.positions_m):
            if not np.all(np.isfinite(value)):
                raise ValueError("continuous decision contains non-finite values")
        if np.any(self.cpu_hz < 0.0) or np.any(self.cpu_hz > config.cpu_max_hz):
            raise ValueError("CPU frequency outside paper bounds")
        if (np.any(self.transmit_power_w < 0.0)
                or np.any(self.transmit_power_w > config.uav_tx_max_w)):
            raise ValueError("transmission power outside paper bounds")
        if not np.allclose(
            self.positions_m[-1], scenario.uav_final_positions_m, atol=1e-8
        ):
            raise ValueError("the final UAV coordinates must satisfy equation (31e)")


@dataclass
class UPTSMResult:
    objective_utility: float
    offloading_selection: np.ndarray  # (N, K, U)
    computation_selection: np.ndarray  # (N, U, U)
    slot_utility: np.ndarray
    cumulative_processing_energy_j: np.ndarray
    cumulative_offloaded_bits: np.ndarray
    task_processing_time_s: np.ndarray  # (N, U), indexed by receiving UAV
    time_excess_s: np.ndarray
    flight_energy_j: np.ndarray
    speed_excess_m_s: float
    separation_deficit_m: float
    final_energy_excess_j: float
    penalty_total: float
    fitness: float
    task_fairness: float
    battery_fairness: float

    @property
    def feasible(self) -> bool:
        maximum_time_excess = (
            float(np.max(self.time_excess_s)) if self.time_excess_s.size else 0.0
        )
        return bool(
            maximum_time_excess <= 1e-9
            and self.speed_excess_m_s <= 1e-9
            and self.separation_deficit_m <= 1e-9
            and self.final_energy_excess_j <= 1e-9
        )


def _safe_duration(bits: float, rate: float) -> float:
    if bits <= 0.0:
        return 0.0
    if not np.isfinite(rate):
        return 0.0
    if rate <= 0.0:
        return 1e12
    return float(bits / rate)


def _trajectory_penalties(
    config: PaperConfig,
    scenario: Scenario,
    positions_m: np.ndarray,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    previous = np.concatenate(
        [scenario.uav_initial_positions_m[None, :, :], positions_m[:-1]],
        axis=0,
    )
    speeds = np.linalg.norm(positions_m - previous, axis=2) / config.slot_seconds
    speed_excess = float(np.maximum(
        speeds - config.uav_speed_max_m_s, 0.0
    ).sum())
    separation_deficit = 0.0
    for slot in range(config.num_slots):
        for first in range(config.num_uavs):
            for second in range(first + 1, config.num_uavs):
                distance = float(np.linalg.norm(
                    positions_m[slot, first] - positions_m[slot, second]
                ))
                separation_deficit += max(
                    config.minimum_separation_m - distance, 0.0
                )
    flight_energy = flight_power(speeds, config) * config.slot_seconds
    return speeds, speed_excess, separation_deficit, flight_energy


def solve_uptsm(
    config: PaperConfig,
    scenario: Scenario,
    decision: ContinuousDecision,
    *,
    cooperative: bool = True,
) -> UPTSMResult:
    """Algorithm 1 and equations (5)-(40), evaluated chronologically."""
    config.validate()
    scenario.validate(config)
    decision.validate(config, scenario)
    n_slots, num_users, num_uavs = (
        config.num_slots, config.num_users, config.num_uavs
    )
    a = np.zeros((n_slots, num_users, num_uavs), dtype=np.int8)
    b = np.zeros((n_slots, num_uavs, num_uavs), dtype=np.int8)
    slot_utility = np.zeros(n_slots, dtype=np.float64)
    processing_times = np.zeros((n_slots, num_uavs), dtype=np.float64)
    time_excess = np.zeros((n_slots, num_uavs), dtype=np.float64)
    cumulative_energy = np.asarray(
        scenario.initial_processing_energy_used_j, dtype=np.float64
    ).copy()
    cumulative_offloaded = np.zeros(num_users, dtype=np.float64)
    _, speed_excess, separation_deficit, flight_energy = _trajectory_penalties(
        config, scenario, decision.positions_m
    )

    for slot in range(n_slots):
        tasks = scenario.task_bits[slot]
        positions = decision.positions_m[slot]
        upload_rates = offloading_rates(
            scenario.ue_positions_m, positions, config
        )
        fairness_increment = config.fairness_subsidy_price * np.log(
            (cumulative_offloaded + tasks + 1.0)
            / (cumulative_offloaded + 1.0)
        )
        # Equation (34): UE-UAV potential profit.
        edge_profit = (
            config.offloaded_bit_price * tasks[:, None]
            - config.time_cost_weight * tasks[:, None] / upload_rates
            + fairness_increment[:, None]
        )
        pairs = maximum_weight_uav_matching(
            edge_profit, allow_idle=config.allow_idle_uavs
        )
        selected_user = np.full(num_uavs, -1, dtype=np.int64)
        selected_bits = np.zeros(num_uavs, dtype=np.float64)
        upload_time = np.zeros(num_uavs, dtype=np.float64)
        for user, origin in pairs:
            a[slot, user, origin] = 1
            selected_user[origin] = user
            selected_bits[origin] = tasks[user]
            upload_time[origin] = tasks[user] / upload_rates[user, origin]

        energy_weight = dynamic_energy_weight(cumulative_energy, config)
        edge_cost = np.zeros((num_uavs, num_uavs), dtype=np.float64)
        tx_time_candidates = np.zeros_like(edge_cost)
        comp_time_candidates = np.zeros_like(edge_cost)
        tx_energy_candidates = np.zeros_like(edge_cost)
        comp_energy_candidates = np.zeros_like(edge_cost)

        for origin in range(num_uavs):
            bits = selected_bits[origin]
            for destination in range(num_uavs):
                rate = cooperation_rate(
                    origin,
                    destination,
                    positions,
                    decision.transmit_power_w[slot, origin],
                    config,
                )
                tx_time = _safe_duration(bits, rate)
                cpu = decision.cpu_hz[slot, destination]
                comp_time = (
                    0.0 if bits <= 0.0 else
                    config.cycles_per_bit * bits / max(cpu, 1e-12)
                )
                tx_energy = decision.transmit_power_w[slot, origin] * tx_time
                comp_energy = (
                    config.effective_capacitance * cpu ** 3 * comp_time
                )
                tx_time_candidates[origin, destination] = tx_time
                comp_time_candidates[origin, destination] = comp_time
                tx_energy_candidates[origin, destination] = tx_energy
                comp_energy_candidates[origin, destination] = comp_energy

                cost = config.time_cost_weight * (tx_time + comp_time)
                cost += energy_weight[origin] * tx_energy
                cost += energy_weight[destination] * comp_energy
                cost += config.penalty_time * max(
                    upload_time[origin] + tx_time + comp_time
                    - config.slot_seconds,
                    0.0,
                )
                # Literal edge penalties in equation (40).
                cost += config.penalty_energy * max(
                    cumulative_energy[origin] + tx_energy
                    - config.processing_energy_budget_j,
                    0.0,
                )
                cost += config.penalty_energy * max(
                    cumulative_energy[destination] + comp_energy
                    - config.processing_energy_budget_j,
                    0.0,
                )
                edge_cost[origin, destination] = min(cost, 1e300)

        if cooperative:
            origins, destinations = linear_sum_assignment(edge_cost)
        else:
            origins = np.arange(num_uavs, dtype=np.int64)
            destinations = origins.copy()

        tx_energy_by_uav = np.zeros(num_uavs, dtype=np.float64)
        comp_energy_by_uav = np.zeros(num_uavs, dtype=np.float64)
        tx_time_by_origin = np.zeros(num_uavs, dtype=np.float64)
        comp_time_by_origin = np.zeros(num_uavs, dtype=np.float64)
        for origin, destination in zip(origins.tolist(), destinations.tolist()):
            b[slot, origin, destination] = 1
            tx_time_by_origin[origin] = tx_time_candidates[origin, destination]
            comp_time_by_origin[origin] = comp_time_candidates[origin, destination]
            tx_energy_by_uav[origin] += tx_energy_candidates[origin, destination]
            comp_energy_by_uav[destination] += comp_energy_candidates[
                origin, destination
            ]

        processing_times[slot] = (
            upload_time + tx_time_by_origin + comp_time_by_origin
        )
        time_excess[slot] = np.maximum(
            processing_times[slot] - config.slot_seconds, 0.0
        )
        processing_energy = tx_energy_by_uav + comp_energy_by_uav
        energy_cost = float(np.sum(energy_weight * processing_energy))
        cumulative_energy += processing_energy

        selected_mask = selected_user >= 0
        for origin in np.flatnonzero(selected_mask):
            cumulative_offloaded[selected_user[origin]] += selected_bits[origin]
        offload_reward = config.offloaded_bit_price * float(
            np.sum(selected_bits)
        )
        fairness_subsidy = config.fairness_subsidy_price * float(
            np.log(cumulative_offloaded + 1.0).sum()
        )
        time_cost = config.time_cost_weight * float(
            np.sum(processing_times[slot])
        )
        flight_cost = config.flight_cost_weight * float(
            np.sum(flight_energy[slot])
        )
        slot_utility[slot] = (
            offload_reward + fairness_subsidy
            - energy_cost - time_cost - flight_cost
        )

    final_energy_excess = float(np.maximum(
        cumulative_energy - config.processing_energy_budget_j, 0.0
    ).sum())
    penalty_total = (
        config.penalty_speed * speed_excess
        + config.penalty_position * separation_deficit
        + config.penalty_time * float(time_excess.sum())
        + config.penalty_energy * final_energy_excess
    )
    objective = float(slot_utility.sum())
    remaining = np.maximum(
        config.processing_energy_budget_j - cumulative_energy, 0.0
    )
    return UPTSMResult(
        objective_utility=objective,
        offloading_selection=a,
        computation_selection=b,
        slot_utility=slot_utility,
        cumulative_processing_energy_j=cumulative_energy,
        cumulative_offloaded_bits=cumulative_offloaded,
        task_processing_time_s=processing_times,
        time_excess_s=time_excess,
        flight_energy_j=flight_energy,
        speed_excess_m_s=speed_excess,
        separation_deficit_m=separation_deficit,
        final_energy_excess_j=final_energy_excess,
        penalty_total=float(penalty_total),
        fitness=float(objective - penalty_total),
        task_fairness=jain_fairness(cumulative_offloaded),
        battery_fairness=jain_fairness(remaining),
    )
