"""SAC adapter for the paper's continuous 4UN optimization problem.

The adapter deliberately keeps the UPTSM evaluator authoritative.  SAC only
proposes the continuous variables; Algorithm 1 still resolves both discrete
matchings, and Algorithm 2's time-variable repair can be applied identically
before the final score is reported.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PaperConfig
from .scenario import Scenario
from .uptsm import UPTSMResult, solve_uptsm
from .vcwoa import DecisionCodec


@dataclass
class AdapterEvaluation:
    action: np.ndarray
    reward: float
    constraint_cost: float
    result: UPTSMResult


class SACPaperProblemAdapter:
    """A contextual-bandit view of one complete paper optimization instance.

    Observation
        Normalized UE coordinates, all per-slot task sizes, UAV endpoints and
        initial remaining processing-energy fractions.

    Action
        The same normalized 4UN continuous vector optimized by VC-WOA:
        CPU frequencies, cooperation powers and horizontal coordinates.

    The learning reward is numerically scaled for neural-network stability.
    Publication metrics always use ``result.objective_utility`` and the raw
    feasibility constraints rather than this shaped reward.
    """

    def __init__(
        self,
        config: PaperConfig,
        scenario: Scenario,
        *,
        cooperative: bool = True,
        repair_time_variables: bool = True,
        project_trajectory: bool = False,
        constraint_penalty: float = 50.0,
    ):
        config.validate()
        scenario.validate(config)
        self.config = config
        self.scenario = scenario
        self.cooperative = bool(cooperative)
        self.repair_time_variables = bool(repair_time_variables)
        self.project_trajectory = bool(project_trajectory)
        self.constraint_penalty = float(constraint_penalty)
        if self.constraint_penalty < 0.0:
            raise ValueError("constraint_penalty must be nonnegative")
        self.codec = DecisionCodec(config, scenario)
        self.action_dim = self.codec.dimension
        self._observation = self._encode_observation()
        self.state_dim = int(self._observation.size)

    def _encode_observation(self) -> np.ndarray:
        config = self.config
        scenario = self.scenario
        task_span = max(config.task_bits_max - config.task_bits_min, 1.0)
        task_values = (
            scenario.task_bits - config.task_bits_min
        ) / task_span
        remaining_energy = 1.0 - (
            scenario.initial_processing_energy_used_j
            / config.processing_energy_budget_j
        )
        values = np.concatenate([
            scenario.ue_positions_m.reshape(-1) / config.area_size_m,
            task_values.reshape(-1),
            scenario.uav_initial_positions_m.reshape(-1)
            / config.area_size_m,
            scenario.uav_final_positions_m.reshape(-1)
            / config.area_size_m,
            remaining_energy.reshape(-1),
        ])
        return np.asarray(np.clip(values, 0.0, 1.0), dtype=np.float32)

    def observation(self) -> np.ndarray:
        return self._observation.copy()

    def action_from_vector(self, normalized_vector: np.ndarray) -> np.ndarray:
        vector = self.codec.sanitize(normalized_vector)
        action = 2.0 * vector - 1.0
        if self.project_trajectory:
            decision = self.codec.decode(vector)
            previous = np.concatenate([
                self.scenario.uav_initial_positions_m[None, :, :],
                decision.positions_m[:-1],
            ], axis=0)
            step_limit = (
                self.config.uav_speed_max_m_s * self.config.slot_seconds
            )
            commands = np.clip(
                (decision.positions_m - previous) / max(step_limit, 1e-12),
                -1.0,
                1.0,
            )
            action[2 * self.codec.scalar_size:] = commands.reshape(-1)
        return np.asarray(action, dtype=np.float32)

    @staticmethod
    def _project_to_disk(point, centre, radius):
        delta = np.asarray(point, dtype=np.float64) - centre
        distance = float(np.linalg.norm(delta))
        if distance <= radius or distance <= 1e-15:
            return np.asarray(point, dtype=np.float64)
        return centre + delta * (radius / distance)

    def _project_point(self, point, previous, final, remaining_steps):
        """Project onto speed, terminal-reachability and area constraints."""
        step_limit = (
            self.config.uav_speed_max_m_s * self.config.slot_seconds
        )
        reach_radius = float(remaining_steps) * step_limit
        desired = np.clip(point, 0.0, self.config.area_size_m)
        # The equal-progress point is feasible whenever the previous point
        # satisfies the induction invariant d(previous, final) <= (r+1)L.
        base = previous + (final - previous) / float(remaining_steps + 1)
        base = np.clip(base, 0.0, self.config.area_size_m)

        def feasible(candidate):
            return (
                np.linalg.norm(candidate - previous) <= step_limit + 1e-10
                and np.linalg.norm(candidate - final) <= reach_radius + 1e-10
                and np.all(candidate >= -1e-12)
                and np.all(candidate <= self.config.area_size_m + 1e-12)
            )

        if feasible(desired):
            return desired
        # The feasible set is convex.  Intersecting the segment from the
        # guaranteed-feasible base to the desired point therefore reduces to
        # a one-dimensional boundary search with a feasible lower endpoint.
        lower = 0.0
        upper = 1.0
        direction = desired - base
        for _ in range(60):
            middle = 0.5 * (lower + upper)
            candidate = base + middle * direction
            if feasible(candidate):
                lower = middle
            else:
                upper = middle
        return base + lower * direction

    def _separate_slot(self, points, previous, remaining_steps):
        if self.config.num_uavs < 2:
            return points
        adjusted = np.asarray(points, dtype=np.float64).copy()
        minimum = self.config.minimum_separation_m
        finals = self.scenario.uav_final_positions_m
        for iteration in range(16):
            changed = False
            for first in range(self.config.num_uavs):
                for second in range(first + 1, self.config.num_uavs):
                    delta = adjusted[first] - adjusted[second]
                    distance = float(np.linalg.norm(delta))
                    if distance + 1e-10 >= minimum:
                        continue
                    changed = True
                    if distance <= 1e-12:
                        angle = 2.0 * np.pi * (
                            first + second + iteration + 1
                        ) / max(self.config.num_uavs + 3, 4)
                        direction = np.array([
                            np.cos(angle), np.sin(angle)
                        ])
                    else:
                        direction = delta / distance
                    shift = 0.5 * (minimum - distance + 1e-9)
                    adjusted[first] += direction * shift
                    adjusted[second] -= direction * shift
                    for index in (first, second):
                        adjusted[index] = self._project_point(
                            adjusted[index],
                            previous[index],
                            finals[index],
                            remaining_steps,
                        )
            if not changed:
                break
        return adjusted

    def _trajectory_from_commands(self, commands):
        config = self.config
        n_slots = config.num_slots
        step_limit = config.uav_speed_max_m_s * config.slot_seconds
        previous = self.scenario.uav_initial_positions_m.copy()
        final = self.scenario.uav_final_positions_m
        positions = np.empty(
            (n_slots, config.num_uavs, 2), dtype=np.float64
        )
        for slot in range(n_slots):
            remaining_steps = n_slots - slot - 1
            if remaining_steps == 0:
                positions[slot] = final
                break
            command = np.asarray(commands[slot], dtype=np.float64)
            norms = np.linalg.norm(command, axis=1, keepdims=True)
            command = command / np.maximum(norms, 1.0)
            proposed = previous + step_limit * command
            for uav in range(config.num_uavs):
                proposed[uav] = self._project_point(
                    proposed[uav],
                    previous[uav],
                    final[uav],
                    remaining_steps,
                )
            proposed = self._separate_slot(
                proposed, previous, remaining_steps
            )
            positions[slot] = proposed
            previous = proposed
        positions[-1] = final
        return positions

    def vector_from_action(self, action: np.ndarray) -> np.ndarray:
        value = np.asarray(action, dtype=np.float64).reshape(-1)
        if value.shape != (self.action_dim,):
            raise ValueError(
                "action expected shape %r, got %r"
                % ((self.action_dim,), value.shape)
            )
        clipped = np.clip(value, -1.0, 1.0)
        normalized = (clipped + 1.0) / 2.0
        if not self.project_trajectory:
            return self.codec.sanitize(normalized)
        scalar_size = self.codec.scalar_size
        commands = clipped[2 * scalar_size:].reshape(
            self.config.num_slots, self.config.num_uavs, 2
        )
        decision = self.codec.decode(normalized)
        decision.positions_m = self._trajectory_from_commands(commands)
        return self.codec.encode(decision)

    def random_feasible_prior_action(
        self, rng: np.random.RandomState
    ) -> np.ndarray:
        """Return the paper optimizer's smooth endpoint-fixed initialization."""
        return self.action_from_vector(self.codec.random_vector(rng))

    def _repair_time(self, vector, result):
        if not self.repair_time_variables or not np.any(
            result.time_excess_s > 1e-9
        ):
            return vector, result
        decision = self.codec.decode(vector)
        slots, origins = np.where(result.time_excess_s > 1e-9)
        for slot, origin in zip(slots.tolist(), origins.tolist()):
            destinations = np.flatnonzero(
                result.computation_selection[slot, origin]
            )
            if destinations.size:
                decision.transmit_power_w[slot, origin] = (
                    self.config.uav_tx_max_w
                )
                decision.cpu_hz[slot, int(destinations[0])] = (
                    self.config.cpu_max_hz
                )
        repaired = self.codec.encode(decision)
        reevaluated = solve_uptsm(
            self.config,
            self.scenario,
            self.codec.decode(repaired),
            cooperative=self.cooperative,
        )
        return repaired, reevaluated

    def normalized_constraint_cost(self, result: UPTSMResult) -> float:
        config = self.config
        pairs = config.num_uavs * max(config.num_uavs - 1, 1) / 2.0
        time_cost = float(result.time_excess_s.sum()) / max(
            config.num_slots * config.num_uavs * config.slot_seconds, 1e-12
        )
        speed_cost = result.speed_excess_m_s / max(
            config.num_slots * config.num_uavs
            * config.uav_speed_max_m_s,
            1e-12,
        )
        separation_cost = result.separation_deficit_m / max(
            config.num_slots * pairs * config.minimum_separation_m,
            1e-12,
        )
        energy_cost = result.final_energy_excess_j / max(
            config.num_uavs * config.processing_energy_budget_j, 1e-12
        )
        return float(time_cost + speed_cost + separation_cost + energy_cost)

    def evaluate_action(self, action: np.ndarray) -> AdapterEvaluation:
        vector = self.vector_from_action(action)
        result = solve_uptsm(
            self.config,
            self.scenario,
            self.codec.decode(vector),
            cooperative=self.cooperative,
        )
        vector, result = self._repair_time(vector, result)
        repaired_action = self.action_from_vector(vector)
        cost = self.normalized_constraint_cost(result)
        scaled_utility = result.objective_utility / max(
            float(self.config.num_slots), 1.0
        )
        reward = scaled_utility - self.constraint_penalty * cost
        return AdapterEvaluation(
            action=repaired_action,
            reward=float(np.clip(reward, -1e4, 1e4)),
            constraint_cost=cost,
            result=result,
        )
