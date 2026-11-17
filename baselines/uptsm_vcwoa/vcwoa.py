from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import numpy as np

from .config import PaperConfig
from .scenario import Scenario
from .uptsm import ContinuousDecision, UPTSMResult, solve_uptsm


@dataclass
class OptimizationResult:
    algorithm: str
    decision: ContinuousDecision
    evaluation: UPTSMResult
    best_fitness_history: np.ndarray
    best_utility_history: np.ndarray
    feasible_iteration: int | None
    elapsed_seconds: float
    seed: int


class DecisionCodec:
    """Normalized [0,1] representation of the paper's 4UN search vector."""

    def __init__(self, config: PaperConfig, scenario: Scenario):
        self.config = config
        self.scenario = scenario
        self.scalar_size = config.num_slots * config.num_uavs
        self.position_size = self.scalar_size * 2
        self.dimension = self.scalar_size * 2 + self.position_size

    def sanitize(self, vector: np.ndarray) -> np.ndarray:
        value = np.clip(np.asarray(vector, dtype=np.float64), 0.0, 1.0).copy()
        if value.shape != (self.dimension,):
            raise ValueError(
                f"search vector expected {(self.dimension,)}, got {value.shape}"
            )
        positions = value[2 * self.scalar_size:].reshape(
            self.config.num_slots, self.config.num_uavs, 2
        )
        positions[-1] = (
            self.scenario.uav_final_positions_m / self.config.area_size_m
        )
        return value

    def decode(self, vector: np.ndarray) -> ContinuousDecision:
        value = self.sanitize(vector)
        cpu = value[:self.scalar_size].reshape(
            self.config.num_slots, self.config.num_uavs
        ) * self.config.cpu_max_hz
        power = value[self.scalar_size:2 * self.scalar_size].reshape(
            self.config.num_slots, self.config.num_uavs
        ) * self.config.uav_tx_max_w
        positions = value[2 * self.scalar_size:].reshape(
            self.config.num_slots, self.config.num_uavs, 2
        ) * self.config.area_size_m
        positions[-1] = self.scenario.uav_final_positions_m
        return ContinuousDecision(cpu, power, positions)

    def encode(self, decision: ContinuousDecision) -> np.ndarray:
        return self.sanitize(np.concatenate([
            np.asarray(decision.cpu_hz).reshape(-1) / self.config.cpu_max_hz,
            np.asarray(decision.transmit_power_w).reshape(-1)
            / self.config.uav_tx_max_w,
            np.asarray(decision.positions_m).reshape(-1)
            / self.config.area_size_m,
        ]))

    def random_vector(self, rng: np.random.RandomState) -> np.ndarray:
        # Algorithm 2 states that the population is generated randomly over
        # the legal variable ranges.  Keeping the full [0,1] range is also
        # essential for exposing the task-time violations that VC-WOA repairs.
        cpu = rng.uniform(0.0, 1.0, size=self.scalar_size)
        power = rng.uniform(0.0, 1.0, size=self.scalar_size)
        positions = self._smooth_random_trajectory(rng).reshape(-1)
        return self.sanitize(np.concatenate([cpu, power, positions]))

    def _smooth_random_trajectory(self, rng: np.random.RandomState) -> np.ndarray:
        n_slots = self.config.num_slots
        num_uavs = self.config.num_uavs
        initial = self.scenario.uav_initial_positions_m
        final = self.scenario.uav_final_positions_m
        progress = np.linspace(1.0 / n_slots, 1.0, n_slots)[:, None, None]
        base = initial[None, :, :] + progress * (final - initial)[None, :, :]
        directions = rng.normal(size=(num_uavs, 2))
        directions /= np.maximum(
            np.linalg.norm(directions, axis=1, keepdims=True), 1e-12
        )
        max_amplitude = min(
            self.config.area_size_m * 0.08,
            self.config.uav_speed_max_m_s * self.config.slot_seconds
            * n_slots / (4.0 * np.pi),
        )
        amplitudes = rng.uniform(0.0, max_amplitude, size=(1, num_uavs, 1))
        envelope = np.sin(np.pi * progress)
        positions = base + envelope * amplitudes * directions[None, :, :]
        positions = np.clip(positions, 0.0, self.config.area_size_m)
        positions[-1] = final
        return positions / self.config.area_size_m


def _evaluate_population(
    population: np.ndarray,
    codec: DecisionCodec,
    evaluator: Callable[[ContinuousDecision], UPTSMResult],
) -> list[UPTSMResult]:
    return [evaluator(codec.decode(vector)) for vector in population]


def _repair_best_time_variables(
    vector: np.ndarray,
    evaluation: UPTSMResult,
    codec: DecisionCodec,
    evaluator: Callable[[ContinuousDecision], UPTSMResult],
) -> tuple[np.ndarray, UPTSMResult]:
    """Algorithm 2 step 24: set the offending P and F entries to maxima."""
    decision = codec.decode(vector)
    violating_slots, origins = np.where(evaluation.time_excess_s > 1e-9)
    for slot, origin in zip(violating_slots.tolist(), origins.tolist()):
        destinations = np.flatnonzero(
            evaluation.computation_selection[slot, origin]
        )
        if destinations.size == 0:
            continue
        decision.transmit_power_w[slot, origin] = codec.config.uav_tx_max_w
        decision.cpu_hz[slot, int(destinations[0])] = codec.config.cpu_max_hz
    repaired_vector = codec.encode(decision)
    return repaired_vector, evaluator(codec.decode(repaired_vector))


def _optimize_woa(
    config: PaperConfig,
    scenario: Scenario,
    seed: int,
    variable_constraints: bool,
    cooperative: bool,
) -> OptimizationResult:
    start = time.perf_counter()
    rng = np.random.RandomState(seed)
    codec = DecisionCodec(config, scenario)
    evaluator = lambda decision: solve_uptsm(
        config, scenario, decision, cooperative=cooperative
    )
    population = np.stack([
        codec.random_vector(rng) for _ in range(config.population_size)
    ])
    evaluations = _evaluate_population(population, codec, evaluator)
    best_index = int(np.argmax([item.fitness for item in evaluations]))
    best_vector = population[best_index].copy()
    best_evaluation = evaluations[best_index]

    fitness_history = []
    utility_history = []
    feasible_iteration = 0 if best_evaluation.feasible else None

    for iteration in range(config.woa_iterations):
        alpha = 2.0 * (1.0 - iteration / config.woa_iterations)
        next_population = np.empty_like(population)
        for index, current in enumerate(population):
            gamma = float(rng.random_sample())
            coefficient_a = 2.0 * alpha * gamma - alpha
            coefficient_c = 2.0 * gamma
            probability = float(rng.random_sample())
            if probability < 0.5:
                if abs(coefficient_a) < 1.0:
                    distance = np.abs(
                        coefficient_c * best_vector - current
                    )
                    updated = best_vector - coefficient_a * distance
                else:
                    random_index = int(rng.randint(config.population_size))
                    random_agent = population[random_index]
                    distance = np.abs(
                        coefficient_c * random_agent - current
                    )
                    updated = random_agent - coefficient_a * distance
            else:
                spiral_l = float(rng.uniform(-1.0, 1.0))
                distance = np.abs(best_vector - current)
                updated = (
                    np.exp(config.spiral_beta * spiral_l)
                    * np.cos(2.0 * np.pi * spiral_l)
                    * distance + best_vector
                )
            next_population[index] = codec.sanitize(updated)

        next_evaluations = _evaluate_population(
            next_population, codec, evaluator
        )
        current_best_index = int(np.argmax(
            [item.fitness for item in next_evaluations]
        ))
        if variable_constraints:
            repaired_vector, repaired_evaluation = _repair_best_time_variables(
                next_population[current_best_index],
                next_evaluations[current_best_index],
                codec,
                evaluator,
            )
            next_population[current_best_index] = repaired_vector
            next_evaluations[current_best_index] = repaired_evaluation

        population = next_population
        evaluations = next_evaluations
        current_best_index = int(np.argmax(
            [item.fitness for item in evaluations]
        ))
        if evaluations[current_best_index].fitness > best_evaluation.fitness:
            best_vector = population[current_best_index].copy()
            best_evaluation = evaluations[current_best_index]
        if feasible_iteration is None and best_evaluation.feasible:
            feasible_iteration = iteration + 1
        fitness_history.append(best_evaluation.fitness)
        utility_history.append(best_evaluation.objective_utility)

    return OptimizationResult(
        algorithm="vcwoa" if variable_constraints else "woa",
        decision=codec.decode(best_vector),
        evaluation=best_evaluation,
        best_fitness_history=np.asarray(fitness_history),
        best_utility_history=np.asarray(utility_history),
        feasible_iteration=feasible_iteration,
        elapsed_seconds=time.perf_counter() - start,
        seed=seed,
    )


def _optimize_pso(
    config: PaperConfig,
    scenario: Scenario,
    seed: int,
    cooperative: bool,
) -> OptimizationResult:
    """Conventional PSO baseline used in the paper's Figure 6."""
    start = time.perf_counter()
    rng = np.random.RandomState(seed)
    codec = DecisionCodec(config, scenario)
    evaluator = lambda decision: solve_uptsm(
        config, scenario, decision, cooperative=cooperative
    )
    positions = np.stack([
        codec.random_vector(rng) for _ in range(config.population_size)
    ])
    velocities = rng.normal(0.0, 0.05, size=positions.shape)
    evaluations = _evaluate_population(positions, codec, evaluator)
    personal_positions = positions.copy()
    personal_evaluations = list(evaluations)
    best_index = int(np.argmax([item.fitness for item in evaluations]))
    global_position = positions[best_index].copy()
    global_evaluation = evaluations[best_index]
    fitness_history = []
    utility_history = []
    feasible_iteration = 0 if global_evaluation.feasible else None

    for iteration in range(config.woa_iterations):
        r1 = rng.random_sample(size=positions.shape)
        r2 = rng.random_sample(size=positions.shape)
        velocities = (
            0.7 * velocities
            + 1.5 * r1 * (personal_positions - positions)
            + 1.5 * r2 * (global_position[None, :] - positions)
        )
        positions = np.stack([
            codec.sanitize(vector)
            for vector in np.clip(positions + velocities, 0.0, 1.0)
        ])
        evaluations = _evaluate_population(positions, codec, evaluator)
        for index, evaluation in enumerate(evaluations):
            if evaluation.fitness > personal_evaluations[index].fitness:
                personal_positions[index] = positions[index].copy()
                personal_evaluations[index] = evaluation
            if evaluation.fitness > global_evaluation.fitness:
                global_position = positions[index].copy()
                global_evaluation = evaluation
        if feasible_iteration is None and global_evaluation.feasible:
            feasible_iteration = iteration + 1
        fitness_history.append(global_evaluation.fitness)
        utility_history.append(global_evaluation.objective_utility)

    return OptimizationResult(
        algorithm="pso",
        decision=codec.decode(global_position),
        evaluation=global_evaluation,
        best_fitness_history=np.asarray(fitness_history),
        best_utility_history=np.asarray(utility_history),
        feasible_iteration=feasible_iteration,
        elapsed_seconds=time.perf_counter() - start,
        seed=seed,
    )


def optimize(
    config: PaperConfig,
    scenario: Scenario,
    *,
    algorithm: str = "vcwoa",
    seed: int | None = None,
    cooperative: bool = True,
) -> OptimizationResult:
    """Run VC-WOA (Algorithm 2), standard WOA, or the paper's PSO baseline."""
    name = algorithm.strip().lower()
    actual_seed = config.seed if seed is None else int(seed)
    if name in {"vcwoa", "vc-woa"}:
        return _optimize_woa(
            config, scenario, actual_seed, True, cooperative
        )
    if name == "woa":
        return _optimize_woa(
            config, scenario, actual_seed, False, cooperative
        )
    if name == "pso":
        return _optimize_pso(config, scenario, actual_seed, cooperative)
    raise ValueError(f"unsupported algorithm: {algorithm}")
