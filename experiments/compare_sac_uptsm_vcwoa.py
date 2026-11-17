"""Compare projected SAC, VC-WOA, standard WOA and PSO on one problem.

This is a contextual-bandit optimizer benchmark.  It intentionally excludes
the native LSTM energy predictor and emergency-return controller because the
paper model has neither wind observations nor a return-to-base safety state.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "baselines"))

from uptsm_vcwoa.config import PaperConfig, paper_default_path
from uptsm_vcwoa.sac_adapter import SACPaperProblemAdapter
from uptsm_vcwoa.scenario import build_scenario
from uptsm_vcwoa.vcwoa import optimize


def parse_seed_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Shared-problem SAC versus paper VC-WOA benchmark"
    )
    parser.add_argument("--config", default=str(paper_default_path()))
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--evaluation-scenarios", type=int, default=10)
    parser.add_argument("--train-episodes", type=int, default=3000)
    parser.add_argument("--warmup-samples", type=int, default=512)
    parser.add_argument("--bc-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--sac-variants",
        default="raw,projected",
        help="Comma-separated SAC action modes: raw, projected",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--population-algorithms",
        default="vcwoa,woa,pso",
        help="Comma-separated population baselines: vcwoa, woa, pso",
    )
    parser.add_argument(
        "--skip-population-baselines", "--skip-vcwoa",
        dest="skip_population_baselines", action="store_true",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "results" / "sac_vs_uptsm_vcwoa"),
    )
    return parser.parse_args()


def quick_config(config):
    return replace(
        config,
        num_users=6,
        num_uavs=2,
        num_slots=6,
        population_size=6,
        woa_iterations=3,
    )


def sac_config(config, args):
    return {
        "action_projection": "identity",
        "cmdp_enabled": True,
        "cmdp_cost_limit": 0.0,
        "lagrange_init": 0.1,
        "lagrange_lr": 1e-4,
        "lagrange_max": 100.0,
        "gamma": 0.0,
        "tau": 0.005,
        "lr": 3e-4,
        "actor_lr": 1e-4,
        "alpha_lr": 3e-4,
        "actor_update_delay": 1,
        "warmup_steps": 0,
        "batch_size": int(args.batch_size),
        "buffer_capacity": max(
            10000, int(args.warmup_samples + args.train_episodes + 1)
        ),
        "replay_alpha": 0.0,
        "target_entropy": None,
        "num_users": int(config.num_users),
    }


def training_scenario_seed(policy_seed, index):
    return int(policy_seed) * 1000003 + int(index)


def evaluation_scenario_seeds(count):
    return [9000001 + index for index in range(int(count))]


def make_adapter(config, scenario_seed, project_trajectory=False):
    return SACPaperProblemAdapter(
        config,
        build_scenario(config, seed=scenario_seed),
        cooperative=True,
        repair_time_variables=True,
        project_trajectory=project_trajectory,
    )


def evaluation_row(name, policy_seed, scenario_seed, evaluated, elapsed,
                   objective_evaluations=1):
    result = evaluated.result
    return {
        "algorithm": name,
        "policy_seed": policy_seed,
        "scenario_seed": int(scenario_seed),
        "paper_utility": float(result.objective_utility),
        "paper_fitness": float(result.fitness),
        "feasible": bool(result.feasible),
        "constraint_cost": float(evaluated.constraint_cost),
        "time_excess_s": float(result.time_excess_s.sum()),
        "speed_excess_m_s": float(result.speed_excess_m_s),
        "separation_deficit_m": float(result.separation_deficit_m),
        "energy_excess_j": float(result.final_energy_excess_j),
        "task_fairness": float(result.task_fairness),
        "battery_fairness": float(result.battery_fairness),
        "elapsed_seconds": float(elapsed),
        "objective_evaluations": int(objective_evaluations),
    }


def train_sac_policy(
    config, args, policy_seed, scenario_seeds, output, variant
):
    # Torch and the native SAC package remain optional for baseline-only use.
    from agent.sac import SAC
    from utils.seeding import seed_everything

    seed_everything(policy_seed)
    rng = np.random.RandomState(policy_seed)
    projected = variant == "projected"
    algorithm_name = (
        "SAC + trajectory projection" if projected
        else "SAC raw action adapter"
    )
    first_adapter = make_adapter(
        config, training_scenario_seed(policy_seed, 0), projected
    )
    agent_settings = sac_config(config, args)
    agent = SAC(
        first_adapter.state_dim,
        first_adapter.action_dim,
        agent_settings,
    )

    # Smooth endpoint-fixed paper initializations keep BC and early replay
    # inside the same variable domain used by VC-WOA.
    for index in range(int(args.warmup_samples)):
        adapter = make_adapter(
            config, training_scenario_seed(policy_seed, index), projected
        )
        state = adapter.observation()
        action = adapter.random_feasible_prior_action(rng)
        evaluated = adapter.evaluate_action(action)
        agent.replay_buffer.push(
            state,
            evaluated.action,
            evaluated.reward,
            state,
            True,
            cost=evaluated.constraint_cost,
        )
    if args.bc_steps > 0 and args.warmup_samples >= args.batch_size:
        agent.pretrain_actor_with_bc(
            agent.replay_buffer,
            steps=int(args.bc_steps),
            batch_size=int(args.batch_size),
        )

    history = []
    started = time.perf_counter()
    for episode in range(int(args.train_episodes)):
        adapter = make_adapter(
            config,
            training_scenario_seed(
                policy_seed, int(args.warmup_samples) + episode
            ),
            projected,
        )
        state = adapter.observation()
        exploration = max(
            0.05, 0.50 * (1.0 - episode / max(args.train_episodes, 1))
        )
        if rng.random_sample() < exploration:
            action = adapter.random_feasible_prior_action(rng)
        else:
            action = agent.select_action(state, eval=False)
        evaluated = adapter.evaluate_action(action)
        agent.replay_buffer.push(
            state,
            evaluated.action,
            evaluated.reward,
            state,
            True,
            cost=evaluated.constraint_cost,
        )
        losses = agent.update()
        agent.update_lagrange(evaluated.constraint_cost)
        history.append({
            "algorithm": algorithm_name,
            "variant": variant,
            "policy_seed": int(policy_seed),
            "episode": episode + 1,
            "learning_reward": float(evaluated.reward),
            "paper_utility": float(evaluated.result.objective_utility),
            "feasible": bool(evaluated.result.feasible),
            "constraint_cost": float(evaluated.constraint_cost),
            "alpha": float(losses.get("alpha", agent.alpha.item())),
            "lagrange": float(agent.lagrange_multiplier.item()),
        })
        if (episode + 1) % max(1, args.train_episodes // 10) == 0:
            recent = history[-min(100, len(history)):]
            print(
                "SAC-%s seed=%d episode=%d/%d reward=%.3f feasible=%.1f%%"
                % (
                    variant,
                    policy_seed,
                    episode + 1,
                    args.train_episodes,
                    np.mean([row["learning_reward"] for row in recent]),
                    100.0 * np.mean([row["feasible"] for row in recent]),
                ),
                flush=True,
            )

    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    agent.save(str(model_dir / (
        "sac_paper013_%s_seed%d.pt" % (variant, policy_seed)
    )))
    training_seconds = time.perf_counter() - started

    rows = []
    for scenario_seed in scenario_seeds:
        adapter = make_adapter(config, scenario_seed, projected)
        started_evaluation = time.perf_counter()
        action = agent.select_action(adapter.observation(), eval=True)
        evaluated = adapter.evaluate_action(action)
        rows.append(evaluation_row(
            algorithm_name,
            int(policy_seed),
            scenario_seed,
            evaluated,
            time.perf_counter() - started_evaluation,
        ))
    return history, rows, training_seconds, agent_settings


def run_population_baselines(config, scenario_seeds, algorithms):
    rows = []
    names = {
        "vcwoa": "VC-WOA (Algorithm 2)",
        "woa": "WOA",
        "pso": "PSO",
    }
    nominal_evaluations = int(
        config.population_size * (config.woa_iterations + 1)
    )
    for algorithm in algorithms:
        for scenario_seed in scenario_seeds:
            scenario = build_scenario(config, seed=scenario_seed)
            started = time.perf_counter()
            optimized = optimize(
                config,
                scenario,
                algorithm=algorithm,
                seed=scenario_seed,
                cooperative=True,
            )
            result = optimized.evaluation
            adapter = SACPaperProblemAdapter(config, scenario)
            action = adapter.action_from_vector(
                adapter.codec.encode(optimized.decision)
            )
            evaluated = adapter.evaluate_action(action)
            rows.append(evaluation_row(
                names[algorithm],
                None,
                scenario_seed,
                evaluated,
                time.perf_counter() - started,
                objective_evaluations=nominal_evaluations,
            ))
            print(
                "%s scenario=%d utility=%.3f feasible=%s"
                % (
                    names[algorithm], scenario_seed,
                    result.objective_utility, result.feasible,
                ),
                flush=True,
            )
    return rows


def run_vcwoa(config, scenario_seeds):
    """Backward-compatible wrapper for earlier experiment scripts."""
    return run_population_baselines(config, scenario_seeds, ["vcwoa"])


def aggregate(rows, name):
    selected = [row for row in rows if row["algorithm"] == name]
    utilities = np.asarray(
        [row["paper_utility"] for row in selected], dtype=np.float64
    )
    feasible = np.asarray(
        [row["feasible"] for row in selected], dtype=np.float64
    )
    elapsed = np.asarray(
        [row["elapsed_seconds"] for row in selected], dtype=np.float64
    )
    feasible_utilities = utilities[feasible > 0.5]
    payload = {
        "samples": int(len(selected)),
        "mean_paper_utility_all": float(np.mean(utilities)),
        "std_paper_utility_all": float(np.std(utilities)),
        "mean_paper_utility_feasible": (
            float(np.mean(feasible_utilities))
            if feasible_utilities.size else None
        ),
        "feasibility_rate": float(np.mean(feasible)),
        "mean_evaluation_seconds": float(np.mean(elapsed)),
        "mean_objective_evaluations": float(np.mean([
            row["objective_evaluations"] for row in selected
        ])),
    }
    for field in (
        "constraint_cost",
        "time_excess_s",
        "speed_excess_m_s",
        "separation_deficit_m",
        "energy_excess_j",
    ):
        payload["mean_" + field] = float(np.mean([
            row[field] for row in selected
        ]))
    return payload


def save_outputs(output, config, args, history, rows, training_times,
                 agent_settings):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "evaluation_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    names = list(dict.fromkeys(row["algorithm"] for row in rows))
    summary = {name: aggregate(rows, name) for name in names}
    payload = {
        "status": "complete",
        "benchmark": "paper013_shared_system_model_and_objective",
        "scope": (
            "Projected SAC versus VC-WOA, WOA and PSO; native LSTM predictor and emergency "
            "return controller are not applicable to the paper state model"
        ),
        "paper_metrics_are_authoritative": True,
        "learning_reward_is_only_numerical_scaling": True,
        "sac_action_modes": {
            "raw": "absolute paper coordinates without trajectory projection",
            "projected": (
                "velocity-bounded commands with terminal reachability and "
                "minimum-separation projection"
            ),
        },
        "quick": bool(args.quick),
        "config": config.to_dict(),
        "seeds": parse_seed_list(args.seeds),
        "evaluation_scenario_seeds": evaluation_scenario_seeds(
            args.evaluation_scenarios
        ),
        "agent_config": agent_settings,
        "training_seconds_by_seed": training_times,
        "summary": summary,
        "nominal_objective_evaluations": {
            "sac_training_per_variant_and_policy_seed": int(
                args.warmup_samples + args.train_episodes
            ),
            "vcwoa_per_scenario": int(
                config.population_size * (config.woa_iterations + 1)
            ),
            "woa_per_scenario": int(
                config.population_size * (config.woa_iterations + 1)
            ),
            "pso_per_scenario": int(
                config.population_size * (config.woa_iterations + 1)
            ),
            "vcwoa_note": (
                "Algorithm-2 repair can add at most one objective evaluation "
                "per iteration; report this separately from the common nominal budget."
            ),
        },
    }
    sac_mean = summary.get("SAC + trajectory projection", {}).get(
        "mean_paper_utility_feasible"
    )
    differences = {}
    for label in ("VC-WOA (Algorithm 2)", "WOA", "PSO"):
        baseline_mean = summary.get(label, {}).get(
            "mean_paper_utility_feasible"
        )
        differences[label] = (
            None if sac_mean is None or baseline_mean is None
            else float(sac_mean - baseline_mean)
        )
    payload["paired_interpretation"] = {
        "utility_difference_projected_sac_minus_baseline": differences,
        "warning": (
            "Compare utility only when both methods are feasible; also "
            "report objective-evaluation and wall-clock budgets."
        ),
    }
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    save_plot(output, history, rows)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def save_plot(output, history, rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    history_algorithms = list(dict.fromkeys(
        row["algorithm"] for row in history
    ))
    for algorithm in history_algorithms:
        seed_values = []
        for seed in sorted(set(
            row["policy_seed"] for row in history
            if row["algorithm"] == algorithm
        )):
            selected = [
                row for row in history
                if row["algorithm"] == algorithm
                and row["policy_seed"] == seed
            ]
            seed_values.append(np.asarray([
                row["learning_reward"] for row in selected
            ], dtype=np.float64))
        values = np.stack(seed_values)
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        window = min(50, max(1, mean.size))
        kernel = np.ones(window) / window
        smooth = np.convolve(mean, kernel, mode="valid")
        smooth_std = np.convolve(std, kernel, mode="valid")
        x_values = np.arange(window, mean.size + 1)
        axes[0].plot(
            x_values, smooth,
            linewidth=1.5, alpha=0.9, label=algorithm,
        )
        axes[0].fill_between(
            x_values, smooth - smooth_std, smooth + smooth_std, alpha=0.12
        )
    axes[0].set_title("SAC learning reward (moving mean)")
    axes[0].set_xlabel("Training episodes")
    axes[0].set_ylabel("Scaled learning reward")
    axes[0].legend(fontsize=8)

    names = list(dict.fromkeys(row["algorithm"] for row in rows))
    for index, name in enumerate(names):
        selected = [row for row in rows if row["algorithm"] == name]
        jitter = np.linspace(-0.08, 0.08, len(selected))
        for offset, row in zip(jitter, selected):
            axes[1].scatter(
                index + offset,
                row["paper_utility"],
                alpha=0.8,
                marker="o" if row["feasible"] else "x",
                color="#2ca02c" if row["feasible"] else "#d62728",
            )
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=12, ha="right")
    axes[1].set_ylabel("Paper network utility")
    axes[1].set_title("Raw paper utility (circle=feasible, x=infeasible)")

    feasibility = [
        np.mean([
            row["feasible"] for row in rows if row["algorithm"] == name
        ]) * 100.0
        for name in names
    ]
    axes[2].bar(range(len(names)), feasibility, alpha=0.8)
    axes[2].set_xticks(range(len(names)))
    axes[2].set_xticklabels(names, rotation=12, ha="right")
    axes[2].set_ylim(0.0, 105.0)
    axes[2].set_ylabel("Feasible evaluations (%)")
    axes[2].set_title("Paper constraints")
    for axis in axes:
        axis.grid(True, linestyle="--", color="0.82", linewidth=0.7)
    figure.tight_layout()
    figure.savefig(str(output / "sac_vs_vcwoa_shared_problem.png"), dpi=220)
    plt.close(figure)


def main():
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    variants = [
        item.strip().lower() for item in args.sac_variants.split(",")
        if item.strip()
    ]
    if not variants or any(
        item not in ("raw", "projected") for item in variants
    ):
        raise ValueError("--sac-variants must contain raw and/or projected")
    if not seeds:
        raise ValueError("at least one SAC policy seed is required")
    population_algorithms = [
        item.strip().lower()
        for item in args.population_algorithms.split(",")
        if item.strip()
    ]
    if any(item not in ("vcwoa", "woa", "pso")
           for item in population_algorithms):
        raise ValueError(
            "--population-algorithms must contain vcwoa, woa and/or pso"
        )
    if args.quick:
        args.train_episodes = min(args.train_episodes, 500)
        args.warmup_samples = min(args.warmup_samples, 128)
        args.bc_steps = min(args.bc_steps, 100)
        args.batch_size = min(args.batch_size, 64)
        args.evaluation_scenarios = min(args.evaluation_scenarios, 3)
    config = PaperConfig.from_json(args.config)
    if args.quick:
        config = quick_config(config)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scenario_seeds = evaluation_scenario_seeds(args.evaluation_scenarios)

    all_history = []
    all_rows = []
    training_times = {}
    settings = None
    for variant in variants:
        for policy_seed in seeds:
            history, rows, seconds, settings = train_sac_policy(
                config, args, policy_seed, scenario_seeds, output, variant
            )
            all_history.extend(history)
            all_rows.extend(rows)
            training_times["%s_seed%d" % (variant, policy_seed)] = float(
                seconds
            )
    if not args.skip_population_baselines:
        all_rows.extend(run_population_baselines(
            config, scenario_seeds, population_algorithms
        ))
    save_outputs(
        output, config, args, all_history, all_rows,
        training_times, settings,
    )


if __name__ == "__main__":
    main()
