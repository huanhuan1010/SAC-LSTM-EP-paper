"""Compare SAC with strict UAV-DDPG results in the paper's exact MDP.

This benchmark intentionally reuses the UAV-DDPG environment and state
normalizer instead of reimplementing their equations.  It measures the
algorithmic change from DDPG to SAC.  The LSTM/emergency modules remain part
of the project's native benchmark because the paper MDP does not expose the
queue, wind, return, or energy-margin interfaces required by those modules.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.sac import SAC  # noqa: E402
from utils.seeding import seed_everything  # noqa: E402


def parse_seeds(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if "benchmark" not in payload or "agent" not in payload:
        raise ValueError("config must contain benchmark and agent sections")
    return payload


def import_paper_environment(root):
    ddpg_dir = Path(root).resolve() / "DDPG"
    required = [
        ddpg_dir / "UAV_env.py",
        ddpg_dir / "state_normalization.py",
        ddpg_dir / "paper_env_config.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "UAV-DDPG paper environment is incomplete: %s" % missing
        )
    sys.path.insert(0, str(ddpg_dir))
    from UAV_env import UAVEnv
    from state_normalization import StateNormalization
    from paper_env_config import PAPER_PARAMETERS, configure_uav_env
    return UAVEnv, StateNormalization, PAPER_PARAMETERS, configure_uav_env


def configure_profile(modules, benchmark):
    UAVEnv, StateNormalization, parameters, configure_uav_env = modules
    configure_uav_env(
        UAVEnv,
        task_size_mb=int(benchmark["task_size_mb"]),
        f_ue_ghz=float(benchmark["ue_cpu_ghz"]),
        users=int(benchmark["users"]),
        alpha0_db=float(benchmark["alpha0_db"]),
    )
    env = UAVEnv()
    normalizer = StateNormalization()
    actual = {
        "profile": str(benchmark["profile"]),
        "state_dim": int(env.state_dim),
        "action_dim": int(env.action_dim),
        "users": int(env.M),
        "slots": int(env.slot_num),
        "task_size_mb": int(env.__class__.sum_task_size // 1048576),
        "ue_cpu_ghz": float(env.f_ue / 1e9),
        "alpha0_db": float(10.0 * np.log10(env.alpha0)),
        "paper_parameters": parameters,
    }
    expected = {
        "state_dim": 20,
        "action_dim": 4,
        "users": 4,
        "slots": 40,
        "task_size_mb": 100,
        "ue_cpu_ghz": 0.6,
        "alpha0_db": -50.0,
    }
    for key, value in expected.items():
        if not np.allclose(actual[key], value, rtol=1e-12, atol=1e-12):
            raise RuntimeError(
                "paper profile mismatch for %s: %r != %r" %
                (key, actual[key], value)
            )
    return env, normalizer, actual


def normalized(normalizer, state):
    return np.asarray(
        normalizer.state_normal(state), dtype=np.float32
    )


def run_episode(env, normalizer, agent, training):
    state = env.reset()
    total_reward = 0.0
    accepted_steps = 0
    while accepted_steps < int(env.slot_num):
        state_input = normalized(normalizer, state)
        action = agent.select_action(state_input, eval=not training)
        action = np.asarray(action, dtype=np.float32)
        (next_state, reward, is_terminal, step_redo,
         offloading_ratio_change, reset_dist) = env.step(action)
        if step_redo:
            continue

        executed_action = action.copy()
        if reset_dist:
            executed_action[2] = -1.0
        if offloading_ratio_change:
            executed_action[3] = -1.0

        final_slot = accepted_steps == int(env.slot_num) - 1
        done = bool(is_terminal or final_slot)
        if training:
            agent.replay_buffer.push(
                state_input,
                executed_action,
                float(reward),
                normalized(normalizer, next_state),
                done,
                cost=0.0,
            )
            agent.update()

        total_reward += float(reward)
        state = next_state
        accepted_steps += 1
        if is_terminal:
            break
    return -total_reward


def evaluate_policy(env, normalizer, agent, runs, evaluation_seed,
                    restore_rng=True):
    training_rng = np.random.get_state() if restore_rng else None
    try:
        np.random.seed(int(evaluation_seed))
        values = [
            run_episode(env, normalizer, agent, training=False)
            for _ in range(int(runs))
        ]
    finally:
        if training_rng is not None:
            np.random.set_state(training_rng)
    return np.asarray(values, dtype=np.float64)


def result_complete(path, seed, benchmark):
    if not path.is_file():
        return False
    try:
        with np.load(str(path), allow_pickle=True) as data:
            return (
                int(data["seed"]) == int(seed)
                and int(data["episodes"]) == int(benchmark["episodes"])
                and str(data["paper_profile"])
                == str(benchmark["profile"])
                and len(data["test_eps"])
                == int(benchmark["episodes"])
                // int(benchmark["test_every"])
                and len(data["final_test_delays"])
                == int(benchmark["final_test_runs"])
            )
    except Exception:
        return False


def train_seed(modules, config, seed, raw_dir, model_dir, resume=True):
    benchmark = config["benchmark"]
    output = raw_dir / ("sac_paper_seed%d.npz" % seed)
    if resume and result_complete(output, seed, benchmark):
        print("sac_paper_seed%d skipped (complete result exists)" % seed,
              flush=True)
        return output

    seed_everything(seed)
    env, normalizer, profile = configure_profile(modules, benchmark)
    agent_config = dict(config["agent"])
    agent = SAC(env.state_dim, env.action_dim, agent_config)
    episodes = int(benchmark["episodes"])
    test_every = int(benchmark["test_every"])
    test_runs = int(benchmark["test_runs"])
    final_test_runs = int(benchmark["final_test_runs"])
    checkpoint_seed = int(seed) + 1000003

    rewards = []
    test_eps = []
    test_delays = []
    started = time.time()
    for episode in range(episodes):
        delay = run_episode(env, normalizer, agent, training=True)
        rewards.append(-delay)
        if (episode + 1) % test_every == 0:
            samples = evaluate_policy(
                env, normalizer, agent, test_runs, checkpoint_seed,
                restore_rng=True,
            )
            test_eps.append(episode + 1)
            test_delays.append(float(np.mean(samples)))
        if (episode + 1) % 100 == 0:
            print(
                "SAC seed=%d episode=%d/%d delay=%.2f elapsed=%.1f min" %
                (seed, episode + 1, episodes, delay,
                 (time.time() - started) / 60.0),
                flush=True,
            )

    final_test_delays = evaluate_policy(
        env, normalizer, agent, final_test_runs, checkpoint_seed,
        restore_rng=False,
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / ("sac_paper_seed%d.pt" % seed)
    agent.save(str(model_path))
    np.savez_compressed(
        str(output),
        rewards=np.asarray(rewards, dtype=np.float64),
        test_eps=np.asarray(test_eps, dtype=np.int32),
        test_delays=np.asarray(test_delays, dtype=np.float64),
        final_test_delays=final_test_delays,
        seed=int(seed),
        episodes=episodes,
        test_every=test_every,
        test_runs=test_runs,
        final_test_runs=final_test_runs,
        paper_profile=str(benchmark["profile"]),
        alpha0_db=float(benchmark["alpha0_db"]),
        algorithm="SAC",
        curve_source="deterministic checkpoint evaluation",
        profile_json=json.dumps(profile, sort_keys=True),
        agent_config_json=json.dumps(agent_config, sort_keys=True),
    )
    print(
        "SAC seed=%d complete: final %.2f +/- %.2f s -> %s" %
        (seed, np.mean(final_test_delays), np.std(final_test_delays), output),
        flush=True,
    )
    return output


def load_algorithm_data(paths):
    checkpoint_x = None
    checkpoint_rows = []
    seed_final = {}
    for path in paths:
        with np.load(str(path), allow_pickle=True) as data:
            seed = int(data["seed"])
            current_x = np.asarray(data["test_eps"], dtype=np.int32)
            current_y = np.asarray(data["test_delays"], dtype=np.float64)
            if checkpoint_x is None:
                checkpoint_x = current_x
            elif not np.array_equal(checkpoint_x, current_x):
                raise ValueError("checkpoint mismatch in %s" % path)
            checkpoint_rows.append(current_y)
            seed_final[seed] = float(np.mean(
                np.asarray(data["final_test_delays"], dtype=np.float64)
            ))
    return checkpoint_x, np.stack(checkpoint_rows), seed_final


def ddpg_paths(ddpg_results, seeds):
    root = Path(ddpg_results)
    paths = [root / ("ddpg_base_seed%d.npz" % seed) for seed in seeds]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "strict DDPG results are missing: %s" % missing
        )
    return paths


def summarize_and_plot(sac_paths, ddpg_result_paths, seeds, output_root,
                       manifest):
    sac_x, sac_rows, sac_final = load_algorithm_data(sac_paths)
    ddpg_x, ddpg_rows, ddpg_final = load_algorithm_data(ddpg_result_paths)
    if not np.array_equal(sac_x, ddpg_x):
        raise ValueError("SAC and DDPG checkpoint schedules differ")

    algorithms = {
        "SAC": (sac_rows, sac_final, "#1f77b4"),
        "DDPG": (ddpg_rows, ddpg_final, "#d62728"),
    }
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure_path = figures / "sac_vs_uav_ddpg.png"
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for name, (rows, _, color) in algorithms.items():
        mean = np.mean(rows, axis=0)
        std = np.std(rows, axis=0)
        axes[0].plot(sac_x, mean, color=color, linewidth=1.8, label=name)
        axes[0].fill_between(
            sac_x, mean - std, mean + std, color=color, alpha=0.16,
            linewidth=0,
        )
    axes[0].set_xlabel("Episodes")
    axes[0].set_ylabel("Delay (s)")
    axes[0].set_title("Deterministic evaluation: mean $\\pm$ 1 std")
    axes[0].grid(True, linestyle="--", color="0.8", linewidth=0.7)
    axes[0].legend()

    x_positions = np.arange(len(seeds), dtype=np.float64)
    width = 0.36
    for offset, (name, (_, final, color)) in zip(
            (-width / 2.0, width / 2.0), algorithms.items()):
        values = [final[seed] for seed in seeds]
        axes[1].bar(
            x_positions + offset, values, width=width, color=color,
            alpha=0.82, label=name,
        )
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(["seed %d" % seed for seed in seeds])
    axes[1].set_ylabel("Final mean delay (s)")
    axes[1].set_title("100 deterministic episodes per training seed")
    axes[1].grid(True, axis="y", linestyle="--", color="0.8", linewidth=0.7)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(str(figure_path), dpi=240)
    plt.close(fig)

    summary = {}
    for name, (_, final, _) in algorithms.items():
        values = np.asarray([final[seed] for seed in seeds], dtype=np.float64)
        summary[name] = {
            "mean_delay_s": float(np.mean(values)),
            "std_across_seed_means_s": float(
                np.std(values, ddof=1) if len(values) > 1 else 0.0
            ),
            "per_seed_mean_delay_s": {
                str(seed): float(final[seed]) for seed in seeds
            },
        }
    summary["improvement_percent"] = float(
        100.0
        * (summary["DDPG"]["mean_delay_s"]
           - summary["SAC"]["mean_delay_s"])
        / summary["DDPG"]["mean_delay_s"]
    )
    summary["paired_delay_difference_s"] = {
        str(seed): float(ddpg_final[seed] - sac_final[seed])
        for seed in seeds
    }
    payload = {
        **manifest,
        "status": "complete",
        "summary": summary,
        "figure": str(figure_path),
        "interpretation": (
            "Positive improvement_percent means SAC has lower delay. "
            "This paper-MDP benchmark tests SAC versus DDPG; LSTM and "
            "emergency protection must be evaluated in the native environment."
        ),
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)

    with (output_root / "checkpoint_curves.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["algorithm", "seed", "episode", "delay_s"])
        for name, (rows, _, _) in algorithms.items():
            for seed, row in zip(seeds, rows):
                for episode, delay in zip(sac_x, row):
                    writer.writerow([name, seed, int(episode), float(delay)])
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(
        description="Compare SAC against strict UAV-DDPG results"
    )
    parser.add_argument("--uav-ddpg-root", required=True)
    parser.add_argument(
        "--config", default=str(
            PROJECT_ROOT / "configs" / "uav_ddpg_paper.yaml"
        )
    )
    parser.add_argument("--ddpg-results", default=None)
    parser.add_argument(
        "--output-root", default=str(
            PROJECT_ROOT / "results" / "uav_ddpg_paper_benchmark"
        )
    )
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--test-every", type=int, default=None)
    parser.add_argument("--test-runs", type=int, default=None)
    parser.add_argument("--final-test-runs", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--train-only", action="store_true",
        help="train SAC without requiring existing DDPG result files",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    benchmark = config["benchmark"]
    for name, value in (
            ("episodes", args.episodes),
            ("test_every", args.test_every),
            ("test_runs", args.test_runs),
            ("final_test_runs", args.final_test_runs)):
        if value is not None:
            benchmark[name] = int(value)
    seeds = (
        parse_seeds(args.seeds) if args.seeds
        else [int(value) for value in benchmark["seeds"]]
    )
    benchmark["seeds"] = seeds
    if int(benchmark["episodes"]) % int(benchmark["test_every"]):
        raise ValueError("episodes must be divisible by test_every")
    if len(set(seeds)) < 3 and not args.train_only:
        raise ValueError("formal comparison requires at least three seeds")

    output_root = Path(args.output_root).resolve()
    raw_dir = output_root / "raw"
    model_dir = output_root / "models"
    raw_dir.mkdir(parents=True, exist_ok=True)
    modules = import_paper_environment(args.uav_ddpg_root)
    _, _, runtime_profile = configure_profile(modules, benchmark)
    manifest = {
        "benchmark": "uav_ddpg_paper_environment",
        "scope": "algorithmic SAC versus DDPG comparison",
        "config": config,
        "runtime_profile": runtime_profile,
        "python": sys.executable,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "uav_ddpg_root": str(Path(args.uav_ddpg_root).resolve()),
        "seeds": seeds,
    }
    with (output_root / "manifest.json").open(
            "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)

    sac_paths = [
        train_seed(
            modules, config, seed, raw_dir, model_dir,
            resume=not args.no_resume,
        )
        for seed in seeds
    ]
    if args.train_only:
        print("SAC paper-environment training complete", flush=True)
        return

    ddpg_results = (
        Path(args.ddpg_results).resolve()
        if args.ddpg_results
        else Path(args.uav_ddpg_root).resolve()
        / "reproduction" / "results_strict_fig3457" / "raw"
    )
    summarize_and_plot(
        sac_paths,
        ddpg_paths(ddpg_results, seeds),
        seeds,
        output_root,
        manifest,
    )


if __name__ == "__main__":
    main()

