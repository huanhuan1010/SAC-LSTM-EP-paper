"""
对比实验: SAC-LSTM-EP vs 基线算法
运行所有算法, 收集指标, 生成对比图
"""
import sys, os, json, numpy as np
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from baselines.train_baseline import train_baseline, ALGO_MAP
from utils.config import load_config
from agent.lstm_runtime import (
    observation_for_agent,
    append_lstm_history,
    reset_prediction_transform,
)
from agent.oracle_predictor import OracleEnergyTracePredictor
from utils.checkpoints import checkpoint_metadata, load_checkpoint_pair
from utils.evaluation_metrics import (
    annotate_cmdp_projection_risk,
    conditional_safe_return_rate,
    successful_safety_return,
    update_mission_outcome,
    update_safety_return_flags,
)
from utils.seeding import (
    component_seed_map,
    episode_seed,
    isolated_torch_seed,
    normalize_seed,
    seed_everything,
)
from utils.provenance import (
    environment_fingerprint,
    experiment_manifest,
    source_fingerprint,
    write_json_atomic,
)
from utils.ablation_support import (
    configure_profile_environment,
    profile_observation_controller,
    profile_safety_decision,
)
from utils.algorithm_profiles import (
    PRIMARY_ABLATION_ALGORITHMS,
    algorithm_config,
    algorithm_profile,
    paper_algorithm_label,
)
from utils.logger import Logger
from env.uav_env import UAVEnv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


DEFAULT_COMPARISON_ALGORITHMS = (
    *PRIMARY_ABLATION_ALGORITHMS,
    'PPO',
    'DDPG',
    'Greedy',
    'Random',
)

DEFAULT_MULTI_SEED_BASES = (100000, 120000, 140000, 160000, 180000)


def _resolved_cmdp_cost_limit(agent, agent_settings):
    """Return the budget actually used by the evaluated algorithm.

    ``algorithm_config`` may apply an algorithm-specific budget override.  The
    evaluation summary must therefore use the resolved agent settings (or the
    instantiated agent), rather than the unmodified base configuration.
    """
    value = getattr(agent, 'cmdp_cost_limit', None)
    if value is None:
        value = agent_settings.get(
            'cmdp_cost_limit',
            agent_settings.get('constraint_cost_limit', 5.0),
        )
    return float(value)


def validate_multiseed_episode_ranges(train_seeds, train_episodes,
                                      evaluation_seed, evaluation_episodes,
                                      evaluation_seeds=None,
                                      minimum_unique_seeds=3):
    """Reject overlapping actual episode-seed ranges across formal runs."""
    seeds = [int(value) for value in train_seeds]
    minimum_unique_seeds = int(minimum_unique_seeds)
    if minimum_unique_seeds <= 0:
        raise ValueError('minimum_unique_seeds must be positive')
    if len(set(seeds)) < minimum_unique_seeds:
        raise ValueError(
            f'comparison requires at least {minimum_unique_seeds} unique seeds'
        )
    train_episodes = int(train_episodes)
    evaluation_episodes = int(evaluation_episodes)
    if train_episodes <= 0 or evaluation_episodes <= 0:
        raise ValueError('training/evaluation episode counts must be positive')
    ranges = [
        (f'training_seed_{seed}', seed, seed + train_episodes - 1)
        for seed in seeds
    ]
    if evaluation_seeds is not None:
        evaluation_blocks = [int(value) for value in evaluation_seeds]
        if len(evaluation_blocks) != len(seeds):
            raise ValueError(
                'evaluation_seeds must contain one block per training seed'
            )
        if len(set(evaluation_blocks)) != len(evaluation_blocks):
            raise ValueError('evaluation_seeds must be unique')
        ranges.extend(
            (
                f'final_evaluation_{index}', block,
                block + evaluation_episodes - 1,
            )
            for index, block in enumerate(evaluation_blocks)
        )
    else:
        if evaluation_seed is None:
            raise ValueError(
                'evaluation_seed is required when evaluation_seeds is absent'
            )
        ranges.append((
            'final_evaluation', int(evaluation_seed),
            int(evaluation_seed) + evaluation_episodes - 1,
        ))
    for label, start, end in ranges:
        if start < 0 or end >= 2 ** 32:
            raise ValueError(f'{label} episode seed range must fit uint32')
    for index, first in enumerate(ranges):
        for second in ranges[index + 1:]:
            if not (first[2] < second[1] or second[2] < first[1]):
                raise ValueError(
                    f'episode seed ranges overlap: {first[0]} and {second[0]}'
                )
    return seeds


COMPARISON_PLOT_METRICS = (
    ('reward', 'Avg Reward', 'Reward'),
    ('tasks', 'Tasks Completed', 'Tasks / episode'),
    ('mean_task_delay_s', 'Mean Completed-task Delay', 'Delay (s)'),
    ('system_energy', 'System Energy Consumption', 'Energy (J)'),
    ('constraint_exceedance_rate', 'Physical Constraint Violation Rate', 'Rate (%)'),
    ('safety_override_rate', 'Safety-layer Intervention Rate', 'Rate (%)'),
    ('safe_return_rate', 'Safe Return Rate', 'Rate (%)'),
    ('mission_success_rate', 'Mission Success Rate', 'Rate (%)'),
    ('energy_violation_rate', 'Energy-floor Violation Rate', 'Rate (%)'),
)


def comparison_metric_summary(result, metric_name):
    """返回绘图所需的 ``(均值, 标准差)``，比例指标统一为百分数。"""
    episode_count = max(len(result.get('reward', [])), 1)
    count_metrics = {
        'battery_depletion_rate': 'battery_depleted_episodes',
        'energy_violation_rate': 'energy_violation_episodes',
        'constraint_exceedance_rate': 'physical_constraint_violation_episodes',
        'physical_constraint_violation_rate': (
            'physical_constraint_violation_episodes'
        ),
        'cmdp_budget_exceedance_rate': 'cmdp_budget_exceeded_episodes',
        'return_trigger_rate': 'return_triggered',
        'safe_return_rate': 'safe_return_episodes',
        'mission_success_rate': 'mission_success_episodes',
        'mission_safe_abort_rate': 'mission_safe_abort_episodes',
        'mission_deadline_miss_rate': 'mission_deadline_missed_episodes',
    }
    if metric_name in count_metrics:
        count = result.get(count_metrics[metric_name], 0)
        return count / episode_count * 100.0, 0.0
    if metric_name == 'returned_safe':
        triggered = result.get('return_triggered', 0)
        safe_rate = conditional_safe_return_rate(
            result.get('returned_safe', 0), triggered
        ) * 100.0
        return safe_rate, 0.0
    if metric_name == 'safety_override_rate':
        total_steps = max(int(np.sum(result.get('steps', []))), 1)
        return (
            result.get('safety_override_steps', 0)
            / total_steps * 100.0,
            0.0,
        )
    if metric_name == 'mean_task_delay_s':
        data = result.get('completed_task_delays', [])
        if not data:
            return 0.0, 0.0
        return float(np.mean(data)), float(np.std(data))
    ratio_metrics = {
        'task_completion_rate': 'task_completion_ratio',
        'core_task_completion_rate': 'core_task_completion_ratio',
    }
    if metric_name in ratio_metrics:
        data = result.get(ratio_metrics[metric_name], [])
        if not data:
            return 0.0, 0.0
        return float(np.mean(data) * 100.0), float(np.std(data) * 100.0)

    data = result.get(metric_name, [])
    if isinstance(data, list):
        if not data:
            return 0.0, 0.0
        return float(np.mean(data)), float(np.std(data))
    return float(data), 0.0


def run_comparison(config_path='configs/default.yaml',
                   num_train_episodes=500,
                   num_eval_episodes=50,
                   algos=None,
                   seed=None,
                   eval_seed=None):
    """运行完整对比实验"""
    if algos is None:
        algos = list(DEFAULT_COMPARISON_ALGORITHMS)

    config = load_config(config_path)
    train_seed = config.get('seed', 42) if seed is None else seed
    evaluation_seed = (
        config.get('eval_seed', 1042) if eval_seed is None else eval_seed
    )

    results = {}
    for algo in algos:
        print(f'\n{"="*60}')
        print(f'Training: {algo}')
        print(f'{"="*60}')

        save_path = f'models/{algo}_final.pt'

        if algo in ('Random', 'Greedy'):
            # 无需训练, 直接评估
            eval_results = evaluate_algo(
                algo, config_path, save_path, num_eval_episodes,
                seed=evaluation_seed,
            )
        else:
            # 训练 + 评估
            logger, agent = train_baseline(
                algo, config_path,
                num_episodes=num_train_episodes,
                save_path=save_path,
                verbose=True,
                seed=train_seed,
            )
            eval_results = evaluate_algo(
                algo, config_path, save_path, num_eval_episodes,
                seed=evaluation_seed,
            )

        results[algo] = eval_results

    # 生成对比表
    print_comparison_table(results, algos)
    plot_comparison(results, algos)

    return results


def evaluate_algo(algo, config_path, checkpoint, num_episodes, seed=None,
                  policy_seed=None, action_log_path=None,
                  allowed_checkpoint_source_fingerprints=None):
    """评估单个算法"""
    from baselines.train_baseline import create_agent
    from utils.config import load_config

    config = load_config(config_path)
    eval_seed = normalize_seed(
        config.get('eval_seed', 1042) if seed is None else seed
    )
    resolved_policy_seed = seed_everything(
        eval_seed if policy_seed is None else policy_seed
    )
    profile = algorithm_profile(algo)
    agent_settings = algorithm_config(config, algo)
    from env.uav_env import UAVEnv
    env = UAVEnv(agent_settings)
    configure_profile_environment(env, profile)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 评估 (无LSTM/无Emergency的情况下, 紧急处理由环境层完成)
    from agent.emergency import EmergencyManager
    from agent.lstm_predictor import LSTMEnergyPredictor
    from agent.lstm_features import lstm_feature_kwargs

    use_lstm = profile.use_lstm
    use_oracle = profile.use_oracle_prediction
    use_emergency = profile.use_energy_safety_supervisor

    agent_settings.update(component_seed_map(resolved_policy_seed))
    agent_settings['component_seed'] = resolved_policy_seed
    agent = create_agent(algo, state_dim, action_dim, agent_settings)
    lstm_predictor = None
    if use_lstm:
        with isolated_torch_seed(agent_settings['lstm_init_seed']):
            lstm_predictor = LSTMEnergyPredictor(
                hidden_dim=config.get('lstm_hidden_dim', 64),
                num_layers=config.get('lstm_num_layers', 2),
                lookback=config.get('lookback', 10),
                horizon=config.get('horizon', 5),
                battery_capacity=config.get('battery_capacity', 100000.0),
                prediction_mode=config.get('lstm_prediction_mode', 'absolute'),
                residual_correction=config.get(
                    'residual_correction_enabled', True
                ),
                residual_ema_alpha=config.get('residual_ema_alpha', 0.2),
                residual_clip_ratio=config.get('residual_clip_ratio', 0.01),
                rng_seed=agent_settings['lstm_update_seed'],
                **lstm_feature_kwargs(config),
            )
    elif use_oracle:
        trace_path = config.get('oracle_trace_path')
        if not trace_path:
            raise ValueError(
                'SAC-Oracle is offline-only and requires oracle_trace_path'
            )
        lstm_predictor = OracleEnergyTracePredictor(
            trace_path,
            horizon=config.get('horizon', 5),
            battery_capacity=config.get('battery_capacity', 100000.0),
        )

    # 固定策略无需检查点；所有训练型算法必须加载对应策略，使用LSTM的算法
    # 还必须通过配对清单和SHA-256校验，不能新建随机预测器直接评估。
    loaded = None
    if algo not in ('Random', 'Greedy'):
        loaded = load_checkpoint_pair(
            agent,
            lstm_predictor,
            checkpoint,
            expected_algorithm=algo,
            expected_metadata=checkpoint_metadata(
                agent_settings, state_dim, action_dim
            ),
            allowed_source_fingerprints=(
                allowed_checkpoint_source_fingerprints
            ),
        )
        for waiver in loaded.get('metadata_waivers', ()):
            print(
                'WARNING checkpoint metadata waiver: '
                f"{waiver['key']} saved={waiver['saved']} "
                f"current={waiver['expected']}",
                flush=True,
            )
        if use_lstm:
            print(f'Loaded paired LSTM for {algo}: {loaded["lstm"]}')

    emergency = EmergencyManager(config)
    max_steps = config.get('max_steps', 200)
    action_logger = None
    if action_log_path is not None:
        action_log_config = dict(config)
        action_log_config['max_episodes'] = num_episodes
        action_logger = Logger(action_log_config)

    reward_component_keys = (
        'task_reward', 'energy_penalty', 'delay_penalty',
        'proximity_reward',
    )
    metrics = {'reward': [], 'energy': [], 'nominal_energy': [],
               'energy_uncertainty_delta': [],
               'mean_energy_uncertainty_scale': [], 'user_energy': [],
               'system_energy': [], 'delay_cost': [],
               'completed_task_delays': [],
               'deadline_missed_tasks': 0,
               'deadline_evaluable_tasks': 0,
               'deadline_miss_episodes': 0,
               'deadline_overshoot_s_sum': 0.0,
               'deadline_overshoot_normalized_sum': 0.0,
               'cmdp_discounted_cost': [],
               'cmdp_stage_cost_sum': 0.0,
               'cmdp_pre_projection_state_cost_sum': 0.0,
               'cmdp_post_projection_state_cost_sum': 0.0,
               'cmdp_projection_state_cost_delta_sum': 0.0,
               'cmdp_projection_state_cost_samples': 0,
               'cmdp_pre_projection_action_cost_sum': 0.0,
               'cmdp_applied_action_cost_sum': 0.0,
               'cmdp_action_cost_delta_sum': 0.0,
               'cmdp_action_cost_samples': 0,
               'cmdp_pre_projection_critic_risk_sum': 0.0,
               'cmdp_post_projection_critic_risk_sum': 0.0,
               'cmdp_projection_critic_risk_delta_sum': 0.0,
               'cmdp_projection_critic_risk_samples': 0,
               'cmdp_cost_active_steps': 0,
               'cmdp_cost_saturated_steps': 0,
               'cmdp_cost_saturation_threshold': None,
               'raw_action_energy_infeasible_steps': 0,
               'applied_action_energy_infeasible_steps': 0,
               'tasks': [], 'steps': [],
               **{key: [] for key in reward_component_keys},
               'battery_depleted_episodes': 0,
               'energy_violation_episodes': 0,
               'physical_constraint_violation_episodes': 0,
               'physical_constraint_violation_steps': 0,
               'cmdp_budget_exceeded_episodes': 0,
               'safety_override_steps': 0,
               'scheduled_return_override_steps': 0,
               'safety_mode_counts': {
                   'normal': 0, 'yellow': 0, 'orange': 0, 'red': 0,
               },
               'cmdp_cost_limit': _resolved_cmdp_cost_limit(
                   agent, agent_settings
               ),
               'short_horizon_energy_infeasible_steps': 0,
               'return_triggered': 0, 'returned_safe': 0,
               'safe_return_episodes': 0,
               'mission_success_episodes': 0,
               'mission_safe_abort_episodes': 0,
               'mission_deadline_missed_episodes': 0,
               'task_completion_ratio': [],
               'core_task_completion_ratio': [],
               'mission_failure_reason_counts': {},
               'checkpoint_metadata_waivers': (
                   list(loaded.get('metadata_waivers', ()))
                   if loaded is not None else []
               ),
               'evaluation_seed': eval_seed,
               'policy_seed': resolved_policy_seed,
               'environment_fingerprint': environment_fingerprint(config)}

    for ep in range(num_episodes):
        current_env_seed = episode_seed(eval_seed, ep)
        state, info = env.reset(seed=current_env_seed)
        if use_oracle:
            lstm_predictor.reset_runtime_state(current_env_seed)
        elif use_lstm:
            lstm_predictor.reset_runtime_state()
        reset_prediction_transform(
            lstm_predictor, profile.prediction_source, current_env_seed
        )

        ep_reward = 0
        ep_energy = 0
        ep_nominal_energy = 0
        ep_energy_uncertainty_delta = 0
        ep_energy_uncertainty_scales = []
        ep_system_energy = 0
        ep_user_energy = 0
        ep_delay_cost = 0
        ep_tasks = 0
        ep_battery_depleted = False
        ep_energy_violation = False
        ep_cmdp_discounted_cost = 0.0
        ep_physical_constraint_violation = False
        ep_return_triggered = False
        ep_returned = False
        ep_mission_success = False
        ep_mission_deadline_missed = False
        ep_deadline_missed_tasks = 0
        ep_reward_components = {key: 0.0 for key in reward_component_keys}

        for step in range(max_steps):
            state_aug, E_pred = observation_for_agent(
                state, env, lstm_predictor, config,
                use_lstm=use_lstm or use_oracle,
                safety_controller=profile_observation_controller(profile),
                prediction_source=profile.prediction_source,
            )
            E_rem = env.E_rem

            if use_emergency:
                mode, _ = profile_safety_decision(
                    profile, env, emergency, E_pred
                )
            else:
                mode = 'normal'

            env.emergency_mode = mode
            action = agent.select_action(state_aug, eval=True)

            next_state, reward, done, truncated, info = env.step(action)
            applied_mode = str(
                info.get('safety_mode_applied', mode)
            ).strip().lower()
            if applied_mode in metrics['safety_mode_counts']:
                metrics['safety_mode_counts'][applied_mode] += 1
            annotate_cmdp_projection_risk(
                agent, state_aug, info, raw_action=action
            )
            if action_logger is not None:
                action_logger.log_action(ep, step, info)
            ep_return_triggered, ep_returned = update_safety_return_flags(
                ep_return_triggered, ep_returned, info
            )
            ep_mission_success, ep_mission_deadline_missed = (
                update_mission_outcome(
                    ep_mission_success, ep_mission_deadline_missed, info
                )
            )

            if use_lstm:
                append_lstm_history(lstm_predictor, env, info)
            elif use_oracle:
                lstm_predictor.observe_actual_energy(info.get('E_total', 0.0))

            state = next_state
            ep_reward += reward
            ep_energy += info.get('E_total', 0)
            ep_nominal_energy += info.get(
                'E_uav_nominal', info.get('E_total', 0)
            )
            ep_energy_uncertainty_delta += info.get(
                'E_uav_uncertainty_delta', 0.0
            )
            ep_energy_uncertainty_scales.append(float(
                info.get('energy_uncertainty_scale', 1.0)
            ))
            ep_system_energy += info.get(
                'E_system_total', info.get('E_total', 0)
            )
            ep_user_energy += info.get('energy_breakdown', {}).get(
                'E_user_total', 0.0
            )
            ep_delay_cost += float(info.get('delay_cost', 0.0))
            metrics['completed_task_delays'].extend(
                info.get('task_progress', {}).get(
                    'completed_task_delays', []
                )
            )
            task_progress = info.get('task_progress', {})
            step_deadline_misses = int(task_progress.get(
                'deadline_miss_count', info.get('tasks_failed', 0)
            ))
            ep_deadline_missed_tasks += step_deadline_misses
            metrics['deadline_missed_tasks'] += step_deadline_misses
            metrics['deadline_evaluable_tasks'] += (
                int(info.get('tasks_completed', 0))
                + step_deadline_misses
            )
            metrics['deadline_overshoot_s_sum'] += float(
                task_progress.get('deadline_overshoot_s_sum', 0.0)
            )
            metrics['deadline_overshoot_normalized_sum'] += float(
                task_progress.get(
                    'deadline_overshoot_normalized_sum', 0.0
                )
            )
            ep_tasks += info.get('tasks_completed', 0)
            stage_cost = float(info.get('cmdp_training_cost', 0.0))
            ep_cmdp_discounted_cost += (
                float(config.get('gamma', 0.99)) ** step
            ) * stage_cost
            metrics['cmdp_stage_cost_sum'] += stage_cost
            projection_costs = (
                float(info.get('cmdp_pre_projection_state_cost', np.nan)),
                float(info.get('cmdp_post_projection_state_cost', np.nan)),
                float(info.get('cmdp_projection_state_cost_delta', np.nan)),
            )
            if (
                info.get('cmdp_projection_cost_comparison_available', True)
                and all(np.isfinite(value) for value in projection_costs)
            ):
                metrics['cmdp_pre_projection_state_cost_sum'] += (
                    projection_costs[0]
                )
                metrics['cmdp_post_projection_state_cost_sum'] += (
                    projection_costs[1]
                )
                metrics['cmdp_projection_state_cost_delta_sum'] += (
                    projection_costs[2]
                )
                metrics['cmdp_projection_state_cost_samples'] += 1
            action_costs = (
                float(info.get('cmdp_pre_projection_action_cost', np.nan)),
                float(info.get('cmdp_applied_action_cost', np.nan)),
                float(info.get('cmdp_action_cost_delta', np.nan)),
            )
            if all(np.isfinite(value) for value in action_costs):
                metrics['cmdp_pre_projection_action_cost_sum'] += (
                    action_costs[0]
                )
                metrics['cmdp_applied_action_cost_sum'] += action_costs[1]
                metrics['cmdp_action_cost_delta_sum'] += action_costs[2]
                metrics['cmdp_action_cost_samples'] += 1
            raw_margin = float(info.get(
                'cmdp_raw_action_margin_ratio', np.nan
            ))
            applied_margin = float(info.get(
                'cmdp_applied_action_margin_ratio', np.nan
            ))
            if np.isfinite(raw_margin):
                metrics['raw_action_energy_infeasible_steps'] += int(
                    raw_margin < 0.0
                )
            if np.isfinite(applied_margin):
                metrics['applied_action_energy_infeasible_steps'] += int(
                    applied_margin < 0.0
                )
            paired_risks = (
                float(info.get('cmdp_pre_projection_critic_risk', np.nan)),
                float(info.get('cmdp_post_projection_critic_risk', np.nan)),
                float(info.get('cmdp_projection_critic_risk_delta', np.nan)),
            )
            if all(np.isfinite(value) for value in paired_risks):
                metrics['cmdp_pre_projection_critic_risk_sum'] += paired_risks[0]
                metrics['cmdp_post_projection_critic_risk_sum'] += paired_risks[1]
                metrics['cmdp_projection_critic_risk_delta_sum'] += paired_risks[2]
                metrics['cmdp_projection_critic_risk_samples'] += 1
            metrics['cmdp_cost_active_steps'] += int(stage_cost > 0.0)
            saturation_threshold = float(info.get(
                'cmdp_cost_saturation_threshold', 1.0
            ))
            metrics['cmdp_cost_saturation_threshold'] = (
                saturation_threshold
            )
            metrics['cmdp_cost_saturated_steps'] += int(
                stage_cost >= saturation_threshold - 1e-12
            )
            physical_violation = bool(
                info.get('physical_constraint_violation', False)
            )
            ep_physical_constraint_violation |= physical_violation
            metrics['physical_constraint_violation_steps'] += int(
                physical_violation
            )
            for key in reward_component_keys:
                ep_reward_components[key] += info.get(
                    'reward_breakdown', {}
                ).get(key, 0.0)
            if info.get('battery_depleted', info.get('crashed', False)):
                ep_battery_depleted = True
            if info.get('energy_violation'):
                ep_energy_violation = True
            if not info.get('short_horizon_energy_feasible', True):
                metrics['short_horizon_energy_infeasible_steps'] += 1
            metrics['safety_override_steps'] += int(
                info.get('safety_action_overridden', False)
            )
            metrics['scheduled_return_override_steps'] += int(
                info.get('scheduled_return_action_overridden', False)
            )
            if done or truncated:
                break

        metrics['reward'].append(ep_reward)
        metrics['energy'].append(ep_energy)
        metrics['nominal_energy'].append(ep_nominal_energy)
        metrics['energy_uncertainty_delta'].append(
            ep_energy_uncertainty_delta
        )
        metrics['mean_energy_uncertainty_scale'].append(float(
            np.mean(ep_energy_uncertainty_scales)
            if ep_energy_uncertainty_scales else 1.0
        ))
        metrics['system_energy'].append(ep_system_energy)
        metrics['user_energy'].append(ep_user_energy)
        metrics['delay_cost'].append(ep_delay_cost)
        metrics['cmdp_discounted_cost'].append(ep_cmdp_discounted_cost)
        metrics['tasks'].append(ep_tasks)
        metrics['steps'].append(step + 1)
        for key in reward_component_keys:
            metrics[key].append(ep_reward_components[key])
        if ep_battery_depleted:
            metrics['battery_depleted_episodes'] += 1
        if ep_energy_violation:
            metrics['energy_violation_episodes'] += 1
        if ep_physical_constraint_violation:
            metrics['physical_constraint_violation_episodes'] += 1
        if ep_cmdp_discounted_cost > metrics['cmdp_cost_limit']:
            metrics['cmdp_budget_exceeded_episodes'] += 1
        if ep_return_triggered:
            metrics['return_triggered'] += 1
        if ep_returned:
            metrics['safe_return_episodes'] += 1
        if successful_safety_return(ep_return_triggered, ep_returned):
            metrics['returned_safe'] += 1
        if ep_mission_success:
            metrics['mission_success_episodes'] += 1
        if info.get('mission_safe_abort', False):
            metrics['mission_safe_abort_episodes'] += 1
        if ep_mission_deadline_missed:
            metrics['mission_deadline_missed_episodes'] += 1
        if ep_deadline_missed_tasks > 0:
            metrics['deadline_miss_episodes'] += 1
        metrics['task_completion_ratio'].append(float(
            info.get('task_completion_ratio', 0.0)
        ))
        metrics['core_task_completion_ratio'].append(float(
            info.get('core_task_completion_ratio', 1.0)
        ))
        for reason in info.get('mission_failure_reasons', ()):
            counts = metrics['mission_failure_reason_counts']
            counts[reason] = counts.get(reason, 0) + 1

    if action_logger is not None:
        action_logger.save_actions(action_log_path)
    return metrics


def summarize_evaluation(metrics):
    """把一次训练seed的逐回合评估压缩为一个统计样本。"""
    summary = {}
    for key in (
        'reward', 'energy', 'nominal_energy',
        'energy_uncertainty_delta', 'mean_energy_uncertainty_scale',
        'user_energy', 'system_energy', 'delay_cost',
        'cmdp_discounted_cost', 'tasks', 'steps',
        'task_reward', 'energy_penalty', 'delay_penalty',
        'proximity_reward',
    ):
        values = metrics.get(key, [])
        summary[key] = float(np.mean(values)) if values else 0.0
    delay_samples = np.asarray(
        metrics.get('completed_task_delays', []), dtype=np.float64
    )
    summary['mean_task_delay_s'] = (
        float(np.mean(delay_samples)) if delay_samples.size else None
    )
    summary['p95_task_delay_s'] = (
        float(np.percentile(delay_samples, 95)) if delay_samples.size else None
    )
    summary['completed_task_delay_samples'] = int(delay_samples.size)
    deadline_missed_tasks = int(metrics.get('deadline_missed_tasks', 0))
    deadline_evaluable_tasks = int(metrics.get(
        'deadline_evaluable_tasks', 0
    ))
    summary['deadline_missed_tasks'] = deadline_missed_tasks
    summary['deadline_evaluable_tasks'] = deadline_evaluable_tasks
    summary['task_deadline_violation_rate'] = (
        deadline_missed_tasks / deadline_evaluable_tasks * 100.0
        if deadline_evaluable_tasks > 0 else 0.0
    )
    summary['mean_deadline_overshoot_s'] = (
        float(metrics.get('deadline_overshoot_s_sum', 0.0))
        / deadline_missed_tasks
        if deadline_missed_tasks > 0 else 0.0
    )
    summary['mean_normalized_deadline_overshoot'] = (
        float(metrics.get('deadline_overshoot_normalized_sum', 0.0))
        / deadline_missed_tasks
        if deadline_missed_tasks > 0 else 0.0
    )
    episode_count = max(len(metrics.get('reward', [])), 1)
    summary['battery_depletion_rate'] = (
        metrics.get('battery_depleted_episodes', 0)
        / episode_count * 100.0
    )
    summary['energy_violation_rate'] = (
        metrics.get('energy_violation_episodes', 0) / episode_count * 100.0
    )
    physical_violation_rate = (
        metrics.get('physical_constraint_violation_episodes', 0)
        / episode_count * 100.0
    )
    summary['physical_constraint_violation_rate'] = physical_violation_rate
    # Stable paper/reporting name; now explicitly defined by the common
    # predictor-independent physical criterion rather than the CMDP budget.
    summary['constraint_exceedance_rate'] = physical_violation_rate
    summary['cmdp_budget_exceedance_rate'] = (
        metrics.get('cmdp_budget_exceeded_episodes', 0)
        / episode_count * 100.0
    )
    summary['cmdp_cost_limit'] = float(metrics.get('cmdp_cost_limit', 5.0))
    total_steps = max(int(np.sum(metrics.get('steps', []))), 1)
    summary['cmdp_stage_cost_mean'] = (
        float(metrics.get('cmdp_stage_cost_sum', 0.0)) / total_steps
    )
    projection_cost_samples = int(metrics.get(
        'cmdp_projection_state_cost_samples', total_steps
    ))
    for output_name, sum_name in (
        ('cmdp_pre_projection_state_cost_mean',
         'cmdp_pre_projection_state_cost_sum'),
        ('cmdp_post_projection_state_cost_mean',
         'cmdp_post_projection_state_cost_sum'),
        ('cmdp_projection_state_cost_delta_mean',
         'cmdp_projection_state_cost_delta_sum'),
    ):
        summary[output_name] = (
            float(metrics.get(sum_name, 0.0)) / projection_cost_samples
            if projection_cost_samples > 0 else None
        )
    action_cost_samples = int(metrics.get('cmdp_action_cost_samples', 0))
    for output_name, sum_name in (
        ('cmdp_pre_projection_action_cost_mean',
         'cmdp_pre_projection_action_cost_sum'),
        ('cmdp_applied_action_cost_mean', 'cmdp_applied_action_cost_sum'),
        ('cmdp_action_cost_delta_mean', 'cmdp_action_cost_delta_sum'),
    ):
        summary[output_name] = (
            float(metrics.get(sum_name, 0.0)) / action_cost_samples
            if action_cost_samples > 0 else None
        )
    critic_risk_samples = int(metrics.get(
        'cmdp_projection_critic_risk_samples', 0
    ))
    for output_name, sum_name in (
        ('cmdp_pre_projection_critic_risk_mean',
         'cmdp_pre_projection_critic_risk_sum'),
        ('cmdp_post_projection_critic_risk_mean',
         'cmdp_post_projection_critic_risk_sum'),
        ('cmdp_projection_critic_risk_delta_mean',
         'cmdp_projection_critic_risk_delta_sum'),
    ):
        summary[output_name] = (
            float(metrics.get(sum_name, 0.0)) / critic_risk_samples
            if critic_risk_samples > 0 else None
        )
    summary['cmdp_cost_activation_rate'] = (
        metrics.get('cmdp_cost_active_steps', 0) / total_steps * 100.0
    )
    summary['cmdp_cost_saturation_rate'] = (
        metrics.get('cmdp_cost_saturated_steps', 0) / total_steps * 100.0
    )
    summary['cmdp_cost_saturation_threshold'] = metrics.get(
        'cmdp_cost_saturation_threshold'
    )
    summary['raw_action_energy_infeasible_rate'] = (
        metrics.get('raw_action_energy_infeasible_steps', 0)
        / total_steps * 100.0
    )
    summary['applied_action_energy_infeasible_rate'] = (
        metrics.get('applied_action_energy_infeasible_steps', 0)
        / total_steps * 100.0
    )
    summary['physical_constraint_violation_step_rate'] = (
        metrics.get('physical_constraint_violation_steps', 0)
        / total_steps * 100.0
    )
    summary['short_horizon_energy_infeasible_rate'] = (
        metrics.get('short_horizon_energy_infeasible_steps', 0)
        / total_steps * 100.0
    )
    summary['safety_override_rate'] = (
        metrics.get('safety_override_steps', 0) / total_steps * 100.0
    )
    summary['scheduled_return_override_rate'] = (
        metrics.get('scheduled_return_override_steps', 0)
        / total_steps * 100.0
    )
    mode_counts = metrics.get('safety_mode_counts', {})
    for mode in ('normal', 'yellow', 'orange', 'red'):
        summary[f'safety_mode_{mode}_rate'] = (
            float(mode_counts.get(mode, 0)) / total_steps * 100.0
        )
    summary['returned_safe'] = conditional_safe_return_rate(
        metrics.get('returned_safe', 0),
        metrics.get('return_triggered', 0),
    ) * 100.0
    summary['return_trigger_rate'] = (
        metrics.get('return_triggered', 0) / episode_count * 100.0
    )
    summary['safe_return_rate'] = (
        metrics.get('safe_return_episodes', 0) / episode_count * 100.0
    )
    summary['mission_success_rate'] = (
        metrics.get('mission_success_episodes', 0) / episode_count * 100.0
    )
    summary['mission_safe_abort_rate'] = (
        metrics.get('mission_safe_abort_episodes', 0)
        / episode_count * 100.0
    )
    summary['mission_deadline_miss_rate'] = (
        metrics.get('mission_deadline_missed_episodes', 0)
        / episode_count * 100.0
    )
    summary['task_deadline_miss_episode_rate'] = (
        metrics.get('deadline_miss_episodes', 0)
        / episode_count * 100.0
    )
    summary['task_completion_rate'] = float(
        np.mean(metrics.get('task_completion_ratio', [0.0])) * 100.0
    )
    summary['core_task_completion_rate'] = float(
        np.mean(metrics.get('core_task_completion_ratio', [1.0])) * 100.0
    )
    return summary


def aggregate_seed_results(per_seed_results):
    """按独立训练seed汇总均值和样本标准差。"""
    seed_summaries = {
        str(seed): summarize_evaluation(metrics)
        for seed, metrics in per_seed_results.items()
    }
    aggregate = {}
    metric_names = tuple(seed_summaries[next(iter(seed_summaries))].keys())
    for metric in metric_names:
        raw_values = [item[metric] for item in seed_summaries.values()]
        values = [
            float(value) for value in raw_values
            if value is not None and np.isfinite(float(value))
        ]
        aggregate[metric] = {
            'mean': float(np.mean(values)) if values else None,
            'std': (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            ),
            'values': [
                None if value is None else float(value)
                for value in raw_values
            ],
        }
    return seed_summaries, aggregate


def print_multiseed_table(algorithm_results, algorithms):
    """Print abstract-aligned cross-seed evaluation statistics."""
    print('\n' + '=' * 152)
    print(f'{"Algorithm":<26} {"Reward":^16} {"Tasks":^14} '
          f'{"Delay(s)":^14} {"SystemE(J)":^16} '
          f'{"Constraint%":^14} {"Intervention%":^16} '
          f'{"SafeReturn%":^14} {"Mission%":^14}')
    print('-' * 152)
    for algorithm in algorithms:
        stats = algorithm_results[algorithm]['summary']
        def cell(metric):
            mean = stats[metric]['mean']
            std = stats[metric]['std']
            return 'N/A' if mean is None else f'{mean:.2f}±{std:.2f}'
        label = paper_algorithm_label(algorithm)
        print(f'{label:<26} {cell("reward"):^16} '
              f'{cell("tasks"):^14} {cell("mean_task_delay_s"):^14} '
              f'{cell("system_energy"):^16} '
              f'{cell("constraint_exceedance_rate"):^14} '
              f'{cell("safety_override_rate"):^16} '
              f'{cell("safe_return_rate"):^14} '
              f'{cell("mission_success_rate"):^14}')
    print('=' * 152)


def plot_multiseed_comparison(algorithm_results, algorithms, output_path):
    """用跨seed标准差绘制对比图。"""
    figure, axes = plt.subplots(3, 3, figsize=(15, 14))
    colors = plt.cm.Set2(np.linspace(0, 1, len(algorithms)))
    for axis, (metric, title, ylabel) in zip(
            axes.flat, COMPARISON_PLOT_METRICS):
        means = [
            algorithm_results[name]['summary'][metric]['mean']
            for name in algorithms
        ]
        stds = [
            algorithm_results[name]['summary'][metric]['std']
            for name in algorithms
        ]
        means = [np.nan if value is None else value for value in means]
        bars = axis.bar(
            range(len(algorithms)), means, yerr=stds,
            capsize=5, color=colors,
        )
        axis.set_xticks(range(len(algorithms)))
        axis.set_xticklabels(
            [paper_algorithm_label(name) for name in algorithms],
            rotation=30, ha='right', fontsize=9,
        )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        for bar, mean in zip(bars, means):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + abs(bar.get_height()) * 0.02,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=8,
            )
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def run_multiseed_comparison(config_path='configs/default.yaml',
                             num_train_episodes=500,
                             num_eval_episodes=50,
                             algos=None,
                             seeds=None,
                             eval_seed=None,
                             eval_seeds=None,
                             run_name='paper',
                             publish_latest=True,
                             resume=True,
                             minimum_unique_seeds=3):
    """运行多训练seed公平对比，并保存环境指纹、逐seed结果和汇总。"""
    config = load_config(config_path)
    algorithms = list(algos or DEFAULT_COMPARISON_ALGORITHMS)
    evaluation_seed = None
    if eval_seeds is None:
        evaluation_seed = int(
            config.get('eval_seed', 1042)
            if eval_seed is None else eval_seed
        )
    train_seeds = validate_multiseed_episode_ranges(
        seeds or DEFAULT_MULTI_SEED_BASES,
        num_train_episodes,
        evaluation_seed,
        num_eval_episodes,
        evaluation_seeds=eval_seeds,
        minimum_unique_seeds=minimum_unique_seeds,
    )
    if eval_seeds is None:
        evaluation_blocks = [evaluation_seed] * len(train_seeds)
    else:
        evaluation_blocks = [int(value) for value in eval_seeds]
    fingerprint = environment_fingerprint(config)
    runtime_source_fingerprint = source_fingerprint()
    run_root = (
        Path('results') / 'experiments' / fingerprint[:12] / str(run_name)
    )
    model_root = (
        Path('models') / 'experiments' / fingerprint[:12] / str(run_name)
    )

    base_manifest = experiment_manifest(
        config,
        status='running',
        experiment='multi_seed_comparison',
        run_name=str(run_name),
        algorithms=algorithms,
        train_seeds=train_seeds,
        evaluation_seed=evaluation_seed,
        evaluation_seeds=(
            evaluation_blocks if eval_seeds is not None else None
        ),
        evaluation_seed_design=(
            'paired_independent_blocks'
            if eval_seeds is not None else 'shared_paired_block'
        ),
        train_episodes=int(num_train_episodes),
        evaluation_episodes=int(num_eval_episodes),
    )
    write_json_atomic(run_root / 'manifest.json', base_manifest)

    signature_env = UAVEnv(config)
    signature_state_dim = signature_env.observation_space.shape[0]
    signature_action_dim = signature_env.action_space.shape[0]

    algorithm_results = {}
    for algorithm in algorithms:
        algorithm_metadata = checkpoint_metadata(
            algorithm_config(config, algorithm),
            signature_state_dim,
            signature_action_dim,
        )
        per_seed = {}
        for seed_index, train_seed in enumerate(train_seeds):
            policy_evaluation_seed = evaluation_blocks[seed_index]
            expected_run_metadata = {
                **algorithm_metadata,
                'episodes': int(num_train_episodes),
                'training_seed': normalize_seed(train_seed),
            }
            print(f'\n[{algorithm}] training seed={train_seed}')
            seed_result_dir = run_root / algorithm / f'seed_{train_seed}'
            evaluation_file = seed_result_dir / 'evaluation.json'
            checkpoint = (
                model_root / algorithm / f'seed_{train_seed}' / 'final.pt'
            )
            if resume and evaluation_file.is_file():
                with evaluation_file.open('r', encoding='utf-8') as stream:
                    cached = json.load(stream)
                if (
                    cached.get('environment_fingerprint') == fingerprint
                    and cached.get('source_fingerprint')
                    == runtime_source_fingerprint
                    and cached.get('evaluation_seed')
                    == policy_evaluation_seed
                    and cached.get('policy_seed') == normalize_seed(train_seed)
                    and cached.get('training_episodes')
                    == int(num_train_episodes)
                    and cached.get('training_metadata')
                    == expected_run_metadata
                ):
                    print('  reusing completed evaluation')
                    per_seed[train_seed] = cached
                    continue

            pair_manifest = checkpoint.with_name(
                f'{checkpoint.stem}.pair.json'
            )
            checkpoint_ready = checkpoint.is_file() and pair_manifest.is_file()
            if checkpoint_ready:
                with pair_manifest.open('r', encoding='utf-8') as stream:
                    pair_data = json.load(stream)
                pair_metadata = pair_data.get('metadata', {})
                checkpoint_ready = (
                    pair_data.get('algorithm') == algorithm
                    and all(
                        pair_metadata.get(key) == value
                        for key, value in expected_run_metadata.items()
                    )
                )
            if algorithm not in ('Random', 'Greedy') and not (
                    resume and checkpoint_ready):
                train_baseline(
                    algorithm,
                    config_path,
                    num_episodes=num_train_episodes,
                    save_path=str(checkpoint),
                    verbose=True,
                    seed=train_seed,
                    output_dir=str(seed_result_dir),
                )
            metrics = evaluate_algo(
                algorithm,
                config_path,
                str(checkpoint),
                num_eval_episodes,
                seed=policy_evaluation_seed,
                policy_seed=train_seed,
                action_log_path=str(
                    seed_result_dir / 'evaluation_actions.csv'
                ),
            )
            metrics['source_fingerprint'] = runtime_source_fingerprint
            metrics['training_episodes'] = int(num_train_episodes)
            metrics['training_metadata'] = expected_run_metadata
            per_seed[train_seed] = metrics
            write_json_atomic(evaluation_file, metrics)

        seed_summaries, aggregate = aggregate_seed_results(per_seed)
        algorithm_results[algorithm] = {
            'per_seed': {str(key): value for key, value in per_seed.items()},
            'seed_summaries': seed_summaries,
            'summary': aggregate,
        }

    completed = {
        **base_manifest,
        'status': 'complete',
        'results': algorithm_results,
    }
    result_json = run_root / 'eval_results.json'
    write_json_atomic(result_json, completed)
    write_json_atomic(run_root / 'manifest.json', {
        **base_manifest,
        'status': 'complete',
        'result_file': result_json.name,
    })
    if publish_latest:
        write_json_atomic(Path('results') / 'latest_experiment.json', {
            'environment_fingerprint': fingerprint,
            'result_file': str(result_json.as_posix()),
        })
    print_multiseed_table(algorithm_results, algorithms)
    plot_multiseed_comparison(
        algorithm_results, algorithms, run_root / 'comparison.png'
    )
    return completed


def print_comparison_table(results, algos):
    """Print one-seed evaluation metrics promised by the abstract."""
    print('\n' + '=' * 126)
    print(f'{"Algorithm":<26} {"Reward":^10} {"Tasks":^10} '
          f'{"Delay(s)":^10} {"SystemE(J)":^12} '
          f'{"Constraint%":^12} {"Intervention%":^14} '
          f'{"Safe%":^10} {"Mission%":^10}')
    print('-' * 126)
    for algo in algos:
        r = results[algo]
        tasks, _ = comparison_metric_summary(r, 'tasks')
        delay, _ = comparison_metric_summary(r, 'mean_task_delay_s')
        system_energy, _ = comparison_metric_summary(r, 'system_energy')
        constraint_rate, _ = comparison_metric_summary(
            r, 'constraint_exceedance_rate'
        )
        intervention_rate, _ = comparison_metric_summary(
            r, 'safety_override_rate'
        )
        safe_rate, _ = comparison_metric_summary(r, 'safe_return_rate')
        mission_rate, _ = comparison_metric_summary(r, 'mission_success_rate')
        label = paper_algorithm_label(algo)
        print(f'{label:<26} {np.mean(r["reward"]):+8.2f}  '
              f'{tasks:8.1f}  {delay:8.2f}  {system_energy:10.1f}  '
              f'{constraint_rate:10.1f}%  {intervention_rate:12.1f}%  '
              f'{safe_rate:8.1f}%  '
              f'{mission_rate:8.1f}%')
    print('=' * 126)


def plot_comparison(results, algos):
    """生成对比图"""
    n_algos = len(algos)
    fig, axes = plt.subplots(3, 3, figsize=(15, 14))

    if len(COMPARISON_PLOT_METRICS) != axes.size:
        raise RuntimeError('comparison plot metric count must match subplot count')

    for ax, (mname, title, ylabel) in zip(
            axes.flat, COMPARISON_PLOT_METRICS):
        means = []
        stds = []
        for algo in algos:
            mean, std = comparison_metric_summary(results[algo], mname)
            means.append(mean)
            stds.append(std)

        colors = plt.cm.Set2(np.linspace(0, 1, n_algos))
        bars = ax.bar(range(n_algos), means, yerr=stds, capsize=5, color=colors)
        ax.set_xticks(range(n_algos))
        ax.set_xticklabels(
            [paper_algorithm_label(name) for name in algos],
            rotation=30, ha='right', fontsize=9,
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)

        # 在柱状图上标数值
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + abs(bar.get_height()) * 0.02,
                    f'{mean:.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/comparison.png', dpi=150)
    plt.close()
    print('\nComparison plot saved to results/comparison.png')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run comparison experiments')
    parser.add_argument('--train_episodes', type=int, default=200,
                        help='Training episodes per algo (use small for quick test)')
    parser.add_argument('--eval_episodes', type=int, default=20)
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--algos', nargs='+', default=None)
    parser.add_argument('--seed', type=int, default=None,
                        help='Shared training seed for every algorithm')
    parser.add_argument('--eval-seed', type=int, default=None,
                        help='Shared independent evaluation seed')
    parser.add_argument(
        '--seeds', type=int, nargs='+', default=None,
        help='Run a paper-grade multi-seed comparison (at least 3 seeds)',
    )
    parser.add_argument('--run-name', default='paper')
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args()

    if args.seeds:
        run_multiseed_comparison(
            config_path=args.config,
            num_train_episodes=args.train_episodes,
            num_eval_episodes=args.eval_episodes,
            algos=args.algos,
            seeds=args.seeds,
            eval_seed=args.eval_seed,
            run_name=args.run_name,
            resume=not args.no_resume,
        )
    else:
        run_comparison(
            config_path=args.config,
            num_train_episodes=args.train_episodes,
            num_eval_episodes=args.eval_episodes,
            algos=args.algos,
            seed=args.seed,
            eval_seed=args.eval_seed,
        )
