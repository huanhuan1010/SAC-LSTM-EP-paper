"""
消融实验: 逐一关闭创新点, 分析贡献
消融项:
  1. NoLSTM: 去掉LSTM能量预测
  2. NoEmergency: 去掉三级应急迫降
  3. NoConstraint: 去掉CMDP代价评论家与拉格朗日约束
"""
import sys, os, json, numpy as np
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from env.uav_env import UAVEnv
from agent.sac import SAC
from agent.lstm_predictor import LSTMEnergyPredictor
from agent.lstm_features import lstm_feature_kwargs
from agent.lstm_runtime import (
    observation_for_agent,
    append_lstm_history,
    append_state_frame,
    maybe_update_lstm,
)
from agent.emergency import EmergencyManager
from utils.config import load_config
from utils.logger import Logger
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
    seed_everything,
)
from utils.provenance import (
    environment_fingerprint,
    experiment_manifest,
    write_json_atomic,
)
from experiments.compare import (
    DEFAULT_MULTI_SEED_BASES,
    validate_multiseed_episode_ranges,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class SACNoLSTM(SAC):
    """消融1: 去掉LSTM预测 (但保留SAC+Emergency)"""
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.name = 'Ablation-NoLSTM'


class SACNoEmergency(SAC):
    """消融2: 去掉三级应急迫降 (但保留SAC+LSTM)"""
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.name = 'Ablation-NoEmergency'


class SACNoConstraint(SAC):
    """Remove learned CMDP optimization while retaining the safety shield."""

    def __init__(self, state_dim, action_dim, config):
        unconstrained_config = dict(config)
        unconstrained_config['cmdp_enabled'] = False
        super().__init__(state_dim, action_dim, unconstrained_config)
        self.name = 'Ablation-NoConstraint'


def evaluate_variant(agent, env, lstm_predictor, emergency, config,
                     no_lstm=False, no_emergency=False, num_episodes=50,
                     seed=None, action_log_path=None):
    """用冻结策略在独立回合中评估一个消融变体。"""
    if num_episodes <= 0:
        raise ValueError('num_eval must be a positive integer')
    eval_seed = seed_everything(
        config.get('eval_seed', 1042) if seed is None else seed
    )
    env.set_energy_safety_supervisor(not no_emergency)

    max_steps = config.get('max_steps', 200)
    action_logger = None
    if action_log_path is not None:
        action_log_config = dict(config)
        action_log_config['max_episodes'] = num_episodes
        action_logger = Logger(action_log_config)
    episode_metrics = {
        'reward': [],
        'energy': [],
        'uav_energy': [],
        'system_energy': [],
        'cmdp_discounted_cost': [],
        'cmdp_budget_exceeded': [],
        'physical_constraint_violation': [],
        'physical_constraint_violation_rate': [],
        'safety_override_rate': [],
        'mean_task_delay_s': [],
        'tasks_completed': [],
        'steps': [],
        'battery_depleted': [],
        'energy_violation': [],
        'return_triggered': [],
        'returned_safe_episode': [],
        'returned_safe_after_trigger': [],
        'mission_success': [],
        'mission_safe_abort': [],
        'mission_deadline_missed': [],
        'task_completion_ratio': [],
        'core_task_completion_ratio': [],
        'short_horizon_energy_infeasible_steps': [],
    }

    for episode in range(num_episodes):
        state, _ = env.reset(seed=episode_seed(eval_seed, episode))
        if not no_lstm:
            lstm_predictor.reset_runtime_state()

        totals = {
            'reward': 0.0,
            'energy': 0.0,
            'system_energy': 0.0,
            'tasks_completed': 0,
            'short_horizon_energy_infeasible_steps': 0,
            'cmdp_discounted_cost': 0.0,
            'physical_constraint_violation_steps': 0,
            'safety_override_steps': 0,
            'completed_task_delays': [],
        }
        battery_depleted = False
        energy_violation = False
        return_triggered = False
        returned_safe = False
        mission_success = False
        mission_deadline_missed = False

        for step in range(max_steps):
            use_lstm = not no_lstm
            state_aug, E_pred = observation_for_agent(
                state,
                env,
                lstm_predictor if use_lstm else None,
                config,
                use_lstm=use_lstm,
                emergency_mode='normal' if no_emergency else None,
            )
            energy_ctx = env.get_energy_context()
            mode, constraints = emergency.check(
                env.E_rem,
                E_pred,
                return_energy=energy_ctx['return_energy'],
                safety_reserve=energy_ctx['safety_reserve'],
            )
            if no_emergency:
                mode = 'normal'
                constraints = {
                    'lambda_max': 1.0,
                    'speed_min': 0.0,
                    'speed_max': 1.0,
                    'cpu_max': 1.0,
                    'p_tx_max': 1.0,
                }
            env.emergency_mode = mode

            action = agent.select_action(state_aug, eval=True)
            next_state, reward, done, truncated, info = env.step(action)
            annotate_cmdp_projection_risk(
                agent, state_aug, info, raw_action=action
            )
            if action_logger is not None:
                action_logger.log_action(episode, step, info)

            if use_lstm:
                # 评估期只保留在线残差校正，不更新LSTM网络权重。
                append_lstm_history(lstm_predictor, env, info)

            totals['reward'] += reward
            totals['energy'] += info.get('E_total', 0.0)
            totals['system_energy'] += info.get(
                'E_system_total', info.get('E_total', 0.0)
            )
            totals['tasks_completed'] += info.get('tasks_completed', 0)
            totals['cmdp_discounted_cost'] += (
                float(config.get('gamma', 0.99)) ** step
            ) * float(info.get('cmdp_training_cost', 0.0))
            totals['physical_constraint_violation_steps'] += int(
                info.get('physical_constraint_violation', False)
            )
            totals['safety_override_steps'] += int(
                info.get('safety_action_overridden', False)
            )
            totals['completed_task_delays'].extend(
                info.get('task_progress', {}).get(
                    'completed_task_delays', []
                )
            )
            if not info.get('short_horizon_energy_feasible', True):
                totals['short_horizon_energy_infeasible_steps'] += 1
            battery_depleted = battery_depleted or bool(
                info.get('battery_depleted', info.get('crashed', False))
            )
            energy_violation = energy_violation or bool(
                info.get('energy_violation')
            )
            return_triggered, returned_safe = update_safety_return_flags(
                return_triggered, returned_safe, info
            )
            mission_success, mission_deadline_missed = update_mission_outcome(
                mission_success, mission_deadline_missed, info
            )
            state = next_state

            if done or truncated:
                break

        episode_metrics['reward'].append(totals['reward'])
        episode_metrics['energy'].append(totals['energy'])
        episode_metrics['uav_energy'].append(totals['energy'])
        episode_metrics['system_energy'].append(totals['system_energy'])
        episode_metrics['tasks_completed'].append(totals['tasks_completed'])
        episode_metrics['cmdp_discounted_cost'].append(
            totals['cmdp_discounted_cost']
        )
        episode_metrics['cmdp_budget_exceeded'].append(float(
            totals['cmdp_discounted_cost']
            > float(config.get(
                'cmdp_cost_limit', config.get('constraint_cost_limit', 5.0)
            ))
        ))
        episode_metrics['physical_constraint_violation'].append(float(
            totals['physical_constraint_violation_steps'] > 0
        ))
        episode_metrics['physical_constraint_violation_rate'].append(float(
            totals['physical_constraint_violation_steps'] / (step + 1)
        ))
        episode_metrics['safety_override_rate'].append(
            totals['safety_override_steps'] / (step + 1)
        )
        episode_metrics['mean_task_delay_s'].append(
            float(np.mean(totals['completed_task_delays']))
            if totals['completed_task_delays'] else np.nan
        )
        episode_metrics['steps'].append(step + 1)
        episode_metrics['battery_depleted'].append(float(battery_depleted))
        episode_metrics['energy_violation'].append(float(energy_violation))
        episode_metrics['return_triggered'].append(float(return_triggered))
        episode_metrics['returned_safe_episode'].append(float(returned_safe))
        episode_metrics['returned_safe_after_trigger'].append(float(
            successful_safety_return(return_triggered, returned_safe)
        ))
        episode_metrics['mission_success'].append(float(mission_success))
        episode_metrics['mission_safe_abort'].append(float(
            info.get('mission_safe_abort', False)
        ))
        episode_metrics['mission_deadline_missed'].append(float(
            mission_deadline_missed
        ))
        episode_metrics['task_completion_ratio'].append(float(
            info.get('task_completion_ratio', 0.0)
        ))
        episode_metrics['core_task_completion_ratio'].append(float(
            info.get('core_task_completion_ratio', 1.0)
        ))
        episode_metrics['short_horizon_energy_infeasible_steps'].append(
            totals['short_horizon_energy_infeasible_steps']
        )

    summary = {}
    for key, values in episode_metrics.items():
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        summary[key] = float(np.mean(finite)) if finite.size else None
    triggered = int(np.sum(episode_metrics['return_triggered']))
    returned = int(np.sum(episode_metrics['returned_safe_after_trigger']))
    summary['returned_safe'] = conditional_safe_return_rate(
        returned, triggered
    )
    summary['evaluation_episodes'] = int(num_episodes)
    summary['evaluation_seed'] = eval_seed
    summary['return_triggered_episodes'] = triggered
    summary['returned_safe_episodes'] = returned
    if action_logger is not None:
        action_logger.save_actions(action_log_path)
    return summary


def run_ablation(config_path='configs/default.yaml',
                 num_episodes=500,
                 num_eval=50,
                 seed=None,
                 eval_seed=None,
                 output_root='results/ablation'):
    """运行所有消融实验"""
    config = load_config(config_path)
    train_seed = config.get('seed', 42) if seed is None else seed
    evaluation_seed = (
        config.get('eval_seed', 1042) if eval_seed is None else eval_seed
    )

    # 定义消融组
    ablations = {
        'Full (SAC-LSTM-EP)': SAC,
        'Abl-NoLSTM': SACNoLSTM,
        'Abl-NoEmergency': SACNoEmergency,
        'Abl-NoConstraint': SACNoConstraint,
    }

    results = {}
    for name, algo_cls in ablations.items():
        # 每个变体从相同全局随机状态开始，确保网络初始化和训练场景可比。
        normalized_train_seed = seed_everything(train_seed)
        print(f'\n{"="*60}')
        print(f'Running: {name}')
        print(f'=' * 60)

        env = UAVEnv(config)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        agent_settings = dict(config)
        agent_settings.update(component_seed_map(normalized_train_seed))
        agent_settings['component_seed'] = normalized_train_seed
        agent = algo_cls(state_dim, action_dim, agent_settings)
        with isolated_torch_seed(agent_settings['lstm_init_seed']):
            lstm_predictor = LSTMEnergyPredictor(
                hidden_dim=64, num_layers=2,
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
        lstm_frame_cap = lstm_predictor.lookback + lstm_predictor.horizon
        state_frames = []
        emergency = EmergencyManager(config)
        logger_config = dict(config)
        logger_config['max_episodes'] = num_episodes
        logger = Logger(logger_config)

        no_emergency = ('NoEmergency' in name)
        no_lstm = ('NoLSTM' in name)
        env.set_energy_safety_supervisor(not no_emergency)

        max_steps = config.get('max_steps', 200)

        # #4: 训练调度参数统一从config读, 与 experiments/train.py 完全一致
        update_interval = config.get('update_interval', 1)
        updates_per_step = config.get('updates_per_step', 1)

        # 训练
        for episode in range(num_episodes):
            current_env_seed = episode_seed(normalized_train_seed, episode)
            state, info = env.reset(seed=current_env_seed)
            if not no_lstm:
                lstm_predictor.reset_runtime_state()
                state_frames.clear()
            ep_reward = 0
            ep_energy = 0
            ep_system_energy = 0
            ep_tasks = 0
            ep_battery_depleted = False
            ep_return_triggered = False
            ep_returned = False
            ep_short_horizon_infeasible_steps = 0
            ep_cmdp_discounted_cost = 0.0
            ep_physical_constraint_violation = False
            ep_physical_constraint_violation_steps = 0
            ep_safety_override_steps = 0

            for step in range(max_steps):
                use_lstm = not no_lstm
                state_aug, E_pred = observation_for_agent(
                    state, env, lstm_predictor if use_lstm else None,
                    config, use_lstm=use_lstm,
                    # NoEmergency消融: 状态中zone不写入真实紧急模式, 避免信息泄漏
                    emergency_mode='normal' if no_emergency else None,
                )
                E_rem = env.E_rem
                energy_ctx = env.get_energy_context()
                mode, constraints = emergency.check(
                    E_rem, E_pred,
                    return_energy=energy_ctx['return_energy'],
                    safety_reserve=energy_ctx['safety_reserve'],
                )

                if no_emergency:
                    mode = 'normal'
                    constraints = {
                        'lambda_max': 1.0, 'speed_min': 0.0,
                        'speed_max': 1.0, 'cpu_max': 1.0, 'p_tx_max': 1.0,
                    }

                env.emergency_mode = mode

                action = agent.select_action(state_aug, eval=False)

                next_state, reward, done, truncated, info = env.step(action)
                annotate_cmdp_projection_risk(
                    agent, state_aug, info, raw_action=action
                )
                logger.log_action(episode, step, info)
                ep_return_triggered, ep_returned = update_safety_return_flags(
                    ep_return_triggered, ep_returned, info
                )

                if use_lstm:
                    append_lstm_history(lstm_predictor, env, info)
                    append_state_frame(
                        state_frames, env, lstm_frame_cap, info=info,
                        feature_schema=lstm_predictor.feature_schema,
                    )
                    maybe_update_lstm(lstm_predictor, state_frames, step)

                next_state_aug, _ = observation_for_agent(
                    next_state, env, lstm_predictor if use_lstm else None,
                    config, use_lstm=use_lstm,
                    emergency_mode='normal' if no_emergency else None,
                )

                requested_action = np.asarray(
                    info.get('raw_action', action), dtype=np.float32
                )
                applied_action = np.asarray(
                    info.get('applied_action', requested_action),
                    dtype=np.float32,
                )
                agent.replay_buffer.push(
                    state_aug, requested_action, reward,
                    next_state_aug, done,
                    cost=info.get('cmdp_training_cost', 0.0),
                    applied_action=applied_action,
                )

                # #4: 调度与 train.py 一致 (update_interval/updates_per_step + 算法内部warmup)
                if step % update_interval == 0:
                    for _ in range(updates_per_step):
                        agent.update()

                state = next_state
                ep_reward += reward
                ep_energy += info.get('E_total', 0)
                ep_system_energy += info.get(
                    'E_system_total', info.get('E_total', 0)
                )
                ep_tasks += info.get('tasks_completed', 0)
                ep_cmdp_discounted_cost += (
                    agent.gamma ** step
                ) * float(info.get('cmdp_training_cost', 0.0))
                physical_violation = bool(
                    info.get('physical_constraint_violation', False)
                )
                ep_physical_constraint_violation |= physical_violation
                ep_physical_constraint_violation_steps += int(
                    physical_violation
                )
                ep_safety_override_steps += int(
                    info.get('safety_action_overridden', False)
                )
                if info.get('battery_depleted', info.get('crashed', False)):
                    ep_battery_depleted = True
                if not info.get('short_horizon_energy_feasible', True):
                    ep_short_horizon_infeasible_steps += 1
                if done or truncated:
                    break

            cmdp_info = agent.update_lagrange(ep_cmdp_discounted_cost)

            logger.log_episode(episode, {
                'reward': ep_reward, 'energy': ep_energy,
                'uav_energy': ep_energy, 'system_energy': ep_system_energy,
                'tasks_completed': ep_tasks,
                'battery_depleted': ep_battery_depleted,
                'cmdp_discounted_cost': ep_cmdp_discounted_cost,
                'cmdp_cost_limit': agent.cmdp_cost_limit,
                'cmdp_budget_exceeded': bool(
                    ep_cmdp_discounted_cost > agent.cmdp_cost_limit
                ),
                'physical_constraint_violation': (
                    ep_physical_constraint_violation
                ),
                'physical_constraint_violation_steps': (
                    ep_physical_constraint_violation_steps
                ),
                'physical_constraint_violation_rate': (
                    ep_physical_constraint_violation_steps / (step + 1)
                ),
                'safety_override_steps': ep_safety_override_steps,
                'safety_override_rate': (
                    ep_safety_override_steps / (step + 1)
                ),
                **cmdp_info,
                'return_triggered': ep_return_triggered,
                'returned_safe': successful_safety_return(
                    ep_return_triggered, ep_returned
                ),
                'returned_safe_episode': ep_returned,
                'mission_success': bool(info.get('mission_success', False)),
                'mission_safe_abort': bool(
                    info.get('mission_safe_abort', False)
                ),
                'mission_deadline_missed': bool(
                    info.get('mission_deadline_missed', False)
                ),
                'mission_outcome': info.get('mission_outcome', 'disabled'),
                'mission_failure_reasons': ';'.join(
                    info.get('mission_failure_reasons', ())
                ),
                'tasks_generated': int(
                    info.get('episode_tasks_generated', 0)
                ),
                'task_completion_ratio': float(
                    info.get('task_completion_ratio', 0.0)
                ),
                'core_task_completion_ratio': float(
                    info.get('core_task_completion_ratio', 1.0)
                ),
                'environment_seed': current_env_seed,
                'short_horizon_energy_infeasible_steps': (
                    ep_short_horizon_infeasible_steps
                ),
                'steps': step + 1,
            })

            if (episode + 1) % 100 == 0:
                avg_r = np.mean(logger.data['reward'][-100:])
                print(f'  Ep {episode+1}: avg_reward={avg_r:.2f}')

        # 保存训练日志
        log_dir = str(Path(output_root) / name)
        os.makedirs(log_dir, exist_ok=True)
        logger.save(f'{log_dir}/log.csv')
        logger.save_actions(f'{log_dir}/actions.csv')
        logger.plot(f'{log_dir}/curves.png')

        # 使用冻结策略进行独立评估。训练日志只用于观察收敛过程，不再把最后
        # 100个训练回合的均值当作测试结果。
        final = evaluate_variant(
            agent,
            env,
            lstm_predictor,
            emergency,
            config,
            no_lstm=no_lstm,
            no_emergency=no_emergency,
            num_episodes=num_eval,
            seed=evaluation_seed,
            action_log_path=f'{log_dir}/evaluation_actions.csv',
        )
        results[name] = final

    # 打印消融表
    print('\n' + '=' * 112)
    print(f'{"Ablation":<25} {"Reward":^10} {"Tasks":^8} '
          f'{"Delay(s)":^10} {"SystemE":^12} {"Constraint%":^12} '
          f'{"Intervene%":^12} {"Mission%":^10}')
    print('-' * 112)
    for name, final in results.items():
        delay = final.get('mean_task_delay_s')
        delay_text = 'N/A' if delay is None else f'{delay:.2f}'
        print(f'{name:<25} {final.get("reward", 0):+8.2f}  '
              f'{final.get("tasks_completed", 0):6.1f}  '
              f'{delay_text:^10} {final.get("system_energy", 0):10.1f}  '
              f'{100 * final.get("physical_constraint_violation", 0):10.1f}%  '
              f'{100 * final.get("safety_override_rate", 0):10.1f}%  '
              f'{100 * final.get("mission_success", 0):8.1f}%')
    print('=' * 112)

    # 画消融对比图
    plot_ablation(results, Path(output_root) / 'ablation.png')
    return results


def plot_ablation(results, output_path='results/ablation.png'):
    """Plot the abstract-aligned native evaluation metrics."""
    names = list(results.keys())
    panels = [
        ('reward', 'Evaluation Reward', 'Reward', False),
        ('tasks_completed', 'Tasks Completed', 'Tasks / episode', False),
        ('mean_task_delay_s', 'Mean Completed-task Delay', 'Delay (s)', False),
        ('system_energy', 'System Energy Consumption', 'Energy (J)', False),
        ('physical_constraint_violation', 'Physical Constraint Violation Rate', 'Rate (%)', True),
        ('mission_success', 'Joint Task-return Success', 'Rate (%)', True),
        ('returned_safe_episode', 'Safe Return Rate', 'Rate (%)', True),
        ('safety_override_rate', 'Safety-layer Intervention Rate', 'Rate (%)', True),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, (metric, title, ylabel, is_rate) in zip(axes.flat, panels):
        values = [results[n].get(metric) for n in names]
        values = [np.nan if v is None else float(v) for v in values]
        if is_rate:
            values = [v * 100 for v in values]
        colors = ['#2ecc71' if i == 0 else '#e74c3c' for i in range(len(names))]
        bars = ax.bar(range(len(names)), values, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha='right', fontsize=8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + abs(bar.get_height()) * 0.02,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150)
    plt.close()
    print(f'\nAblation plot saved to {output}')


def plot_multiseed_ablation(summary, output_path):
    """Plot abstract-aligned metrics with across-training-seed uncertainty."""
    variants = list(summary.keys())
    panels = [
        ('reward', 'Evaluation Reward', 'Reward', False),
        ('tasks_completed', 'Tasks Completed', 'Tasks / episode', False),
        ('mean_task_delay_s', 'Mean Completed-task Delay', 'Delay (s)', False),
        ('system_energy', 'System Energy Consumption', 'Energy (J)', False),
        ('physical_constraint_violation', 'Physical Constraint Violation Rate', 'Rate (%)', True),
        ('mission_success', 'Joint Task-return Success', 'Rate (%)', True),
        ('returned_safe', 'Safe Return Rate', 'Rate (%)', True),
        ('safety_override_rate', 'Safety-layer Intervention Rate', 'Rate (%)', True),
    ]
    figure, axes = plt.subplots(2, 4, figsize=(18, 9))
    for axis, (metric, title, ylabel, is_rate) in zip(axes.flat, panels):
        means = [summary[name][metric]['mean'] for name in variants]
        stds = [summary[name][metric]['std'] for name in variants]
        means = [np.nan if value is None else value for value in means]
        if is_rate:
            means = [value * 100.0 for value in means]
            stds = [value * 100.0 for value in stds]
        axis.bar(range(len(variants)), means, yerr=stds, capsize=4)
        axis.set_xticks(range(len(variants)))
        axis.set_xticklabels(variants, rotation=25, ha='right', fontsize=8)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis='y', alpha=0.3)
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def run_multiseed_ablation(config_path='configs/default.yaml',
                           num_episodes=500,
                           num_eval=50,
                           seeds=None,
                           eval_seed=None,
                           run_name='paper',
                           publish_latest=True,
                           resume=True):
    """以至少3个独立训练seed运行消融并报告跨seed统计。"""
    config = load_config(config_path)
    evaluation_seed = int(
        config.get('eval_seed', 1042) if eval_seed is None else eval_seed
    )
    train_seeds = validate_multiseed_episode_ranges(
        seeds or DEFAULT_MULTI_SEED_BASES,
        num_episodes,
        evaluation_seed,
        num_eval,
    )
    fingerprint = environment_fingerprint(config)
    run_root = (
        Path('results') / 'ablation_experiments'
        / fingerprint[:12] / str(run_name)
    )
    manifest = experiment_manifest(
        config,
        status='running',
        experiment='multi_seed_ablation',
        run_name=str(run_name),
        train_seeds=train_seeds,
        evaluation_seed=evaluation_seed,
        train_episodes=int(num_episodes),
        evaluation_episodes=int(num_eval),
    )
    write_json_atomic(run_root / 'manifest.json', manifest)

    per_seed = {}
    for train_seed in train_seeds:
        seed_root = run_root / f'seed_{train_seed}'
        seed_summary_file = seed_root / 'summary.json'
        if resume and seed_summary_file.is_file():
            with seed_summary_file.open('r', encoding='utf-8') as stream:
                cached = json.load(stream)
            if (
                cached.get('environment_fingerprint') == fingerprint
                and cached.get('train_seed') == train_seed
                and cached.get('evaluation_seed') == evaluation_seed
            ):
                print(f'Reusing completed ablation seed={train_seed}')
                per_seed[str(train_seed)] = cached['results']
                continue

        seed_results = run_ablation(
                config_path=config_path,
                num_episodes=num_episodes,
                num_eval=num_eval,
                seed=train_seed,
                eval_seed=evaluation_seed,
                output_root=seed_root,
            )
        per_seed[str(train_seed)] = seed_results
        write_json_atomic(seed_summary_file, {
            'environment_fingerprint': fingerprint,
            'train_seed': train_seed,
            'evaluation_seed': evaluation_seed,
            'results': seed_results,
        })

    variants = list(next(iter(per_seed.values())).keys())
    aggregate = {}
    for variant in variants:
        aggregate[variant] = {}
        metric_names = next(iter(per_seed.values()))[variant].keys()
        for metric in metric_names:
            raw_values = [per_seed[str(seed)][variant][metric]
                          for seed in train_seeds]
            if not all(isinstance(value, (int, float, np.number))
                       for value in raw_values):
                continue
            values = [float(value) for value in raw_values]
            aggregate[variant][metric] = {
                'mean': float(np.mean(values)),
                'std': (
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                ),
                'values': values,
            }

    completed = {
        **manifest,
        'status': 'complete',
        'per_seed': per_seed,
        'summary': aggregate,
    }
    result_file = run_root / 'ablation_results.json'
    write_json_atomic(result_file, completed)
    write_json_atomic(run_root / 'manifest.json', {
        **manifest,
        'status': 'complete',
        'result_file': result_file.name,
    })
    if publish_latest:
        write_json_atomic(Path('results') / 'latest_ablation.json', {
            'environment_fingerprint': fingerprint,
            'result_file': str(result_file.as_posix()),
        })

    plot_multiseed_ablation(aggregate, run_root / 'ablation.png')
    return completed


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run ablation experiments')
    parser.add_argument('--episodes', type=int, default=200,
                        help='Training episodes per ablation (small for quick test)')
    parser.add_argument('--eval-episodes', type=int, default=50,
                        help='Independent evaluation episodes per ablation')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--eval-seed', type=int, default=None)
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Run multi-seed ablation (at least 3 seeds)')
    parser.add_argument('--run-name', default='paper')
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args()
    if args.seeds:
        run_multiseed_ablation(
            num_episodes=args.episodes,
            num_eval=args.eval_episodes,
            seeds=args.seeds,
            eval_seed=args.eval_seed,
            run_name=args.run_name,
            resume=not args.no_resume,
        )
    else:
        run_ablation(
            num_episodes=args.episodes,
            num_eval=args.eval_episodes,
            seed=args.seed,
            eval_seed=args.eval_seed,
        )
