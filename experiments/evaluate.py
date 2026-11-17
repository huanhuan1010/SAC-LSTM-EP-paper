"""
评估脚本: SAC-LSTM-EP 模型评估
用法: python experiments/evaluate.py --checkpoint models/sac_final.pt [--episodes 50]
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from env.uav_env import UAVEnv
from agent.sac import SAC
from agent.lstm_predictor import LSTMEnergyPredictor
from agent.lstm_features import lstm_feature_kwargs
from agent.lstm_runtime import (
    observation_for_agent,
    append_lstm_history,
)
from agent.emergency import EmergencyManager
from utils.config import load_config
from utils.logger import Logger
from utils.checkpoints import checkpoint_metadata, load_checkpoint_pair
from utils.evaluation_metrics import (
    conditional_safe_return_rate,
    successful_safety_return,
    update_mission_outcome,
    update_safety_return_flags,
)
from utils.seeding import episode_seed, seed_everything


def evaluate(config_path='configs/default.yaml', checkpoint='models/sac_final.pt',
             num_episodes=50, seed=None, lstm_checkpoint=None,
             action_log_path='results/evaluation_actions.csv'):
    """评估已训练的SAC-LSTM-EP模型"""
    config = load_config(config_path)
    eval_seed = seed_everything(
        config.get('eval_seed', 1042) if seed is None else seed
    )

    env = UAVEnv(config)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 构造策略与LSTM后按配对清单统一加载。缺少任一文件时直接终止评估，
    # 禁止随机初始化模型静默产生“评估结果”。
    agent = SAC(state_dim, action_dim, config)
    lstm_predictor = LSTMEnergyPredictor(
        hidden_dim=config.get('lstm_hidden_dim', 64),
        num_layers=config.get('lstm_num_layers', 2),
        lookback=config.get('lookback', 10),
        horizon=config.get('horizon', 5),
        battery_capacity=config.get('battery_capacity', 100000.0),
        prediction_mode=config.get('lstm_prediction_mode', 'absolute'),
        residual_correction=config.get('residual_correction_enabled', True),
        residual_ema_alpha=config.get('residual_ema_alpha', 0.2),
        residual_clip_ratio=config.get('residual_clip_ratio', 0.01),
        **lstm_feature_kwargs(config),
    )
    loaded = load_checkpoint_pair(
        agent,
        lstm_predictor,
        checkpoint,
        lstm_checkpoint=lstm_checkpoint,
        expected_algorithm='SAC-LSTM-EP',
        expected_metadata=checkpoint_metadata(config, state_dim, action_dim),
        allow_legacy_explicit=lstm_checkpoint is not None,
    )
    print(f'Loaded policy checkpoint: {loaded["policy"]}')
    print(f'Loaded paired LSTM checkpoint: {loaded["lstm"]}')
    emergency = EmergencyManager(config)
    action_logger = None
    if action_log_path is not None:
        action_log_config = dict(config)
        action_log_config['max_episodes'] = num_episodes
        action_logger = Logger(action_log_config)

    # 评估指标
    metrics = {
        'reward': [], 'energy': [], 'nominal_energy': [],
        'energy_uncertainty_delta': [],
        'mean_energy_uncertainty_scale': [],
        'system_energy': [], 'tasks': [], 'steps': [],
        'cmdp_discounted_cost': [],
        'cmdp_cost_limit': float(config.get(
            'cmdp_cost_limit', config.get('constraint_cost_limit', 5.0)
        )),
        'cmdp_budget_violations': 0,
        'physical_constraint_violation_episodes': 0,
        'physical_constraint_violation_steps': 0,
        'safety_override_steps': 0,
        'battery_depleted_episodes': 0, 'energy_violation_episodes': 0,
        'short_horizon_energy_infeasible_steps': 0,
        'return_triggered': 0, 'returned_safe': 0,
        'safe_return_episodes': 0,
        'mission_success_episodes': 0,
        'mission_safe_abort_episodes': 0,
        'mission_deadline_missed_episodes': 0,
        'task_completion_ratio': [],
        'core_task_completion_ratio': [],
        'mission_failure_reason_counts': {},
        'avg_lambda': [], 'avg_speed': [],
        'emergency_counts': {'normal': 0, 'yellow': 0, 'orange': 0, 'red': 0},
        'evaluation_seed': eval_seed,
    }

    for ep in range(num_episodes):
        state, info = env.reset(seed=episode_seed(eval_seed, ep))
        lstm_predictor.reset_runtime_state()
        ep_reward = 0
        ep_energy = 0
        ep_nominal_energy = 0
        ep_energy_uncertainty_delta = 0
        ep_energy_uncertainty_scales = []
        ep_system_energy = 0
        ep_tasks = 0
        ep_battery_depleted = False
        ep_energy_violation = False
        ep_cmdp_discounted_cost = 0.0
        ep_physical_constraint_violation = False
        ep_return_triggered = False
        ep_returned = False
        ep_mission_success = False
        ep_mission_deadline_missed = False
        ep_emergency = {'normal': 0, 'yellow': 0, 'orange': 0, 'red': 0}
        ep_lam = []
        ep_speed = []

        for step in range(config.get('max_steps', 200)):
            state_aug, E_pred = observation_for_agent(
                state, env, lstm_predictor, config, use_lstm=True,
            )
            E_rem = env.E_rem
            energy_ctx = env.get_energy_context()
            mode, _ = emergency.check(
                E_rem, E_pred,
                return_energy=energy_ctx['return_energy'],
                safety_reserve=energy_ctx['safety_reserve'],
            )
            env.emergency_mode = mode

            action = agent.select_action(state_aug, eval=True)

            next_state, reward, done, truncated, info = env.step(action)
            if action_logger is not None:
                action_logger.log_action(ep, step, info)
            executed_action = np.asarray(
                info.get('applied_action', action), dtype=np.float32
            )
            ep_lam.append(np.mean(np.abs(
                executed_action[:env.num_users]
            )))
            ep_speed.append(abs(executed_action[env.num_users + 2]))
            applied_mode = info.get('safety_mode_applied', mode)
            ep_emergency[applied_mode] += 1
            ep_return_triggered, ep_returned = update_safety_return_flags(
                ep_return_triggered, ep_returned, info
            )
            ep_mission_success, ep_mission_deadline_missed = (
                update_mission_outcome(
                    ep_mission_success, ep_mission_deadline_missed, info
                )
            )
            append_lstm_history(lstm_predictor, env, info)

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
            ep_tasks += info.get('tasks_completed', 0)
            ep_cmdp_discounted_cost += (
                float(config.get('gamma', 0.99)) ** step
            ) * float(info.get('cmdp_training_cost', 0.0))
            physical_violation = bool(
                info.get('physical_constraint_violation', False)
            )
            ep_physical_constraint_violation |= physical_violation
            metrics['physical_constraint_violation_steps'] += int(
                physical_violation
            )

            if info.get('battery_depleted', info.get('crashed', False)):
                ep_battery_depleted = True
            if info.get('energy_violation'):
                ep_energy_violation = True
            if not info.get('short_horizon_energy_feasible', True):
                metrics['short_horizon_energy_infeasible_steps'] += 1
            metrics['safety_override_steps'] += int(
                info.get('safety_action_overridden', False)
            )
            if done or truncated:
                break

        # 记录本episode
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
        metrics['cmdp_discounted_cost'].append(ep_cmdp_discounted_cost)
        metrics['tasks'].append(ep_tasks)
        metrics['steps'].append(step + 1)
        metrics['avg_lambda'].append(np.mean(ep_lam) if ep_lam else 0)
        metrics['avg_speed'].append(np.mean(ep_speed) if ep_speed else 0)
        if ep_battery_depleted:
            metrics['battery_depleted_episodes'] += 1
        if ep_energy_violation:
            metrics['energy_violation_episodes'] += 1
        if ep_physical_constraint_violation:
            metrics['physical_constraint_violation_episodes'] += 1
        if ep_cmdp_discounted_cost > metrics['cmdp_cost_limit']:
            metrics['cmdp_budget_violations'] += 1
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
        metrics['task_completion_ratio'].append(float(
            info.get('task_completion_ratio', 0.0)
        ))
        metrics['core_task_completion_ratio'].append(float(
            info.get('core_task_completion_ratio', 1.0)
        ))
        for reason in info.get('mission_failure_reasons', ()):
            counts = metrics['mission_failure_reason_counts']
            counts[reason] = counts.get(reason, 0) + 1
        for k in ep_emergency:
            metrics['emergency_counts'][k] += ep_emergency[k]

    # 汇总
    total_eps = num_episodes
    print('\n' + '=' * 60)
    print(f'Evaluation Results ({num_episodes} episodes)')
    print('=' * 60)
    print(f'Avg Reward:          {np.mean(metrics["reward"]):+8.2f} ± {np.std(metrics["reward"]):.2f}')
    print(f'Avg UAV Energy:      {np.mean(metrics["energy"]):8.1f} J')
    print(
        'Avg Nominal UAV E:   '
        f'{np.mean(metrics["nominal_energy"]):8.1f} J'
    )
    print(
        'Avg Energy Delta:    '
        f'{np.mean(metrics["energy_uncertainty_delta"]):+8.1f} J'
    )
    print(
        'Avg Energy Scale:    '
        f'{np.mean(metrics["mean_energy_uncertainty_scale"]):8.3f}'
    )
    print(f'Avg System Energy:   {np.mean(metrics["system_energy"]):8.1f} J')
    print(f'Avg Tasks Completed: {np.mean(metrics["tasks"]):8.1f}')
    print(f'Avg Steps:           {np.mean(metrics["steps"]):8.1f}')
    battery_depletion_rate = metrics['battery_depleted_episodes'] / total_eps
    energy_violation_rate = metrics['energy_violation_episodes'] / total_eps
    physical_constraint_violation_rate = (
        metrics['physical_constraint_violation_episodes'] / total_eps
    )
    cmdp_budget_exceedance_rate = (
        metrics['cmdp_budget_violations'] / total_eps
    )
    safe_return_rate = metrics['safe_return_episodes'] / total_eps
    mission_success_rate = metrics['mission_success_episodes'] / total_eps
    mission_safe_abort_rate = (
        metrics['mission_safe_abort_episodes'] / total_eps
    )
    task_completion_rate = float(np.mean(metrics['task_completion_ratio']))
    core_task_completion_rate = float(
        np.mean(metrics['core_task_completion_ratio'])
    )
    print(f'Battery Depletion:   {battery_depletion_rate*100:7.1f}%')
    print(f'Energy Violation:    {energy_violation_rate*100:7.1f}%')
    print(f'Physical Constraint: {physical_constraint_violation_rate*100:7.1f}%')
    print(f'CMDP Budget Exceed:  {cmdp_budget_exceedance_rate*100:7.1f}%')
    print(f'Safe Return:         {safe_return_rate*100:7.1f}%')
    print(f'Mission Success:     {mission_success_rate*100:7.1f}%')
    print(f'Task Completion:     {task_completion_rate*100:7.1f}%')
    print(f'Core Completion:     {core_task_completion_rate*100:7.1f}%')
    print(f'Mission Safe Abort:  {mission_safe_abort_rate*100:7.1f}%')
    # Machine-readable 0..1 ratios consumed by AutoPaper.
    print(f'METRIC battery_depletion_rate={battery_depletion_rate:.6f}')
    print(f'METRIC energy_violation_rate={energy_violation_rate:.6f}')
    print(
        'METRIC constraint_exceedance_rate='
        f'{physical_constraint_violation_rate:.6f}'
    )
    print(
        'METRIC physical_constraint_violation_rate='
        f'{physical_constraint_violation_rate:.6f}'
    )
    print(
        'METRIC cmdp_budget_exceedance_rate='
        f'{cmdp_budget_exceedance_rate:.6f}'
    )
    print(f'METRIC safe_return_rate={safe_return_rate:.6f}')
    print(f'METRIC mission_success_rate={mission_success_rate:.6f}')
    print(f'METRIC task_completion_rate={task_completion_rate:.6f}')
    print(f'METRIC core_task_completion_rate={core_task_completion_rate:.6f}')
    print(f'METRIC mission_safe_abort_rate={mission_safe_abort_rate:.6f}')
    print(
        'Avg CMDP Training Cost: '
        f'{np.mean(metrics["cmdp_discounted_cost"]):8.3f} '
        f'(budget={metrics["cmdp_cost_limit"]:.3f})'
    )
    print(
        'CMDP Budget Violation:'
        f'{metrics["cmdp_budget_violations"]/total_eps*100:7.1f}%'
    )
    print(f'Final Multiplier:    {float(agent.lagrange_multiplier.item()):8.4f}')
    print(
        'Safety Override Rate:'
        f'{metrics["safety_override_steps"]/max(sum(metrics["steps"]), 1)*100:7.1f}%'
    )
    total_steps = max(sum(metrics['steps']), 1)
    print(
        'Short-Horizon Energy Infeasible Steps: '
        f'{metrics["short_horizon_energy_infeasible_steps"]/total_steps*100:7.1f}%'
    )
    triggered = metrics['return_triggered']
    safe_rate = conditional_safe_return_rate(
        metrics['returned_safe'], triggered
    ) * 100.0
    print(
        'Return Trigger Rate: '
        f'{triggered/total_eps*100:7.1f}% ({triggered}/{total_eps})'
    )
    print(
        'Safe Return Rate:    '
        f'{metrics["safe_return_episodes"]/total_eps*100:7.1f}% '
        f'({metrics["safe_return_episodes"]}/{total_eps})'
    )
    print(f'Conditional Return:  {safe_rate:7.1f}% '
          f'({metrics["returned_safe"]}/{triggered} triggered episodes)')
    print(
        'Mission Success:     '
        f'{metrics["mission_success_episodes"]/total_eps*100:7.1f}% '
        f'({metrics["mission_success_episodes"]}/{total_eps})'
    )
    print(
        'Mission Deadline Miss:'
        f'{metrics["mission_deadline_missed_episodes"]/total_eps*100:7.1f}% '
        f'({metrics["mission_deadline_missed_episodes"]}/{total_eps})'
    )
    print(f'Avg |λ|:             {np.mean(metrics["avg_lambda"]):8.3f}')
    print(f'Avg Speed (norm):    {np.mean(metrics["avg_speed"]):8.3f}')
    print(f'Emergency Dist:      normal={metrics["emergency_counts"]["normal"]} '
          f'yellow={metrics["emergency_counts"]["yellow"]} '
          f'orange={metrics["emergency_counts"]["orange"]} '
          f'red={metrics["emergency_counts"]["red"]}')
    print('=' * 60)

    if action_logger is not None:
        action_logger.save_actions(action_log_path)
    return metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate SAC-LSTM-EP')
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--checkpoint', default='models/sac_final.pt')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument(
        '--lstm-checkpoint', default=None,
        help='Explicit legacy LSTM path; new paired checkpoints derive it from the manifest',
    )
    parser.add_argument(
        '--action-log',
        default='results/evaluation_actions.csv',
        help='CSV path for per-step raw/applied action traces',
    )
    args = parser.parse_args()

    evaluate(
        args.config,
        args.checkpoint,
        args.episodes,
        args.seed,
        args.lstm_checkpoint,
        args.action_log,
    )
