"""
统一基线训练脚本
训练/评估任意基线算法并输出结果
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from env.uav_env import UAVEnv
from agent.sac import SAC
from agent.lstm_predictor import LSTMEnergyPredictor
from agent.lstm_features import lstm_feature_kwargs
from agent.oracle_predictor import OracleEnergyTracePredictor
from agent.lstm_runtime import (
    observation_for_agent,
    append_lstm_history,
    append_state_frame,
    maybe_update_lstm,
    reset_prediction_transform,
)
from agent.emergency import EmergencyManager
from utils.config import load_config
from utils.logger import Logger
from utils.checkpoints import checkpoint_metadata, save_checkpoint_pair
from utils.evaluation_metrics import (
    annotate_cmdp_projection_risk,
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
from utils.ablation_support import (
    configure_profile_environment,
    profile_observation_controller,
    profile_safety_decision,
)
from utils.algorithm_profiles import algorithm_config, algorithm_profile

# 基线算法
from baselines.random_baseline import RandomPolicy
from baselines.greedy import GreedyPolicy
from baselines.ddpg import DDPG
from baselines.sac_ablated import (
    SACNoLSTM,
    SACNoEmergency,
    SACNoConstraint,
)
from baselines.ppo import PPO

ALGO_MAP = {
    # Paper-facing v1 identifiers.  They all retain the SAC implementation;
    # component differences are resolved exclusively by AlgorithmProfile.
    'SAC-LSTM-EP-PaperMain': SAC,
    'SAC-LSTM-EP-CMDP': SAC,
    'SAC-LSTM-EP-NoPrediction': SAC,
    'SAC-LSTM-EP-FixedSOC': SAC,
    'SAC-LSTM-EP': SAC,
    'SAC': SAC,
    'SAC+LSTM': SAC,
    'SAC+CMDP': SAC,
    'C-SAC-Alvi-Adapted': SAC,
    'SAC+SafetyProjection': SAC,
    'SAC-LSTM-EP-NoSafety': SAC,
    'SAC-Oracle': SAC,
    'SAC-SOCThresholdReturn': SAC,
    'SAC-Current': SAC,
    'SAC-Persistence-EP': SAC,
    'SAC-Shuffled-EP': SAC,
    'SAC-FixedSOC': SAC,
    'SAC-LSTM-EP-SOC': SAC,
    'SAC-LSTM-EP-PredictiveReturn': SAC,
    'SAC-LSTM-EP-NoCMDP': SAC,
    'SAC-LSTM-EP-SingleLevel': SAC,
    'SAC-NoLSTM': SACNoLSTM,
    'SAC-NoEmergency': SACNoEmergency,
    'SAC-NoConstraint': SACNoConstraint,
    'PPO': PPO,
    'DDPG': DDPG,
    'DDPG-MatchedSafety': DDPG,
    'Random': RandomPolicy,
    'Greedy': GreedyPolicy,
}


def create_agent(name, state_dim, action_dim, config):
    """工厂方法创建算法实例"""
    cls = ALGO_MAP.get(name)
    if cls is None:
        raise ValueError(f'Unknown algorithm: {name}. Available: {list(ALGO_MAP.keys())}')
    return cls(state_dim, action_dim, config)


def train_baseline(algo_name, config_path='configs/default.yaml',
                   num_episodes=500, save_path=None, verbose=True, seed=None,
                   output_dir=None):
    """训练一个基线算法"""
    config = load_config(config_path)
    if num_episodes is not None:
        config['max_episodes'] = num_episodes
    train_seed = seed_everything(
        config.get('seed', 42) if seed is None else seed
    )

    profile = algorithm_profile(algo_name)
    agent_settings = algorithm_config(config, algo_name)
    env = UAVEnv(agent_settings)
    configure_profile_environment(env, profile)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent_settings.update(component_seed_map(train_seed))
    agent_settings['component_seed'] = train_seed
    agent = create_agent(algo_name, state_dim, action_dim, agent_settings)

    # LSTM & Emergency (SAC-NoLSTM / Random / Greedy 不用 LSTM)
    use_lstm = profile.use_lstm
    use_oracle = profile.use_oracle_prediction
    use_emergency = profile.use_energy_safety_supervisor

    lstm_predictor = None
    state_frames = []
    lstm_frame_cap = 0
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
        lstm_frame_cap = lstm_predictor.lookback + lstm_predictor.horizon
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

    emergency = EmergencyManager(config)

    # #4: 训练调度参数统一从config读, 与 experiments/train.py 完全一致
    update_interval = config.get('update_interval', 1)   # SAC原论文: 每1步learn
    updates_per_step = config.get('updates_per_step', 1)

    logger = Logger(config)
    max_steps = config.get('max_steps', 200)
    print_interval = max(1, int(config.get('baseline_print_interval', 50)))

    for episode in range(max(1, num_episodes)):
        current_env_seed = episode_seed(train_seed, episode)
        state, info = env.reset(seed=current_env_seed)
        if use_oracle:
            lstm_predictor.reset_runtime_state(current_env_seed)
        elif use_lstm:
            lstm_predictor.reset_runtime_state()
        reset_prediction_transform(
            lstm_predictor, profile.prediction_source, current_env_seed
        )
        if use_lstm:
            state_frames.clear()
        ep_reward = 0
        ep_energy = 0
        ep_system_energy = 0
        ep_user_energy = 0
        ep_completed_task_delays = []
        ep_deadline_missed_tasks = 0
        ep_deadline_evaluable_tasks = 0
        ep_deadline_overshoot_s_sum = 0.0
        ep_deadline_overshoot_normalized_sum = 0.0
        ep_tasks = 0
        ep_battery_depleted = False
        ep_energy_violation = False
        ep_return_triggered = False
        ep_returned = False
        ep_mission_success = False
        ep_mission_deadline_missed = False
        ep_short_horizon_infeasible_steps = 0
        ep_cmdp_discounted_cost = 0.0
        ep_cmdp_stage_cost_sum = 0.0
        ep_cmdp_pre_projection_state_cost_sum = 0.0
        ep_cmdp_post_projection_state_cost_sum = 0.0
        ep_cmdp_projection_state_cost_delta_sum = 0.0
        ep_cmdp_projection_state_cost_samples = 0
        ep_cmdp_pre_projection_critic_risks = []
        ep_cmdp_post_projection_critic_risks = []
        ep_cmdp_projection_critic_risk_deltas = []
        ep_cmdp_cost_active_steps = 0
        ep_cmdp_cost_saturated_steps = 0
        ep_update_diagnostics = {
            key: [] for key in (
                'cost_critic_loss', 'cmdp_actor_penalty_active',
                'cmdp_actor_penalty_scale',
                'policy_cost_q_mean', 'policy_cost_q_std',
                'replay_cost_q_mean',
                'cmdp_pre_projection_q_mean',
                'cmdp_post_projection_q_mean',
                'constraint_term_mean', 'cost_critic_update_count',
                'cmdp_positive_cost_samples',
                'actor_reward_gradient_norm',
                'actor_cost_gradient_norm',
                'actor_cost_to_reward_gradient_ratio',
            )
        }
        ep_physical_constraint_violation = False
        ep_physical_constraint_violation_steps = 0
        ep_safety_override_steps = 0
        ep_scheduled_return_override_steps = 0
        ep_reward_components = {
            key: 0.0 for key in (
                'task_reward', 'energy_penalty', 'delay_penalty',
                'proximity_reward',
            )
        }

        for step in range(max_steps):
            state_aug, E_pred = observation_for_agent(
                state, env, lstm_predictor, config,
                use_lstm=use_lstm or use_oracle,
                safety_controller=profile_observation_controller(profile),
                prediction_source=profile.prediction_source,
            )
            E_rem = env.E_rem
            mode = 'normal'
            constraints = {
                'lambda_max': 1.0, 'speed_min': 0.0, 'speed_max': 1.0,
                'cpu_max': 1.0, 'p_tx_max': 1.0,
            }

            if use_emergency:
                mode, constraints = profile_safety_decision(
                    profile, env, emergency, E_pred
                )
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
            ep_mission_success, ep_mission_deadline_missed = (
                update_mission_outcome(
                    ep_mission_success, ep_mission_deadline_missed, info
                )
            )

            if use_lstm:
                append_lstm_history(lstm_predictor, env, info)
                append_state_frame(
                    state_frames, env, lstm_frame_cap, info=info,
                    feature_schema=lstm_predictor.feature_schema,
                )
                maybe_update_lstm(
                    lstm_predictor,
                    state_frames,
                    step,
                    config.get('lstm_update_interval', 10),
                )
            elif use_oracle:
                lstm_predictor.observe_actual_energy(info.get('E_total', 0.0))

            next_state_aug, _ = observation_for_agent(
                next_state, env, lstm_predictor, config,
                use_lstm=use_lstm or use_oracle,
                safety_controller=profile_observation_controller(profile),
                prediction_source=profile.prediction_source,
            )

            requested_action = np.asarray(
                info.get('raw_action', action), dtype=np.float32
            )
            applied_action = np.asarray(
                info.get('applied_action', requested_action),
                dtype=np.float32,
            )
            if hasattr(agent, 'store_transition'):
                # PPO's log probability was computed for the requested
                # action, so storing the projected action would make its
                # policy-ratio update mathematically inconsistent.
                agent.store_transition(
                    state_aug, requested_action, reward, next_state_aug, done
                )
            elif hasattr(agent, 'replay_buffer'):
                if hasattr(agent.replay_buffer, 'costs'):
                    agent.replay_buffer.push(
                        state_aug, requested_action, reward, next_state_aug,
                        done, cost=info.get('cmdp_training_cost', 0.0),
                        applied_action=applied_action,
                    )
                else:
                    agent.replay_buffer.push(
                        state_aug, requested_action, reward,
                        next_state_aug, done
                    )

            # 更新 (PPO: episode结束时; SAC/DDPG: 每update_interval步) — 调度与train.py一致
            if hasattr(agent, 'store_transition'):
                # PPO: 仅episode结束时更新
                if done or truncated:
                    update_info = agent.update()
                    if isinstance(update_info, dict):
                        for key in ep_update_diagnostics:
                            value = update_info.get(key)
                            if value is not None and np.isfinite(value):
                                ep_update_diagnostics[key].append(float(value))
            elif hasattr(agent, 'update') and step % update_interval == 0:
                for _ in range(updates_per_step):
                    update_info = agent.update()
                    if isinstance(update_info, dict):
                        for key in ep_update_diagnostics:
                            value = update_info.get(key)
                            if value is not None and np.isfinite(value):
                                ep_update_diagnostics[key].append(float(value))

            state = next_state
            ep_reward += reward
            ep_energy += info.get('E_total', 0)
            ep_system_energy += info.get('E_system_total', info.get('E_total', 0))
            ep_user_energy += info.get('energy_breakdown', {}).get(
                'E_user_total', 0.0
            )
            ep_completed_task_delays.extend(
                info.get('task_progress', {}).get(
                    'completed_task_delays', []
                )
            )
            task_progress = info.get('task_progress', {})
            step_deadline_misses = int(task_progress.get(
                'deadline_miss_count', info.get('tasks_failed', 0)
            ))
            ep_deadline_missed_tasks += step_deadline_misses
            ep_deadline_evaluable_tasks += (
                int(info.get('tasks_completed', 0))
                + step_deadline_misses
            )
            ep_deadline_overshoot_s_sum += float(
                task_progress.get('deadline_overshoot_s_sum', 0.0)
            )
            ep_deadline_overshoot_normalized_sum += float(
                task_progress.get(
                    'deadline_overshoot_normalized_sum', 0.0
                )
            )
            ep_tasks += info.get('tasks_completed', 0)
            stage_cost = float(info.get('cmdp_training_cost', 0.0))
            ep_cmdp_discounted_cost += (
                float(config.get('gamma', 0.99)) ** step
            ) * stage_cost
            ep_cmdp_stage_cost_sum += stage_cost
            projection_costs = (
                float(info.get('cmdp_pre_projection_state_cost', np.nan)),
                float(info.get('cmdp_post_projection_state_cost', np.nan)),
                float(info.get('cmdp_projection_state_cost_delta', np.nan)),
            )
            if (
                info.get('cmdp_projection_cost_comparison_available', True)
                and all(np.isfinite(value) for value in projection_costs)
            ):
                ep_cmdp_pre_projection_state_cost_sum += projection_costs[0]
                ep_cmdp_post_projection_state_cost_sum += projection_costs[1]
                ep_cmdp_projection_state_cost_delta_sum += projection_costs[2]
                ep_cmdp_projection_state_cost_samples += 1
            paired_risks = (
                float(info.get('cmdp_pre_projection_critic_risk', np.nan)),
                float(info.get('cmdp_post_projection_critic_risk', np.nan)),
                float(info.get('cmdp_projection_critic_risk_delta', np.nan)),
            )
            if all(np.isfinite(value) for value in paired_risks):
                ep_cmdp_pre_projection_critic_risks.append(paired_risks[0])
                ep_cmdp_post_projection_critic_risks.append(paired_risks[1])
                ep_cmdp_projection_critic_risk_deltas.append(paired_risks[2])
            ep_cmdp_cost_active_steps += int(stage_cost > 0.0)
            saturation_threshold = float(info.get(
                'cmdp_cost_saturation_threshold', 1.0
            ))
            ep_cmdp_cost_saturated_steps += int(
                stage_cost >= saturation_threshold - 1e-12
            )
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
            ep_scheduled_return_override_steps += int(
                info.get('scheduled_return_action_overridden', False)
            )
            for key in ep_reward_components:
                ep_reward_components[key] += info.get(
                    'reward_breakdown', {}
                ).get(key, 0.0)
            if info.get('battery_depleted', info.get('crashed', False)):
                ep_battery_depleted = True
            if info.get('energy_violation'):
                ep_energy_violation = True
            if not info.get('short_horizon_energy_feasible', True):
                ep_short_horizon_infeasible_steps += 1

            if done or truncated:
                break

        if hasattr(agent, 'update_lagrange'):
            cmdp_info = agent.update_lagrange(ep_cmdp_discounted_cost)
            cmdp_limit = agent.cmdp_cost_limit
        else:
            cmdp_limit = float(config.get(
                'cmdp_cost_limit', config.get('constraint_cost_limit', 5.0)
            ))
            cmdp_info = {
                'lagrange_multiplier': 0.0,
                'cmdp_cost_ema': ep_cmdp_discounted_cost,
                'cmdp_cost_gap': ep_cmdp_discounted_cost - cmdp_limit,
                'lagrange_loss': 0.0,
            }

        logger.log_episode(episode, {
            'reward': ep_reward, 'energy': ep_energy,
            'uav_energy': ep_energy, 'user_energy': ep_user_energy,
            'system_energy': ep_system_energy,
            'mean_task_delay_s': (
                float(np.mean(ep_completed_task_delays))
                if ep_completed_task_delays else np.nan
            ),
            'p95_task_delay_s': (
                float(np.percentile(ep_completed_task_delays, 95))
                if ep_completed_task_delays else np.nan
            ),
            'deadline_missed_tasks': ep_deadline_missed_tasks,
            'deadline_evaluable_tasks': ep_deadline_evaluable_tasks,
            'task_deadline_violation_rate': (
                ep_deadline_missed_tasks / ep_deadline_evaluable_tasks
                if ep_deadline_evaluable_tasks > 0 else 0.0
            ),
            'deadline_overshoot_mean_s': (
                ep_deadline_overshoot_s_sum / ep_deadline_missed_tasks
                if ep_deadline_missed_tasks > 0 else 0.0
            ),
            'normalized_deadline_overshoot_mean': (
                ep_deadline_overshoot_normalized_sum
                / ep_deadline_missed_tasks
                if ep_deadline_missed_tasks > 0 else 0.0
            ),
            'tasks_completed': ep_tasks,
            'battery_depleted': ep_battery_depleted,
            'energy_violation': ep_energy_violation,
            'cmdp_discounted_cost': ep_cmdp_discounted_cost,
            'cmdp_stage_cost_mean': ep_cmdp_stage_cost_sum / (step + 1),
            'cmdp_pre_projection_state_cost_mean': (
                ep_cmdp_pre_projection_state_cost_sum
                / ep_cmdp_projection_state_cost_samples
                if ep_cmdp_projection_state_cost_samples > 0 else np.nan
            ),
            'cmdp_post_projection_state_cost_mean': (
                ep_cmdp_post_projection_state_cost_sum
                / ep_cmdp_projection_state_cost_samples
                if ep_cmdp_projection_state_cost_samples > 0 else np.nan
            ),
            'cmdp_projection_state_cost_delta_mean': (
                ep_cmdp_projection_state_cost_delta_sum
                / ep_cmdp_projection_state_cost_samples
                if ep_cmdp_projection_state_cost_samples > 0 else np.nan
            ),
            'cmdp_projection_state_cost_samples': (
                ep_cmdp_projection_state_cost_samples
            ),
            'cmdp_pre_projection_critic_risk_mean': (
                float(np.mean(ep_cmdp_pre_projection_critic_risks))
                if ep_cmdp_pre_projection_critic_risks else np.nan
            ),
            'cmdp_post_projection_critic_risk_mean': (
                float(np.mean(ep_cmdp_post_projection_critic_risks))
                if ep_cmdp_post_projection_critic_risks else np.nan
            ),
            'cmdp_projection_critic_risk_delta_mean': (
                float(np.mean(ep_cmdp_projection_critic_risk_deltas))
                if ep_cmdp_projection_critic_risk_deltas else np.nan
            ),
            'cmdp_cost_activation_rate': (
                ep_cmdp_cost_active_steps / (step + 1)
            ),
            'cmdp_cost_saturation_rate': (
                ep_cmdp_cost_saturated_steps / (step + 1)
            ),
            'cmdp_cost_limit': cmdp_limit,
            'cost_critic_loss': (
                float(np.mean(ep_update_diagnostics['cost_critic_loss']))
                if ep_update_diagnostics['cost_critic_loss'] else np.nan
            ),
            'cmdp_actor_penalty_active_rate': (
                float(np.mean(ep_update_diagnostics[
                    'cmdp_actor_penalty_active'
                ]))
                if ep_update_diagnostics['cmdp_actor_penalty_active']
                else 0.0
            ),
            'cmdp_actor_penalty_scale_mean': (
                float(np.mean(ep_update_diagnostics[
                    'cmdp_actor_penalty_scale'
                ]))
                if ep_update_diagnostics['cmdp_actor_penalty_scale']
                else 0.0
            ),
            'policy_cost_q_mean': (
                float(np.mean(ep_update_diagnostics['policy_cost_q_mean']))
                if ep_update_diagnostics['policy_cost_q_mean'] else np.nan
            ),
            'policy_cost_q_std': (
                float(np.mean(ep_update_diagnostics['policy_cost_q_std']))
                if ep_update_diagnostics['policy_cost_q_std'] else np.nan
            ),
            'replay_cost_q_mean': (
                float(np.mean(ep_update_diagnostics['replay_cost_q_mean']))
                if ep_update_diagnostics['replay_cost_q_mean'] else np.nan
            ),
            'cmdp_pre_projection_q_mean': (
                float(np.mean(ep_update_diagnostics[
                    'cmdp_pre_projection_q_mean'
                ]))
                if ep_update_diagnostics['cmdp_pre_projection_q_mean']
                else np.nan
            ),
            'cmdp_post_projection_q_mean': (
                float(np.mean(ep_update_diagnostics[
                    'cmdp_post_projection_q_mean'
                ]))
                if ep_update_diagnostics['cmdp_post_projection_q_mean']
                else np.nan
            ),
            'constraint_term_mean': (
                float(np.mean(ep_update_diagnostics['constraint_term_mean']))
                if ep_update_diagnostics['constraint_term_mean'] else 0.0
            ),
            'actor_reward_gradient_norm': (
                float(np.mean(ep_update_diagnostics[
                    'actor_reward_gradient_norm'
                ]))
                if ep_update_diagnostics['actor_reward_gradient_norm']
                else np.nan
            ),
            'actor_cost_gradient_norm': (
                float(np.mean(ep_update_diagnostics[
                    'actor_cost_gradient_norm'
                ]))
                if ep_update_diagnostics['actor_cost_gradient_norm']
                else np.nan
            ),
            'actor_cost_to_reward_gradient_ratio': (
                float(np.mean(ep_update_diagnostics[
                    'actor_cost_to_reward_gradient_ratio'
                ]))
                if ep_update_diagnostics[
                    'actor_cost_to_reward_gradient_ratio'
                ] else np.nan
            ),
            'cost_critic_update_count': (
                ep_update_diagnostics['cost_critic_update_count'][-1]
                if ep_update_diagnostics['cost_critic_update_count'] else 0.0
            ),
            'cmdp_positive_cost_samples': (
                ep_update_diagnostics['cmdp_positive_cost_samples'][-1]
                if ep_update_diagnostics['cmdp_positive_cost_samples'] else 0.0
            ),
            'cmdp_budget_exceeded': bool(
                ep_cmdp_discounted_cost > cmdp_limit
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
            'safety_override_rate': ep_safety_override_steps / (step + 1),
            'scheduled_return_override_steps': (
                ep_scheduled_return_override_steps
            ),
            'scheduled_return_override_rate': (
                ep_scheduled_return_override_steps / (step + 1)
            ),
            **cmdp_info,
            'return_triggered': ep_return_triggered,
            'returned_safe': successful_safety_return(
                ep_return_triggered, ep_returned
            ),
            'returned_safe_episode': ep_returned,
            'mission_success': ep_mission_success,
            'mission_safe_abort': bool(info.get('mission_safe_abort', False)),
            'mission_deadline_missed': ep_mission_deadline_missed,
            'mission_outcome': info.get('mission_outcome', 'disabled'),
            'mission_failure_reasons': ';'.join(
                info.get('mission_failure_reasons', ())
            ),
            'tasks_generated': int(info.get('episode_tasks_generated', 0)),
            'tasks_failed': int(info.get('episode_tasks_failed', 0)),
            'core_tasks_completed': int(
                info.get('episode_core_tasks_completed', 0)
            ),
            'core_tasks_failed': int(
                info.get('episode_core_tasks_failed', 0)
            ),
            'core_tasks_generated': int(
                info.get('episode_core_tasks_generated', 0)
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
            **ep_reward_components,
        })

        if verbose and (episode + 1) % print_interval == 0:
            avg_r = np.mean(logger.data['reward'][-print_interval:])
            window = slice(-print_interval, None)
            energy_rate = np.mean(logger.data['energy_violation'][window])
            constraint_rate = np.mean(
                logger.data['physical_constraint_violation'][window]
            )
            cmdp_budget_rate = np.mean(
                logger.data['cmdp_budget_exceeded'][window]
            )
            cmdp_activation_rate = np.mean(
                logger.data['cmdp_cost_activation_rate'][window]
            )
            cmdp_stage_cost = np.mean(
                logger.data['cmdp_stage_cost_mean'][window]
            )
            cost_critic_values = np.asarray(
                logger.data['cost_critic_loss'][window], dtype=float
            )
            finite_cost_critic_values = cost_critic_values[
                np.isfinite(cost_critic_values)
            ]
            cost_critic_loss = (
                float(np.mean(finite_cost_critic_values))
                if finite_cost_critic_values.size else float('nan')
            )
            lagrange_multiplier = np.mean(
                logger.data['lagrange_multiplier'][window]
            )
            intervention_rate = np.mean(
                logger.data['safety_override_rate'][window]
            )
            mean_tasks = np.mean(logger.data['tasks_completed'][window])
            mean_system_energy = np.mean(
                logger.data['system_energy'][window]
            )
            safe_return_rate = np.mean(
                logger.data['returned_safe_episode'][window]
            )
            mission_rate = np.mean(logger.data['mission_success'][window])
            task_deadline_violation_rate = np.mean(
                logger.data['task_deadline_violation_rate'][window]
            )
            print(
                f'[{algo_name}] Ep {episode+1}/{num_episodes}: '
                f'avg_reward={avg_r:.2f}, '
                f'tasks={mean_tasks:.1f}, '
                f'system_energy={mean_system_energy:.1f}J, '
                f'energy_violation_rate={energy_rate*100:.1f}%, '
                f'constraint_exceedance_rate={constraint_rate*100:.1f}%, '
                f'cmdp_activation_rate={cmdp_activation_rate*100:.1f}%, '
                f'cmdp_stage_cost={cmdp_stage_cost:.3f}, '
                f'task_deadline_violation_rate='
                f'{task_deadline_violation_rate*100:.1f}%, '
                f'cost_q_loss={cost_critic_loss:.4f}, '
                f'lambda={lagrange_multiplier:.4f}, '
                f'intervention_rate={intervention_rate*100:.1f}%, '
                f'safe_return_rate={safe_return_rate*100:.1f}%, '
                f'mission_success_rate={mission_rate*100:.1f}%'
            )
            # Canonical 0..1 ratios for automated research pipelines.
            print(f'METRIC energy_violation_rate={energy_rate:.6f}')
            print(f'METRIC constraint_exceedance_rate={constraint_rate:.6f}')
            print(
                'METRIC physical_constraint_violation_rate='
                f'{constraint_rate:.6f}'
            )
            print(
                'METRIC cmdp_budget_exceedance_rate='
                f'{cmdp_budget_rate:.6f}'
            )
            print(
                'METRIC cmdp_cost_activation_rate='
                f'{cmdp_activation_rate:.6f}'
            )
            print(f'METRIC cmdp_stage_cost_mean={cmdp_stage_cost:.6f}')
            print(
                'METRIC task_deadline_violation_rate='
                f'{task_deadline_violation_rate:.6f}'
            )
            print(f'METRIC cost_critic_loss={cost_critic_loss:.6f}')
            print(f'METRIC lagrange_multiplier={lagrange_multiplier:.6f}')
            print(f'METRIC safety_override_rate={intervention_rate:.6f}')
            print(f'METRIC safe_return_rate={safe_return_rate:.6f}')
            print(f'METRIC mission_success_rate={mission_rate:.6f}')

    # 保存模型
    if save_path is not None:
        if hasattr(agent, 'save'):
            save_checkpoint_pair(
                agent,
                lstm_predictor if use_lstm else None,
                save_path,
                algorithm=algo_name,
                metadata={
                    **checkpoint_metadata(
                        agent_settings, state_dim, action_dim
                    ),
                    'episodes': max(1, num_episodes),
                    'training_seed': train_seed,
                },
            )
        elif verbose:
            print(f'[{algo_name}] is a fixed policy; no checkpoint was saved')

    # 保存日志
    log_dir = output_dir or f'results/{algo_name}'
    os.makedirs(log_dir, exist_ok=True)
    logger.save(f'{log_dir}/log.csv')
    logger.save_actions(f'{log_dir}/actions.csv')
    logger.plot(f'{log_dir}/curves.png')

    return logger, agent


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train baseline')
    parser.add_argument('--algo', type=str, default='Random',
                        choices=list(ALGO_MAP.keys()))
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    save_path = args.save or f'models/{args.algo}_final.pt'
    logger, agent = train_baseline(
        args.algo,
        config_path=args.config,
        num_episodes=args.episodes,
        save_path=save_path,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(f'{args.algo} training done!')
