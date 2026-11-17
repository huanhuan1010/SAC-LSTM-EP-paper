"""配置管理"""
import json
import os

import yaml


def load_config(path='configs/default.yaml'):
    """加载YAML配置文件"""
    default_config = {
        # === 环境参数 ===
        'num_users': 5,
        'area_size': 500.0,
        'dt': 0.5,
        'max_steps': 200,
        'policy_action_mode': 'joint_control',
        'common_mobility_speed': 0.0,
        'common_mobility_heading_deg': 0.0,
        'common_mobility_pitch_deg': 0.0,
        # Optional paper-grade two-phase mission.  Disabled in the nominal
        # environment; dedicated mission configs enable it.
        'mission_return_enabled': False,
        'mission_service_steps': 160,
        'mission_min_completed_tasks': 1,
        'mission_min_completion_ratio': 0.80,
        'mission_min_core_completion_ratio': 0.80,
        'mission_require_full_service_window': True,
        'mission_require_energy_feasible': True,
        'task_interval': 15,
        'lookback': 10,
        'horizon': 5,
        'seed': 42,
        'eval_seed': 1042,
        'channel_db_normalizer': 100.0,
        'queue_length_normalizer': 5.0,

        # === 任务参数 (与 default.yaml 保持一致) ===
        'D_mean': 1e6,
        'D_std': 0.5e6,
        'C_mean': 8e9,
        'C_std': 1e9,
        'tau_mean': 8.0,
        'tau_std': 1.0,
        'core_task_ratio': 0.3,

        # === 无人机参数 ===
        'v_max': 20.0,
        'a_max': 5.0,
        'h_min': 30.0,
        'h_max': 300.0,
        'h_default': 100.0,
        'greedy_speed': 10.0,
        # Wind is disabled in the reference environment. Dedicated pressure
        # configs enable the temporally correlated Gauss-Markov process.
        'wind_enabled': False,
        'wind_mean_speed': 3.0,
        'wind_speed_std': 1.0,
        'wind_temporal_correlation': 0.90,
        'wind_max_speed': 8.0,
        'wind_direction_deg': None,
        'wind_seed_offset': 7919,

        # Bounded AR(1) multiplicative uncertainty on actual UAV battery
        # energy. Disabled in the nominal reference environment and enabled
        # by dedicated stress configurations.
        'energy_uncertainty_enabled': False,
        'energy_uncertainty_log_std': 0.10,
        'energy_uncertainty_temporal_correlation': 0.85,
        'energy_uncertainty_min_scale': 0.75,
        'energy_uncertainty_max_scale': 1.35,
        'energy_uncertainty_seed_offset': 15401,

        # === 电池参数 (P235参考, 与 default.yaml 保持一致) ===
        'battery_capacity': 100000.0,   # J (~500s hover)
        'initial_energy_min': 0.80,
        'initial_energy_max': 1.00,
        'P0': 100.0,                 # 悬停桨叶剖面功率
        'P1': 90.0,                  # 悬停诱导功率
        'v_tip': 120.0,
        'v0': 4.03,
        'd0': 0.6,
        'rho': 1.225,
        's': 0.05,
        'A': 0.79,
        'kappa': 1e-26,              # 计算能耗系数
        'kappa_user': 1e-28,         # 用户本地计算能耗系数
        'f0': 2.0e9,                 # 地面计算频率
        'f_user': 1e9,               # 用户设备固定频率
        'E_min_ratio': 0.15,         # 最低安全能量占比
        'soc_return_threshold': 0.20,
        'predictive_return_threshold': 0.20,
        'single_level_margin_threshold': 0.40,
        'safety_reserve_ratio': 0.05,
        'safety_power_gain': 0.5,
        'safety_power_scale_max': 2.0,
        'prediction_uncertainty_gain': 1.0,
        'return_speed': 10.0,
        'return_descent_speed': 3.0,
        'vertical_power_model': 'axial_momentum',
        'zone_threshold_high': 0.60,
        'zone_threshold_low': 0.30,
        'energy_penalty_scale': 1.0,
        'energy_normalizer': 100.0,
        'delay_penalty_scale': 2.0,
        'reward_energy_weight': 0.5,
        'reward_delay_weight': 0.5,
        'reward_user_energy_weight': 1.0,

        # === 通信参数 (与 default.yaml 保持一致) ===
        'bandwidth': 1e6,
        'beta0': -20,
        'alpha': 2.5,
        'noise_power': 1e-13,
        'f_c': 2.4e9,
        'pa_efficiency': 0.3,   # A2: 功放效率
        'p_circuit': 5.0,       # A2: 电路功耗 (W)
        'p_rx_uav': 3.0,        # UAV接收机功耗 (W)
        'p_tx_min': 0.1,        # A2: 发射功率下限 (W)
        'p_tx_max': 5.0,        # A2: 发射功率上限 (W)

        # === SAC参数 ===
        'gamma': 0.99,
        'tau': 0.005,
        'lr': 1e-4,
        'warmup_steps': 10000,
        'batch_size': 512,
        'actor_lr': 1e-4,
        'actor_update_delay': 2,
        'target_entropy': None,
        'buffer_capacity': 200000,
        'replay_alpha': 0.0,
        'max_episodes': 2000,
        'update_interval': 1,           # SAC原论文: 每1步learn
        'updates_per_step': 1,
        'baseline_print_interval': 50,

        # === CMDP / Lagrangian SAC parameters ===
        'cmdp_enabled': True,
        'cmdp_cost_mode': 'energy_margin',
        'cmdp_margin_target': 0.20,
        'cmdp_cost_scale': 0.20,
        'cmdp_energy_cost_source': 'post_transition_state',
        'cmdp_energy_cost_max': 1.0,
        'cmdp_terminal_cost': 1.0,
        'cmdp_latency_cost_scale': 0.05,
        'cmdp_latency_miss_weight': 1.0,
        'cmdp_latency_overshoot_weight': 1.0,
        'cmdp_cost_limit': 5.0,
        'lagrange_init': 0.1,
        'lagrange_lr': 1e-4,
        'lagrange_max': 100.0,
        'lagrange_cost_ema_alpha': 0.10,

        # === LSTM参数 ===
        'lstm_hidden_dim': 64,
        'lstm_num_layers': 2,
        'lstm_lr': 1e-3,
        'lstm_update_interval': 10,
        'residual_correction_enabled': True,
        'residual_ema_alpha': 0.2,
        'residual_clip_ratio': 0.01,
        'oracle_trace_path': None,
        'prediction_shuffle_min_history': 2,
        'prediction_shuffle_buffer_size': 32,

        # === 基于能量裕度的分级安全返航参数 ===
        'threshold_yellow': 0.40,
        'threshold_orange': 0.30,
        'threshold_red': 0.20,
        'yellow_lambda_max': 0.70,
        'yellow_speed_min': 0.00,
        'yellow_speed_max': 0.70,
        'yellow_cpu_max': 0.80,
        'yellow_p_tx_max': 0.80,
        'yellow_pause_noncore_uav': False,
        'orange_lambda_max': 0.50,
        'orange_speed_min': 0.00,
        'orange_speed_max': 0.60,
        'orange_cpu_max': 0.50,
        'orange_p_tx_max': 0.50,
        'orange_pause_noncore_uav': False,
        'red_lambda_max': 0.20,
        'red_speed_min': 0.50,
        'red_speed_max': 0.50,
        'red_cpu_max': 0.25,
        'red_p_tx_max': 0.25,
        'red_pause_noncore_uav': True,
    }

    # 如果有配置文件，合并
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            if os.path.splitext(os.fspath(path))[1].lower() == '.json':
                file_config = json.load(f)
            else:
                file_config = yaml.safe_load(f)
        if file_config:
            if not isinstance(file_config, dict):
                raise ValueError(
                    f'configuration root must be a mapping: {path}'
                )
            file_config = dict(file_config)
            inherited_path = file_config.pop('extends', None)
            legacy_cmdp_keys = {
                'constraint_margin_target': 'cmdp_margin_target',
                'constraint_cost_scale': 'cmdp_cost_scale',
                'constraint_terminal_cost': 'cmdp_terminal_cost',
                'constraint_cost_limit': 'cmdp_cost_limit',
            }
            for legacy_key, cmdp_key in legacy_cmdp_keys.items():
                if legacy_key in file_config and cmdp_key not in file_config:
                    file_config[cmdp_key] = file_config[legacy_key]
            if inherited_path:
                if not os.path.isabs(inherited_path):
                    inherited_path = os.path.join(
                        os.path.dirname(os.path.abspath(path)), inherited_path
                    )
                inherited_config = load_config(inherited_path)
                inherited_config.update(file_config)
                return inherited_config
            default_config.update(file_config)

    return default_config
