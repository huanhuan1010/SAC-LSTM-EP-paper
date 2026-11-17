"""实验环境指纹与可追溯清单。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


ENVIRONMENT_SCHEMA_VERSION = 16

SOURCE_ROOTS = ('agent', 'env', 'baselines', 'utils')
SOURCE_FILES = (
    'experiments/compare.py',
    'scripts/run_csac_alvi_smoke.py',
    'configs/default.yaml',
)

# 仅包含会改变MDP、能耗、通信、奖励或安全约束的参数。训练轮数、随机种子、
# 学习率等运行参数不属于环境，因此不会导致同一环境产生不同指纹。
ENVIRONMENT_CONFIG_KEYS = (
    'num_users', 'area_size', 'dt', 'max_steps', 'task_interval',
    'policy_action_mode', 'common_mobility_speed',
    'common_mobility_heading_deg', 'common_mobility_pitch_deg',
    'mission_return_enabled', 'mission_service_steps',
    'mission_min_completed_tasks', 'mission_min_completion_ratio',
    'mission_min_core_completion_ratio',
    'mission_require_full_service_window',
    'mission_require_energy_feasible',
    'lookback', 'horizon', 'channel_db_normalizer',
    'queue_length_normalizer',
    'D_mean', 'D_std', 'C_mean', 'C_std', 'tau_mean', 'tau_std',
    'core_task_ratio',
    'v_max', 'a_max', 'h_min', 'h_max', 'h_default', 'base_pos',
    'wind_enabled', 'wind_mean_speed', 'wind_speed_std',
    'wind_temporal_correlation', 'wind_max_speed',
    'wind_direction_deg', 'wind_seed_offset',
    'energy_uncertainty_enabled', 'energy_uncertainty_log_std',
    'energy_uncertainty_temporal_correlation',
    'energy_uncertainty_min_scale', 'energy_uncertainty_max_scale',
    'energy_uncertainty_seed_offset',
    'battery_capacity', 'initial_energy_min', 'initial_energy_max',
    'P0', 'P1', 'v_tip', 'U_tip', 'v0', 'd0', 'rho', 's', 'A',
    'kappa', 'kappa_user', 'f0', 'f_user', 'E_min_ratio',
    'soc_return_threshold', 'predictive_return_threshold',
    'single_level_margin_threshold',
    'safety_reserve_ratio', 'safety_power_gain', 'safety_power_scale_max',
    'prediction_uncertainty_gain', 'return_speed', 'return_descent_speed',
    'vertical_power_model',
    'cmdp_cost_mode', 'cmdp_margin_target', 'cmdp_cost_scale',
    'cmdp_energy_cost_source', 'cmdp_energy_cost_max',
    'cmdp_terminal_cost', 'cmdp_latency_cost_scale',
    'cmdp_latency_miss_weight', 'cmdp_latency_overshoot_weight',
    # Legacy aliases remain fingerprinted for old custom configuration files.
    'constraint_margin_target', 'constraint_cost_scale',
    'constraint_terminal_cost',
    'zone_threshold_high', 'zone_threshold_low',
    'energy_penalty_scale', 'energy_normalizer', 'delay_penalty_scale',
    'reward_energy_weight', 'reward_delay_weight',
    'reward_user_energy_weight',
    'bandwidth', 'beta0', 'alpha', 'noise_power', 'f_c',
    'pa_efficiency', 'p_circuit', 'p_rx_uav', 'p_tx_min', 'p_tx_max',
    'threshold_yellow', 'threshold_orange', 'threshold_red',
    'safety_projection_policy', 'safety_min_energy_saving',
    'safety_red_requires_infeasibility',
    'yellow_lambda_max', 'yellow_speed_min', 'yellow_speed_max',
    'yellow_cpu_max', 'yellow_p_tx_max', 'yellow_pause_noncore_uav',
    'orange_lambda_max', 'orange_speed_min', 'orange_speed_max',
    'orange_cpu_max', 'orange_p_tx_max', 'orange_pause_noncore_uav',
    'red_lambda_max', 'red_speed_min', 'red_speed_max',
    'red_cpu_max', 'red_p_tx_max', 'red_pause_noncore_uav',
)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def environment_payload(config):
    """返回参与环境指纹计算的规范化配置。"""
    return {
        'environment_schema_version': ENVIRONMENT_SCHEMA_VERSION,
        'parameters': {
            key: _jsonable(config[key])
            for key in ENVIRONMENT_CONFIG_KEYS
            if key in config
        },
    }


def environment_fingerprint(config):
    """返回当前环境配置的完整SHA-256指纹。"""
    encoded = json.dumps(
        environment_payload(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def source_fingerprint(project_root=None):
    """Hash runtime source files that define training and evaluation behavior."""
    root = Path(project_root or Path(__file__).resolve().parents[1])
    paths = []
    for directory in SOURCE_ROOTS:
        paths.extend((root / directory).rglob('*.py'))
    paths.extend(root / relative for relative in SOURCE_FILES)

    digest = hashlib.sha256()
    for path in sorted(set(paths), key=lambda item: item.as_posix()):
        if not path.is_file():
            raise FileNotFoundError(f'provenance source file not found: {path}')
        relative = path.relative_to(root).as_posix().encode('utf-8')
        digest.update(len(relative).to_bytes(4, 'big'))
        digest.update(relative)
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


def experiment_manifest(config, **metadata):
    """构造带环境指纹的实验清单。"""
    return {
        'manifest_schema_version': 1,
        'environment_fingerprint': environment_fingerprint(config),
        'source_fingerprint': source_fingerprint(),
        'environment': environment_payload(config),
        'config': _jsonable(dict(config)),
        **_jsonable(metadata),
    }


def write_json_atomic(path, payload):
    """原子写入JSON，避免中断后留下半个结果文件。"""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        # Evaluation metrics frequently contain NumPy scalar values.  Convert
        # the complete payload at the serialization boundary so every result
        # writer follows the same JSON-safe convention.
        json.dump(
            _jsonable(payload), stream,
            ensure_ascii=False, indent=2, sort_keys=True,
        )
    os.replace(temporary, destination)
    return str(destination)
