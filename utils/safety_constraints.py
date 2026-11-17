"""环境与策略共享的分级安全约束。"""
import numpy as np


MODE_SEVERITY = {'normal': 0, 'yellow': 1, 'orange': 2, 'red': 3}


def validate_safety_thresholds(config):
    """Validate and return the ordered graded energy-margin thresholds."""
    thresholds = {
        'yellow': float(config.get('threshold_yellow', 0.40)),
        'orange': float(config.get('threshold_orange', 0.30)),
        'red': float(config.get('threshold_red', 0.20)),
    }
    for name, value in thresholds.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f'threshold_{name} must lie in [0, 1]')
    if not (
        thresholds['red'] < thresholds['orange'] < thresholds['yellow']
    ):
        raise ValueError(
            'safety thresholds must satisfy red < orange < yellow'
        )
    return thresholds


def mode_from_margin_ratio(margin_ratio, config):
    """按能量裕度比例返回权威安全模式。"""
    ratio = float(margin_ratio)
    if ratio < float(config.get('threshold_red', 0.20)):
        return 'red'
    if ratio < float(config.get('threshold_orange', 0.30)):
        return 'orange'
    if ratio < float(config.get('threshold_yellow', 0.40)):
        return 'yellow'
    return 'normal'


def stricter_mode(first, second):
    """返回两个模式中更保守的一档。"""
    first = first if first in MODE_SEVERITY else 'normal'
    second = second if second in MODE_SEVERITY else 'normal'
    return first if MODE_SEVERITY[first] >= MODE_SEVERITY[second] else second


def safety_constraints(config, mode):
    """读取给定安全模式的归一化动作边界。"""
    mode = mode if mode in MODE_SEVERITY else 'normal'
    defaults = {
        'normal': (1.0, 0.0, 1.0, 1.0, 1.0, False, 'normal'),
        'yellow': (0.7, 0.0, 0.7, 0.8, 0.8, False, 'normal'),
        # Orange tightens resource and motion bounds but keeps task service
        # active.  Pausing all non-core work here previously made the shield
        # dominate the policy and drove intervention rates towards 100%.
        'orange': (0.5, 0.0, 0.6, 0.5, 0.5, False, 'prefer_base'),
        # 红色模式使用固定返航速度0.5*v_max；默认配置下等于10m/s，
        # 与返航能耗估计的return_speed保持一致。
        'red': (0.2, 0.5, 0.5, 0.25, 0.25, True, 'to_base'),
    }
    (lambda_default, speed_min_default, speed_max_default, cpu_default,
     p_tx_default, pause_default, heading_mode) = defaults[mode]

    constraints = {
        'lambda_max': float(config.get(f'{mode}_lambda_max', lambda_default)),
        'speed_min': float(config.get(f'{mode}_speed_min', speed_min_default)),
        'speed_max': float(config.get(f'{mode}_speed_max', speed_max_default)),
        'cpu_max': float(config.get(f'{mode}_cpu_max', cpu_default)),
        'p_tx_max': float(config.get(f'{mode}_p_tx_max', p_tx_default)),
        'pause_noncore_uav': bool(
            config.get(f'{mode}_pause_noncore_uav', pause_default)
        ),
        'heading_mode': heading_mode,
    }
    for key in ('lambda_max', 'speed_min', 'speed_max', 'cpu_max', 'p_tx_max'):
        if not 0.0 <= constraints[key] <= 1.0:
            raise ValueError(f'{mode}_{key} must lie in [0, 1]')
    if constraints['speed_min'] > constraints['speed_max']:
        raise ValueError(f'{mode} speed_min must not exceed speed_max')
    constraints['action_mask'] = (
        'noncore_uav_pause' if constraints['pause_noncore_uav'] else None
    )
    return constraints


def project_normalized_action(action, num_users, constraints):
    """把归一化动作投影到安全集合，布局为[N个λ, θ, φ, v, f, P_tx]。"""
    projected = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    expected_dim = int(num_users) + 5
    if projected.size != expected_dim:
        raise ValueError(
            f'action dimension mismatch: expected {expected_dim}, got {projected.size}'
        )
    if not np.all(np.isfinite(projected)):
        raise ValueError('action must contain only finite values')
    projected = np.clip(projected, -1.0, 1.0)

    lambda_upper = 2.0 * constraints.get('lambda_max', 1.0) - 1.0
    projected[:num_users] = np.clip(
        projected[:num_users], -1.0, lambda_upper
    )

    speed_index = num_users + 2
    speed_lower = 2.0 * constraints.get('speed_min', 0.0) - 1.0
    speed_upper = 2.0 * constraints.get('speed_max', 1.0) - 1.0
    projected[speed_index] = np.clip(
        projected[speed_index], speed_lower, speed_upper
    )

    cpu_index = num_users + 3
    cpu_upper = 2.0 * constraints.get('cpu_max', 1.0) - 1.0
    projected[cpu_index] = np.clip(projected[cpu_index], -1.0, cpu_upper)

    p_tx_index = num_users + 4
    p_tx_upper = 2.0 * constraints.get('p_tx_max', 1.0) - 1.0
    projected[p_tx_index] = np.clip(projected[p_tx_index], -1.0, p_tx_upper)
    return projected
