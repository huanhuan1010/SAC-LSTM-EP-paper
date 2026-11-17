"""Shared causal feature construction for LSTM energy prediction."""
from __future__ import annotations

import numpy as np


LEGACY_SCHEMA = 'legacy_v1'
PHYSICAL_CONTROL_SCHEMA = 'physical_control_v1'

_STATE_FEATURE_NAMES = {
    LEGACY_SCHEMA: (
        'remaining_energy_fraction',
        'ground_velocity_x_over_vmax',
        'ground_velocity_y_over_vmax',
        'ground_velocity_z_over_vmax',
        'altitude_over_hmax',
    ),
    PHYSICAL_CONTROL_SCHEMA: (
        'remaining_energy_fraction',
        'ground_velocity_x_over_vmax',
        'ground_velocity_y_over_vmax',
        'ground_velocity_z_over_vmax',
        'altitude_over_hmax',
        'air_velocity_x_over_vmax',
        'air_velocity_y_over_vmax',
        'air_velocity_z_over_vmax',
        'wind_velocity_x_over_windmax',
        'wind_velocity_y_over_windmax',
        'mean_applied_offload_fraction',
        'applied_speed_fraction',
        'applied_uav_cpu_fraction',
        'applied_tx_power_fraction',
    ),
}


def normalize_lstm_feature_schema(schema):
    """Return a validated canonical feature-schema name."""
    value = str(schema or LEGACY_SCHEMA).strip().lower()
    if value not in _STATE_FEATURE_NAMES:
        raise ValueError(
            f'unsupported lstm_feature_schema: {value}; expected one of '
            f'{sorted(_STATE_FEATURE_NAMES)}'
        )
    return value


def lstm_feature_names(schema):
    """Names in the exact tensor-column order, including step energy."""
    value = normalize_lstm_feature_schema(schema)
    return ('step_energy_fraction', *_STATE_FEATURE_NAMES[value])


def lstm_feature_input_dim(schema):
    return len(lstm_feature_names(schema))


def lstm_feature_spec(config):
    """Resolve and validate the feature schema declared by a config."""
    schema = normalize_lstm_feature_schema(
        config.get('lstm_feature_schema', LEGACY_SCHEMA)
    )
    input_dim = lstm_feature_input_dim(schema)
    configured_dim = config.get('lstm_input_dim')
    if configured_dim is not None and int(configured_dim) != input_dim:
        raise ValueError(
            f'lstm_input_dim={configured_dim} conflicts with '
            f'lstm_feature_schema={schema} (expected {input_dim})'
        )
    return {
        'schema': schema,
        'input_dim': input_dim,
        'feature_names': list(lstm_feature_names(schema)),
    }


def lstm_feature_kwargs(config):
    """Constructor arguments shared by every LSTM creation entry point."""
    spec = lstm_feature_spec(config)
    gate_warmup = int(config.get('lstm_history_gate_warmup_steps', 0))
    if gate_warmup < 0:
        raise ValueError('lstm_history_gate_warmup_steps must be nonnegative')
    return {
        'input_dim': spec['input_dim'],
        'feature_schema': spec['schema'],
        'history_gate_warmup_steps': gate_warmup,
    }


def lstm_history_gate_weight(observation_count, lookback, warmup_steps):
    """Causal confidence weight for the learned correction over persistence.

    The first prediction becomes available after ``lookback`` observations.
    With a positive warm-up, its learned correction starts at zero and grows
    linearly to full strength after ``warmup_steps`` additional observations.
    """
    lookback = int(lookback)
    warmup_steps = int(warmup_steps)
    if lookback <= 0:
        raise ValueError('lookback must be positive')
    if warmup_steps < 0:
        raise ValueError('warmup_steps must be nonnegative')
    count = np.asarray(observation_count, dtype=np.float64)
    if np.any(~np.isfinite(count)) or np.any(count < 0.0):
        raise ValueError('observation_count must be finite and nonnegative')
    if warmup_steps == 0:
        weight = np.ones_like(count, dtype=np.float64)
    else:
        weight = np.clip(
            (count - float(lookback)) / float(warmup_steps), 0.0, 1.0
        )
    if weight.ndim == 0:
        return float(weight)
    return weight


def _transition_vector(info, key, fallback, size):
    value = np.asarray(info.get(key, fallback), dtype=np.float64).reshape(-1)
    if value.size < size:
        raise ValueError(f'{key} must contain at least {size} values')
    value = value[:size]
    if not np.all(np.isfinite(value)):
        raise ValueError(f'{key} contains non-finite values')
    return value


def lstm_state_features(env, info, schema=LEGACY_SCHEMA):
    """Build post-step causal state/control features, excluding step energy.

    Both runtime history and offline validation call this function after
    ``env.step``. For the physical/control schema, controls come from
    ``info['applied_action']`` so safety projection and scheduled return are
    represented by what the UAV actually executed, not by the actor proposal.
    """
    value = normalize_lstm_feature_schema(schema)
    info = {} if info is None else info
    capacity = max(float(env.energy.battery_capacity), 1e-12)
    v_max = max(float(env.uav.v_max), 1e-12)
    h_max = max(float(env.uav.h_max), 1e-12)
    ground_velocity = _transition_vector(
        info, 'ground_velocity', env.uav.vel, 3
    )
    position = _transition_vector(info, 'pos', env.uav.pos, 3)
    base = [
        float(info.get('E_rem', env.E_rem)) / capacity,
        *(ground_velocity / v_max).tolist(),
        float(position[2]) / h_max,
    ]
    if value == LEGACY_SCHEMA:
        return np.asarray(base, dtype=np.float32)

    if 'applied_action' not in info:
        raise ValueError(
            'physical_control_v1 requires post-step info["applied_action"]'
        )
    air_velocity = _transition_vector(
        info, 'air_velocity', env.uav.air_vel, 3
    )
    wind_velocity = _transition_vector(
        info, 'wind_velocity', env.wind.velocity, 3
    )
    applied_action = np.asarray(
        info['applied_action'], dtype=np.float64
    ).reshape(-1)
    expected_action_dim = int(env.num_users) + 5
    if applied_action.size != expected_action_dim:
        raise ValueError(
            f'applied_action has {applied_action.size} values; expected '
            f'{expected_action_dim}'
        )
    if not np.all(np.isfinite(applied_action)):
        raise ValueError('applied_action contains non-finite values')

    controls = np.clip((applied_action + 1.0) / 2.0, 0.0, 1.0)
    n = int(env.num_users)
    wind_max = max(float(env.config.get('wind_max_speed', 8.0)), 1e-12)
    extended = [
        *base,
        *(air_velocity / v_max).tolist(),
        *(wind_velocity[:2] / wind_max).tolist(),
        float(np.mean(controls[:n])) if n else 0.0,
        float(controls[n + 2]),
        float(controls[n + 3]),
        float(controls[n + 4]),
    ]
    result = np.asarray(extended, dtype=np.float32)
    expected_state_dim = lstm_feature_input_dim(value) - 1
    if result.size != expected_state_dim:
        raise RuntimeError(
            f'internal feature-size error: {result.size} != '
            f'{expected_state_dim}'
        )
    return result
