"""策略与LSTM预测器的成对检查点管理。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from utils.provenance import (
    ENVIRONMENT_SCHEMA_VERSION,
    environment_fingerprint,
    source_fingerprint,
)
from agent.lstm_features import lstm_feature_spec


PAIR_SCHEMA_VERSION = 1
def checkpoint_metadata(config, state_dim, action_dim):
    """生成训练与评估必须一致的环境/LSTM结构签名。"""
    feature_spec = lstm_feature_spec(config)
    return {
        'environment_schema_version': ENVIRONMENT_SCHEMA_VERSION,
        'environment_fingerprint': environment_fingerprint(config),
        'source_fingerprint': source_fingerprint(),
        'state_dim': int(state_dim),
        'action_dim': int(action_dim),
        'policy_action_mode': str(config.get(
            'policy_action_mode', 'joint_control'
        )),
        'prediction_source': str(config.get('prediction_source', 'none')),
        'num_users': int(config.get('num_users', 5)),
        'dt': float(config.get('dt', 0.5)),
        'lookback': int(config.get('lookback', 10)),
        'horizon': int(config.get('horizon', 5)),
        'lstm_hidden_dim': int(config.get('lstm_hidden_dim', 64)),
        'lstm_num_layers': int(config.get('lstm_num_layers', 2)),
        'lstm_prediction_mode': str(config.get(
            'lstm_prediction_mode', 'absolute'
        )),
        'lstm_feature_schema': feature_spec['schema'],
        'lstm_input_dim': feature_spec['input_dim'],
        'lstm_feature_names': feature_spec['feature_names'],
        'lstm_history_gate_warmup_steps': int(config.get(
            'lstm_history_gate_warmup_steps', 0
        )),
        'battery_capacity': float(config.get('battery_capacity', 100000.0)),
        'cmdp_enabled': bool(config.get('cmdp_enabled', True)),
        'cmdp_cost_mode': str(config.get(
            'cmdp_cost_mode', 'energy_margin'
        )),
        'cmdp_margin_target': float(config.get('cmdp_margin_target', 0.20)),
        'cmdp_cost_scale': float(config.get('cmdp_cost_scale', 0.20)),
        'cmdp_energy_cost_source': str(config.get(
            'cmdp_energy_cost_source', 'post_transition_state'
        )),
        'cmdp_energy_cost_max': float(config.get(
            'cmdp_energy_cost_max', 1.0
        )),
        'cmdp_terminal_cost': float(config.get('cmdp_terminal_cost', 1.0)),
        'cmdp_latency_cost_scale': float(config.get(
            'cmdp_latency_cost_scale', 0.05
        )),
        'cmdp_latency_miss_weight': float(config.get(
            'cmdp_latency_miss_weight', 1.0
        )),
        'cmdp_latency_overshoot_weight': float(config.get(
            'cmdp_latency_overshoot_weight', 1.0
        )),
        'cmdp_cost_limit': float(config.get(
            'cmdp_cost_limit', config.get('constraint_cost_limit', 5.0)
        )),
        'lagrange_init': float(config.get('lagrange_init', 0.1)),
        'lagrange_lr': float(config.get('lagrange_lr', 1e-4)),
        'lagrange_max': float(config.get('lagrange_max', 100.0)),
        'lagrange_cost_ema_alpha': float(
            config.get('lagrange_cost_ema_alpha', 0.10)
        ),
    }


def paired_lstm_checkpoint(policy_checkpoint):
    """返回与策略检查点相邻且同名的LSTM检查点路径。"""
    policy = Path(policy_checkpoint)
    suffix = policy.suffix or '.pt'
    return policy.with_name(f'{policy.stem}.lstm{suffix}')


def pair_manifest_path(policy_checkpoint):
    """返回策略—LSTM配对清单路径。"""
    policy = Path(policy_checkpoint)
    return policy.with_name(f'{policy.stem}.pair.json')


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _require_file(path, label):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'{label} not found: {path}')
    return path


def save_checkpoint_pair(agent, lstm_predictor, policy_checkpoint, algorithm,
                         metadata=None):
    """保存策略、可选LSTM及带SHA-256校验的配对清单。"""
    policy = Path(policy_checkpoint)
    policy.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(agent, 'save'):
        raise TypeError(f'{algorithm} agent does not support checkpoint saving')

    agent.save(str(policy))
    _require_file(policy, 'policy checkpoint')

    uses_lstm = lstm_predictor is not None
    lstm_path = paired_lstm_checkpoint(policy) if uses_lstm else None
    if uses_lstm:
        lstm_predictor.save(str(lstm_path))
        _require_file(lstm_path, 'LSTM checkpoint')

    manifest = {
        'schema_version': PAIR_SCHEMA_VERSION,
        'algorithm': str(algorithm),
        'uses_lstm': uses_lstm,
        'policy_checkpoint': policy.name,
        'policy_sha256': _sha256(policy),
        'lstm_checkpoint': lstm_path.name if lstm_path else None,
        'lstm_sha256': _sha256(lstm_path) if lstm_path else None,
        'metadata': dict(metadata or {}),
    }
    manifest_path = pair_manifest_path(policy)
    temporary_path = manifest_path.with_suffix(manifest_path.suffix + '.tmp')
    with temporary_path.open('w', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary_path, manifest_path)
    return {
        'policy': str(policy),
        'lstm': str(lstm_path) if lstm_path else None,
        'manifest': str(manifest_path),
    }


def load_checkpoint_pair(agent, lstm_predictor, policy_checkpoint, *,
                         lstm_checkpoint=None, expected_algorithm=None,
                         expected_metadata=None, allow_legacy_explicit=False,
                         allowed_source_fingerprints=None):
    """校验配对清单后加载策略与LSTM，禁止随机LSTM静默参与评估。

    旧检查点只有在调用者显式给出 ``lstm_checkpoint`` 且设置
    ``allow_legacy_explicit=True`` 时才能加载；此时无法验证训练配对关系。
    """
    policy = _require_file(policy_checkpoint, 'policy checkpoint')
    manifest_path = pair_manifest_path(policy)
    explicit_lstm = Path(lstm_checkpoint) if lstm_checkpoint else None
    allowed_sources = {
        str(value) for value in (allowed_source_fingerprints or ())
    }
    metadata_waivers = []

    if manifest_path.is_file():
        with manifest_path.open('r', encoding='utf-8') as stream:
            manifest = json.load(stream)
        if manifest.get('schema_version') != PAIR_SCHEMA_VERSION:
            raise ValueError(
                f'unsupported checkpoint pair schema: {manifest.get("schema_version")}'
            )
        if expected_algorithm and manifest.get('algorithm') != expected_algorithm:
            raise ValueError(
                'checkpoint algorithm mismatch: '
                f'expected {expected_algorithm}, got {manifest.get("algorithm")}'
            )
        saved_metadata = manifest.get('metadata', {})
        for key, expected_value in dict(expected_metadata or {}).items():
            if saved_metadata.get(key) != expected_value:
                saved_value = saved_metadata.get(key)
                if (
                    key == 'source_fingerprint'
                    and str(saved_value) in allowed_sources
                ):
                    metadata_waivers.append({
                        'key': key,
                        'expected': expected_value,
                        'saved': saved_value,
                        'reason': (
                            'explicitly preregistered historical checkpoint '
                            'for development-only evaluation'
                        ),
                    })
                    continue
                raise ValueError(
                    f'checkpoint metadata mismatch for {key}: '
                    f'expected {expected_value}, got {saved_value}'
                )
        if manifest.get('policy_checkpoint') != policy.name:
            raise ValueError('policy filename does not match pair manifest')
        if _sha256(policy) != manifest.get('policy_sha256'):
            raise ValueError('policy checkpoint hash does not match pair manifest')

        manifest_uses_lstm = bool(manifest.get('uses_lstm'))
        if manifest_uses_lstm != (lstm_predictor is not None):
            raise ValueError(
                'LSTM usage mismatch between requested algorithm and pair manifest'
            )

        resolved_lstm = None
        if manifest_uses_lstm:
            manifest_lstm = policy.parent / manifest['lstm_checkpoint']
            if explicit_lstm is not None:
                if explicit_lstm.absolute() != manifest_lstm.absolute():
                    raise ValueError(
                        'explicit LSTM checkpoint does not match pair manifest'
                    )
                resolved_lstm = explicit_lstm
            else:
                resolved_lstm = manifest_lstm
            _require_file(resolved_lstm, 'LSTM checkpoint')
            if _sha256(resolved_lstm) != manifest.get('lstm_sha256'):
                raise ValueError('LSTM checkpoint hash does not match pair manifest')
    else:
        if lstm_predictor is not None:
            if explicit_lstm is None or not allow_legacy_explicit:
                raise FileNotFoundError(
                    f'checkpoint pair manifest not found: {manifest_path}. '
                    'Retrain with the paired checkpoint workflow, or explicitly '
                    'supply a legacy LSTM checkpoint.'
                )
            resolved_lstm = _require_file(
                explicit_lstm, 'explicit legacy LSTM checkpoint'
            )
        else:
            resolved_lstm = None

    if not hasattr(agent, 'load'):
        raise TypeError('agent does not support checkpoint loading')
    agent.load(str(policy))
    if lstm_predictor is not None:
        lstm_predictor.load(str(resolved_lstm))
        lstm_predictor.eval()

    return {
        'policy': str(policy),
        'lstm': str(resolved_lstm) if resolved_lstm else None,
        'manifest': str(manifest_path) if manifest_path.is_file() else None,
        'metadata_waivers': metadata_waivers,
    }
