"""Audit frozen formal-policy LSTM forecasts without retraining.

The analysis consumes ``evaluation_actions.csv`` files written by the
independent formal evaluation.  It excludes cold-start bootstrap rows and
compares the LSTM output used online with causal persistence and rolling-mean
baselines on exactly the same observations.  Overlapping forecast windows are
not treated as independent replicates: errors are first summarized by episode,
then by training seed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.provenance import write_json_atomic  # noqa: E402


MODELS = ('runtime_lstm', 'raw_lstm', 'persistence', 'rolling_mean')
SCOPES = ('step', 'cumulative')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(spec_path, value):
    path = Path(value)
    if not path.is_absolute():
        path = spec_path.parent / path
    return path.resolve()


def load_spec(path):
    spec_path = Path(path).resolve()
    with spec_path.open('r', encoding='utf-8') as stream:
        spec = yaml.safe_load(stream)
    if not isinstance(spec, dict):
        raise ValueError('prediction-validation spec must be a mapping')
    if spec.get('protocol_status') != 'frozen_secondary_analysis':
        raise ValueError('protocol_status must be frozen_secondary_analysis')

    horizons = sorted({int(value) for value in spec['horizons']})
    if horizons != [1, 3, 5]:
        raise ValueError('the registered horizons must be exactly [1, 3, 5]')
    train_seeds = [int(value) for value in spec['training_seeds']]
    if len(train_seeds) < 5 or len(train_seeds) != len(set(train_seeds)):
        raise ValueError('at least five distinct formal training seeds required')
    if int(spec.get('rolling_window', 10)) < 1:
        raise ValueError('rolling_window must be positive')

    resolved = dict(spec)
    resolved.update({
        'spec_path': str(spec_path),
        'formal_result_path': str(_resolve(
            spec_path, spec['formal_result_file']
        )),
        'comparison_path': str(_resolve(
            spec_path, spec['comparison_result_file']
        )),
        'output_root_path': str(_resolve(spec_path, spec['output_root'])),
        'horizons': horizons,
        'training_seeds': train_seeds,
        'rolling_window': int(spec.get('rolling_window', 10)),
        'minimum_seed_wins': int(spec.get('minimum_seed_wins', 3)),
    })
    return resolved


def build_plan(spec):
    formal_path = Path(spec['formal_result_path'])
    comparison_path = Path(spec['comparison_path'])
    if not formal_path.is_file():
        raise FileNotFoundError(f'missing formal result: {formal_path}')
    if not comparison_path.is_file():
        raise FileNotFoundError(f'missing comparison result: {comparison_path}')

    formal = json.loads(formal_path.read_text(encoding='utf-8'))
    comparison = json.loads(comparison_path.read_text(encoding='utf-8'))
    comparison_hash = file_sha256(comparison_path)
    checks = {
        'formal_run_complete': formal.get('status') == 'complete',
        'formal_estimates_available': bool(
            formal.get('formal_estimates_available', False)
        ),
        'comparison_hash_matches_formal_result': (
            formal.get('comparison_result_sha256') == comparison_hash
        ),
        'algorithm_registered': str(spec['algorithm']) in set(
            formal.get('algorithms', [])
        ),
        'training_seeds_match_formal_protocol': (
            [int(value) for value in formal.get('training_seeds', [])]
            == list(spec['training_seeds'])
        ),
        'comparison_complete': comparison.get('status') == 'complete',
    }
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError('formal-data prerequisite failed: ' + ', '.join(failed))

    comparison_root = comparison_path.parent
    action_files = {}
    for seed in spec['training_seeds']:
        path = (
            comparison_root / str(spec['algorithm']) / f'seed_{seed}'
            / 'evaluation_actions.csv'
        )
        if not path.is_file():
            raise FileNotFoundError(f'missing formal action trace: {path}')
        action_files[str(seed)] = {
            'path': str(path),
            'sha256': file_sha256(path),
            'size_bytes': int(path.stat().st_size),
        }

    return {
        'status': 'planned',
        'protocol_status': 'frozen_secondary_analysis',
        'development_only': False,
        'formal_claims_authorized': False,
        'analysis_was_registered_before_reading_prediction_errors': True,
        'no_training_or_parameter_update_performed': True,
        'spec_path': spec['spec_path'],
        'spec_sha256': file_sha256(spec['spec_path']),
        'runner_sha256': file_sha256(__file__),
        'formal_result_file': str(formal_path),
        'formal_result_sha256': file_sha256(formal_path),
        'comparison_result_file': str(comparison_path),
        'comparison_result_sha256': comparison_hash,
        'prerequisite_checks': checks,
        'algorithm': str(spec['algorithm']),
        'training_seeds': list(spec['training_seeds']),
        'horizons': list(spec['horizons']),
        'rolling_window': int(spec['rolling_window']),
        'models': list(MODELS),
        'scopes': list(SCOPES),
        'primary_endpoint': {
            'scope': 'cumulative',
            'horizon': 5,
            'metric': 'mae_joule',
            'contrast': 'persistence_minus_runtime_lstm',
            'minimum_seed_wins': int(spec['minimum_seed_wins']),
        },
        'action_files': action_files,
        'interpretation': (
            'Secondary analysis of already-frozen independent evaluation '
            'traces. It supports prediction accuracy but does not replace a '
            'separately preregistered confirmatory prediction experiment.'
        ),
    }


def add_causal_baselines(frame, rolling_window=10):
    """Add baselines using measurements available before the forecast row."""
    ordered = frame.sort_values(['episode', 'step']).copy()
    grouped = ordered.groupby('episode', sort=False)['E_total']
    ordered['persistence_prediction'] = grouped.shift(1)
    ordered['rolling_mean_prediction'] = grouped.transform(
        lambda series: series.shift(1).rolling(
            int(rolling_window), min_periods=1
        ).mean()
    )
    return ordered


def _metric_row(actual, predicted):
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    finite = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[finite]
    predicted = predicted[finite]
    if actual.size == 0:
        return None
    error = predicted - actual
    mae = float(np.mean(np.abs(error)))
    actual_scale = float(np.mean(np.abs(actual)))
    return {
        'sample_count': int(actual.size),
        'mae_joule': mae,
        'rmse_joule': float(np.sqrt(np.mean(np.square(error)))),
        'bias_joule': float(np.mean(error)),
        'nmae_percent': (
            float(100.0 * mae / actual_scale)
            if actual_scale > 1e-12 else float('nan')
        ),
        'underprediction_rate': float(np.mean(predicted < actual)),
        'actual_mean_joule': actual_scale,
    }


def episode_metric_records(frame, train_seed, horizons):
    records = []
    for episode, group in frame.groupby('episode', sort=True):
        for horizon in horizons:
            actual_steps = [
                group[f'actual_energy_step_{index}'].to_numpy()
                for index in range(1, horizon + 1)
            ]
            runtime_steps = [
                group[f'predicted_energy_step_{index}'].to_numpy()
                for index in range(1, horizon + 1)
            ]
            raw_steps = [
                group[f'raw_predicted_energy_step_{index}'].to_numpy()
                for index in range(1, horizon + 1)
            ]
            actual_step = actual_steps[-1]
            predictions = {
                'runtime_lstm': runtime_steps[-1],
                'raw_lstm': raw_steps[-1],
                'persistence': group['persistence_prediction'].to_numpy(),
                'rolling_mean': group['rolling_mean_prediction'].to_numpy(),
            }
            for model, predicted in predictions.items():
                values = _metric_row(actual_step, predicted)
                if values is not None:
                    records.append({
                        'train_seed': int(train_seed),
                        'episode': int(episode),
                        'scope': 'step',
                        'horizon': int(horizon),
                        'model': model,
                        **values,
                    })

            actual_cumulative = np.sum(np.vstack(actual_steps), axis=0)
            cumulative_predictions = {
                'runtime_lstm': np.sum(np.vstack(runtime_steps), axis=0),
                'raw_lstm': np.sum(np.vstack(raw_steps), axis=0),
                'persistence': (
                    float(horizon)
                    * group['persistence_prediction'].to_numpy()
                ),
                'rolling_mean': (
                    float(horizon)
                    * group['rolling_mean_prediction'].to_numpy()
                ),
            }
            for model, predicted in cumulative_predictions.items():
                values = _metric_row(actual_cumulative, predicted)
                if values is not None:
                    records.append({
                        'train_seed': int(train_seed),
                        'episode': int(episode),
                        'scope': 'cumulative',
                        'horizon': int(horizon),
                        'model': model,
                        **values,
                    })
    return records


def aggregate_seed_metrics(episode_frame):
    metric_columns = [
        'mae_joule', 'rmse_joule', 'bias_joule', 'nmae_percent',
        'underprediction_rate', 'actual_mean_joule',
    ]
    keys = ['train_seed', 'scope', 'horizon', 'model']
    grouped = episode_frame.groupby(keys, as_index=False)
    means = grouped[metric_columns].mean()
    counts = grouped.agg(
        episode_count=('episode', 'nunique'),
        sample_count=('sample_count', 'sum'),
    )
    return means.merge(counts, on=keys, validate='one_to_one')


def _critical_95(sample_count):
    values = {
        2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
        6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262,
    }
    return values.get(int(sample_count), 1.96)


def summarize_across_seeds(seed_frame):
    metrics = [
        'mae_joule', 'rmse_joule', 'bias_joule', 'nmae_percent',
        'underprediction_rate',
    ]
    summary = []
    keys = ['scope', 'horizon', 'model']
    for key, group in seed_frame.groupby(keys, sort=True):
        row = dict(zip(keys, key))
        row['seed_count'] = int(group['train_seed'].nunique())
        for metric in metrics:
            values = group[metric].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            half = (
                _critical_95(len(values)) * std / math.sqrt(len(values))
                if values.size > 1 else 0.0
            )
            row[f'{metric}_mean'] = mean
            row[f'{metric}_std'] = std
            row[f'{metric}_ci95_low'] = mean - half
            row[f'{metric}_ci95_high'] = mean + half
        summary.append(row)
    return pd.DataFrame(summary)


def assess_primary_endpoint(seed_frame, minimum_seed_wins=3):
    primary = seed_frame[
        (seed_frame['scope'] == 'cumulative')
        & (seed_frame['horizon'] == 5)
        & (seed_frame['model'].isin(['runtime_lstm', 'persistence']))
    ].pivot(index='train_seed', columns='model', values='mae_joule')
    primary = primary.dropna()
    differences = primary['persistence'] - primary['runtime_lstm']
    wins = int(np.sum(differences > 0.0))
    count = int(len(differences))
    mean = float(np.mean(differences)) if count else float('nan')
    std = float(np.std(differences, ddof=1)) if count > 1 else 0.0
    half = (
        _critical_95(count) * std / math.sqrt(count) if count > 1 else 0.0
    )
    passed = bool(
        count >= 5 and wins >= int(minimum_seed_wins) and mean > 0.0
    )
    return {
        'status': (
            'supports_prediction_accuracy' if passed
            else 'prediction_accuracy_not_supported'
        ),
        'passed': passed,
        'difference_definition': 'persistence_mae_minus_runtime_lstm_mae',
        'mean_difference_joule': mean,
        'std_difference_joule': std,
        'ci95_low_joule': mean - half,
        'ci95_high_joule': mean + half,
        'seed_wins': wins,
        'minimum_seed_wins': int(minimum_seed_wins),
        'seed_count': count,
        'paired_differences_by_seed': {
            str(int(seed)): float(value)
            for seed, value in differences.items()
        },
    }


def read_action_trace(path, horizons, rolling_window):
    columns = ['episode', 'step', 'prediction_source', 'E_total']
    for index in range(1, max(horizons) + 1):
        columns.extend([
            f'predicted_energy_step_{index}',
            f'raw_predicted_energy_step_{index}',
            f'actual_energy_step_{index}',
        ])
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = sorted(set(columns) - set(header))
    if missing:
        raise ValueError(f'{path} is missing columns: {missing}')
    frame = pd.read_csv(path, usecols=columns)
    frame = add_causal_baselines(frame, rolling_window=rolling_window)
    frame = frame[frame['prediction_source'].astype(str) == 'lstm'].copy()
    if frame.empty:
        raise ValueError(f'{path} contains no active LSTM prediction rows')
    return frame


def execute_analysis(spec, registered):
    episode_records = []
    trace_counts = {}
    for seed in spec['training_seeds']:
        path = registered['action_files'][str(seed)]['path']
        print(f'[frozen-lstm] reading seed={seed}: {path}', flush=True)
        frame = read_action_trace(
            path, spec['horizons'], spec['rolling_window']
        )
        trace_counts[str(seed)] = {
            'active_lstm_rows': int(len(frame)),
            'episode_count': int(frame['episode'].nunique()),
        }
        episode_records.extend(episode_metric_records(
            frame, seed, spec['horizons']
        ))

    episode_frame = pd.DataFrame(episode_records)
    seed_frame = aggregate_seed_metrics(episode_frame)
    summary_frame = summarize_across_seeds(seed_frame)
    assessment = assess_primary_endpoint(
        seed_frame, minimum_seed_wins=spec['minimum_seed_wins']
    )
    root = Path(spec['output_root_path'])
    root.mkdir(parents=True, exist_ok=True)
    episode_file = root / 'prediction_metrics_by_episode.csv'
    seed_file = root / 'prediction_metrics_by_seed.csv'
    summary_file = root / 'prediction_metrics_summary.csv'
    episode_frame.to_csv(episode_file, index=False)
    seed_frame.to_csv(seed_file, index=False)
    summary_frame.to_csv(summary_file, index=False)

    result = {
        **registered,
        'status': 'complete',
        'primary_assessment': assessment,
        'trace_counts': trace_counts,
        'outputs': {
            'episode_metrics': str(episode_file),
            'seed_metrics': str(seed_file),
            'summary_metrics': str(summary_file),
        },
        'metric_notes': {
            'persistence': (
                'The last measured per-slot energy before the forecast; this '
                'is the causal current-value baseline.'
            ),
            'rolling_mean': (
                f'Causal mean of up to the preceding '
                f'{spec["rolling_window"]} measured slots.'
            ),
            'runtime_lstm': (
                'Residual-corrected frozen LSTM output actually supplied to '
                'the formal policy.'
            ),
            'raw_lstm': 'Frozen LSTM output before online residual correction.',
            'replication_unit': (
                'Metrics are summarized by episode and then by training seed; '
                'overlapping forecast windows are not independent replicates.'
            ),
        },
    }
    result_file = root / 'prediction_validation_result.json'
    write_json_atomic(result_file, result)
    return result_file, result


def main():
    parser = argparse.ArgumentParser(
        description='Analyze frozen formal LSTM predictions without training'
    )
    parser.add_argument('--spec', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()

    spec = load_spec(args.spec)
    plan = build_plan(spec)
    root = Path(spec['output_root_path'])
    plan_file = root / 'plan.json'
    if not args.execute:
        if plan_file.is_file():
            registered = json.loads(plan_file.read_text(encoding='utf-8'))
            if registered != plan:
                raise RuntimeError('a different validation plan is registered')
        else:
            write_json_atomic(plan_file, plan)
        print(json.dumps({
            'status': 'planned',
            'plan_file': str(plan_file),
            'action_trace_count': len(plan['action_files']),
            'no_training_or_parameter_update_performed': True,
            'execute_command': (
                'python scripts/analyze_frozen_lstm_predictions.py '
                f'--spec {args.spec} --execute'
            ),
        }, ensure_ascii=False, indent=2))
        return

    if not plan_file.is_file():
        raise RuntimeError('run plan-only once before execution')
    registered = json.loads(plan_file.read_text(encoding='utf-8'))
    if registered != plan:
        raise RuntimeError(
            'validation protocol drift detected; spec, runner, formal '
            'result, comparison, or action trace changed'
        )
    result_file, result = execute_analysis(spec, registered)
    print(json.dumps({
        'status': result['status'],
        'result_file': str(result_file),
        'primary_assessment': result['primary_assessment'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
