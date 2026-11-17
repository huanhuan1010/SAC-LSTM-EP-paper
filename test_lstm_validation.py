import hashlib
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts.validate_lstm_prediction import (
    apply_online_residual_correction,
    apply_history_gate,
    baseline_predictions,
    bootstrap_mean_ci,
    build_windows,
    episode_level_metrics,
    evaluate_predictions,
    portable_protocol_manifest,
    validate_frozen_protocol_lock,
    validate_split_seeds,
    validation_gate,
    train_predictor,
)


class LSTMValidationTests(unittest.TestCase):
    def test_persisted_protocol_paths_are_project_relative(self):
        protocol = portable_protocol_manifest({
            'protocol_file': str(
                Path('configs/lstm_prediction_validation_formal_v2.yaml').resolve()
            ),
            'config_path': str(
                Path('configs/wind_mission_energy_stress.yaml').resolve()
            ),
        })
        self.assertEqual(
            protocol['protocol_file'],
            'configs/lstm_prediction_validation_formal_v2.yaml',
        )
        self.assertEqual(
            protocol['config_path'],
            'configs/wind_mission_energy_stress.yaml',
        )

    def test_formal_v2_lock_preserves_yaml_and_rejects_script_drift(self):
        lock = json.loads(
            Path(
                'configs/lstm_prediction_validation_formal_v2.lock.json'
            ).read_text(encoding='utf-8')
        )

        def digest(path):
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()

        self.assertEqual(
            digest(lock['protocol_file']),
            lock['protocol_file_sha256'],
        )
        current_script_digest = digest('scripts/validate_lstm_prediction.py')
        if current_script_digest == lock['validation_script_sha256']:
            self.assertIsNotNone(validate_frozen_protocol_lock({
                'protocol_file': lock['protocol_file'],
            }))
        else:
            with self.assertRaisesRegex(
                    RuntimeError, 'validation script differs'):
                validate_frozen_protocol_lock({
                    'protocol_file': lock['protocol_file'],
                })

    def test_evaluation_builds_history_gated_paired_comparisons(self):
        class DummyPredictor(torch.nn.Module):
            lookback = 2

            def forward(self, inputs):
                return torch.full(
                    (inputs.shape[0], 1), 0.5, dtype=inputs.dtype
                )

        inputs = np.zeros((2, 2, 2), dtype=np.float32)
        inputs[:, -1, 0] = [0.4, 0.6]
        targets = np.array([[5.0], [7.0]], dtype=np.float32)
        metadata = np.array([[0, 0], [1, 1]], dtype=np.int64)
        summary, predictions, _ = evaluate_predictions(
            DummyPredictor(), inputs, targets, metadata,
            {
                'battery_capacity': 10.0,
                'lstm_history_gate_warmup_steps': 4,
                'residual_correction_enabled': False,
            },
            bootstrap_resamples=10,
        )

        self.assertIn('history_gated_lstm', predictions)
        self.assertIn(
            'history_gated_lstm_vs_persistence_baseline',
            summary['paired_episode_cumulative_mae_improvement'],
        )

    def test_early_stopping_keeps_sub_1e10_validation_improvements(self):
        class DummyPredictor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([0.0]))
                self.battery_capacity = 1.0

            def update(self, history, target):
                with torch.no_grad():
                    self.weight.add_(1.0)
                return 0.0

        model = DummyPredictor()
        val_losses = iter((1.160e-8, 1.155e-8, 1.150e-8))
        call_index = {'value': 0}

        def fake_predict(_model, inputs, battery_capacity, batch_size=512):
            call_index['value'] += 1
            # train_predictor calls validation first and training second.
            if call_index['value'] % 2:
                loss = next(val_losses)
            else:
                loss = 1.0e-8
            return np.full((len(inputs), 1), np.sqrt(loss), dtype=np.float64)

        inputs = np.zeros((1, 1, 1), dtype=np.float32)
        targets = np.zeros((1, 1), dtype=np.float32)
        with patch(
            'scripts.validate_lstm_prediction.predict_joule',
            side_effect=fake_predict,
        ):
            history, best_val = train_predictor(
                model, inputs, targets, inputs, targets,
                epochs=3, batch_size=1, patience=1, seed=1,
            )

        self.assertEqual(len(history), 3)
        self.assertAlmostEqual(float(model.weight.item()), 3.0)
        self.assertAlmostEqual(best_val, 1.150e-8, places=15)

    def test_windows_never_cross_episode_boundaries(self):
        first = np.zeros((6, 6), dtype=np.float32)
        second = np.ones((6, 6), dtype=np.float32)
        first[:, 0] = np.arange(6) / 100.0
        second[:, 0] = (100 + np.arange(6)) / 100.0
        inputs, targets, metadata = build_windows(
            [first, second], lookback=3, horizon=2,
            battery_capacity=100.0,
        )
        self.assertEqual(len(inputs), 4)
        self.assertEqual(set(metadata[:, 0]), {0, 1})
        np.testing.assert_allclose(targets[0], [3.0, 4.0])
        np.testing.assert_allclose(targets[2], [103.0, 104.0])

    def test_residual_correction_uses_only_prior_observations(self):
        raw = np.array([[80.0], [80.0], [80.0]])
        actual = np.array([[100.0], [110.0], [120.0]])
        metadata = np.array([[0, 0], [0, 1], [0, 2]])
        corrected = apply_online_residual_correction(
            raw, actual, metadata, alpha=0.5, clip_energy=100.0
        )
        self.assertEqual(corrected[0, 0], 80.0)
        self.assertEqual(corrected[1, 0], 100.0)
        self.assertEqual(corrected[2, 0], 105.0)

    def test_history_gate_uses_only_window_age_and_persistence(self):
        inputs = np.zeros((3, 2, 2), dtype=np.float32)
        inputs[:, -1, 0] = 0.5
        metadata = np.array([[0, 0], [0, 2], [0, 4]])
        ungated = np.full((3, 1), 10.0)
        gated = apply_history_gate(
            ungated, inputs, metadata, battery_capacity=10.0,
            lookback=2, warmup_steps=4,
        )
        np.testing.assert_allclose(gated[:, 0], [5.0, 7.5, 10.0])

    def test_split_seeds_must_be_episode_disjoint(self):
        valid = validate_split_seeds({
            'train': [1, 2], 'validation': [3], 'test': [4, 5],
        })
        self.assertEqual(valid['test'], [4, 5])
        with self.assertRaisesRegex(ValueError, 'leakage'):
            validate_split_seeds({
                'train': [1, 2], 'validation': [2, 3], 'test': [4],
            })

    def test_causal_baselines_use_history_only(self):
        inputs = np.zeros((1, 3, 6), dtype=np.float32)
        inputs[0, :, 0] = [0.1, 0.2, 0.3]
        targets = np.zeros((1, 2), dtype=np.float32)
        predictions = baseline_predictions(
            inputs, targets, battery_capacity=10.0
        )
        np.testing.assert_allclose(
            predictions['persistence_baseline'], [[3.0, 3.0]]
        )
        np.testing.assert_allclose(
            predictions['rolling_mean_baseline'], [[2.0, 2.0]]
        )
        np.testing.assert_allclose(
            predictions['linear_trend_baseline'], [[4.0, 5.0]]
        )

    def test_bootstrap_is_episode_level_and_deterministic(self):
        single = bootstrap_mean_ci([7.0], resamples=20, seed=9)
        self.assertEqual(single, {'mean': 7.0, 'lower': 7.0, 'upper': 7.0})
        first = bootstrap_mean_ci([1.0, 2.0, 4.0], resamples=100, seed=9)
        second = bootstrap_mean_ci([1.0, 2.0, 4.0], resamples=100, seed=9)
        self.assertEqual(first, second)

        targets = np.array([[1.0], [3.0], [10.0]])
        metadata = np.array([[0, 0], [0, 1], [1, 0]])
        rows = episode_level_metrics(
            targets,
            {'predictor': np.array([[2.0], [2.0], [8.0]])},
            metadata,
        )
        self.assertEqual([row['windows'] for row in rows], [2, 1])
        self.assertEqual([row['predictor_mae_joule'] for row in rows], [1.0, 2.0])

    def test_episode_metrics_include_horizon_sum_error(self):
        targets = np.array([[1.0, 2.0], [3.0, 5.0]])
        predictions = {
            'predictor': np.array([[2.0, 2.0], [2.0, 4.0]])
        }
        metadata = np.array([[0, 0], [0, 1]])

        rows = episode_level_metrics(targets, predictions, metadata)

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(
            rows[0]['predictor_cumulative_mae_joule'], 1.5
        )
        self.assertAlmostEqual(
            rows[0]['predictor_cumulative_underprediction_rate'], 0.5
        )

    def test_validation_gate_requires_sample_size_and_bootstrap_evidence(self):
        def result(seed, raw_ci=True, corrected_ci=True):
            paired = {
                'raw_lstm_vs_persistence_baseline': {
                    'ci_excludes_zero_in_favor_of_predictor': raw_ci,
                },
                'history_gated_lstm_vs_persistence_baseline': {
                    'ci_excludes_zero_in_favor_of_predictor': raw_ci,
                },
                'residual_corrected_lstm_vs_linear_trend_baseline': {
                    'ci_excludes_zero_in_favor_of_predictor': corrected_ci,
                },
            }
            return {
                'model_seed': seed,
                'test_metrics': {
                    'raw_lstm': {'mae_joule': 7.0},
                    'history_gated_lstm': {'mae_joule': 6.5},
                    'residual_corrected_lstm': {'mae_joule': 6.0},
                    'persistence_baseline': {'mae_joule': 10.0},
                    'rolling_mean_baseline': {'mae_joule': 9.0},
                    'linear_trend_baseline': {'mae_joule': 8.0},
                    'paired_episode_mae_improvement': paired,
                    'cumulative_horizon': {
                        'raw_lstm': {'mae_joule': 30.0},
                        'history_gated_lstm': {'mae_joule': 28.0},
                        'residual_corrected_lstm': {'mae_joule': 25.0},
                        'persistence_baseline': {'mae_joule': 40.0},
                        'rolling_mean_baseline': {'mae_joule': 35.0},
                        'linear_trend_baseline': {'mae_joule': 32.0},
                    },
                    'paired_episode_cumulative_mae_improvement': paired,
                },
            }

        model_results = [result(1), result(2), result(3, raw_ci=False)]
        passed = validation_gate(
            model_results,
            independent_test_episodes=20,
            minimum_test_episodes=20,
            minimum_model_seed_pass_fraction=2.0 / 3.0,
        )
        self.assertEqual(passed['validation_conclusion'], 'passed_preregistered_gate')

        too_small = validation_gate(
            model_results,
            independent_test_episodes=19,
            minimum_test_episodes=20,
            minimum_model_seed_pass_fraction=2.0 / 3.0,
        )
        self.assertEqual(
            too_small['validation_conclusion'],
            'did_not_pass_preregistered_gate',
        )

        no_interval_evidence = validation_gate(
            [result(1, corrected_ci=False), result(2, corrected_ci=False), result(3)],
            independent_test_episodes=20,
            minimum_test_episodes=20,
            minimum_model_seed_pass_fraction=2.0 / 3.0,
        )
        self.assertFalse(
            no_interval_evidence['checks'][
                'runtime_predictor_seed_pass_fraction_met'
            ]
        )

        cumulative = validation_gate(
            model_results,
            independent_test_episodes=20,
            minimum_test_episodes=20,
            minimum_model_seed_pass_fraction=2.0 / 3.0,
            primary_metric='cumulative_horizon_mae',
        )
        self.assertEqual(
            cumulative['validation_conclusion'],
            'passed_preregistered_gate',
        )
        self.assertEqual(
            cumulative['primary_metric'], 'cumulative_horizon_mae'
        )

        gated = validation_gate(
            model_results,
            independent_test_episodes=20,
            minimum_test_episodes=20,
            minimum_model_seed_pass_fraction=2.0 / 3.0,
            architecture_predictor='history_gated_lstm',
        )
        self.assertEqual(
            gated['architecture_predictor'], 'history_gated_lstm'
        )
        self.assertEqual(
            gated['validation_conclusion'], 'passed_preregistered_gate'
        )


if __name__ == '__main__':
    unittest.main()
