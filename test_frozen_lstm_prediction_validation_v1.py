import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'analyze_frozen_lstm_predictions.py'
SPEC = importlib.util.spec_from_file_location('frozen_lstm_validation', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FrozenLstmPredictionValidationTests(unittest.TestCase):
    def test_causal_baselines_use_only_preceding_rows(self):
        frame = pd.DataFrame({
            'episode': [0, 0, 0, 1, 1],
            'step': [0, 1, 2, 0, 1],
            'E_total': [10.0, 20.0, 1000.0, 7.0, 9.0],
        })
        output = MODULE.add_causal_baselines(frame, rolling_window=2)
        self.assertTrue(np.isnan(output.loc[0, 'persistence_prediction']))
        self.assertEqual(output.loc[1, 'persistence_prediction'], 10.0)
        self.assertEqual(output.loc[2, 'persistence_prediction'], 20.0)
        self.assertEqual(output.loc[2, 'rolling_mean_prediction'], 15.0)
        self.assertTrue(np.isnan(output.loc[3, 'persistence_prediction']))
        self.assertEqual(output.loc[4, 'persistence_prediction'], 7.0)

    def test_primary_gate_requires_mean_improvement_and_three_wins(self):
        records = []
        differences = [2.0, 1.0, 0.5, -0.2, -0.1]
        for seed, difference in enumerate(differences, start=1):
            records.extend([
                {
                    'train_seed': seed, 'scope': 'cumulative',
                    'horizon': 5, 'model': 'runtime_lstm',
                    'mae_joule': 10.0,
                },
                {
                    'train_seed': seed, 'scope': 'cumulative',
                    'horizon': 5, 'model': 'persistence',
                    'mae_joule': 10.0 + difference,
                },
            ])
        result = MODULE.assess_primary_endpoint(
            pd.DataFrame(records), minimum_seed_wins=3
        )
        self.assertTrue(result['passed'])
        self.assertEqual(result['seed_wins'], 3)

    def test_episode_metrics_keep_step_and_cumulative_targets_distinct(self):
        frame = pd.DataFrame({
            'episode': [0, 0],
            'step': [10, 11],
            'persistence_prediction': [1.0, 1.0],
            'rolling_mean_prediction': [1.0, 1.0],
            **{
                f'actual_energy_step_{index}': [float(index)] * 2
                for index in range(1, 6)
            },
            **{
                f'predicted_energy_step_{index}': [float(index)] * 2
                for index in range(1, 6)
            },
            **{
                f'raw_predicted_energy_step_{index}': [float(index)] * 2
                for index in range(1, 6)
            },
        })
        records = MODULE.episode_metric_records(frame, 1, [1, 3, 5])
        runtime = [
            row for row in records
            if row['model'] == 'runtime_lstm'
        ]
        self.assertEqual(len(runtime), 6)
        self.assertTrue(all(row['mae_joule'] == 0.0 for row in runtime))
        cumulative_five = next(
            row for row in runtime
            if row['scope'] == 'cumulative' and row['horizon'] == 5
        )
        self.assertEqual(cumulative_five['actual_mean_joule'], 15.0)

    def test_registered_spec_has_fixed_formal_seeds_and_horizons(self):
        spec = MODULE.load_spec(
            ROOT / 'configs' / 'frozen_lstm_prediction_validation_v1.yaml'
        )
        self.assertEqual(spec['horizons'], [1, 3, 5])
        self.assertEqual(
            spec['training_seeds'],
            [860000, 870000, 880000, 890000, 900000],
        )
        self.assertEqual(spec['minimum_seed_wins'], 3)


if __name__ == '__main__':
    unittest.main()
