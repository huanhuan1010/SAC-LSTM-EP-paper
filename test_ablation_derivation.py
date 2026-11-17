import unittest

from baselines.train_baseline import ALGO_MAP
from experiments.compare import DEFAULT_COMPARISON_ALGORITHMS
from scripts.derive_ablation_results import (
    VARIANT_MAP,
    derive_ablation_package,
)
from utils.provenance import environment_fingerprint, source_fingerprint


def _seed_summary(offset):
    return {
        'reward': 10.0 + offset,
        'energy': 100.0 + offset,
        'system_energy': 120.0 + offset,
        'tasks': 5.0 + offset,
        'steps': 20.0,
        'battery_depletion_rate': 20.0,
        'energy_violation_rate': 10.0,
        'returned_safe': 75.0,
        'return_trigger_rate': 80.0,
        'safe_return_rate': 60.0,
        'mission_success_rate': 50.0,
        'mission_deadline_miss_rate': 25.0,
    }


class AblationDerivationTests(unittest.TestCase):
    def _comparison_package(self):
        results = {}
        for algorithm in VARIANT_MAP.values():
            results[algorithm] = {
                'seed_summaries': {
                    '42': _seed_summary(0.0),
                    '43': _seed_summary(2.0),
                    '44': _seed_summary(4.0),
                }
            }
        config = {
            'num_users': 1,
            'max_steps': 2,
            'task_interval': 1000,
        }
        return {
            'status': 'complete',
            'config': config,
            'environment_fingerprint': environment_fingerprint(config),
            'source_fingerprint': source_fingerprint(),
            'run_name': 'test',
            'train_seeds': [42, 43, 44],
            'evaluation_seed': 1042,
            'train_episodes': 10,
            'evaluation_episodes': 2,
            'results': results,
        }

    def test_ablation_variants_use_unified_training_registry(self):
        self.assertNotIn('SAC-NoDynamicWeight', ALGO_MAP)
        self.assertNotIn('SAC+DynamicWeight', ALGO_MAP)
        self.assertIn('SAC-NoConstraint', ALGO_MAP)
        for algorithm in (
            'SAC-LSTM-EP-SOC',
            'SAC-LSTM-EP-PredictiveReturn',
            'SAC-LSTM-EP-NoCMDP',
            'SAC-LSTM-EP-SingleLevel',
        ):
            self.assertIn(algorithm, ALGO_MAP)
        for algorithm in VARIANT_MAP.values():
            self.assertIn(algorithm, ALGO_MAP)
            self.assertIn(algorithm, DEFAULT_COMPARISON_ALGORITHMS)

    def test_derivation_remaps_units_and_computes_seed_std(self):
        package = derive_ablation_package(self._comparison_package())
        full = package['summary']['Full (SAC-LSTM-EP)']
        self.assertEqual(full['reward']['mean'], 12.0)
        self.assertEqual(full['reward']['std'], 2.0)
        self.assertAlmostEqual(full['battery_depleted']['mean'], 0.2)
        self.assertAlmostEqual(full['returned_safe']['mean'], 0.60)
        self.assertAlmostEqual(
            full['conditional_safe_return']['mean'], 0.75
        )
        self.assertAlmostEqual(full['mission_success']['mean'], 0.5)
        self.assertAlmostEqual(full['mission_deadline_missed']['mean'], 0.25)
        self.assertEqual(full['tasks_completed']['values'], [5.0, 7.0, 9.0])

    def test_incomplete_comparison_is_rejected(self):
        package = self._comparison_package()
        package['status'] = 'running'
        with self.assertRaisesRegex(ValueError, 'complete'):
            derive_ablation_package(package)


if __name__ == '__main__':
    unittest.main()
