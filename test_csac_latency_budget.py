import unittest

import pandas as pd

from scripts.calibrate_csac_latency_budget import derive_latency_budget


class CSACLatencyBudgetTests(unittest.TestCase):
    def test_budget_uses_discounted_episode_cost_and_ceiling_grid(self):
        frame = pd.DataFrame({
            'episode': [0, 0, 1, 1],
            'step': [0, 1, 0, 1],
            'cmdp_latency_violation_cost': [1.0, 1.0, 2.0, 0.0],
        })
        result = derive_latency_budget(
            frame, gamma=0.5, target_fraction=0.9, rounding_grid=0.25
        )
        # Episode costs are 1.5 and 2.0; 0.9 * mean = 1.575, which is
        # conservatively rounded upward to the next 0.25 grid point.
        self.assertAlmostEqual(
            result['reference_distribution']['mean'], 1.75
        )
        self.assertAlmostEqual(result['derivation']['raw_budget'], 1.575)
        self.assertEqual(result['derivation']['selected_budget'], 1.75)

    def test_missing_latency_column_is_rejected(self):
        with self.assertRaises(ValueError):
            derive_latency_budget(pd.DataFrame({
                'episode': [0],
                'step': [0],
            }))


if __name__ == '__main__':
    unittest.main()
