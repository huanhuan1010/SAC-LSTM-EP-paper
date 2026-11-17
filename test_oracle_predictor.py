from pathlib import Path
import tempfile
import unittest

import numpy as np

from agent.oracle_predictor import OracleEnergyTracePredictor


class OraclePredictorTests(unittest.TestCase):
    def test_trace_is_aligned_and_padded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'oracle.npz'
            np.savez(path, seed_42=np.array([10.0, 20.0, 30.0]))
            predictor = OracleEnergyTracePredictor(path, 2, 1000.0)
            predictor.reset_runtime_state(42)
            np.testing.assert_allclose(
                predictor.get_energy_features(), [10.0, 20.0]
            )
            predictor.observe_actual_energy(999.0)
            np.testing.assert_allclose(
                predictor.get_energy_features(), [20.0, 30.0]
            )
            predictor.observe_actual_energy(999.0)
            np.testing.assert_allclose(
                predictor.get_energy_features(), [30.0, 0.0]
            )

    def test_missing_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'oracle.npz'
            np.savez(path, seed_1=np.array([1.0]))
            predictor = OracleEnergyTracePredictor(path, 1, 100.0)
            with self.assertRaises(KeyError):
                predictor.reset_runtime_state(2)


if __name__ == '__main__':
    unittest.main()
