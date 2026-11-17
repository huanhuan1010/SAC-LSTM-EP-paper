import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from env.comm_model import CommModel
from utils.config import load_config
from utils.provenance import environment_fingerprint, write_json_atomic


class ConfigLoadingTests(unittest.TestCase):
    def test_json_scientific_notation_remains_numeric(self):
        payload = {
            'noise_power': 1e-13,
            'kappa': 1e-26,
            'kappa_user': 1e-28,
            'bandwidth': 1e6,
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'candidate.json'
            path.write_text(json.dumps(payload), encoding='utf-8')
            loaded = load_config(path)

        for key in payload:
            self.assertIsInstance(loaded[key], float)
            self.assertEqual(loaded[key], payload[key])
        self.assertGreater(CommModel(loaded).data_rate(1.0, 100.0), 0.0)

    def test_generated_candidate_config_preserves_environment_fingerprint(self):
        base = load_config('configs/wind_mission_energy_stress.yaml')
        candidate = {
            **base,
            'cmdp_cost_limit': 3.5,
            'threshold_yellow': 0.45,
            'threshold_orange': 0.35,
            'threshold_red': 0.25,
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'candidate.json'
            write_json_atomic(path, candidate)
            loaded = load_config(path)

        self.assertEqual(
            environment_fingerprint(loaded),
            environment_fingerprint(candidate),
        )
        self.assertIsInstance(loaded['noise_power'], float)
        self.assertIsInstance(loaded['kappa'], float)


if __name__ == '__main__':
    unittest.main()
