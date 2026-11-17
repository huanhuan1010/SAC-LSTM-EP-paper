import json
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.run_formal_paper_main import (
    REQUIRED_SMOKE_ALGORITHMS,
    build_plan,
    load_formal_spec,
)


class FormalPaperMainProtocolTests(unittest.TestCase):
    def setUp(self):
        self.spec = load_formal_spec(
            'configs/formal_paper_main_core_v1.yaml'
        )

    def test_core_protocol_is_five_seed_confirmatory_comparison(self):
        self.assertEqual(self.spec['protocol_status'], 'frozen_formal')
        self.assertFalse(self.spec['development_only'])
        self.assertEqual(len(self.spec['training_seeds']), 5)
        self.assertEqual(len(self.spec['evaluation_seeds']), 5)
        self.assertEqual(self.spec['training_episodes'], 2000)
        self.assertEqual(self.spec['evaluation_episodes'], 100)
        self.assertEqual(len(self.spec['algorithms']), 6)
        self.assertIn('SAC-LSTM-EP-PaperMain', self.spec['algorithms'])
        self.assertIn('C-SAC-Alvi-Adapted', self.spec['algorithms'])
        self.assertIn('DDPG', self.spec['algorithms'])

    def test_training_and_evaluation_episode_ranges_are_disjoint(self):
        ranges = []
        for seed in self.spec['training_seeds']:
            ranges.append((seed, seed + self.spec['training_episodes'] - 1))
        for seed in self.spec['evaluation_seeds']:
            ranges.append((seed, seed + self.spec['evaluation_episodes'] - 1))
        for index, first in enumerate(ranges):
            for second in ranges[index + 1:]:
                self.assertTrue(
                    first[1] < second[0] or second[1] < first[0],
                    msg=f'overlapping registered ranges: {first}, {second}',
                )

    def test_plan_keeps_smoke_development_only(self):
        base_config_path = Path(self.spec['base_config_path'])
        from utils.config import load_config
        from utils.provenance import environment_fingerprint

        fingerprint = environment_fingerprint(load_config(base_config_path))
        smoke = {
            'status': 'complete',
            'formal_claims_authorized': False,
            'environment_fingerprint': fingerprint,
            'algorithms': sorted(REQUIRED_SMOKE_ALGORITHMS),
        }
        plan = build_plan(self.spec, prerequisite=smoke)
        self.assertFalse(plan['development_only'])
        self.assertFalse(plan['formal_claims_authorized'])
        self.assertFalse(plan['final_evaluation_is_used_for_selection'])
        self.assertEqual(plan['training_runs'], 30)
        self.assertTrue(
            plan['smoke_prerequisite'][
                'development_effects_are_not_formal_evidence'
            ]
        )

    def test_formal_protocol_rejects_less_than_five_seeds(self):
        source = Path('configs/formal_paper_main_core_v1.yaml')
        payload = yaml.safe_load(source.read_text(encoding='utf-8'))
        payload['training_seeds'] = [860000, 870000, 880000]
        payload['minimum_training_seeds'] = 3
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.yaml'
            payload['base_config'] = str(
                Path(self.spec['base_config_path']).resolve()
            )
            payload['smoke_prerequisite'] = str(
                Path(self.spec['smoke_prerequisite_path']).resolve()
            )
            path.write_text(
                yaml.safe_dump(payload, sort_keys=False), encoding='utf-8'
            )
            with self.assertRaisesRegex(ValueError, 'at least five seeds'):
                load_formal_spec(path)


if __name__ == '__main__':
    unittest.main()
