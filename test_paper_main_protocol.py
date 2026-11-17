import unittest
from pathlib import Path

import yaml

from baselines.train_baseline import ALGO_MAP
from scripts.run_prediction_study import (
    build_plan,
    load_study_spec,
    summarize_planned_contrasts,
)
from utils.algorithm_profiles import algorithm_profile
from utils.config import load_config


class PaperMainProtocolTests(unittest.TestCase):
    def test_paper_profiles_are_registered_with_sac_factory(self):
        reference = ALGO_MAP['SAC-LSTM-EP-NoCMDP']
        for name in (
            'SAC-LSTM-EP-PaperMain',
            'SAC-LSTM-EP-NoPrediction',
            'SAC-LSTM-EP-FixedSOC',
            'SAC-LSTM-EP-CMDP',
        ):
            self.assertIn(name, ALGO_MAP)
            self.assertIs(ALGO_MAP[name], reference)

    def test_smoke_is_one_seed_development_only_and_disjoint(self):
        spec = load_study_spec('configs/paper_main_core_smoke_v1.yaml')
        plan = build_plan(spec)
        self.assertEqual(plan['protocol_status'], 'smoke')
        self.assertTrue(plan['development_only'])
        self.assertFalse(plan['formal_claims_authorized'])
        self.assertEqual(plan['minimum_training_seeds'], 1)
        self.assertEqual(plan['training_runs'], 4)
        train = set(range(850000, 850300))
        evaluate = set(range(851000, 851100))
        self.assertFalse(train & evaluate)
        self.assertEqual(
            plan['paper_labels']['SAC-LSTM-EP-PaperMain'],
            'SAC-LSTM-EP',
        )
        self.assertEqual(len(plan['planned_contrasts']), 3)

    def test_smoke_profile_contracts_match_declared_components(self):
        spec = load_study_spec('configs/paper_main_core_smoke_v1.yaml')
        plan = build_plan(spec)
        contracts = plan['profile_contracts']
        self.assertFalse(contracts['SAC-LSTM-EP-PaperMain']['use_cmdp'])
        self.assertTrue(contracts['SAC-LSTM-EP-PaperMain']['use_lstm'])
        self.assertEqual(
            contracts['SAC-LSTM-EP-NoPrediction']['prediction_source'],
            'current',
        )
        self.assertEqual(
            contracts['SAC-LSTM-EP-FixedSOC']['safety_controller'],
            'soc_threshold',
        )
        self.assertTrue(contracts['SAC-LSTM-EP-CMDP']['use_cmdp'])

    def test_paper_environment_is_offloading_only_with_frozen_thresholds(self):
        config = load_config('configs/paper_main_task_offloading_v1.yaml')
        self.assertEqual(config['policy_action_mode'], 'offloading_only')
        self.assertEqual(config['threshold_yellow'], 0.06)
        self.assertEqual(config['threshold_orange'], 0.03)
        self.assertEqual(config['threshold_red'], 0.0)

    def test_formal_protocol_remains_draft(self):
        path = Path('configs/formal_paper_main_v1_draft.yaml')
        spec = yaml.safe_load(path.read_text(encoding='utf-8'))
        self.assertEqual(spec['protocol_status'], 'draft')
        self.assertEqual(len(spec['training_seeds']), 5)
        self.assertEqual(len(spec['evaluation_seeds']), 5)
        for name in spec['algorithms']:
            algorithm_profile(name)
        self.assertIn('C-SAC-Alvi-Adapted', spec['algorithms'])
        self.assertIn('DDPG-MatchedSafety', spec['algorithms'])

    def test_contrast_summary_is_descriptive_and_directional(self):
        def record(value):
            return {'summary': {
                'reward': {'mean': value},
                'tasks': {'mean': None},
            }}

        result = {'results': {
            'Treatment': record(12.0),
            'Control': record(10.0),
        }}
        contrasts = summarize_planned_contrasts(result, [{
            'id': 'test',
            'treatment': 'Treatment',
            'control': 'Control',
            'purpose': 'unit test',
        }])
        self.assertEqual(
            contrasts['test']['mean_differences']['reward'], 2.0
        )
        self.assertIsNone(
            contrasts['test']['mean_differences']['tasks']
        )
        self.assertIn('no superiority claim', contrasts['test']['interpretation'])


if __name__ == '__main__':
    unittest.main()
