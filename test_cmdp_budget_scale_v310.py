import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_matched_cmdp_retraining import (
    BUDGET_SCALE_STABILITY_PROTOCOL_STATUS,
    build_plan,
    load_protocol_spec,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'configs' / 'matched_cmdp_retraining_v3_10.yaml'


class CmdpBudgetScaleV310Tests(unittest.TestCase):
    def test_protocol_is_three_seed_equal_horizon_stability_run(self):
        spec = load_protocol_spec(SPEC)
        self.assertEqual(
            spec['protocol_status'],
            BUDGET_SCALE_STABILITY_PROTOCOL_STATUS,
        )
        self.assertEqual(spec['required_seed_count'], 3)
        self.assertEqual(len(spec['budget_reference']['training_seeds']), 3)
        self.assertEqual(len(spec['pilot']['training_seeds']), 3)
        self.assertEqual(
            spec['budget_reference']['training_episodes'],
            spec['pilot']['training_episodes'],
        )
        self.assertEqual(spec['pilot']['training_episodes'], 500)
        self.assertEqual(
            spec['pilot']['paired_gates'][
                'minimum_primary_improvement_seed_wins'
            ],
            2,
        )
        self.assertIn('budget_scale_gates', spec['pilot'])

    def test_seed_ranges_are_fresh_and_pairwise_disjoint(self):
        spec = load_protocol_spec(SPEC)
        seeds = []
        for stage in ('budget_reference', 'pilot'):
            seeds.extend(spec[stage]['training_seeds'])
            seeds.extend(spec[stage]['evaluation_seeds'])
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertTrue(all(seed >= 780000 for seed in seeds))

    def test_plan_binds_passing_v39_gates_without_formal_claims(self):
        spec = load_protocol_spec(SPEC)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / 'calibration_result.json'
            calibration.write_text(json.dumps({
                'status': 'candidate_selected',
                'passed': True,
                'selected_candidate_id': 'relaxed_y006_o003',
                'selected_parameters': {
                    'threshold_yellow': 0.06,
                    'threshold_orange': 0.03,
                },
                'formal_claims_authorized': False,
                'retraining_required_before_confirmatory_use': True,
            }), encoding='utf-8')
            prior = root / 'pilot_result.json'
            prior.write_text(json.dumps({
                'status': 'ready_for_three_seed_pilot',
                'passed': True,
                'gates': {
                    'full_cmdp_training_mechanism_active': True,
                    'full_directionally_improves_matched_nocmdp': True,
                    'paired_seed_improvement_gate': True,
                    'cmdp_budget_scale_aligned': True,
                },
                'formal_claims_authorized': False,
            }), encoding='utf-8')
            spec['calibration_result_path'] = str(calibration)
            spec['prior_development_result_path'] = str(prior)
            plan = build_plan(spec)

        self.assertEqual(
            plan['protocol_status'],
            BUDGET_SCALE_STABILITY_PROTOCOL_STATUS,
        )
        self.assertTrue(plan['prior_development_handoff']['passed'])
        self.assertFalse(plan['formal_claims_authorized'])
        self.assertEqual(plan['required_seed_count'], 3)


if __name__ == '__main__':
    unittest.main()
