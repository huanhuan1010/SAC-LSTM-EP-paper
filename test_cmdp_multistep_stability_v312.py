import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_matched_cmdp_retraining import (
    ACTION_SENSITIVE_STABILITY_PROTOCOL_STATUS,
    build_plan,
    load_protocol_spec,
)
from utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'configs' / 'matched_cmdp_retraining_v3_12.yaml'
V311_SPEC = ROOT / 'configs' / 'matched_cmdp_retraining_v3_11.yaml'
BASE = ROOT / 'configs' / 'cmdp_multistep_action_environment_v3_11.yaml'


class CmdpMultistepStabilityV312Tests(unittest.TestCase):
    def test_protocol_is_three_seed_equal_horizon_stability(self):
        spec = load_protocol_spec(SPEC)
        self.assertEqual(
            spec['protocol_status'],
            ACTION_SENSITIVE_STABILITY_PROTOCOL_STATUS,
        )
        self.assertTrue(spec['development_only'])
        self.assertEqual(spec['required_seed_count'], 3)
        self.assertEqual(len(spec['budget_reference']['training_seeds']), 3)
        self.assertEqual(len(spec['pilot']['training_seeds']), 3)
        self.assertEqual(
            spec['budget_reference']['training_episodes'],
            spec['pilot']['training_episodes'],
        )
        self.assertEqual(spec['pilot']['training_episodes'], 500)
        self.assertEqual(
            spec['pilot']['effect_gate_mode'], 'paired_consistency'
        )
        self.assertIn('budget_scale_gates', spec['pilot'])

    def test_v311_mechanism_is_frozen(self):
        spec = load_protocol_spec(SPEC)
        config = load_config(BASE)
        self.assertEqual(
            Path(spec['base_config_path']).resolve(), BASE.resolve()
        )
        self.assertEqual(
            config['cmdp_action_forecast_replacement_steps'], 3
        )
        self.assertAlmostEqual(
            spec['budget_reference']['lagrange_init'], 0.30
        )
        self.assertEqual(
            spec['threshold_calibration']['required_parameters'],
            {'threshold_yellow': 0.06, 'threshold_orange': 0.03},
        )

    def test_seed_ranges_are_fresh_and_pairwise_disjoint(self):
        current = load_protocol_spec(SPEC)
        prior = load_protocol_spec(V311_SPEC)
        seeds = []
        for stage in ('budget_reference', 'pilot'):
            seeds.extend(current[stage]['training_seeds'])
            seeds.extend(current[stage]['evaluation_seeds'])
        self.assertEqual(len(seeds), len(set(seeds)))
        for left in current['seed_ranges']:
            for right in prior['seed_ranges']:
                disjoint = (
                    left['end'] < right['start']
                    or right['end'] < left['start']
                )
                self.assertTrue(disjoint, (left, right))

    def test_stability_gates_require_two_of_three_paired_wins(self):
        spec = load_protocol_spec(SPEC)
        paired = spec['pilot']['paired_gates']
        consistency = spec['pilot']['consistency_gates']
        self.assertEqual(
            paired['minimum_primary_improvement_seed_wins'], 2
        )
        self.assertEqual(paired['minimum_improved_metrics_per_seed'], 2)
        self.assertEqual(
            consistency['minimum_mean_improved_primary_metrics'], 2
        )
        self.assertEqual(
            consistency['maximum_significantly_regressed_primary_metrics'],
            0,
        )
        self.assertAlmostEqual(
            consistency['minimum_task_completion_retention_ratio'], 0.90
        )

    def test_plan_binds_passing_v311_without_formal_claims(self):
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
                },
                'formal_claims_authorized': False,
            }), encoding='utf-8')
            spec['calibration_result_path'] = str(calibration)
            spec['prior_development_result_path'] = str(prior)
            plan = build_plan(spec)

        self.assertEqual(
            plan['protocol_status'],
            ACTION_SENSITIVE_STABILITY_PROTOCOL_STATUS,
        )
        self.assertTrue(plan['prior_development_handoff']['passed'])
        self.assertEqual(plan['required_seed_count'], 3)
        self.assertEqual(plan['cmdp_action_cost_contract'][
            'forecast_replacement_steps'
        ], 3)
        self.assertFalse(plan['formal_claims_authorized'])


if __name__ == '__main__':
    unittest.main()

