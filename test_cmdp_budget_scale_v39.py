import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_matched_cmdp_retraining import (
    BUDGET_SCALE_ALIGNMENT_PROTOCOL_STATUS,
    assess_cmdp_budget_scale_alignment,
    build_plan,
    load_protocol_spec,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'configs' / 'matched_cmdp_retraining_v3_9.yaml'


class CmdpBudgetScaleV39Tests(unittest.TestCase):
    def test_protocol_matches_reference_and_pilot_training_horizons(self):
        spec = load_protocol_spec(SPEC)
        self.assertEqual(
            spec['protocol_status'],
            BUDGET_SCALE_ALIGNMENT_PROTOCOL_STATUS,
        )
        self.assertEqual(spec['required_seed_count'], 1)
        self.assertEqual(
            spec['budget_reference']['training_episodes'],
            spec['pilot']['training_episodes'],
        )
        self.assertEqual(spec['pilot']['training_episodes'], 500)
        self.assertIn('budget_scale_gates', spec['pilot'])

    def test_scale_gate_accepts_nondegenerate_equal_horizon_distribution(self):
        result = assess_cmdp_budget_scale_alignment(
            full_metrics={
                'cmdp_discounted_cost': 48.0,
                'cmdp_budget_exceedance_rate': 30.0,
            },
            no_cmdp_metrics={
                'cmdp_discounted_cost': 50.0,
                'cmdp_budget_exceedance_rate': 40.0,
            },
            budget={
                'recommended_cmdp_cost_limit': 50.0,
                'reference_mean': 49.0,
            },
        )
        self.assertTrue(result['passed'])
        self.assertAlmostEqual(
            result['pilot_to_reference_mean_ratio'], 50.0 / 49.0
        )

    def test_scale_gate_rejects_reference_to_pilot_distribution_drift(self):
        result = assess_cmdp_budget_scale_alignment(
            full_metrics={
                'cmdp_discounted_cost': 9.0,
                'cmdp_budget_exceedance_rate': 0.0,
            },
            no_cmdp_metrics={
                'cmdp_discounted_cost': 10.0,
                'cmdp_budget_exceedance_rate': 0.0,
            },
            budget={
                'recommended_cmdp_cost_limit': 50.0,
                'reference_mean': 49.0,
            },
        )
        self.assertFalse(result['passed'])
        self.assertFalse(result['checks'][
            'reference_and_pilot_nocmdp_distributions_align'
        ])
        self.assertFalse(result['checks'][
            'nocmdp_budget_exceedance_is_nondegenerate'
        ])

    def test_plan_binds_v38_failure_without_reusing_effect_size(self):
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
                'status': 'needs_revision',
                'passed': False,
                'gates': {},
                'formal_claims_authorized': False,
            }), encoding='utf-8')
            spec['calibration_result_path'] = str(calibration)
            spec['prior_development_result_path'] = str(prior)
            plan = build_plan(spec)

        self.assertEqual(
            plan['protocol_status'],
            BUDGET_SCALE_ALIGNMENT_PROTOCOL_STATUS,
        )
        self.assertFalse(plan['prior_development_handoff']['passed'])
        self.assertFalse(plan['formal_claims_authorized'])


if __name__ == '__main__':
    unittest.main()
