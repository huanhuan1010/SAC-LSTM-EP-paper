import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_matched_cmdp_retraining import (
    ACTION_SENSITIVE_STABILITY_PROTOCOL_STATUS,
    assess_cmdp_consistency_effect,
    build_plan,
    load_protocol_spec,
    paired_seed_diagnostics,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'configs' / 'matched_cmdp_retraining_v3_8.yaml'
V37_SPEC = ROOT / 'configs' / 'matched_cmdp_retraining_v3_7.yaml'


def _summary(values):
    return {'values': [float(value) for value in values]}


def _comparison(full, no_cmdp):
    return {
        'results': {
            'SAC-LSTM-EP': {'summary': {
                key: _summary(value) for key, value in full.items()
            }},
            'SAC-LSTM-EP-NoCMDP': {'summary': {
                key: _summary(value) for key, value in no_cmdp.items()
            }},
        }
    }


class CmdpActionSensitiveV38Tests(unittest.TestCase):
    def test_protocol_is_three_seed_development_stability(self):
        spec = load_protocol_spec(SPEC)
        self.assertEqual(
            spec['protocol_status'],
            ACTION_SENSITIVE_STABILITY_PROTOCOL_STATUS,
        )
        self.assertTrue(spec['development_only'])
        self.assertEqual(spec['required_seed_count'], 3)
        self.assertEqual(len(spec['budget_reference']['training_seeds']), 3)
        self.assertEqual(len(spec['pilot']['training_seeds']), 3)
        self.assertEqual(spec['pilot']['training_episodes'], 500)
        self.assertEqual(
            spec['pilot']['effect_gate_mode'], 'paired_consistency'
        )

    def test_fresh_seed_ranges_do_not_reuse_v37(self):
        current = load_protocol_spec(SPEC)
        prior = load_protocol_spec(V37_SPEC)
        for left in current['seed_ranges']:
            for right in prior['seed_ranges']:
                disjoint = (
                    left['end'] < right['start']
                    or right['end'] < left['start']
                )
                self.assertTrue(disjoint, (left, right))

    def test_mean_consistency_requires_multiple_risk_improvements(self):
        full = {
            'cmdp_discounted_cost': 4.8,
            'cmdp_budget_exceedance_rate': 30.0,
            'safety_override_rate': 21.0,
            'raw_action_energy_infeasible_rate': 8.0,
            'physical_constraint_violation_rate': 0.0,
            'task_completion_rate': 60.0,
        }
        no_cmdp = {
            'cmdp_discounted_cost': 5.0,
            'cmdp_budget_exceedance_rate': 31.0,
            'safety_override_rate': 20.0,
            'raw_action_energy_infeasible_rate': 10.0,
            'physical_constraint_violation_rate': 0.0,
            'task_completion_rate': 62.0,
        }
        result = assess_cmdp_consistency_effect(full, no_cmdp)
        self.assertTrue(result['passed'])
        self.assertEqual(result['mean_improved_primary_metric_count'], 3)
        self.assertEqual(
            result['mean_significantly_regressed_primary_metric_count'], 0
        )

        equal = dict(no_cmdp)
        result = assess_cmdp_consistency_effect(equal, no_cmdp)
        self.assertFalse(result['passed'])
        self.assertEqual(result['mean_improved_primary_metric_count'], 0)

    def test_paired_gate_requires_two_metrics_in_two_of_three_seeds(self):
        common_full = {
            'physical_constraint_violation_rate': [0, 0, 0],
            'task_completion_rate': [60, 60, 60],
            'mission_success_rate': [50, 50, 50],
        }
        common_reference = {
            'physical_constraint_violation_rate': [0, 0, 0],
            'task_completion_rate': [60, 60, 60],
            'mission_success_rate': [50, 50, 50],
        }
        full = {
            **common_full,
            'cmdp_discounted_cost': [4, 4, 5],
            'cmdp_budget_exceedance_rate': [20, 20, 20],
            'safety_override_rate': [10, 10, 10],
            'raw_action_energy_infeasible_rate': [5, 5, 5],
        }
        no_cmdp = {
            **common_reference,
            'cmdp_discounted_cost': [5, 5, 5],
            'cmdp_budget_exceedance_rate': [21, 20, 20],
            'safety_override_rate': [10, 11, 10],
            'raw_action_energy_infeasible_rate': [5, 6, 5],
        }
        result = paired_seed_diagnostics(
            _comparison(full, no_cmdp),
            minimum_wins=2,
            minimum_metrics_per_seed=2,
        )
        self.assertTrue(result['passed'])
        self.assertEqual(result['primary_improvement_seed_wins'], 2)
        self.assertEqual(
            [row['improved_primary_metric_count']
             for row in result['per_seed_primary_wins']],
            [2, 3, 0],
        )

    def test_plan_binds_failed_v37_result_and_amendment(self):
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
                'gates': {
                    'full_cmdp_training_mechanism_active': True,
                    'full_directionally_improves_matched_nocmdp': False,
                    'paired_seed_improvement_gate': True,
                },
                'formal_claims_authorized': False,
            }), encoding='utf-8')
            spec['calibration_result_path'] = str(calibration)
            spec['prior_development_result_path'] = str(prior)
            plan = build_plan(spec)

        self.assertEqual(plan['effect_gate_mode'], 'paired_consistency')
        self.assertIsNotNone(plan['protocol_amendment'])
        self.assertFalse(plan['prior_development_handoff']['passed'])
        self.assertFalse(plan['formal_claims_authorized'])


if __name__ == '__main__':
    unittest.main()
