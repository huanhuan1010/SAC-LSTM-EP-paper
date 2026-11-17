import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_matched_cmdp_retraining import (
    NORMALIZED_DUAL_SMOKE_PROTOCOL_STATUS,
    assess_cmdp_budget_scale_alignment,
    assess_cmdp_consistency_effect,
    build_plan,
    load_protocol_spec,
)
from utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'configs' / 'matched_cmdp_retraining_v3_14.yaml'
V313_SPEC = ROOT / 'configs' / 'matched_cmdp_retraining_v3_13.yaml'


class CmdpNormalizedDualV314Tests(unittest.TestCase):
    def test_protocol_is_one_seed_normalized_dual_smoke(self):
        spec = load_protocol_spec(SPEC)
        config = load_config(spec['base_config_path'])
        self.assertEqual(
            spec['protocol_status'], NORMALIZED_DUAL_SMOKE_PROTOCOL_STATUS
        )
        self.assertEqual(spec['required_seed_count'], 1)
        self.assertEqual(config['lagrange_update_mode'], 'projected_normalized')
        self.assertGreater(config['cmdp_gradient_diagnostic_interval'], 0)
        self.assertEqual(spec['pilot']['training_episodes'], 500)

    def test_v314_seed_ranges_are_fresh_and_disjoint_from_v313(self):
        current = load_protocol_spec(SPEC)
        prior = load_protocol_spec(V313_SPEC)
        seeds = []
        for stage in ('budget_reference', 'pilot'):
            seeds.extend(current[stage]['training_seeds'])
            seeds.extend(current[stage]['evaluation_seeds'])
        self.assertEqual(len(seeds), len(set(seeds)))
        for left in current['seed_ranges']:
            for right in prior['seed_ranges']:
                self.assertTrue(
                    left['end'] < right['start']
                    or right['end'] < left['start']
                )

    def test_consistency_gate_rejects_joint_success_regression(self):
        gates = load_protocol_spec(SPEC)['pilot']['consistency_gates']
        no_cmdp = {
            'cmdp_discounted_cost': 48.0,
            'cmdp_budget_exceedance_rate': 50.0,
            'safety_override_rate': 12.0,
            'raw_action_energy_infeasible_rate': 5.0,
            'physical_constraint_violation_rate': 0.0,
            'task_completion_rate': 60.0,
            'mission_success_rate': 70.0,
        }
        full = dict(no_cmdp)
        full.update({
            'cmdp_discounted_cost': 47.5,
            'cmdp_budget_exceedance_rate': 47.0,
            'safety_override_rate': 11.0,
            'raw_action_energy_infeasible_rate': 4.5,
            'task_completion_rate': 58.0,
            'mission_success_rate': 64.0,
        })
        result = assess_cmdp_consistency_effect(full, no_cmdp, gates)
        self.assertFalse(result['passed'])
        self.assertFalse(result['mission_success_noninferior'])
        full['mission_success_rate'] = 66.0
        self.assertTrue(
            assess_cmdp_consistency_effect(full, no_cmdp, gates)['passed']
        )

    def test_budget_gate_requires_full_mean_near_budget(self):
        gates = load_protocol_spec(SPEC)['pilot']['budget_scale_gates']
        budget = {
            'recommended_cmdp_cost_limit': 40.0,
            'reference_mean': 42.0,
        }
        no_cmdp = {
            'cmdp_discounted_cost': 42.0,
            'cmdp_budget_exceedance_rate': 50.0,
        }
        failed = assess_cmdp_budget_scale_alignment(
            {'cmdp_discounted_cost': 42.1}, no_cmdp, budget, gates
        )
        self.assertFalse(failed['passed'])
        self.assertFalse(failed['checks']['full_mean_is_near_or_below_budget'])
        passed = assess_cmdp_budget_scale_alignment(
            {'cmdp_discounted_cost': 41.5}, no_cmdp, budget, gates
        )
        self.assertTrue(passed['passed'])

    def test_plan_binds_v313_and_keeps_formal_claims_disabled(self):
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
                    'cmdp_budget_scale_aligned': True,
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
            plan['protocol_status'], NORMALIZED_DUAL_SMOKE_PROTOCOL_STATUS
        )
        self.assertTrue(plan['prior_development_handoff']['passed'])
        self.assertFalse(plan['formal_claims_authorized'])


if __name__ == '__main__':
    unittest.main()
