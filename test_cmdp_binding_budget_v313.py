import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_matched_cmdp_retraining import (
    BINDING_EXPECTATION_BUDGET_PROTOCOL_STATUS,
    assess_budget_reference,
    assess_cmdp_consistency_effect,
    build_plan,
    load_protocol_spec,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'configs' / 'matched_cmdp_retraining_v3_13.yaml'
V312_SPEC = ROOT / 'configs' / 'matched_cmdp_retraining_v3_12.yaml'


class CmdpBindingBudgetV313Tests(unittest.TestCase):
    def test_protocol_is_one_seed_mean_fraction_smoke(self):
        spec = load_protocol_spec(SPEC)
        self.assertEqual(
            spec['protocol_status'],
            BINDING_EXPECTATION_BUDGET_PROTOCOL_STATUS,
        )
        self.assertTrue(spec['development_only'])
        self.assertEqual(spec['required_seed_count'], 1)
        self.assertEqual(
            spec['budget_reference']['budget_rule'], 'fraction_of_mean'
        )
        self.assertAlmostEqual(
            spec['budget_reference']['budget_target_fraction'], 0.95
        )
        self.assertEqual(
            spec['budget_reference']['training_episodes'],
            spec['pilot']['training_episodes'],
        )
        self.assertEqual(spec['pilot']['training_episodes'], 500)

    def test_v313_seeds_are_fresh_and_pairwise_disjoint(self):
        current = load_protocol_spec(SPEC)
        prior = load_protocol_spec(V312_SPEC)
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

    def test_fraction_of_mean_budget_is_binding_by_construction(self):
        spec = load_protocol_spec(SPEC)
        metrics = {
            'cmdp_cost_activation_rate': 70.0,
            'cmdp_cost_saturation_rate': 10.0,
            'physical_constraint_violation_rate': 0.0,
            'safe_return_rate': 98.0,
            'task_completion_rate': 60.0,
            'safety_override_rate': 12.0,
        }
        result = assess_budget_reference(
            metrics, [40.0, 50.0], spec['budget_reference']
        )
        self.assertTrue(result['passed'])
        self.assertEqual(
            result['budget']['budget_rule'],
            'fraction_of_same_safety_nocmdp_reference_mean',
        )
        self.assertAlmostEqual(
            result['budget']['recommended_cmdp_cost_limit'], 42.75
        )
        self.assertGreater(
            result['budget']['reference_mean'],
            result['budget']['recommended_cmdp_cost_limit'],
        )

    def test_effect_gate_requires_discounted_cost_reduction(self):
        spec = load_protocol_spec(SPEC)
        gates = spec['pilot']['consistency_gates']
        no_cmdp = {
            'cmdp_discounted_cost': 48.0,
            'cmdp_budget_exceedance_rate': 50.0,
            'safety_override_rate': 12.0,
            'raw_action_energy_infeasible_rate': 5.0,
            'physical_constraint_violation_rate': 0.0,
            'task_completion_rate': 60.0,
        }
        full = dict(no_cmdp)
        full.update({
            'cmdp_discounted_cost': 47.8,
            'cmdp_budget_exceedance_rate': 49.0,
            'safety_override_rate': 11.0,
            'raw_action_energy_infeasible_rate': 4.5,
            'task_completion_rate': 58.0,
        })
        passing = assess_cmdp_consistency_effect(full, no_cmdp, gates)
        self.assertTrue(passing['passed'])
        self.assertTrue(passing['cmdp_discounted_cost_reduction_gate'])

        full['cmdp_discounted_cost'] = 48.01
        failing = assess_cmdp_consistency_effect(full, no_cmdp, gates)
        self.assertFalse(failing['passed'])
        self.assertFalse(failing['cmdp_discounted_cost_reduction_gate'])

    def test_plan_binds_failed_v312_without_relabeling_it(self):
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
                    'cmdp_budget_scale_aligned': True,
                    'full_cmdp_training_mechanism_active': True,
                    'full_directionally_improves_matched_nocmdp': False,
                    'paired_seed_improvement_gate': True,
                },
                'formal_claims_authorized': False,
            }), encoding='utf-8')
            spec['calibration_result_path'] = str(calibration)
            spec['prior_development_result_path'] = str(prior)
            plan = build_plan(spec)

        self.assertEqual(
            plan['protocol_status'],
            BINDING_EXPECTATION_BUDGET_PROTOCOL_STATUS,
        )
        self.assertFalse(plan['prior_development_handoff']['passed'])
        self.assertEqual(plan['required_seed_count'], 1)
        self.assertFalse(plan['formal_claims_authorized'])


if __name__ == '__main__':
    unittest.main()
