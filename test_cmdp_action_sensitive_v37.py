import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_matched_cmdp_retraining import (
    ACTION_SENSITIVE_PROTOCOL_STATUS,
    build_plan,
    load_protocol_spec,
)
from utils.calibration import assess_cmdp_directional_effect
from utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'configs' / 'matched_cmdp_retraining_v3_7.yaml'
BASE = ROOT / 'configs' / 'cmdp_action_sensitive_environment_v3_7.yaml'


class CmdpActionSensitiveV37Tests(unittest.TestCase):
    def test_protocol_is_one_seed_development_only(self):
        spec = load_protocol_spec(SPEC)
        self.assertEqual(
            spec['protocol_status'], ACTION_SENSITIVE_PROTOCOL_STATUS
        )
        self.assertTrue(spec['development_only'])
        self.assertEqual(spec['required_seed_count'], 1)
        self.assertEqual(len(spec['budget_reference']['training_seeds']), 1)
        self.assertEqual(len(spec['pilot']['training_seeds']), 1)
        self.assertEqual(spec['pilot']['training_episodes'], 300)

    def test_environment_contract_is_action_sensitive_and_low_soc(self):
        config = load_config(BASE)
        self.assertEqual(
            config['cmdp_energy_cost_source'],
            'raw_action_counterfactual',
        )
        self.assertGreater(config['cmdp_energy_cost_max'], 1.0)
        self.assertGreaterEqual(
            config['cmdp_stage_cost_max'],
            config['cmdp_energy_cost_max'],
        )
        self.assertLessEqual(config['initial_energy_min'], 0.30)
        self.assertGreaterEqual(config['initial_energy_max'], 0.45)

    def test_plan_hash_locks_action_cost_contract(self):
        spec = load_protocol_spec(SPEC)
        with tempfile.TemporaryDirectory() as directory:
            calibration = Path(directory) / 'calibration_result.json'
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
            spec['calibration_result_path'] = str(calibration)
            plan = build_plan(spec)

        contract = plan['cmdp_action_cost_contract']
        self.assertEqual(contract['source'], 'raw_action_counterfactual')
        self.assertEqual(contract['energy_cost_max'], 4.0)
        self.assertEqual(contract['stage_cost_max'], 4.0)
        self.assertFalse(plan['formal_claims_authorized'])

    def test_directional_gate_can_use_raw_action_infeasibility(self):
        full = {
            'cmdp_discounted_cost': 5.0,
            'cmdp_budget_exceedance_rate': 30.0,
            'raw_action_energy_infeasible_rate': 8.0,
            'safety_override_rate': 20.0,
            'physical_constraint_violation_rate': 0.0,
            'task_completion_rate': 60.0,
        }
        no_cmdp = {
            'cmdp_discounted_cost': 5.0,
            'cmdp_budget_exceedance_rate': 30.0,
            'raw_action_energy_infeasible_rate': 12.0,
            'safety_override_rate': 20.0,
            'physical_constraint_violation_rate': 0.0,
            'task_completion_rate': 62.0,
        }
        result = assess_cmdp_directional_effect(full, no_cmdp, {
            'minimum_cmdp_discounted_cost_reduction': 0.10,
            'minimum_budget_exceedance_reduction_percentage_points': 2.0,
            'minimum_raw_action_energy_infeasible_reduction_percentage_points': 2.0,
            'minimum_safety_override_reduction_percentage_points': 2.0,
            'minimum_physical_violation_reduction_percentage_points': 2.0,
            'maximum_physical_violation_increase_percentage_points': 2.0,
            'minimum_task_completion_retention_ratio': 0.90,
        })
        self.assertTrue(result['passed'])
        self.assertTrue(result['improvement_checks'][
            'raw_action_energy_infeasible_rate'
        ])


if __name__ == '__main__':
    unittest.main()

