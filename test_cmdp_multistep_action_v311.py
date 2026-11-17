import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from env.uav_env import UAVEnv
from scripts.run_matched_cmdp_retraining import (
    ACTION_SENSITIVE_PROTOCOL_STATUS,
    build_plan,
    load_protocol_spec,
)
from utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'configs' / 'matched_cmdp_retraining_v3_11.yaml'
BASE = ROOT / 'configs' / 'cmdp_multistep_action_environment_v3_11.yaml'


class CmdpMultistepActionV311Tests(unittest.TestCase):
    def test_protocol_is_one_seed_development_smoke(self):
        spec = load_protocol_spec(SPEC)
        self.assertEqual(
            spec['protocol_status'], ACTION_SENSITIVE_PROTOCOL_STATUS
        )
        self.assertTrue(spec['development_only'])
        self.assertEqual(spec['required_seed_count'], 1)
        self.assertEqual(spec['budget_reference']['training_episodes'], 300)
        self.assertEqual(spec['pilot']['training_episodes'], 300)
        self.assertEqual(
            spec['budget_reference']['training_episodes'],
            spec['pilot']['training_episodes'],
        )

    def test_cmdp_counterfactual_replaces_three_forecast_slots(self):
        config = load_config(BASE)
        self.assertEqual(config['cmdp_action_forecast_replacement_steps'], 3)
        env = UAVEnv(config)
        env.E_rem = 1000.0
        env._estimate_action_uav_energy = lambda action: {
            'estimated_uav_energy': 20.0,
            'estimated_flight_energy': 20.0,
            'estimated_compute_energy': 0.0,
            'estimated_receiver_energy': 0.0,
            'estimated_uav_cycles': 0.0,
        }
        context = {
            'predicted_energy_vector': np.full(5, 10.0),
            'predicted_energy': 50.0,
            'return_energy': 0.0,
            'safety_reserve': 0.0,
        }
        action = np.zeros(env.full_action_dim, dtype=np.float32)

        safety = env._action_conditioned_energy_context(
            action, context, replacement_steps=1
        )
        cmdp = env._action_conditioned_energy_context(
            action, context, replacement_steps=3
        )

        self.assertEqual(safety['action_forecast_replacement_steps'], 1)
        self.assertEqual(cmdp['action_forecast_replacement_steps'], 3)
        self.assertAlmostEqual(safety['predicted_energy'], 60.0)
        self.assertAlmostEqual(cmdp['predicted_energy'], 80.0)
        self.assertLess(
            cmdp['energy_margin_ratio'], safety['energy_margin_ratio']
        )

    def test_plan_records_isolated_multistep_contract(self):
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

        contract = plan['cmdp_action_cost_contract']
        self.assertEqual(contract['source'], 'raw_action_counterfactual')
        self.assertEqual(contract['forecast_replacement_steps'], 3)
        self.assertFalse(plan['formal_claims_authorized'])


if __name__ == '__main__':
    unittest.main()

