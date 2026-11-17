import unittest

from scripts.scenario_sweep import load_scenario_spec


FULL = 'SAC-LSTM-EP-PaperMain'
NO_PREDICTION = 'SAC-LSTM-EP-NoPrediction'
SPEC = 'configs/paper_main_prediction_stress_v1.yaml'


class PaperMainPredictionStressV1Tests(unittest.TestCase):
    def setUp(self):
        _, self.spec = load_scenario_spec(SPEC)

    def test_uses_only_frozen_full_and_matched_no_prediction(self):
        self.assertEqual(self.spec['algorithms'], [FULL, NO_PREDICTION])
        self.assertTrue(self.spec['require_frozen_plan'])
        self.assertEqual(self.spec['evaluation_episodes'], 100)
        self.assertEqual(self.spec['evaluation_seed'], 940000)
        self.assertEqual(
            self.spec['comparison_file'],
            'results/experiments/1ab64f30becd/'
            'paper-main-core-formal-v1/eval_results.json',
        )
        handoff = self.spec['comparison_handoff']
        self.assertEqual(handoff['required_status'], 'complete')
        self.assertFalse(handoff['required_formal_claims_authorized'])

    def test_registered_contrast_is_prediction_contribution(self):
        self.assertEqual(self.spec['contrasts'], [{
            'id': 'prediction_contribution_under_stress',
            'treatment': FULL,
            'control': NO_PREDICTION,
        }])
        primary = self.spec['acceptance_criteria']['primary_metrics']
        self.assertEqual(set(primary), {
            'mission_success_rate',
            'short_horizon_energy_infeasible_rate',
            'task_completion_rate',
        })

    def test_soc_grid_is_exactly_30_50_70_90_percent(self):
        observed = []
        for name in ('soc_30', 'soc_50', 'soc_70', 'soc_90'):
            overrides = self.spec['scenarios'][name]['overrides']
            self.assertEqual(
                overrides['initial_energy_min'],
                overrides['initial_energy_max'],
            )
            observed.append(overrides['initial_energy_min'])
        self.assertEqual(observed, [0.30, 0.50, 0.70, 0.90])

    def test_load_and_uncertainty_stresses_are_predeclared(self):
        scenarios = self.spec['scenarios']
        self.assertEqual(scenarios['load_medium']['overrides'], {
            'task_interval': 12,
        })
        self.assertEqual(scenarios['load_high']['overrides'], {
            'task_interval': 10,
        })
        uncertainty = scenarios['uncertainty_high']['overrides']
        self.assertTrue(uncertainty['energy_uncertainty_enabled'])
        self.assertEqual(uncertainty['energy_uncertainty_log_std'], 0.15)
        self.assertEqual(uncertainty['energy_uncertainty_min_scale'], 0.65)
        self.assertEqual(uncertainty['energy_uncertainty_max_scale'], 1.50)

    def test_scenarios_cannot_change_learned_or_safety_controls(self):
        locked = set(self.spec['locked_control_keys'])
        self.assertTrue({
            'policy_action_mode',
            'threshold_yellow',
            'threshold_orange',
            'threshold_red',
            'cmdp_enabled',
            'lookback',
            'horizon',
            'lstm_update_interval',
            'reward_energy_weight',
            'reward_delay_weight',
        }.issubset(locked))
        allowed_exogenous = {
            'initial_energy_min',
            'initial_energy_max',
            'task_interval',
            'energy_uncertainty_enabled',
            'energy_uncertainty_log_std',
            'energy_uncertainty_min_scale',
            'energy_uncertainty_max_scale',
        }
        for scenario in self.spec['scenarios'].values():
            self.assertTrue(
                set(scenario.get('overrides', {})).issubset(
                    allowed_exogenous
                )
            )


if __name__ == '__main__':
    unittest.main()
