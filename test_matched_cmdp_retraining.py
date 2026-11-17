import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_matched_cmdp_retraining import (
    FULL,
    NO_CMDP,
    assess_budget_reference,
    assess_pilot,
    build_plan,
    load_protocol_spec,
    paired_seed_diagnostics,
)
from utils.algorithm_profiles import algorithm_config
from utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'configs' / 'matched_cmdp_retraining_v3_4.yaml'
BASE = ROOT / 'configs' / 'csac_alvi_environment_v4.yaml'


def metric(mean, values):
    return {'mean': float(mean), 'values': [float(value) for value in values]}


def synthetic_comparison():
    full_values = {
        'cmdp_discounted_cost': [4.0, 4.5, 5.0],
        'cmdp_budget_exceedance_rate': [15.0, 20.0, 25.0],
        'safety_override_rate': [20.0, 21.0, 22.0],
        'physical_constraint_violation_rate': [0.0, 0.0, 1.0],
        'task_completion_rate': [60.0, 61.0, 62.0],
        'mission_success_rate': [55.0, 56.0, 57.0],
    }
    no_cmdp_values = {
        'cmdp_discounted_cost': [6.0, 6.5, 7.0],
        'cmdp_budget_exceedance_rate': [35.0, 40.0, 45.0],
        'safety_override_rate': [25.0, 26.0, 27.0],
        'physical_constraint_violation_rate': [0.0, 1.0, 1.0],
        'task_completion_rate': [61.0, 62.0, 63.0],
        'mission_success_rate': [50.0, 51.0, 52.0],
    }

    def record(values):
        return {
            name: metric(sum(items) / len(items), items)
            for name, items in values.items()
        }

    return {
        'results': {
            FULL: {'summary': record(full_values)},
            NO_CMDP: {'summary': record(no_cmdp_values)},
        }
    }


class MatchedCmdpRetrainingTests(unittest.TestCase):
    def calibration_result(self, path, **overrides):
        payload = {
            'status': 'candidate_selected',
            'passed': True,
            'selected_candidate_id': 'relaxed_y006_o003',
            'selected_parameters': {
                'threshold_yellow': 0.06,
                'threshold_orange': 0.03,
            },
            'formal_claims_authorized': False,
            'retraining_required_before_confirmatory_use': True,
        }
        payload.update(overrides)
        path.write_text(json.dumps(payload), encoding='utf-8')

    def resolved_spec_with_handoff(self, directory, **overrides):
        result = Path(directory) / 'calibration_result.json'
        self.calibration_result(result, **overrides)
        spec = load_protocol_spec(SPEC)
        spec['calibration_result_path'] = str(result)
        return spec

    def test_v4_environment_freezes_selected_thresholds(self):
        config = load_config(BASE)
        self.assertEqual(config['policy_action_mode'], 'offloading_only')
        self.assertEqual(config['threshold_yellow'], 0.06)
        self.assertEqual(config['threshold_orange'], 0.03)
        self.assertEqual(config['threshold_red'], 0.0)
        self.assertTrue(algorithm_config(config, FULL)['cmdp_enabled'])
        self.assertFalse(algorithm_config(config, NO_CMDP)['cmdp_enabled'])

    def test_plan_binds_calibration_and_disjoint_seed_ranges(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = self.resolved_spec_with_handoff(directory)
            plan = build_plan(spec)
        self.assertFalse(plan['formal_claims_authorized'])
        self.assertEqual(
            plan['threshold_calibration_handoff']['selected_candidate_id'],
            'relaxed_y006_o003',
        )
        ranges = plan['seed_ranges']
        for left, right in zip(ranges, ranges[1:]):
            self.assertLess(left['end'], right['start'])

    def test_plan_rejects_different_calibration_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = self.resolved_spec_with_handoff(
                directory, selected_candidate_id='unexpected'
            )
            with self.assertRaisesRegex(RuntimeError, 'different candidate'):
                build_plan(spec)

    def test_budget_comes_from_matched_nocmdp_episode_distribution(self):
        spec = load_protocol_spec(SPEC)
        metrics = {
            'cmdp_cost_activation_rate': 50.0,
            'cmdp_cost_saturation_rate': 10.0,
            'physical_constraint_violation_rate': 0.0,
            'safe_return_rate': 100.0,
            'task_completion_rate': 60.0,
            'safety_override_rate': 20.0,
        }
        result = assess_budget_reference(
            metrics, [1.0, 2.0, 3.0, 4.0], spec['budget_reference']
        )
        self.assertTrue(result['passed'])
        self.assertEqual(
            result['budget']['reference_algorithm'], NO_CMDP
        )
        self.assertEqual(
            result['budget']['reference_scope'], 'same_safety_layer'
        )
        self.assertEqual(
            result['budget']['recommended_cmdp_cost_limit'], 3.25
        )

    def test_pilot_requires_mechanism_direction_and_paired_seed_wins(self):
        comparison = synthetic_comparison()
        diagnostics = {
            FULL: {
                'mean': {
                    'cmdp_cost_activation_rate': 0.50,
                    'cmdp_stage_cost_mean': 0.10,
                    'cost_critic_loss': 0.50,
                    'lagrange_multiplier': 0.20,
                    'cmdp_actor_penalty_active_rate': 0.80,
                    'cmdp_actor_penalty_scale_mean': 0.80,
                    'constraint_term_mean': 0.10,
                    'cost_critic_update_count': 5000.0,
                    'cmdp_positive_cost_samples': 1000.0,
                }
            }
        }
        config = load_config(BASE)
        config.update({
            'cmdp_actor_penalty_warmup_updates': 1500,
            'cmdp_actor_penalty_min_positive_samples': 256,
            'lagrange_init': 0.10,
        })
        stage = load_protocol_spec(SPEC)['pilot']
        result = assess_pilot(comparison, diagnostics, config, stage)
        self.assertTrue(result['passed'])
        self.assertEqual(result['status'], 'ready_for_freeze_candidate')
        self.assertEqual(
            result['paired_seed_diagnostics'][
                'primary_improvement_seed_wins'
            ],
            3,
        )
        self.assertFalse(result['formal_claims_authorized'])

    def test_paired_seed_gate_does_not_use_unpaired_means(self):
        comparison = synthetic_comparison()
        result = paired_seed_diagnostics(comparison, minimum_wins=2)
        self.assertTrue(result['passed'])
        self.assertEqual(result['seed_count'], 3)


if __name__ == '__main__':
    unittest.main()
