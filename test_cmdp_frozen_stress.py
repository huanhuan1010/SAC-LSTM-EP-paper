import json
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.scenario_sweep import (
    assess_cmdp_binding_stress,
    load_scenario_spec,
    paired_seed_contrasts,
    run_scenarios,
    validate_comparison_handoff,
)
from scripts.soc_sweep import aggregate_seed_summaries


FULL = 'SAC-LSTM-EP'
NO_CMDP = 'SAC-LSTM-EP-NoCMDP'
SEEDS = [720000, 722000, 724000]


class CmdpFrozenStressTests(unittest.TestCase):
    @staticmethod
    def criteria():
        return {
            'contrast': 'full_vs_nocmdp',
            'treatment': FULL,
            'control': NO_CMDP,
            'minimum_binding_scenarios': 2,
            'binding_minimums': {
                'cmdp_budget_exceedance_rate': 10.0,
                'raw_action_energy_infeasible_rate': 1.0,
            },
            'minimum_improved_scenarios': 2,
            'minimum_primary_improvement_seed_wins': 2,
            'primary_maximum_treatment_minus_control': {
                'cmdp_budget_exceedance_rate': -2.0,
                'cmdp_discounted_cost': -0.10,
                'raw_action_energy_infeasible_rate': -2.0,
            },
            'noninferiority': {
                'maximum_physical_violation_increase_percentage_points': 2.0,
                'minimum_safe_return_difference_percentage_points': -2.0,
                'minimum_mission_success_difference_percentage_points': -5.0,
                'minimum_task_completion_retention_ratio': 0.90,
            },
        }

    @staticmethod
    def seed_record(*, budget, cost, raw, physical=0.0, safe=96.0,
                    mission=80.0, completion=70.0):
        return {
            'cmdp_budget_exceedance_rate': budget,
            'cmdp_discounted_cost': cost,
            'raw_action_energy_infeasible_rate': raw,
            'physical_constraint_violation_rate': physical,
            'safe_return_rate': safe,
            'mission_success_rate': mission,
            'task_completion_rate': completion,
        }

    def scenario(self, improved=True):
        control_rows = {}
        treatment_rows = {}
        for index, seed in enumerate(SEEDS):
            control_rows[str(seed)] = self.seed_record(
                budget=40.0 + index,
                cost=8.0 + 0.1 * index,
                raw=12.0 + index,
            )
            if improved:
                treatment_rows[str(seed)] = self.seed_record(
                    budget=35.0 + index,
                    cost=7.7 + 0.1 * index,
                    raw=8.0 + index,
                    safe=95.0,
                    mission=78.0,
                    completion=68.0,
                )
            else:
                treatment_rows[str(seed)] = self.seed_record(
                    budget=40.0 + index,
                    cost=8.0 + 0.1 * index,
                    raw=12.0 + index,
                )
        return {
            'algorithms': {
                FULL: {
                    'per_seed': treatment_rows,
                    'summary': aggregate_seed_summaries(treatment_rows),
                },
                NO_CMDP: {
                    'per_seed': control_rows,
                    'summary': aggregate_seed_summaries(control_rows),
                },
            }
        }

    def contrast(self, results):
        contrasts, _ = paired_seed_contrasts(
            results,
            [{
                'id': 'full_vs_nocmdp',
                'treatment': FULL,
                'control': NO_CMDP,
            }],
            SEEDS,
        )
        return contrasts

    def test_registered_protocol_changes_only_initial_soc(self):
        _, spec = load_scenario_spec(
            'configs/cmdp_frozen_energy_stress_v3_6.yaml'
        )
        self.assertTrue(spec['require_frozen_plan'])
        self.assertEqual(spec['assessment_mode'], 'cmdp_binding_stress')
        self.assertEqual(
            list(spec['scenarios']),
            ['soc_30', 'soc_35', 'soc_40', 'soc_45'],
        )
        for scenario in spec['scenarios'].values():
            self.assertEqual(
                set(scenario['overrides']),
                {'initial_energy_min', 'initial_energy_max'},
            )
        self.assertNotIn('initial_energy_min', spec['locked_control_keys'])
        self.assertNotIn('initial_energy_max', spec['locked_control_keys'])

    def test_binding_improvement_requires_mean_and_paired_seed_wins(self):
        results = {
            'soc_30': self.scenario(improved=True),
            'soc_35': self.scenario(improved=True),
        }
        assessment = assess_cmdp_binding_stress(
            results, self.contrast(results), self.criteria(), SEEDS
        )
        self.assertTrue(assessment['passed'])
        self.assertEqual(
            assessment['classification'],
            'binding_constraint_robustness_identified',
        )
        self.assertEqual(assessment['improved_binding_scenario_count'], 2)

    def test_equal_risk_under_binding_stress_is_not_a_pass(self):
        results = {
            'soc_30': self.scenario(improved=False),
            'soc_35': self.scenario(improved=False),
        }
        assessment = assess_cmdp_binding_stress(
            results, self.contrast(results), self.criteria(), SEEDS
        )
        self.assertFalse(assessment['passed'])
        self.assertTrue(assessment['gates']['stress_is_binding'])
        self.assertEqual(
            assessment['classification'],
            'cmdp_policy_effect_not_identified',
        )

    def test_handoff_authorizes_only_registered_comparison_source(self):
        source = 'a' * 64
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            comparison = root / 'eval_results.json'
            comparison.write_text(json.dumps({
                'status': 'complete',
                'source_fingerprint': source,
                'environment_fingerprint': 'b' * 64,
                'train_seeds': SEEDS,
            }), encoding='utf-8')
            pilot = root / 'pilot_result.json'
            pilot.write_text(json.dumps({
                'status': 'needs_revision',
                'passed': False,
                'formal_claims_authorized': False,
                'source_fingerprint': source,
                'comparison_result_file': str(comparison),
                'gates': {
                    'mechanism': True,
                    'direction': False,
                },
            }), encoding='utf-8')
            handoff = validate_comparison_handoff({
                'comparison_handoff': {
                    'result_file': str(pilot),
                    'required_status': 'needs_revision',
                    'required_passed': False,
                    'required_formal_claims_authorized': False,
                    'required_gates': {
                        'mechanism': True,
                        'direction': False,
                    },
                }
            })
        self.assertEqual(
            handoff['authorized_checkpoint_source_fingerprint'], source
        )
        self.assertEqual(handoff['comparison_train_seeds'], SEEDS)

    def test_plan_is_hash_locked_before_execution(self):
        source = 'a' * 64
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            comparison = root / 'eval_results.json'
            comparison.write_text(json.dumps({
                'status': 'complete',
                'source_fingerprint': source,
                'environment_fingerprint': 'b' * 64,
                'train_seeds': SEEDS,
            }), encoding='utf-8')
            pilot = root / 'pilot_result.json'
            pilot.write_text(json.dumps({
                'status': 'needs_revision',
                'passed': False,
                'formal_claims_authorized': False,
                'source_fingerprint': source,
                'comparison_result_file': str(comparison),
                'gates': {'mechanism': True},
            }), encoding='utf-8')
            output = root / 'stress'
            spec_path = root / 'stress.yaml'
            spec_path.write_text(yaml.safe_dump({
                'protocol_status': 'development_frozen_cmdp_stress',
                'development_only': True,
                'require_frozen_plan': True,
                'output_root': str(output),
                'evaluation_seed': 730000,
                'evaluation_episodes': 10,
                'comparison_handoff': {
                    'result_file': str(pilot),
                    'required_status': 'needs_revision',
                    'required_passed': False,
                    'required_formal_claims_authorized': False,
                    'required_gates': {'mechanism': True},
                },
                'algorithms': [FULL, NO_CMDP],
                'contrasts': [{
                    'id': 'full_vs_nocmdp',
                    'treatment': FULL,
                    'control': NO_CMDP,
                }],
                'scenarios': {
                    'soc_40': {
                        'overrides': {
                            'initial_energy_min': 0.4,
                            'initial_energy_max': 0.4,
                        }
                    }
                },
            }), encoding='utf-8')
            plan = run_scenarios(spec_path, execute=False)
            registered = json.loads(
                (output / 'plan.json').read_text(encoding='utf-8')
            )
        self.assertEqual(plan['status'], 'planned')
        self.assertEqual(plan['plan_file'], str(output / 'plan.json'))
        self.assertEqual(registered['spec_sha256'], plan['spec_sha256'])
        self.assertEqual(
            registered['comparison_handoff']['comparison_sha256'],
            plan['comparison_handoff']['comparison_sha256'],
        )


if __name__ == '__main__':
    unittest.main()
