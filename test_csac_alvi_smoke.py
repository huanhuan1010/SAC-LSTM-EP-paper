import unittest

from scripts.run_csac_alvi_smoke import (
    REQUIRED_ALGORITHMS,
    assess_result,
    build_plan,
    load_smoke_spec,
)


class CSACAlviSmokeTests(unittest.TestCase):
    def test_protocol_is_one_seed_development_smoke(self):
        spec = load_smoke_spec(
            'configs/csac_alvi_functional_smoke_v2.yaml'
        )
        plan = build_plan(spec)
        self.assertTrue(plan['development_only'])
        self.assertEqual(tuple(plan['algorithms']), REQUIRED_ALGORITHMS)
        self.assertEqual(len(plan['training_seeds']), 1)
        self.assertGreaterEqual(plan['training_episodes'], 200)
        self.assertLessEqual(plan['training_episodes'], 300)
        self.assertEqual(
            plan['cmdp_cost_modes']['C-SAC-Alvi-Adapted'],
            'latency_violation',
        )
        self.assertEqual(
            plan['cmdp_cost_modes']['SAC-LSTM-EP'], 'energy_margin'
        )

    def test_assessment_uses_declared_directions(self):
        def row(latency, energy, physical, mission):
            metrics = {
                'task_deadline_violation_rate': latency,
                'task_completion_rate': 50.0,
                'short_horizon_energy_infeasible_rate': energy,
                'physical_constraint_violation_rate': physical,
                'mission_success_rate': mission,
            }
            return {
                'summary': {
                    key: {'mean': value, 'std': 0.0, 'values': [value]}
                    for key, value in metrics.items()
                }
            }

        result = {'results': {
            'SAC': row(30.0, 20.0, 10.0, 20.0),
            'C-SAC-Alvi-Adapted': row(20.0, 18.0, 8.0, 30.0),
            'SAC-LSTM-EP': row(25.0, 5.0, 1.0, 40.0),
        }}
        assessment = assess_result(result)
        self.assertEqual(assessment['status'], 'ready_for_energy_stress')
        self.assertEqual(
            assessment['energy_gate']['status'],
            'deferred_to_frozen_policy_stress_evaluation',
        )
        self.assertEqual(
            assessment['contrasts_percentage_points'][
                'csac_minus_sac_task_deadline_violation'
            ],
            -10.0,
        )

    def test_v3_budget_and_three_seed_gate_are_preregistered(self):
        spec = load_smoke_spec(
            'configs/csac_alvi_functional_pilot_v3.yaml'
        )
        plan = build_plan(spec)
        self.assertEqual(plan['required_seed_count'], 3)
        self.assertEqual(len(plan['training_seeds']), 3)
        self.assertEqual(len(plan['evaluation_seeds']), 3)
        self.assertEqual(
            plan['cmdp_cost_limits']['C-SAC-Alvi-Adapted'], 2.75
        )
        self.assertEqual(
            plan['cmdp_cost_limits']['SAC-LSTM-EP'], 10.0
        )
        self.assertEqual(
            plan['cmdp_latency_budget_derivation']['rounding_rule'],
            'ceiling',
        )

    def test_three_seed_assessment_requires_paired_stability(self):
        def row(latency, completion, energy, physical, mission):
            metrics = {
                'task_deadline_violation_rate': latency,
                'task_completion_rate': completion,
                'short_horizon_energy_infeasible_rate': energy,
                'physical_constraint_violation_rate': physical,
                'mission_success_rate': mission,
            }
            return {
                'summary': {
                    key: {
                        'mean': sum(value) / len(value),
                        'std': 0.0,
                        'values': value,
                    }
                    for key, value in metrics.items()
                }
            }

        result = {'results': {
            'SAC': row(
                [40.0, 42.0, 44.0], [70.0, 72.0, 74.0],
                [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [20.0, 30.0, 40.0],
            ),
            'C-SAC-Alvi-Adapted': row(
                [35.0, 43.0, 39.0], [68.0, 70.0, 71.0],
                [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [30.0, 30.0, 30.0],
            ),
            'SAC-LSTM-EP': row(
                [34.0, 37.0, 38.0], [72.0, 74.0, 75.0],
                [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [50.0, 45.0, 30.0],
            ),
        }}
        gates = {
            'nominal_latency': {
                'minimum_paired_wins': 2,
                'maximum_task_completion_loss_percentage_points': 5.0,
            },
            'nominal_joint_success': {'minimum_paired_wins': 2},
        }
        assessment = assess_result(result, development_gates=gates)
        self.assertEqual(assessment['status'], 'ready_for_energy_stress')
        self.assertEqual(
            assessment['paired_seed_diagnostics']['csac_latency_wins'], 2
        )
        self.assertEqual(
            assessment['paired_seed_diagnostics'][
                'full_joint_success_wins'
            ],
            2,
        )

    def test_three_seed_gate_rejects_single_outlier_mean(self):
        def row(latency, completion, mission):
            metrics = {
                'task_deadline_violation_rate': latency,
                'task_completion_rate': completion,
                'short_horizon_energy_infeasible_rate': [0.0] * 3,
                'physical_constraint_violation_rate': [0.0] * 3,
                'mission_success_rate': mission,
            }
            return {'summary': {
                key: {
                    'mean': sum(value) / len(value),
                    'std': 0.0,
                    'values': value,
                }
                for key, value in metrics.items()
            }}

        result = {'results': {
            'SAC': row([40.0, 40.0, 40.0], [70.0] * 3, [20.0] * 3),
            'C-SAC-Alvi-Adapted': row(
                [0.0, 45.0, 45.0], [70.0] * 3, [25.0] * 3
            ),
            'SAC-LSTM-EP': row(
                [30.0] * 3, [75.0] * 3, [40.0] * 3
            ),
        }}
        assessment = assess_result(result, development_gates={
            'nominal_latency': {'minimum_paired_wins': 2},
            'nominal_joint_success': {'minimum_paired_wins': 2},
        })
        self.assertEqual(assessment['status'], 'needs_revision')
        self.assertFalse(
            assessment['gates']['csac_latency_paired_wins_met']
        )

    def test_diagnostic_joint_gate_does_not_block_energy_stress(self):
        def row(latency, completion, mission):
            metrics = {
                'task_deadline_violation_rate': latency,
                'task_completion_rate': completion,
                'short_horizon_energy_infeasible_rate': [0.0] * 3,
                'physical_constraint_violation_rate': [0.0] * 3,
                'mission_success_rate': mission,
            }
            return {'summary': {
                key: {
                    'mean': sum(value) / len(value),
                    'std': 0.0,
                    'values': value,
                }
                for key, value in metrics.items()
            }}

        result = {'results': {
            'SAC': row([40.0] * 3, [70.0] * 3, [60.0] * 3),
            'C-SAC-Alvi-Adapted': row(
                [30.0, 35.0, 36.0], [72.0] * 3, [70.0] * 3
            ),
            'SAC-LSTM-EP': row(
                [32.0] * 3, [71.0] * 3, [60.0, 80.0, 65.0]
            ),
        }}
        assessment = assess_result(result, development_gates={
            'nominal_latency': {
                'minimum_paired_wins': 2,
                'maximum_task_completion_loss_percentage_points': 5.0,
            },
            'nominal_joint_success': {
                'role': 'diagnostic',
                'minimum_paired_wins': 2,
            },
        })
        self.assertEqual(assessment['status'], 'ready_for_energy_stress')
        self.assertTrue(all(assessment['blocking_gates'].values()))
        self.assertFalse(
            assessment['diagnostic_gates'][
                'full_joint_success_higher_than_csac'
            ]
        )
        self.assertEqual(
            assessment['gate_roles']['nominal_joint_success'], 'diagnostic'
        )

    def test_unknown_joint_gate_role_is_rejected(self):
        def row(value):
            metrics = {
                'task_deadline_violation_rate': [value],
                'task_completion_rate': [value],
                'short_horizon_energy_infeasible_rate': [0.0],
                'physical_constraint_violation_rate': [0.0],
                'mission_success_rate': [value],
            }
            return {'summary': {
                key: {'mean': item[0], 'std': 0.0, 'values': item}
                for key, item in metrics.items()
            }}

        result = {'results': {
            'SAC': row(30.0),
            'C-SAC-Alvi-Adapted': row(20.0),
            'SAC-LSTM-EP': row(40.0),
        }}
        with self.assertRaisesRegex(ValueError, 'blocking or diagnostic'):
            assess_result(result, development_gates={
                'nominal_joint_success': {'role': 'optional'},
            })


if __name__ == '__main__':
    unittest.main()

