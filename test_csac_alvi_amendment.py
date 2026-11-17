import unittest
from pathlib import Path

import yaml

from scripts.reassess_csac_alvi_pilot import (
    build_amended_assessment,
    load_amendment_spec,
)


class CSACAlviAmendmentTests(unittest.TestCase):
    @staticmethod
    def row(latency, completion, energy, physical, mission, stored_limit):
        metrics = {
            'task_deadline_violation_rate': latency,
            'task_completion_rate': completion,
            'short_horizon_energy_infeasible_rate': energy,
            'physical_constraint_violation_rate': physical,
            'mission_success_rate': mission,
        }
        summary = {
            key: {
                'mean': sum(values) / len(values),
                'std': 0.0,
                'values': values,
            }
            for key, values in metrics.items()
        }
        per_seed = {}
        for index, seed in enumerate((1, 2, 3)):
            costs = [2.0, 3.0] if stored_limit == 10.0 else [1.0, 2.0]
            per_seed[str(seed)] = {
                'cmdp_discounted_cost': costs,
                'cmdp_cost_limit': stored_limit,
                'cmdp_budget_exceeded_episodes': 0,
            }
        return {'summary': summary, 'per_seed': per_seed}

    def packages(self, identifiable=False):
        environment = 'a' * 64
        result = {
            'run_name': 'pilot-v3',
            'environment_fingerprint': environment,
            'algorithms': [
                'SAC', 'C-SAC-Alvi-Adapted', 'SAC-LSTM-EP'
            ],
            'train_seeds': [1, 2, 3],
            'evaluation_seeds': [11, 12, 13],
            'results': {
                'SAC': self.row(
                    [40.0] * 3, [70.0] * 3, [0.0] * 3,
                    [0.0] * 3, [60.0] * 3, 10.0,
                ),
                'C-SAC-Alvi-Adapted': self.row(
                    [30.0, 35.0, 36.0], [72.0] * 3, [0.0] * 3,
                    [0.0] * 3, [70.0] * 3, 10.0,
                ),
                'SAC-LSTM-EP': self.row(
                    [32.0] * 3, [71.0] * 3,
                    ([1.0, 0.0, 0.0] if identifiable else [0.0] * 3),
                    [0.0] * 3, [60.0, 80.0, 65.0], 10.0,
                ),
            },
        }
        plan = {
            'run_name': 'pilot-v3',
            'environment_fingerprint': environment,
            'algorithms': result['algorithms'],
            'required_seed_count': 3,
            'training_seeds': [1, 2, 3],
            'evaluation_seeds': [11, 12, 13],
            'cmdp_cost_limits': {
                'SAC': 10.0,
                'C-SAC-Alvi-Adapted': 2.75,
                'SAC-LSTM-EP': 10.0,
            },
        }
        original = {
            'status': 'needs_revision',
            'gates': {
                'csac_latency_lower_than_sac': True,
                'csac_latency_paired_wins_met': True,
                'csac_task_completion_noninferior': True,
                'full_joint_success_higher_than_csac': False,
                'full_joint_success_paired_wins_met': False,
            },
        }
        spec = {
            'source_run_name': 'pilot-v3',
            'source_environment_fingerprint': environment,
            'required_seed_count': 3,
            'expected_original_assessment_status': 'needs_revision',
            'amendment': {
                'id': 'registered-amendment',
                'timing': 'post_nominal_pre_stress_pre_confirmatory',
            },
            'amended_development_gates': {
                'nominal_latency': {
                    'minimum_paired_wins': 2,
                    'maximum_task_completion_loss_percentage_points': 5.0,
                },
                'nominal_joint_success': {
                    'role': 'diagnostic',
                    'minimum_paired_wins': 2,
                },
            },
            'energy_stress_protocol': {'frozen_policy_only': True},
        }
        return spec, result, plan, original

    def test_amendment_preserves_failure_as_diagnostic_and_audits_budget(self):
        assessment = build_amended_assessment(*self.packages())
        self.assertEqual(assessment['status'], 'ready_for_energy_stress')
        self.assertFalse(
            assessment['diagnostic_gates'][
                'full_joint_success_higher_than_csac'
            ]
        )
        audit = assessment['cmdp_budget_metadata_audit']
        self.assertTrue(audit['evaluation_budget_metadata_bug_detected'])
        csac = audit['algorithms']['C-SAC-Alvi-Adapted']
        self.assertEqual(csac['corrected_exceedance_rate_percent'], 50.0)
        self.assertTrue(
            assessment['original_assessment'][
                'preserved_without_modification'
            ]
        )

    def test_amendment_is_rejected_when_nominal_energy_is_identifiable(self):
        with self.assertRaisesRegex(ValueError, 'unidentifiable'):
            build_amended_assessment(*self.packages(identifiable=True))

    def test_registered_amendment_preserves_v3_stress_design(self):
        spec = load_amendment_spec(
            'configs/csac_alvi_assessment_amendment_v3_1.yaml'
        )
        self.assertEqual(
            spec['amendment']['timing'],
            'post_nominal_pre_stress_pre_confirmatory',
        )
        with Path('configs/csac_alvi_energy_stress_v3.yaml').open(
                'r', encoding='utf-8') as stream:
            original = yaml.safe_load(stream)
        with Path('configs/csac_alvi_energy_stress_v3_1.yaml').open(
                'r', encoding='utf-8') as stream:
            amended = yaml.safe_load(stream)
        unchanged_fields = (
            'evaluation_seed',
            'evaluation_episodes',
            'algorithms',
            'locked_control_keys',
            'contrasts',
            'acceptance_criteria',
            'scenarios',
        )
        for field in unchanged_fields:
            self.assertEqual(amended[field], original[field], field)
        self.assertIn('prerequisite_assessment', amended)
        self.assertEqual(
            amended['allowed_checkpoint_source_fingerprints'],
            [
                'c085de889a4bf3ab35a8a21fd1087d3cdf02338c401a03d4a'
                '31322bbcea0fdd2'
            ],
        )
        self.assertNotEqual(amended['output_root'], original['output_root'])

    def test_v3_2_boundary_scan_is_frozen_and_uses_fresh_soc_grid(self):
        with Path('configs/csac_alvi_energy_stress_v3_1.yaml').open(
                'r', encoding='utf-8') as stream:
            stress = yaml.safe_load(stream)
        with Path('configs/csac_alvi_safety_boundary_v3_2.yaml').open(
                'r', encoding='utf-8') as stream:
            boundary = yaml.safe_load(stream)

        self.assertTrue(boundary['development_only'])
        self.assertEqual(
            boundary['protocol_status'],
            'development_boundary_mapping',
        )
        self.assertEqual(boundary['comparison_file'], stress['comparison_file'])
        self.assertEqual(boundary['algorithms'], stress['algorithms'])
        self.assertEqual(
            boundary['locked_control_keys'], stress['locked_control_keys']
        )
        self.assertEqual(
            boundary['allowed_checkpoint_source_fingerprints'],
            stress['allowed_checkpoint_source_fingerprints'],
        )
        self.assertEqual(
            boundary['prerequisite_assessment'],
            stress['prerequisite_assessment'],
        )
        self.assertGreaterEqual(
            boundary['evaluation_seed'],
            stress['evaluation_seed'] + stress['evaluation_episodes'],
        )
        self.assertEqual(boundary['evaluation_episodes'], 50)

        expected = {
            'soc_35': 0.35,
            'soc_40': 0.40,
            'soc_45': 0.45,
            'soc_50': 0.50,
            'soc_55': 0.55,
        }
        locked = set(boundary['locked_control_keys'])
        self.assertEqual(list(boundary['scenarios']), list(expected))
        for scenario, soc in expected.items():
            overrides = boundary['scenarios'][scenario]['overrides']
            self.assertEqual(
                overrides,
                {
                    'initial_energy_min': soc,
                    'initial_energy_max': soc,
                },
            )
            self.assertFalse(locked.intersection(overrides))

        gates = boundary['acceptance_criteria'][
            'full_feasible_service_boundary'
        ]
        self.assertEqual(gates['algorithm'], 'SAC-LSTM-EP')
        self.assertEqual(
            gates['ordered_scenarios'], list(expected)
        )
        self.assertEqual(
            gates['selection_rule'],
            'lowest_initial_soc_passing_all_absolute_gates',
        )
        self.assertGreater(gates['minimum_mission_success_rate'], 0.0)
        self.assertLess(gates['maximum_safety_override_rate'], 100.0)
        diagnostic = boundary['acceptance_criteria'][
            'paired_reference_diagnostic'
        ]
        self.assertEqual(diagnostic['role'], 'diagnostic_not_selection')
        self.assertNotEqual(boundary['output_root'], stress['output_root'])


if __name__ == '__main__':
    unittest.main()
