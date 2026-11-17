import unittest

from utils.calibration import (
    analyse_cmdp_margin_signal,
    assess_cmdp_ablation_identifiability,
    assess_cmdp_directional_effect,
    assess_calibration_identifiability,
    calibration_completion_status,
    derive_margin_scan_recommendations,
    derive_nosafety_service_gates,
    generate_threshold_candidates,
    recommend_cmdp_budget,
    recommend_matched_nocmdp_budget,
    select_safety_candidate,
    validate_disjoint_seed_ranges,
)


class CalibrationTests(unittest.TestCase):
    @staticmethod
    def metrics(**overrides):
        values = {
            'physical_constraint_violation_rate': 1.0,
            'battery_depletion_rate': 0.0,
            'mission_deadline_miss_rate': 5.0,
            'safe_return_rate': 95.0,
            'mission_success_rate': 70.0,
            'task_completion_rate': 85.0,
            'core_task_completion_rate': 90.0,
            'mean_task_delay_s': 3.0,
            'system_energy': 1000.0,
            'safety_override_rate': 10.0,
        }
        values.update(overrides)
        return values

    def test_budget_uses_ci_upper_bound_and_upward_grid_rounding(self):
        result = recommend_cmdp_budget([0.0, 2.0], grid=0.5)
        self.assertGreater(result['mean_ci_upper_bound'], 2.9)
        self.assertEqual(result['recommended_cmdp_cost_limit'], 3.0)

    def test_budget_rejects_invalid_samples(self):
        with self.assertRaises(ValueError):
            recommend_cmdp_budget([])
        with self.assertRaises(ValueError):
            recommend_cmdp_budget([1.0, -1.0])

    def test_matched_nocmdp_budget_uses_same_safety_reference_mean(self):
        result = recommend_matched_nocmdp_budget(
            [4.0, 5.0, 6.0], target_fraction=0.9, grid=0.5
        )
        self.assertEqual(result['reference_scope'], 'same_safety_layer')
        self.assertEqual(result['recommended_cmdp_cost_limit'], 4.5)
        self.assertEqual(
            result['budget_rule'],
            'fraction_of_same_safety_nocmdp_reference_mean',
        )
        with self.assertRaisesRegex(ValueError, 'all zero'):
            recommend_matched_nocmdp_budget([0.0, 0.0])

    def test_matched_nocmdp_quantile_budget_declares_exceedance(self):
        result = recommend_matched_nocmdp_budget(
            [1.0, 2.0, 3.0, 4.0],
            grid=0.25,
            rule='empirical_quantile',
            quantile=0.75,
        )
        self.assertEqual(
            result['budget_rule'],
            'empirical_quantile_of_same_safety_nocmdp_cost',
        )
        self.assertEqual(result['reference_quantile'], 0.75)
        self.assertEqual(result['recommended_cmdp_cost_limit'], 3.25)
        self.assertEqual(result['reference_budget_exceedance_rate'], 25.0)

    def test_operational_cmdp_signal_is_nonzero_and_budget_is_positive(self):
        episodes = [
            {
                'trajectory': [
                    {'margin': margin}
                    for margin in (0.50, 0.43, 0.40, 0.34, 0.25)
                ]
            }
            for _ in range(30)
        ]
        result = analyse_cmdp_margin_signal(
            episodes,
            margin_targets=[0.40, 0.42, 0.45],
            cost_scale=0.20,
            selected_margin_target=0.42,
            budget_grid=0.5,
        )
        self.assertTrue(result['passed'])
        self.assertEqual(result['selected_margin_target'], 0.42)
        self.assertGreater(
            result['selected_candidate']['cost_activation_rate'], 0.0
        )
        self.assertGreater(result['recommended_cmdp_cost_limit'], 0.0)
        self.assertEqual(
            result['budget_calibration']['budget_rule'],
            'fraction_of_unconstrained_reference_mean',
        )
        self.assertLess(
            result['budget_calibration']['unrounded_target'],
            result['budget_calibration']['mean_ci_upper_bound'],
        )

    def test_threshold_grid_keeps_only_strictly_ordered_candidates(self):
        candidates = generate_threshold_candidates(
            yellow_values=[0.3, 0.4],
            orange_values=[0.3],
            red_values=[0.2, 0.3],
        )
        self.assertEqual(candidates, [{
            'threshold_yellow': 0.4,
            'threshold_orange': 0.3,
            'threshold_red': 0.2,
        }])

    def test_safety_gates_precede_performance_ranking(self):
        unsafe_fast = {
            'candidate_id': 'unsafe-fast',
            'parameters': {'threshold_red': 0.1},
            'metrics': self.metrics(
                physical_constraint_violation_rate=20.0,
                mission_success_rate=99.0,
            ),
        }
        safe_slower = {
            'candidate_id': 'safe-slower',
            'parameters': {'threshold_red': 0.2},
            'metrics': self.metrics(mission_success_rate=60.0),
        }
        selected = select_safety_candidate([unsafe_fast, safe_slower])
        self.assertEqual(
            selected['selection_status'], 'feasible_candidate_selected'
        )
        self.assertEqual(selected['selected_candidate_id'], 'safe-slower')

    def test_no_feasible_candidate_is_reported_explicitly(self):
        candidates = [
            {
                'candidate_id': 'worse',
                'metrics': self.metrics(
                    physical_constraint_violation_rate=10.0
                ),
            },
            {
                'candidate_id': 'less-unsafe',
                'metrics': self.metrics(
                    physical_constraint_violation_rate=6.0
                ),
            },
        ]
        selected = select_safety_candidate(candidates)
        self.assertEqual(selected['selection_status'], 'no_feasible_candidate')
        self.assertEqual(selected['selected_candidate_id'], 'less-unsafe')

    def test_task_completion_floor_rejects_trivially_safe_early_return(self):
        candidate = {
            'candidate_id': 'always-return',
            'metrics': self.metrics(
                safe_return_rate=100.0,
                physical_constraint_violation_rate=0.0,
                task_completion_rate=20.0,
            ),
        }
        selected = select_safety_candidate(
            [candidate], {'min_task_completion_rate': 40.0}
        )
        self.assertEqual(
            selected['selection_status'], 'no_feasible_candidate'
        )
        self.assertFalse(
            selected['candidates'][0]['safety_gate_checks'][
                'task_completion_rate'
            ]
        )

    def test_override_ceiling_rejects_shield_that_replaces_policy(self):
        candidate = {
            'candidate_id': 'always-overridden',
            'metrics': self.metrics(safety_override_rate=95.0),
        }
        selected = select_safety_candidate(
            [candidate], {'max_safety_override_rate': 70.0}
        )
        self.assertEqual(selected['selection_status'], 'no_feasible_candidate')
        self.assertFalse(
            selected['candidates'][0]['safety_gate_checks'][
                'safety_override_rate'
            ]
        )

    def test_soc_infeasibility_does_not_block_primary_calibration(self):
        graded = {'selection_status': 'feasible_candidate_selected'}
        soc = {'selection_status': 'no_feasible_candidate'}
        self.assertEqual(
            calibration_completion_status(graded), 'complete'
        )
        self.assertEqual(soc['selection_status'], 'no_feasible_candidate')

    def test_identifiability_gate_detects_non_degenerate_sweep(self):
        graded = {
            'candidates': [
                {
                    'metrics': self.metrics(
                        mission_success_rate=20.0,
                        task_completion_rate=55.0,
                        safety_override_rate=40.0,
                    )
                },
                {
                    'metrics': self.metrics(
                        mission_success_rate=40.0,
                        task_completion_rate=65.0,
                        safety_override_rate=30.0,
                    )
                },
            ]
        }
        soc = {
            'candidates': [
                {
                    'parameters': {'soc_return_threshold': 0.2},
                    'metrics': self.metrics(
                        safe_return_rate=80.0,
                        physical_constraint_violation_rate=8.0,
                    ),
                },
                {
                    'parameters': {'soc_return_threshold': 0.3},
                    'metrics': self.metrics(
                        safe_return_rate=95.0,
                        physical_constraint_violation_rate=2.0,
                    ),
                },
            ]
        }
        result = assess_calibration_identifiability(
            graded,
            soc,
            {'minimum_distinct_graded_outcome_fraction': 1.0},
        )
        self.assertTrue(result['passed'])
        self.assertEqual(
            calibration_completion_status(
                {'selection_status': 'feasible_candidate_selected'}, result
            ),
            'complete',
        )

    def test_identifiability_gate_rejects_all_zero_mission_results(self):
        metrics = self.metrics(
            mission_success_rate=0.0,
            task_completion_rate=50.0,
        )
        graded = {'candidates': [{'metrics': metrics}, {'metrics': metrics}]}
        soc = {
            'candidates': [
                {
                    'parameters': {'soc_return_threshold': value},
                    'metrics': metrics,
                }
                for value in (0.2, 0.3)
            ]
        }
        result = assess_calibration_identifiability(graded, soc)
        self.assertFalse(result['passed'])
        self.assertFalse(
            result['checks']['mission_success_is_discriminative']
        )
        self.assertFalse(
            result['checks']['soc_frontier_is_non_degenerate']
        )
        self.assertFalse(result['checks']['soc_frontier_is_monotone'])
        self.assertEqual(
            calibration_completion_status(
                {'selection_status': 'feasible_candidate_selected'}, result
            ),
            'failed_identifiability',
        )

    def test_nosafety_ceiling_derives_relative_service_floor(self):
        result = derive_nosafety_service_gates(
            76.4, 81.9, target_fraction=0.90, rounding_grid=1.0
        )
        self.assertEqual(
            result['recommended_min_task_completion_rate'], 68.0
        )
        self.assertEqual(
            result['recommended_min_core_task_completion_rate'], 73.0
        )

    def test_margin_scan_derives_ordered_threshold_and_soc_grids(self):
        episodes = [
            {
                'margin0': 0.10 + 0.01 * index,
                'ret0': 0.20,
                'safe0': 0.08 + 0.001 * index,
                'service_steps': 160,
                'trajectory': [
                    {
                        'step': 0,
                        'margin': 0.40 + 0.005 * index,
                        'soc': 0.75 + 0.002 * index,
                    },
                    {
                        'step': 159,
                        'margin': 0.04 + 0.006 * index,
                        'soc': 0.30 + 0.008 * index,
                    },
                ],
            }
            for index in range(30)
        ]
        result = derive_margin_scan_recommendations(episodes)
        self.assertEqual(len(result['threshold_candidates']), 5)
        for candidate in result['threshold_candidates']:
            self.assertLess(
                candidate['threshold_red'], candidate['threshold_orange']
            )
            self.assertLess(
                candidate['threshold_orange'],
                candidate['threshold_yellow'],
            )
        self.assertGreaterEqual(
            len(result['soc_threshold_candidates']), 4
        )
        self.assertIn('service_boundary_margin', result)
        self.assertIn('service_boundary_soc', result)
        with self.assertRaisesRegex(ValueError, 'at least 30'):
            derive_margin_scan_recommendations(episodes[:29])

    def test_cmdp_identifiability_does_not_require_favorable_direction(self):
        full = {
            'cmdp_discounted_cost': 8.0,
            'cmdp_budget_exceedance_rate': 40.0,
            'physical_constraint_violation_rate': 5.0,
            'safety_override_rate': 20.0,
        }
        no_cmdp = dict(full, cmdp_discounted_cost=9.0)
        result = assess_cmdp_ablation_identifiability(full, no_cmdp)
        self.assertTrue(result['passed'])
        reversed_result = assess_cmdp_ablation_identifiability(no_cmdp, full)
        self.assertTrue(reversed_result['passed'])
        self.assertEqual(
            result['absolute_differences']['cmdp_discounted_cost'], 1.0
        )

    def test_directional_cmdp_gate_rejects_reversed_result(self):
        no_cmdp = {
            'cmdp_discounted_cost': 8.0,
            'cmdp_budget_exceedance_rate': 40.0,
            'physical_constraint_violation_rate': 2.0,
            'safety_override_rate': 30.0,
            'task_completion_rate': 60.0,
        }
        full = dict(
            no_cmdp,
            cmdp_discounted_cost=6.0,
            cmdp_budget_exceedance_rate=30.0,
            task_completion_rate=55.0,
        )
        self.assertTrue(
            assess_cmdp_directional_effect(full, no_cmdp)['passed']
        )
        self.assertFalse(
            assess_cmdp_directional_effect(no_cmdp, full)['passed']
        )

    def test_seed_ranges_must_be_disjoint(self):
        ranges = validate_disjoint_seed_ranges(
            [1000, 2000, 3000], 100, 4000, 50, 5000, 50
        )
        self.assertEqual(len(ranges), 5)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_disjoint_seed_ranges(
                [1000, 1050, 3000], 100, 4000, 50, 5000, 50
            )


if __name__ == '__main__':
    unittest.main()
