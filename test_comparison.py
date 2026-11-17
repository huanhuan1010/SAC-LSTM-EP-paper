import unittest

from experiments.compare import (
    COMPARISON_PLOT_METRICS,
    _resolved_cmdp_cost_limit,
    aggregate_seed_results,
    comparison_metric_summary,
    summarize_evaluation,
    validate_multiseed_episode_ranges,
)
from utils.evaluation_metrics import (
    conditional_safe_return_rate,
    successful_safety_return,
    update_safety_return_flags,
)


class ComparisonPlotTests(unittest.TestCase):
    def setUp(self):
        self.result = {
            'reward': [1.0, 3.0, 5.0, 7.0],
            'energy': [100.0, 200.0, 300.0, 400.0],
            'system_energy': [120.0, 220.0, 320.0, 420.0],
            'tasks': [1, 2, 3, 4],
            'steps': [10, 20, 30, 40],
            'completed_task_delays': [2.0, 4.0, 6.0],
            'deadline_missed_tasks': 3,
            'deadline_evaluable_tasks': 12,
            'deadline_miss_episodes': 2,
            'deadline_overshoot_s_sum': 1.5,
            'deadline_overshoot_normalized_sum': 0.3,
            'safety_override_steps': 25,
            'battery_depleted_episodes': 1,
            'energy_violation_episodes': 3,
            'physical_constraint_violation_episodes': 2,
            'cmdp_budget_exceeded_episodes': 4,
            'cmdp_budget_exceeded_episodes': 4,
            'return_triggered': 2,
            'returned_safe': 1,
            'safe_return_episodes': 1,
            'mission_success_episodes': 1,
            'mission_safe_abort_episodes': 2,
            'mission_deadline_missed_episodes': 2,
            'task_completion_ratio': [0.5, 0.7, 0.9, 1.0],
            'core_task_completion_ratio': [0.25, 0.75, 1.0, 1.0],
        }

    def test_nine_subplots_cover_abstract_evaluation_metrics(self):
        metric_names = [item[0] for item in COMPARISON_PLOT_METRICS]
        self.assertEqual(len(metric_names), 9)
        self.assertEqual(len(set(metric_names)), 9)
        self.assertIn('tasks', metric_names)
        self.assertIn('mean_task_delay_s', metric_names)
        self.assertIn('system_energy', metric_names)
        self.assertIn('safety_override_rate', metric_names)
        self.assertIn('safe_return_rate', metric_names)
        self.assertIn('energy_violation_rate', metric_names)
        self.assertIn('constraint_exceedance_rate', metric_names)
        self.assertIn('mission_success_rate', metric_names)

    def test_safe_return_plot_uses_all_episodes_as_denominator(self):
        mean, std = comparison_metric_summary(
            self.result, 'safe_return_rate'
        )
        self.assertEqual(mean, 25.0)
        self.assertEqual(std, 0.0)

    def test_trigger_and_conditional_return_rates_remain_distinct(self):
        trigger, _ = comparison_metric_summary(
            self.result, 'return_trigger_rate'
        )
        conditional, _ = comparison_metric_summary(
            self.result, 'returned_safe'
        )
        self.assertEqual(trigger, 50.0)
        self.assertEqual(conditional, 50.0)

    def test_mission_success_plot_uses_all_episodes_as_denominator(self):
        mean, std = comparison_metric_summary(
            self.result, 'mission_success_rate'
        )
        self.assertEqual(mean, 25.0)
        self.assertEqual(std, 0.0)

    def test_strict_mission_support_metrics_use_episode_denominator(self):
        task_rate, _ = comparison_metric_summary(
            self.result, 'task_completion_rate'
        )
        core_rate, _ = comparison_metric_summary(
            self.result, 'core_task_completion_rate'
        )
        abort_rate, _ = comparison_metric_summary(
            self.result, 'mission_safe_abort_rate'
        )
        self.assertAlmostEqual(task_rate, 77.5)
        self.assertEqual(core_rate, 75.0)
        self.assertEqual(abort_rate, 50.0)

    def test_battery_depletion_rate_is_reported_as_percentage(self):
        mean, std = comparison_metric_summary(
            self.result, 'battery_depletion_rate'
        )
        self.assertEqual(mean, 25.0)
        self.assertEqual(std, 0.0)

    def test_primary_safety_rates_share_episode_denominator(self):
        energy, _ = comparison_metric_summary(
            self.result, 'energy_violation_rate'
        )
        constraint, _ = comparison_metric_summary(
            self.result, 'constraint_exceedance_rate'
        )
        self.assertEqual(energy, 75.0)
        self.assertEqual(constraint, 50.0)

    def test_physical_constraint_rate_is_not_cmdp_budget_rate(self):
        physical, _ = comparison_metric_summary(
            self.result, 'constraint_exceedance_rate'
        )
        cmdp, _ = comparison_metric_summary(
            self.result, 'cmdp_budget_exceedance_rate'
        )
        self.assertEqual(physical, 50.0)
        self.assertEqual(cmdp, 100.0)

    def test_evaluation_uses_resolved_algorithm_cmdp_budget(self):
        class AgentWithoutBudget:
            pass

        self.assertEqual(
            _resolved_cmdp_cost_limit(
                AgentWithoutBudget(),
                {'cmdp_cost_limit': 2.75, 'constraint_cost_limit': 10.0},
            ),
            2.75,
        )

        class AgentWithBudget:
            cmdp_cost_limit = 3.25

        self.assertEqual(
            _resolved_cmdp_cost_limit(
                AgentWithBudget(), {'cmdp_cost_limit': 2.75}
            ),
            3.25,
        )

    def test_physical_constraint_rate_is_not_cmdp_budget_rate(self):
        physical, _ = comparison_metric_summary(
            self.result, 'constraint_exceedance_rate'
        )
        cmdp, _ = comparison_metric_summary(
            self.result, 'cmdp_budget_exceedance_rate'
        )
        self.assertEqual(physical, 50.0)
        self.assertEqual(cmdp, 100.0)

    def test_list_metric_keeps_mean_and_standard_deviation(self):
        mean, std = comparison_metric_summary(self.result, 'reward')
        self.assertEqual(mean, 4.0)
        self.assertGreater(std, 0.0)

    def test_delay_and_intervention_metrics_use_correct_denominators(self):
        delay, delay_std = comparison_metric_summary(
            self.result, 'mean_task_delay_s'
        )
        intervention, _ = comparison_metric_summary(
            self.result, 'safety_override_rate'
        )
        self.assertEqual(delay, 4.0)
        self.assertGreater(delay_std, 0.0)
        self.assertEqual(intervention, 25.0)

    def test_task_deadline_metrics_use_task_outcomes_not_episodes(self):
        summary = summarize_evaluation(self.result)
        self.assertEqual(summary['task_deadline_violation_rate'], 25.0)
        self.assertEqual(summary['task_deadline_miss_episode_rate'], 50.0)
        self.assertEqual(summary['mean_deadline_overshoot_s'], 0.5)
        self.assertAlmostEqual(
            summary['mean_normalized_deadline_overshoot'], 0.1
        )

    def test_environment_applied_red_is_authoritative_trigger(self):
        triggered, returned = update_safety_return_flags(
            False,
            False,
            {
                'requested_emergency_mode': 'normal',
                'autonomous_emergency_mode': 'red',
                'safety_mode_applied': 'red',
                'returned_safe': False,
            },
        )
        self.assertTrue(triggered)
        self.assertFalse(returned)

        triggered, returned = update_safety_return_flags(
            triggered,
            returned,
            {'safety_mode_applied': 'normal', 'returned_safe': True},
        )
        self.assertTrue(successful_safety_return(triggered, returned))

    def test_arrival_without_red_trigger_is_not_safety_return_success(self):
        triggered, returned = update_safety_return_flags(
            False,
            False,
            {'safety_mode_applied': 'normal', 'returned_safe': True},
        )
        self.assertFalse(successful_safety_return(triggered, returned))

    def test_safe_return_rate_uses_triggered_episodes_as_denominator(self):
        self.assertEqual(conditional_safe_return_rate(3, 4), 0.75)
        self.assertEqual(conditional_safe_return_rate(0, 0), 0.0)

    def test_multiseed_std_is_computed_from_seed_level_means(self):
        first = dict(self.result)
        second = dict(self.result)
        first['reward'] = [1.0, 3.0]
        second['reward'] = [5.0, 7.0]

        seed_summaries, aggregate = aggregate_seed_results({
            42: first,
            43: second,
        })

        self.assertEqual(seed_summaries['42']['reward'], 2.0)
        self.assertEqual(seed_summaries['43']['reward'], 6.0)
        self.assertEqual(aggregate['reward']['mean'], 4.0)
        self.assertAlmostEqual(aggregate['reward']['std'], 2.82842712)

    def test_multiseed_empty_delay_samples_are_reported_as_missing(self):
        first = dict(self.result)
        second = dict(self.result)
        first['completed_task_delays'] = []
        second['completed_task_delays'] = []

        _, aggregate = aggregate_seed_results({42: first, 43: second})

        self.assertIsNone(aggregate['mean_task_delay_s']['mean'])
        self.assertEqual(
            aggregate['mean_task_delay_s']['values'], [None, None]
        )

    def test_multiseed_validation_rejects_overlapping_episode_ranges(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_multiseed_episode_ranges(
                [42, 43, 44], 2000, 1042, 100
            )
        accepted = validate_multiseed_episode_ranges(
            [100000, 120000, 140000], 2000, 1042, 100
        )
        self.assertEqual(accepted, [100000, 120000, 140000])

    def test_development_runner_can_explicitly_allow_one_seed(self):
        accepted = validate_multiseed_episode_ranges(
            [650000], 250, None, 50,
            evaluation_seeds=[651000],
            minimum_unique_seeds=1,
        )
        self.assertEqual(accepted, [650000])

    def test_independent_evaluation_blocks_require_one_unique_block_per_seed(self):
        accepted = validate_multiseed_episode_ranges(
            [100000, 120000, 140000], 2000, None, 100,
            evaluation_seeds=[200000, 201000, 202000],
        )
        self.assertEqual(accepted, [100000, 120000, 140000])
        with self.assertRaisesRegex(ValueError, 'one block per training seed'):
            validate_multiseed_episode_ranges(
                [100000, 120000, 140000], 2000, None, 100,
                evaluation_seeds=[200000, 201000],
            )
        with self.assertRaisesRegex(ValueError, 'unique'):
            validate_multiseed_episode_ranges(
                [100000, 120000, 140000], 2000, None, 100,
                evaluation_seeds=[200000, 200000, 202000],
            )

    def test_mode_specific_safety_rates_are_reported(self):
        result = dict(self.result)
        result['safety_mode_counts'] = {
            'normal': 50,
            'yellow': 20,
            'orange': 20,
            'red': 10,
        }
        summary = summarize_evaluation(result)
        self.assertEqual(summary['safety_mode_normal_rate'], 50.0)
        self.assertEqual(summary['safety_mode_yellow_rate'], 20.0)
        self.assertEqual(summary['safety_mode_orange_rate'], 20.0)
        self.assertEqual(summary['safety_mode_red_rate'], 10.0)


if __name__ == '__main__':
    unittest.main()
