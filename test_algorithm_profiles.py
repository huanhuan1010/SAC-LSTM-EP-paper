import unittest

import yaml

from utils.algorithm_profiles import (
    BASELINE_ALGORITHMS,
    MATCHED_CONTROL_ALGORITHMS,
    PREDICTION_ABLATION_ALGORITHMS,
    PRIMARY_ABLATION_ALGORITHMS,
    PAPER_CMDP_VARIANT,
    PAPER_CORE_ALGORITHMS,
    PAPER_MAIN_ALGORITHM,
    PAPER_PREDICTION_CONTROLS,
    algorithm_config,
    algorithm_profile,
    paper_algorithm_label,
)


class AlgorithmProfileTests(unittest.TestCase):
    def test_paper_main_profiles_preserve_legacy_results(self):
        self.assertEqual(
            algorithm_profile(PAPER_MAIN_ALGORITHM),
            algorithm_profile('SAC-LSTM-EP-NoCMDP'),
        )
        self.assertEqual(
            algorithm_profile(PAPER_CMDP_VARIANT),
            algorithm_profile('SAC-LSTM-EP'),
        )
        self.assertEqual(
            PAPER_CORE_ALGORITHMS,
            (
                'SAC-LSTM-EP-PaperMain',
                'SAC-LSTM-EP-NoPrediction',
                'SAC-LSTM-EP-FixedSOC',
                'SAC-LSTM-EP-CMDP',
            ),
        )

    def test_paper_prediction_controls_are_strictly_matched(self):
        main = algorithm_profile(PAPER_MAIN_ALGORITHM)
        no_prediction = algorithm_profile(PAPER_PREDICTION_CONTROLS[0])
        fixed_soc = algorithm_profile(PAPER_PREDICTION_CONTROLS[1])
        self.assertFalse(main.use_cmdp)
        self.assertTrue(main.use_lstm)
        self.assertEqual(main.prediction_source, 'lstm')
        self.assertEqual(main.safety_controller, 'margin')
        self.assertFalse(no_prediction.use_cmdp)
        self.assertFalse(no_prediction.use_lstm)
        self.assertEqual(no_prediction.prediction_source, 'current')
        self.assertEqual(no_prediction.safety_controller, 'margin')
        self.assertFalse(fixed_soc.use_cmdp)
        self.assertFalse(fixed_soc.use_lstm)
        self.assertEqual(fixed_soc.prediction_source, 'current')
        self.assertEqual(fixed_soc.safety_controller, 'soc_threshold')

    def test_paper_labels_do_not_change_internal_identifiers(self):
        self.assertEqual(
            paper_algorithm_label('SAC-LSTM-EP-PaperMain'),
            'SAC-LSTM-EP',
        )
        self.assertEqual(
            paper_algorithm_label('SAC-LSTM-EP-CMDP'),
            'SAC-LSTM-EP-CMDP',
        )
        self.assertEqual(paper_algorithm_label('DDPG'), 'DDPG')

    def test_primary_incremental_ablation_is_explicit(self):
        plain = algorithm_profile('SAC')
        self.assertFalse(plain.use_lstm)
        self.assertFalse(plain.use_cmdp)
        self.assertEqual(plain.safety_controller, 'none')

        self.assertTrue(algorithm_profile('SAC+LSTM').use_lstm)
        self.assertTrue(algorithm_profile('SAC+CMDP').use_cmdp)
        self.assertEqual(
            algorithm_profile('SAC+SafetyProjection').safety_controller,
            'margin',
        )
        self.assertIn('SAC-LSTM-EP-NoSafety', PRIMARY_ABLATION_ALGORITHMS)
        self.assertNotIn('SAC+DynamicWeight', PRIMARY_ABLATION_ALGORITHMS)
        with self.assertRaises(ValueError):
            algorithm_profile('SAC+DynamicWeight')

    def test_yaml_registry_matches_runtime_primary_variants(self):
        with open('configs/ablation.yaml', encoding='utf-8') as stream:
            registry = yaml.safe_load(stream)
        self.assertEqual(
            tuple(registry['primary_variants']),
            PRIMARY_ABLATION_ALGORITHMS,
        )
        for name in PRIMARY_ABLATION_ALGORITHMS:
            self.assertIn(name, registry['definitions'])

    def test_full_no_safety_differs_only_in_controller(self):
        full = algorithm_profile('SAC-LSTM-EP')
        no_safety = algorithm_profile('SAC-LSTM-EP-NoSafety')
        self.assertEqual(full.use_lstm, no_safety.use_lstm)
        self.assertEqual(full.use_cmdp, no_safety.use_cmdp)
        self.assertEqual(no_safety.safety_controller, 'none')
        self.assertTrue(no_safety.offline_only)
        self.assertTrue(no_safety.observe_margin_state)

    def test_matched_controls_change_only_the_declared_component(self):
        full = algorithm_profile('SAC-LSTM-EP')
        soc = algorithm_profile('SAC-LSTM-EP-SOC')
        predictive = algorithm_profile('SAC-LSTM-EP-PredictiveReturn')
        no_cmdp = algorithm_profile('SAC-LSTM-EP-NoCMDP')
        single = algorithm_profile('SAC-LSTM-EP-SingleLevel')

        for profile in (soc, predictive, no_cmdp, single):
            self.assertTrue(profile.use_lstm)
            self.assertEqual(profile.prediction_source, 'lstm')
            self.assertFalse(profile.offline_only)

        self.assertTrue(soc.use_cmdp)
        self.assertTrue(predictive.use_cmdp)
        self.assertEqual(soc.safety_controller, 'soc_threshold')
        self.assertEqual(
            predictive.safety_controller, 'margin_threshold'
        )

        self.assertFalse(no_cmdp.use_cmdp)
        self.assertEqual(no_cmdp.safety_controller, full.safety_controller)

        self.assertTrue(single.use_cmdp)
        self.assertEqual(single.safety_controller, 'margin_single_level')
        self.assertEqual(
            set(MATCHED_CONTROL_ALGORITHMS),
            {
                'SAC-LSTM-EP-SOC',
                'SAC-LSTM-EP-PredictiveReturn',
                'SAC-LSTM-EP-NoCMDP',
                'SAC-LSTM-EP-SingleLevel',
            },
        )
        for name in MATCHED_CONTROL_ALGORITHMS:
            self.assertIn(name, PRIMARY_ABLATION_ALGORITHMS)

    def test_soc_and_oracle_baselines_are_unambiguous(self):
        self.assertEqual(
            algorithm_profile('SAC-SOCThresholdReturn').safety_controller,
            'soc_threshold',
        )
        oracle = algorithm_profile('SAC-Oracle')
        self.assertTrue(oracle.use_oracle_prediction)
        self.assertTrue(oracle.offline_only)

    def test_algorithm_config_sets_cmdp_from_profile(self):
        base = {'cmdp_enabled': True}
        self.assertFalse(algorithm_config(base, 'SAC')['cmdp_enabled'])
        self.assertTrue(algorithm_config(base, 'SAC+CMDP')['cmdp_enabled'])
        self.assertFalse(
            algorithm_config(base, 'SAC-LSTM-EP-NoCMDP')['cmdp_enabled']
        )

    def test_alvi_adapted_baseline_uses_latency_cmdp_only(self):
        profile = algorithm_profile('C-SAC-Alvi-Adapted')
        self.assertTrue(profile.use_cmdp)
        self.assertFalse(profile.use_lstm)
        self.assertEqual(profile.safety_controller, 'none')
        self.assertEqual(profile.cmdp_cost_mode, 'latency_violation')
        self.assertEqual(
            algorithm_config({}, 'C-SAC-Alvi-Adapted')['cmdp_cost_mode'],
            'latency_violation',
        )
        self.assertEqual(
            algorithm_config({}, 'SAC-LSTM-EP')['cmdp_cost_mode'],
            'energy_margin',
        )
        self.assertIn('C-SAC-Alvi-Adapted', BASELINE_ALGORITHMS)

    def test_algorithm_specific_budget_does_not_change_full(self):
        base = {
            'cmdp_cost_limit': 10.0,
            'algorithm_overrides': {
                'C-SAC-Alvi-Adapted': {'cmdp_cost_limit': 2.75},
            },
        }
        self.assertEqual(
            algorithm_config(base, 'C-SAC-Alvi-Adapted')[
                'cmdp_cost_limit'
            ],
            2.75,
        )
        self.assertEqual(
            algorithm_config(base, 'SAC-LSTM-EP')['cmdp_cost_limit'],
            10.0,
        )

    def test_algorithm_override_cannot_change_component_definition(self):
        base = {
            'algorithm_overrides': {
                'C-SAC-Alvi-Adapted': {
                    'cmdp_cost_mode': 'energy_margin',
                },
            },
        }
        with self.assertRaises(ValueError):
            algorithm_config(base, 'C-SAC-Alvi-Adapted')

    def test_prediction_ablation_changes_only_prediction_source(self):
        expected_sources = {
            'SAC-Current': 'current',
            'SAC-Persistence-EP': 'persistence',
            'SAC-Shuffled-EP': 'shuffled_lstm',
            'SAC-LSTM-EP': 'lstm',
        }
        self.assertEqual(
            tuple(expected_sources), PREDICTION_ABLATION_ALGORITHMS
        )
        for name, source in expected_sources.items():
            profile = algorithm_profile(name)
            self.assertTrue(profile.use_cmdp)
            self.assertEqual(profile.safety_controller, 'margin')
            self.assertEqual(profile.prediction_source, source)
            self.assertEqual(
                profile.use_prediction,
                source != 'current',
            )

    def test_traditional_baselines_do_not_inherit_proposed_modules(self):
        for name in ('PPO', 'DDPG', 'Greedy', 'Random'):
            profile = algorithm_profile(name)
            self.assertFalse(profile.use_lstm)
            self.assertFalse(profile.use_energy_safety_supervisor)

    def test_ddpg_matched_safety_is_an_explicit_fairness_control(self):
        plain = algorithm_profile('DDPG')
        matched = algorithm_profile('DDPG-MatchedSafety')
        self.assertFalse(plain.use_lstm)
        self.assertTrue(matched.use_lstm)
        self.assertEqual(matched.prediction_source, 'lstm')
        self.assertEqual(matched.safety_controller, 'margin')
        self.assertFalse(matched.use_cmdp)
        self.assertIn('DDPG-MatchedSafety', BASELINE_ALGORITHMS)


if __name__ == '__main__':
    unittest.main()
