import unittest

import numpy as np

from env.energy_model import EnergyModel
from env.energy_uncertainty import EnergyUncertaintyModel
from env.uav_env import UAVEnv
from utils.config import load_config
from utils.provenance import environment_fingerprint


class EnergyUncertaintyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_config = load_config('configs/default.yaml')

    def uncertainty_config(self):
        config = dict(self.base_config)
        config.update({
            'energy_uncertainty_enabled': True,
            'energy_uncertainty_log_std': 0.20,
            'energy_uncertainty_temporal_correlation': 0.80,
            'energy_uncertainty_min_scale': 0.70,
            'energy_uncertainty_max_scale': 1.40,
            'energy_uncertainty_seed_offset': 15401,
        })
        return config

    def test_disabled_model_is_exactly_deterministic_identity(self):
        model = EnergyUncertaintyModel(self.base_config)
        self.assertEqual(model.reset(42), 1.0)
        self.assertTrue(all(model.advance() == 1.0 for _ in range(10)))
        self.assertEqual(model.return_energy_scale, 1.0)

    def test_same_seed_reproduces_bounded_correlated_sequence(self):
        first = EnergyUncertaintyModel(self.uncertainty_config())
        second = EnergyUncertaintyModel(self.uncertainty_config())
        first_values = [first.reset(123)]
        second_values = [second.reset(123)]
        for _ in range(20):
            first_values.append(first.advance())
            second_values.append(second.advance())
        np.testing.assert_allclose(first_values, second_values)
        self.assertGreaterEqual(min(first_values), first.min_scale)
        self.assertLessEqual(max(first_values), first.max_scale)

    def test_different_seed_changes_only_dedicated_disturbance_stream(self):
        first = EnergyUncertaintyModel(self.uncertainty_config())
        second = EnergyUncertaintyModel(self.uncertainty_config())
        self.assertNotEqual(first.reset(100), second.reset(101))

        reference = UAVEnv(self.base_config)
        uncertain = UAVEnv(self.uncertainty_config())
        reference.reset(seed=77)
        uncertain.reset(seed=77)
        np.testing.assert_allclose(reference.uav.pos, uncertain.uav.pos)
        self.assertEqual(reference.E_rem, uncertain.E_rem)
        np.testing.assert_allclose(
            [user['pos'] for user in reference.task_gen.users],
            [user['pos'] for user in uncertain.task_gen.users],
        )

    def test_energy_model_preserves_nominal_and_applies_actual_scale(self):
        model = EnergyModel(self.base_config)
        E_next, actual, breakdown, _ = model.step_energy(
            80_000.0,
            np.array([5.0, 0.0, 0.0]),
            100.0,
            1.5e9,
            2.0,
            0.5,
            uav_cycles_executed=2.0e8,
            uav_rx_active_time=0.2,
            local_cycles_executed=1.0e8,
            uav_energy_scale=1.20,
        )
        self.assertAlmostEqual(actual, 1.20 * breakdown['E_uav_nominal'])
        self.assertAlmostEqual(
            breakdown['E_uav_uncertainty_delta'],
            actual - breakdown['E_uav_nominal'],
        )
        self.assertAlmostEqual(
            breakdown['E_system_total'],
            actual + breakdown['E_user_total'],
        )
        self.assertAlmostEqual(E_next, 80_000.0 - actual)

    def test_environment_logs_actual_nominal_and_hidden_scale(self):
        env = UAVEnv(self.uncertainty_config())
        env.reset(seed=19)
        env.energy_uncertainty.scale = 1.20
        action = np.zeros(env.action_space.shape[0], dtype=np.float32)
        _, _, _, _, info = env.step(action)
        self.assertTrue(info['energy_uncertainty_enabled'])
        self.assertFalse(info['energy_uncertainty_observed_by_policy'])
        self.assertAlmostEqual(info['energy_uncertainty_scale'], 1.20)
        self.assertAlmostEqual(
            info['E_uav_total'], 1.20 * info['E_uav_nominal']
        )
        self.assertAlmostEqual(
            info['E_uav_uncertainty_delta'],
            info['E_uav_total'] - info['E_uav_nominal'],
        )

    def test_return_checks_use_declared_uncertainty_upper_bound(self):
        config = self.uncertainty_config()
        env = UAVEnv(config)
        env.reset(seed=31)
        horizontal_distance = env.uav.horizontal_distance_to_base()
        return_direction = env.uav.base_pos[:2] - env.uav.pos[:2]
        nominal = env.energy.return_energy(
            horizontal_distance,
            altitude=env.uav.pos[2],
            landing_altitude=env.uav.h_min,
            return_direction=return_direction,
            wind_velocity=env.wind.velocity,
        )
        safety = env.get_energy_context()
        physical = env.get_physical_constraint_context()
        expected = nominal * config['energy_uncertainty_max_scale']
        self.assertAlmostEqual(safety['return_energy'], expected)
        self.assertAlmostEqual(physical['physical_return_energy'], expected)
        self.assertTrue(safety['return_energy_uses_uncertainty_bound'])
        self.assertTrue(
            physical['physical_return_energy_uses_uncertainty_bound']
        )

    def test_uncertainty_parameters_change_environment_fingerprint(self):
        enabled = self.uncertainty_config()
        self.assertNotEqual(
            environment_fingerprint(self.base_config),
            environment_fingerprint(enabled),
        )
        changed = dict(enabled)
        changed['energy_uncertainty_log_std'] = 0.25
        self.assertNotEqual(
            environment_fingerprint(enabled), environment_fingerprint(changed)
        )

    def test_formal_stress_and_matched_ablation_differ_only_by_switch(self):
        enabled = load_config('configs/wind_mission_energy_stress.yaml')
        disabled = load_config(
            'configs/wind_mission_energy_stress_no_uncertainty.yaml'
        )
        self.assertTrue(enabled['energy_uncertainty_enabled'])
        self.assertFalse(disabled['energy_uncertainty_enabled'])
        differences = {
            key for key in set(enabled) | set(disabled)
            if enabled.get(key) != disabled.get(key)
        }
        self.assertEqual(differences, {'energy_uncertainty_enabled'})

    def test_invalid_uncertainty_parameters_fail_fast(self):
        invalid = self.uncertainty_config()
        invalid['energy_uncertainty_temporal_correlation'] = 1.0
        with self.assertRaises(ValueError):
            EnergyUncertaintyModel(invalid)
        invalid = self.uncertainty_config()
        invalid['energy_uncertainty_min_scale'] = 1.1
        with self.assertRaises(ValueError):
            EnergyUncertaintyModel(invalid)


if __name__ == '__main__':
    unittest.main()
