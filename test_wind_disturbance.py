import unittest

import numpy as np

from env.energy_model import EnergyModel
from env.uav_env import UAVEnv
from env.wind_model import WindModel
from utils.config import load_config
from utils.provenance import environment_fingerprint


class WindDisturbanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_config = load_config('configs/default.yaml')

    def wind_config(self):
        config = dict(self.base_config)
        config.update({
            'wind_enabled': True,
            'wind_mean_speed': 4.0,
            'wind_speed_std': 1.5,
            'wind_temporal_correlation': 0.9,
            'wind_max_speed': 8.0,
            'wind_direction_deg': None,
        })
        return config

    def test_default_reference_environment_has_no_wind(self):
        model = WindModel(self.base_config)
        np.testing.assert_allclose(model.reset(42), np.zeros(3))
        np.testing.assert_allclose(model.advance(), np.zeros(3))

    def test_same_seed_reproduces_full_wind_sequence(self):
        first = WindModel(self.wind_config())
        second = WindModel(self.wind_config())
        first_sequence = [first.reset(123)]
        second_sequence = [second.reset(123)]
        for _ in range(10):
            first_sequence.append(first.advance())
            second_sequence.append(second.advance())
        np.testing.assert_allclose(first_sequence, second_sequence)

    def test_different_seeds_change_wind_but_not_base_scenario(self):
        first = WindModel(self.wind_config())
        second = WindModel(self.wind_config())
        self.assertFalse(np.allclose(first.reset(100), second.reset(101)))

        calm_env = UAVEnv(self.base_config)
        windy_env = UAVEnv(self.wind_config())
        calm_env.reset(seed=77)
        windy_env.reset(seed=77)
        np.testing.assert_allclose(calm_env.uav.pos, windy_env.uav.pos)
        np.testing.assert_allclose(
            [user['pos'] for user in calm_env.task_gen.users],
            [user['pos'] for user in windy_env.task_gen.users],
        )

    def test_wind_changes_required_air_velocity_and_flight_energy(self):
        config = self.wind_config()
        config.update({
            'wind_mean_speed': 4.0,
            'wind_speed_std': 0.0,
            'wind_direction_deg': 0.0,
        })
        env = UAVEnv(config)
        env.reset(seed=19)
        action = np.zeros(env.action_space.shape[0], dtype=np.float32)
        _, _, _, _, info = env.step(action)

        np.testing.assert_allclose(
            info['ground_velocity'],
            info['air_velocity'] + info['wind_velocity'],
        )
        expected_flight_energy = (
            env.energy.propulsion_power(info['air_velocity'], env.uav.pos[2])
            * env.dt
        )
        self.assertAlmostEqual(
            info['energy_breakdown']['E_fly'], expected_flight_energy
        )
        self.assertAlmostEqual(info['wind_speed'], 4.0)

    def test_return_estimate_is_direction_and_wind_aware(self):
        model = EnergyModel(self.base_config)
        distance = 200.0
        direction = np.array([1.0, 0.0])
        calm = model.return_energy(
            distance,
            return_direction=direction,
            wind_velocity=np.zeros(2),
        )
        headwind = model.return_energy(
            distance,
            return_direction=direction,
            wind_velocity=np.array([-5.0, 0.0]),
        )
        tailwind = model.return_energy(
            distance,
            return_direction=direction,
            wind_velocity=np.array([5.0, 0.0]),
        )
        self.assertGreater(headwind, calm)
        self.assertGreater(headwind, tailwind)
        # Rotary-wing power is U-shaped in airspeed. At a fixed 10 m/s ground
        # return speed, a tailwind can move airspeed away from the minimum-power
        # point, so it is not generally valid to assert tailwind < calm.
        self.assertNotAlmostEqual(calm, tailwind)

    def test_wind_parameters_are_part_of_environment_fingerprint(self):
        windy = self.wind_config()
        self.assertNotEqual(
            environment_fingerprint(self.base_config),
            environment_fingerprint(windy),
        )

    def test_invalid_wind_parameters_fail_fast(self):
        config = self.wind_config()
        config['wind_temporal_correlation'] = 1.0
        with self.assertRaises(ValueError):
            WindModel(config)


if __name__ == '__main__':
    unittest.main()
