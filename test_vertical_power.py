import unittest

import numpy as np

from env.energy_model import EnergyModel
from utils.config import load_config


class VerticalPowerTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config('configs/default.yaml')
        self.model = EnergyModel(self.config)

    def test_vertical_model_is_continuous_with_hover(self):
        self.assertAlmostEqual(
            self.model.vertical_flight_power(0.0),
            self.model.flight_power(0.0),
            places=10,
        )

    def test_climb_and_descent_have_distinct_physical_power(self):
        hover = self.model.vertical_flight_power(0.0)
        climb = self.model.vertical_flight_power(3.0)
        descent = self.model.vertical_flight_power(-3.0)

        self.assertGreater(climb, hover)
        self.assertLess(descent, hover)
        self.assertGreater(descent, self.model.P0)

    def test_return_energy_uses_axial_descent_not_hover(self):
        horizontal_distance = 100.0
        altitude = 100.0
        landing_altitude = 30.0
        horizontal_time = horizontal_distance / self.model.return_speed
        descent_time = (
            (altitude - landing_altitude) / self.model.descent_speed
        )
        expected = (
            self.model.flight_power(self.model.return_speed) * horizontal_time
            + self.model.vertical_flight_power(-self.model.descent_speed)
            * descent_time
        )
        old_hover_approximation = (
            self.model.flight_power(self.model.return_speed) * horizontal_time
            + self.model.flight_power(0.0) * descent_time
        )

        actual = self.model.return_energy(
            horizontal_distance,
            altitude=altitude,
            landing_altitude=landing_altitude,
        )
        self.assertAlmostEqual(actual, expected, places=8)
        self.assertNotAlmostEqual(actual, old_hover_approximation, places=4)

    def test_three_dimensional_power_has_correct_axis_limits(self):
        horizontal = np.array([6.0, 8.0, 0.0])
        vertical = np.array([0.0, 0.0, -3.0])
        mixed = np.array([6.0, 8.0, -3.0])
        hover = self.model.flight_power(0.0)

        self.assertAlmostEqual(
            self.model.propulsion_power(horizontal),
            self.model.flight_power(10.0),
        )
        self.assertAlmostEqual(
            self.model.propulsion_power(vertical),
            self.model.vertical_flight_power(-3.0),
        )
        self.assertAlmostEqual(
            self.model.propulsion_power(mixed),
            self.model.flight_power(10.0)
            + self.model.vertical_flight_power(-3.0)
            - hover,
        )

    def test_step_energy_uses_three_dimensional_power(self):
        velocity = np.array([0.0, 0.0, -3.0])
        dt = 0.5
        _, energy, breakdown, _ = self.model.step_energy(
            self.model.battery_capacity,
            velocity,
            100.0,
            1.0e9,
            1.0,
            dt,
        )
        expected = self.model.vertical_flight_power(-3.0) * dt
        self.assertAlmostEqual(breakdown['E_fly'], expected)
        self.assertAlmostEqual(energy, expected)

    def test_invalid_vertical_model_parameters_fail_fast(self):
        bad_v0 = dict(self.config, v0=0.0)
        bad_model = dict(self.config, vertical_power_model='hover_proxy')
        with self.assertRaises(ValueError):
            EnergyModel(bad_v0)
        with self.assertRaises(ValueError):
            EnergyModel(bad_model)


if __name__ == '__main__':
    unittest.main()
