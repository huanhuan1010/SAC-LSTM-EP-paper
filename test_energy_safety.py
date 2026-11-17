import unittest

import numpy as np
import torch

from agent.emergency import EmergencyManager
from agent.lstm_predictor import LSTMEnergyPredictor
from agent.lstm_runtime import observation_for_agent
from env.energy_model import EnergyModel
from env.uav_env import UAVEnv
from utils.config import load_config


class EnergySafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config('configs/default.yaml')

    def test_energy_margin_subtracts_all_required_energy(self):
        model = EnergyModel(self.config)
        reserve = model.safety_reserve()
        margin, ratio = model.energy_margin(
            80_000.0,
            predicted_energy=2_000.0,
            return_energy=3_000.0,
            safety_reserve=reserve,
        )
        self.assertAlmostEqual(margin, 80_000.0 - 2_000.0 - 3_000.0 - reserve)
        self.assertAlmostEqual(ratio, margin / model.battery_capacity)

    def test_minimum_safe_energy_contains_floor_and_is_load_aware(self):
        model = EnergyModel(self.config)
        hover_power = model.flight_power(0.0)
        nominal = model.safety_reserve(average_power=hover_power)
        high_load = model.safety_reserve(average_power=3.0 * hover_power)

        expected_nominal = (
            model.E_min
            + model.safety_reserve_ratio * model.battery_capacity
        )
        self.assertAlmostEqual(nominal, expected_nominal)
        self.assertGreater(high_load, nominal)
        self.assertLessEqual(high_load, model.battery_capacity)

    def test_idle_slot_only_charges_uav_flight_energy(self):
        model = EnergyModel(self.config)
        E_rem = 80_000.0
        dt = self.config['dt']
        E_next, E_uav, breakdown, _ = model.step_energy(
            E_rem,
            np.zeros(3),
            100.0,
            model.f0,
            5.0,
            dt,
        )

        self.assertAlmostEqual(E_uav, model.flight_power(0.0) * dt)
        self.assertEqual(breakdown['E_comp'], 0.0)
        self.assertEqual(breakdown['E_comm'], 0.0)
        self.assertEqual(breakdown['E_user_total'], 0.0)
        self.assertAlmostEqual(E_next, E_rem - E_uav)

    def test_energy_is_proportional_to_actual_workload(self):
        model = EnergyModel(self.config)
        frequency = 1.5e9
        uav_cycles = 0.25 * frequency
        local_cycles = 2.0e8
        uplink_time = 0.2
        rx_time = 0.15
        E_rem = 80_000.0

        E_next, E_uav, breakdown, _ = model.step_energy(
            E_rem,
            np.zeros(3),
            100.0,
            frequency,
            2.0,
            0.5,
            uav_cycles_executed=uav_cycles,
            uplink_time_sum=uplink_time,
            uav_rx_active_time=rx_time,
            local_cycles_executed=local_cycles,
        )

        self.assertAlmostEqual(
            breakdown['E_comp'], model.kappa * frequency**2 * uav_cycles
        )
        self.assertAlmostEqual(breakdown['E_comm'], model.p_rx_uav * rx_time)
        self.assertAlmostEqual(
            breakdown['E_user_local'],
            model.kappa_user * model.f_user**2 * local_cycles,
        )
        self.assertAlmostEqual(
            breakdown['E_user_comm'],
            model.communication_power(2.0) * uplink_time,
        )
        self.assertAlmostEqual(
            breakdown['E_system_total'],
            breakdown['E_uav_total'] + breakdown['E_user_total'],
        )
        self.assertAlmostEqual(E_next, E_rem - E_uav)

    def test_energy_margin_time_is_zero_when_short_horizon_condition_fails(self):
        model = EnergyModel(self.config)
        self.assertAlmostEqual(model.energy_margin_time(1_000.0, 200.0), 5.0)
        self.assertEqual(model.energy_margin_time(-1_000.0, 200.0), 0.0)

    def test_emergency_uses_margin_instead_of_raw_soc(self):
        manager = EmergencyManager(self.config)
        mode, constraints = manager.check(
            E_rem=60_000.0,
            E_pred=np.array([5_000.0, 6_000.0]),
            return_energy=20_000.0,
            safety_reserve=10_000.0,
        )
        self.assertEqual(mode, 'red')
        self.assertAlmostEqual(constraints['energy_margin_ratio'], 0.19)
        self.assertTrue(constraints['short_horizon_energy_feasible'])
        self.assertEqual(constraints['prediction_horizon_steps'], 2)
        self.assertAlmostEqual(
            constraints['prediction_horizon_seconds'], 2 * self.config['dt']
        )
        self.assertAlmostEqual(constraints['lambda_max'], 0.2)
        self.assertAlmostEqual(constraints['speed_min'], 0.5)
        self.assertAlmostEqual(constraints['speed_max'], 0.5)
        self.assertAlmostEqual(constraints['cpu_max'], 0.25)
        self.assertAlmostEqual(constraints['p_tx_max'], 0.25)
        self.assertNotIn('mission_feasible', constraints)

    def test_environment_rejects_unordered_safety_thresholds(self):
        config = dict(self.config)
        config.update({
            'threshold_yellow': 0.20,
            'threshold_orange': 0.30,
            'threshold_red': 0.40,
        })
        with self.assertRaisesRegex(ValueError, 'red < orange < yellow'):
            UAVEnv(config)

    def test_lstm_energy_prediction_is_nonnegative(self):
        predictor = LSTMEnergyPredictor(
            lookback=10, horizon=5,
            battery_capacity=self.config['battery_capacity'],
        )
        prediction = predictor(torch.zeros(10, 6))
        self.assertTrue(torch.all(prediction >= 0))

    def test_persistence_residual_lstm_starts_at_causal_baseline(self):
        predictor = LSTMEnergyPredictor(
            lookback=3,
            horizon=2,
            battery_capacity=self.config['battery_capacity'],
            prediction_mode='persistence_residual',
        )
        history = torch.zeros(2, 3, 6)
        history[:, -1, 0] = torch.tensor([0.0010, 0.0015])

        prediction = predictor(history)

        expected = torch.tensor([[0.0010, 0.0010], [0.0015, 0.0015]])
        torch.testing.assert_close(prediction, expected)

    def test_persistence_residual_update_keeps_positive_gradient_path(self):
        predictor = LSTMEnergyPredictor(
            lookback=3,
            horizon=2,
            battery_capacity=self.config['battery_capacity'],
            prediction_mode='persistence_residual',
        )
        history = torch.zeros(2, 3, 6)
        history[:, -1, 0] = 0.001
        target = torch.full((2, 2), 80.0)

        predictor.update(history, target)
        prediction = predictor(history)

        self.assertTrue(torch.all(prediction > 0.0))
        self.assertTrue(torch.all(prediction < 0.001))

    def test_lstm_rejects_unknown_prediction_mode(self):
        with self.assertRaisesRegex(ValueError, 'prediction_mode'):
            LSTMEnergyPredictor(prediction_mode='unknown')

    def test_lstm_cold_start_uses_nonzero_physical_bootstrap(self):
        env = UAVEnv(self.config)
        state, _ = env.reset(seed=19)
        predictor = LSTMEnergyPredictor(
            lookback=10, horizon=5,
            battery_capacity=self.config['battery_capacity'],
        )

        state_aug, prediction = observation_for_agent(
            state, env, predictor, self.config, use_lstm=True
        )
        context = env.get_energy_context()

        self.assertTrue(np.all(prediction > 0.0))
        self.assertEqual(context['prediction_source'], 'hover_bootstrap')
        self.assertGreater(context['predicted_energy'], 0.0)
        self.assertTrue(np.any(state_aug != state))

    def test_lstm_prediction_uses_recent_residual_correction(self):
        predictor = LSTMEnergyPredictor(
            lookback=10, horizon=5,
            battery_capacity=self.config['battery_capacity'],
            residual_correction=True,
        )
        predictor.history.extend([
            (100.0, 0.8, 0.0, 0.0, 0.0, 0.3)
            for _ in range(predictor.lookback)
        ])
        first = predictor.get_energy_features()
        predictor.observe_actual_energy(float(first[0]) + 50.0)
        corrected = predictor.get_energy_features()

        self.assertAlmostEqual(predictor.residual_bias_joule, 50.0, places=4)
        np.testing.assert_allclose(
            corrected,
            predictor.last_raw_prediction_joule + 50.0,
            rtol=0.0,
            atol=1e-3,
        )

    def test_disabled_bias_correction_still_tracks_uncertainty(self):
        predictor = LSTMEnergyPredictor(
            lookback=10,
            horizon=5,
            battery_capacity=self.config['battery_capacity'],
            prediction_mode='persistence_residual',
            residual_correction=False,
        )
        predictor.history.extend([
            (100.0, 0.8, 0.0, 0.0, 0.0, 0.3)
            for _ in range(predictor.lookback)
        ])
        first = predictor.get_energy_features()
        predictor.observe_actual_energy(float(first[0]) + 50.0)
        runtime_output = predictor.get_energy_features()

        self.assertAlmostEqual(predictor.residual_bias_joule, 50.0, places=4)
        self.assertAlmostEqual(
            predictor.residual_abs_ema_joule, 50.0, places=4
        )
        np.testing.assert_allclose(
            runtime_output,
            predictor.last_raw_prediction_joule,
            rtol=0.0,
            atol=1e-6,
        )

    def test_environment_reports_scoped_short_horizon_energy_condition(self):
        env = UAVEnv(self.config)
        env.reset(seed=17)
        env.set_energy_prediction(
            np.full(env.horizon, 100.0),
            raw_prediction=np.full(env.horizon, 90.0),
            residual_bias=10.0,
            residual_uncertainty=5.0,
        )
        context = env.get_energy_context()

        self.assertEqual(
            context['short_horizon_energy_feasible'],
            context['energy_margin'] >= 0.0,
        )
        self.assertNotIn('mission_feasible', context)
        self.assertEqual(
            context['energy_feasibility_scope'],
            'prediction_horizon_plus_current_position_return',
        )
        self.assertTrue(context['return_energy_from_current_position'])
        self.assertAlmostEqual(
            context['prediction_horizon_seconds'], env.horizon * env.dt
        )
        self.assertGreaterEqual(
            context['minimum_safe_energy'], env.energy.E_min
        )
        self.assertAlmostEqual(
            context['energy_after_predicted_load_and_return'],
            env.E_rem - context['predicted_energy'] - context['return_energy'],
        )
        self.assertAlmostEqual(
            context['required_energy_with_reserve'],
            context['predicted_energy']
            + context['return_energy']
            + context['safety_reserve'],
        )
        self.assertAlmostEqual(
            context['estimated_energy_margin_time'],
            max(context['energy_margin'], 0.0)
            / context['average_power_estimate'],
        )

        env.E_rem = context['required_energy_with_reserve'] - 1.0
        infeasible_context = env.get_energy_context()
        self.assertFalse(
            infeasible_context['short_horizon_energy_feasible']
        )
        self.assertAlmostEqual(infeasible_context['energy_margin'], -1.0)
        self.assertEqual(
            infeasible_context['estimated_energy_margin_time'], 0.0
        )

    def test_red_mode_overrides_heading_toward_base(self):
        env = UAVEnv(self.config)
        env.reset(seed=7)
        env.uav.pos = np.array([400.0, 250.0, 100.0])
        env.uav.vel = np.zeros(3)
        env.emergency_mode = 'red'
        action = np.zeros(env.action_space.shape[0], dtype=np.float32)
        x_before = env.uav.pos[0]
        env.step(action)
        self.assertLess(env.uav.pos[0], x_before)

    def test_environment_hard_projects_every_red_resource_dimension(self):
        env = UAVEnv(self.config)
        env.reset(seed=29)
        env.uav.pos = np.array([400.0, 250.0, 100.0])
        env.uav.vel = np.zeros(3)
        env.emergency_mode = 'red'
        action = np.ones(env.action_space.shape[0], dtype=np.float32)
        x_before = env.uav.pos[0]

        _, _, _, _, info = env.step(action)
        applied = info['applied_action']

        self.assertEqual(info['safety_mode_applied'], 'red')
        self.assertTrue(info['safety_action_overridden'])
        self.assertTrue(np.all(applied[:env.num_users] <= -0.6 + 1e-6))
        self.assertAlmostEqual(applied[env.num_users + 2], 0.0, places=6)
        self.assertLessEqual(applied[env.num_users + 3], -0.5 + 1e-6)
        self.assertLessEqual(applied[env.num_users + 4], -0.5 + 1e-6)
        self.assertLess(env.uav.pos[0], x_before)

    def test_environment_autonomously_enforces_red_mode(self):
        env = UAVEnv(self.config)
        env.reset(seed=31)
        context = env.get_energy_context()
        env.E_rem = (
            context['required_energy_with_reserve']
            + 0.10 * env.energy.battery_capacity
        )
        env.emergency_mode = 'normal'

        action = np.ones(env.action_space.shape[0], dtype=np.float32)
        _, _, _, _, info = env.step(action)

        self.assertEqual(info['requested_emergency_mode'], 'normal')
        self.assertEqual(info['autonomous_emergency_mode'], 'red')
        self.assertEqual(info['safety_mode_applied'], 'red')

    def test_disabled_supervisor_does_not_force_emergency_projection(self):
        env = UAVEnv(self.config)
        env.reset(seed=37)
        env.set_energy_safety_supervisor(False)
        context = env.get_energy_context()
        env.E_rem = (
            context['required_energy_with_reserve']
            + 0.10 * env.energy.battery_capacity
        )
        action = np.ones(env.action_space.shape[0], dtype=np.float32)

        _, _, _, _, first = env.step(action)
        _, _, _, _, second = env.step(action)

        for info in (first, second):
            self.assertFalse(info['energy_safety_supervisor_enabled'])
            self.assertEqual(info['autonomous_emergency_mode'], 'normal')
            self.assertEqual(info['requested_emergency_mode'], 'normal')
            self.assertEqual(info['safety_mode_applied'], 'normal')
            np.testing.assert_allclose(info['applied_action'], action)

    def test_safe_return_uses_horizontal_distance(self):
        env = UAVEnv(self.config)
        env.reset(seed=11)
        env.uav.pos = np.array([
            env.uav.base_pos[0], env.uav.base_pos[1], env.uav.h_min,
        ])
        env.uav.vel = np.zeros(3)
        action = -np.ones(env.action_space.shape[0], dtype=np.float32)
        _, _, done, _, info = env.step(action)
        self.assertTrue(done)
        self.assertTrue(info['returned_safe'])


if __name__ == '__main__':
    unittest.main()
