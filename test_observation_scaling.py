import unittest

import numpy as np

from agent.lstm_predictor import LSTMEnergyPredictor
from agent.lstm_features import (
    lstm_feature_kwargs,
    lstm_feature_spec,
    lstm_history_gate_weight,
    lstm_state_features,
)
from agent.lstm_runtime import append_lstm_history, append_state_frame
from env.uav_env import UAVEnv
from utils.config import load_config


class ObservationScalingTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config('configs/default.yaml')
        self.env = UAVEnv(self.config)
        self.env.reset(seed=123)

    def test_observation_uses_comparable_channel_and_queue_scales(self):
        obs = self.env._get_obs()
        n = self.env.num_users
        channels = obs[10:10 + n]
        queues = obs[10 + n:10 + 2 * n]
        self.assertTrue(np.all(channels >= -2.0))
        self.assertTrue(np.all(channels <= 0.0))
        self.assertTrue(np.all(queues >= 0.0))
        self.assertTrue(np.all(queues <= 1.0))
        self.assertAlmostEqual(
            float(obs[6]),
            float(self.env.uav.pos[2] / self.env.uav.h_max),
            places=6,
        )

    def test_history_gate_weight_ramps_after_lookback(self):
        weights = lstm_history_gate_weight(
            np.array([9, 10, 30, 50, 80]),
            lookback=10,
            warmup_steps=40,
        )
        np.testing.assert_allclose(weights, [0.0, 0.0, 0.5, 1.0, 1.0])

    def test_lstm_runtime_normalizes_velocity_by_vmax(self):
        predictor = LSTMEnergyPredictor(
            lookback=2,
            horizon=1,
            battery_capacity=self.env.energy.battery_capacity,
        )
        self.env.uav.vel = np.array([
            self.env.uav.v_max,
            -self.env.uav.v_max,
            0.5 * self.env.uav.v_max,
        ])
        append_lstm_history(predictor, self.env, {'E_total': 100.0})
        self.assertEqual(predictor.history_observation_count, 1)
        self.assertEqual(tuple(predictor.history[-1][2:5]), (1.0, -1.0, 0.5))

        frames = []
        append_state_frame(frames, self.env, max_len=2)
        np.testing.assert_allclose(frames[-1][1:4], [1.0, -1.0, 0.5])
        predictor.reset_runtime_state()
        self.assertEqual(predictor.history_observation_count, 0)

    def test_physical_control_features_match_history_and_update_frame(self):
        config = dict(self.config)
        config['lstm_feature_schema'] = 'physical_control_v1'
        env = UAVEnv(config)
        env.reset(seed=321)
        n = env.num_users
        applied = np.linspace(-1.0, 1.0, n + 5, dtype=np.float32)
        info = {
            'E_total': 125.0,
            'E_rem': 0.75 * env.energy.battery_capacity,
            'pos': np.array([1.0, 2.0, 0.5 * env.uav.h_max]),
            'ground_velocity': np.array([
                0.5 * env.uav.v_max, -0.25 * env.uav.v_max, 0.1,
            ]),
            'air_velocity': np.array([
                0.25 * env.uav.v_max, -0.5 * env.uav.v_max, 0.2,
            ]),
            'wind_velocity': np.array([
                0.5 * env.wind.max_speed, -0.25 * env.wind.max_speed, 0.0,
            ]),
            'applied_action': applied,
        }
        predictor = LSTMEnergyPredictor(
            lookback=2, horizon=1,
            battery_capacity=env.energy.battery_capacity,
            **lstm_feature_kwargs(config),
        )
        expected = lstm_state_features(
            env, info, predictor.feature_schema
        )
        append_lstm_history(predictor, env, info)
        frames = []
        append_state_frame(
            frames, env, max_len=2, info=info,
            feature_schema=predictor.feature_schema,
        )

        self.assertEqual(lstm_feature_spec(config)['input_dim'], 15)
        self.assertEqual(len(predictor.history[-1]), predictor.input_dim)
        np.testing.assert_allclose(predictor.history[-1][1:], expected)
        np.testing.assert_allclose(frames[-1], expected)
        self.assertAlmostEqual(float(expected[10]), float(np.mean(
            np.clip((applied[:n] + 1.0) / 2.0, 0.0, 1.0)
        )))

    def test_physical_control_schema_rejects_missing_applied_action(self):
        with self.assertRaisesRegex(ValueError, 'applied_action'):
            lstm_state_features(
                self.env, {'E_total': 1.0}, 'physical_control_v1'
            )


if __name__ == '__main__':
    unittest.main()
