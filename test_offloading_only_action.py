import unittest

import numpy as np

from agent.lstm_features import (
    PHYSICAL_CONTROL_SCHEMA,
    lstm_feature_input_dim,
    lstm_state_features,
)
from baselines.greedy import GreedyPolicy
from env.uav_env import UAVEnv
from utils.config import load_config


class OffloadingOnlyActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_config = load_config('configs/default.yaml')

    def make_env(self, **overrides):
        config = dict(self.base_config)
        config.update({
            'num_users': 2,
            'policy_action_mode': 'offloading_only',
            'common_mobility_speed': 0.0,
            'task_interval': 1000,
        })
        config.update(overrides)
        env = UAVEnv(config)
        env.reset(seed=31)
        for user in env.task_gen.users:
            user['queue'] = []
        return env

    def test_compact_policy_action_expands_to_full_physical_action(self):
        env = self.make_env()
        env.set_energy_safety_supervisor(False)
        requested = np.array(
            [-0.5, 0.5, 0.25, -0.25], dtype=np.float32
        )

        _, _, _, _, info = env.step(requested)

        self.assertEqual(env.action_space.shape, (env.num_users + 2,))
        np.testing.assert_allclose(info['raw_action'], requested)
        np.testing.assert_allclose(info['applied_action'], requested)
        self.assertEqual(
            info['full_applied_action'].shape, (env.num_users + 5,)
        )
        np.testing.assert_allclose(
            info['full_raw_action'][:env.num_users],
            requested[:env.num_users],
        )
        self.assertAlmostEqual(info['full_raw_action'][env.num_users], 0.0)
        self.assertAlmostEqual(
            info['full_raw_action'][env.num_users + 1], 0.0
        )
        self.assertAlmostEqual(
            info['full_raw_action'][env.num_users + 2], -1.0
        )
        np.testing.assert_allclose(
            info['full_raw_action'][env.num_users + 3:],
            requested[env.num_users:],
        )

    def test_physical_lstm_features_use_full_executed_action(self):
        env = self.make_env()
        env.set_energy_safety_supervisor(False)
        requested = np.zeros(env.action_space.shape[0], dtype=np.float32)

        _, _, _, _, info = env.step(requested)
        features = lstm_state_features(
            env, info, schema=PHYSICAL_CONTROL_SCHEMA
        )

        self.assertEqual(
            features.size,
            lstm_feature_input_dim(PHYSICAL_CONTROL_SCHEMA) - 1,
        )
        # The applied speed feature reflects the shared hover controller,
        # not the compact policy's CPU coordinate.
        self.assertAlmostEqual(float(features[-3]), 0.0)

    def test_red_takeover_changes_full_mobility_not_action_dimension(self):
        env = self.make_env()
        env.emergency_mode = 'red'
        requested = np.ones(env.action_space.shape[0], dtype=np.float32)

        _, _, _, _, info = env.step(requested)

        self.assertEqual(info['raw_action'].shape, requested.shape)
        self.assertEqual(info['applied_action'].shape, requested.shape)
        self.assertEqual(
            info['full_applied_action'].shape, (env.num_users + 5,)
        )
        self.assertEqual(info['safety_mode_applied'], 'red')
        self.assertTrue(info['safety_action_overridden'])
        self.assertGreater(info['full_safety_projection_l2'], 0.0)

    def test_greedy_emits_compact_action_under_matched_mobility(self):
        env = self.make_env()
        state, _ = env.reset(seed=37)
        policy = GreedyPolicy(
            env.observation_space.shape[0],
            env.action_space.shape[0],
            env.config,
        )

        action = policy.select_action(state)

        self.assertEqual(action.shape, env.action_space.shape)
        self.assertTrue(np.all(action >= -1.0))
        self.assertTrue(np.all(action <= 1.0))


if __name__ == '__main__':
    unittest.main()
