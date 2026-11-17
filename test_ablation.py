import unittest

import numpy as np

from agent.emergency import EmergencyManager
from env.uav_env import UAVEnv
from experiments.ablation import evaluate_variant
from utils.config import load_config


class ZeroPolicy:
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def select_action(self, state, eval=False, **kwargs):
        if not eval:
            raise AssertionError('independent evaluation must use eval=True')
        return np.zeros(self.action_dim, dtype=np.float32)


class AblationExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_config = load_config('configs/default.yaml')

    def test_evaluate_variant_runs_requested_independent_episodes(self):
        config = dict(self.base_config)
        config['num_users'] = 1
        config['max_steps'] = 2
        config['task_interval'] = 1000
        env = UAVEnv(config)
        policy = ZeroPolicy(env.action_space.shape[0])

        summary = evaluate_variant(
            policy,
            env,
            lstm_predictor=None,
            emergency=EmergencyManager(config),
            config=config,
            no_lstm=True,
            no_emergency=True,
            num_episodes=3,
        )

        self.assertEqual(summary['evaluation_episodes'], 3)
        self.assertIn('reward', summary)
        self.assertIn('system_energy', summary)
        self.assertIn('returned_safe', summary)

    def test_evaluate_variant_rejects_empty_evaluation(self):
        config = dict(self.base_config)
        config['num_users'] = 1
        env = UAVEnv(config)
        with self.assertRaisesRegex(ValueError, 'num_eval'):
            evaluate_variant(
                ZeroPolicy(env.action_space.shape[0]),
                env,
                lstm_predictor=None,
                emergency=EmergencyManager(config),
                config=config,
                no_lstm=True,
                no_emergency=True,
                num_episodes=0,
            )


if __name__ == '__main__':
    unittest.main()
