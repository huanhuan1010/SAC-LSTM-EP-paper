import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from agent.sac import SAC
from baselines.ddpg import DDPG
from baselines.ppo import PPO
from env.uav_env import UAVEnv
from utils.config import load_config
from utils.logger import Logger
from utils.safety_constraints import project_normalized_action


class UnifiedActionProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config('configs/default.yaml')

    def test_sac_and_ddpg_return_raw_actions_despite_constraint_kwargs(self):
        config = {
            'num_users': 2,
            'batch_size': 2,
            'buffer_capacity': 16,
            'warmup_steps': 5,
            'cmdp_enabled': False,
        }
        expected = np.linspace(-0.8, 0.8, 7, dtype=np.float32)

        sac = SAC(4, 7, config)
        # SAC owns an isolated exploration stream; patch the instance stream
        # rather than NumPy's global RNG.
        with patch.object(sac, 'exploration_rng') as exploration_rng:
            exploration_rng.uniform.return_value = expected.copy()
            sac_action = sac.select_action(
                np.zeros(4, dtype=np.float32),
                lambda_max=0.0,
                speed_max=0.0,
            )
        np.testing.assert_allclose(sac_action, expected)

        ddpg = DDPG(4, 7, config)
        with patch(
            'baselines.ddpg.np.random.uniform', return_value=expected.copy()
        ):
            ddpg_action = ddpg.select_action(
                np.zeros(4, dtype=np.float32),
                lambda_max=0.0,
                speed_max=0.0,
            )
        np.testing.assert_allclose(ddpg_action, expected)

    def test_ppo_returns_its_sample_without_safety_projection(self):
        agent = PPO(4, 7, {'num_users': 2})
        action = agent.select_action(
            np.zeros(4, dtype=np.float32),
            lambda_max=0.0,
            speed_max=0.0,
        )
        sampled = agent._pending_action.numpy().squeeze(0)
        np.testing.assert_allclose(action, sampled)

    def test_environment_projects_once_and_reports_exact_delta(self):
        env = UAVEnv(dict(self.config))
        env.reset(seed=7)
        env.emergency_mode = 'red'
        requested = np.ones(env.action_space.shape[0], dtype=np.float32)

        with patch(
            'env.uav_env.project_normalized_action',
            wraps=project_normalized_action,
        ) as projector:
            _, _, _, _, info = env.step(requested)

        self.assertEqual(projector.call_count, 1)
        np.testing.assert_allclose(info['raw_action'], requested)
        expected_delta = info['applied_action'] - requested
        np.testing.assert_allclose(
            info['action_projection_delta'], expected_delta
        )
        self.assertAlmostEqual(
            info['action_projection_l1'],
            float(np.linalg.norm(expected_delta, ord=1)),
        )
        self.assertAlmostEqual(
            info['action_projection_l2'],
            float(np.linalg.norm(expected_delta)),
        )
        self.assertAlmostEqual(
            info['action_projection_linf'],
            float(np.linalg.norm(expected_delta, ord=np.inf)),
        )
        self.assertTrue(info['safety_action_overridden'])
        self.assertFalse(info['scheduled_return_action_overridden'])
        np.testing.assert_allclose(
            info['safety_applied_action'], info['applied_action']
        )

    def test_action_logger_writes_raw_applied_and_delta_columns(self):
        logger = Logger({
            'max_episodes': 1,
            'max_steps': 1,
            'log_action_steps': True,
        })
        raw_action = np.array([-0.5, 0.5], dtype=np.float32)
        applied_action = np.array([-0.5, 0.0], dtype=np.float32)
        logger.log_action(3, 4, {
            'raw_action': raw_action,
            'applied_action': applied_action,
            'action_projection_delta': applied_action - raw_action,
            'safety_mode_applied': 'orange',
            'safety_action_overridden': True,
            'action_conditioned_emergency_mode': 'orange',
            'safety_candidate_mode_before_guard': 'red',
            'safety_candidate_mode': 'orange',
            'safety_red_feasibility_guard_applied': True,
            'safety_action_conditioned_feasible': True,
            'safety_projection_policy': 'action_aware',
            'safety_projection_reason': 'positive_energy_saving',
            'safety_action_conditioned_margin_ratio': 0.29,
            'safety_raw_action_energy_estimate': 120.0,
            'safety_projected_action_energy_estimate': 100.0,
            'safety_projected_energy_saving': 20.0,
            'energy_margin_ratio': 0.31,
            'cmdp_margin_target': 0.42,
            'cmdp_cost_scale': 0.20,
            'cmdp_energy_cost_source': 'raw_action_counterfactual',
            'cmdp_energy_cost_max': 4.0,
            'cmdp_margin_cost': 0.55,
            'cmdp_margin_cost_unclipped': 0.55,
            'cmdp_training_cost': 0.55,
            'cmdp_cost_active': True,
            'cmdp_pre_projection_action_cost': 0.55,
            'cmdp_safety_applied_action_cost': 0.35,
            'cmdp_applied_action_cost': 0.30,
            'cmdp_action_cost_delta': -0.25,
            'cmdp_raw_action_margin_ratio': 0.09,
            'cmdp_safety_applied_action_margin_ratio': 0.11,
            'cmdp_applied_action_margin_ratio': 0.12,
        })

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'actions.csv')
            frame = logger.save_actions(path)
            self.assertTrue(os.path.exists(path))

        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.loc[0, 'episode'], 3)
        self.assertEqual(frame.loc[0, 'step'], 4)
        self.assertEqual(frame.loc[0, 'safety_mode_applied'], 'orange')
        self.assertEqual(
            frame.loc[0, 'action_conditioned_emergency_mode'], 'orange'
        )
        self.assertEqual(
            frame.loc[0, 'safety_candidate_mode_before_guard'], 'red'
        )
        self.assertEqual(frame.loc[0, 'safety_candidate_mode'], 'orange')
        self.assertTrue(
            frame.loc[0, 'safety_red_feasibility_guard_applied']
        )
        self.assertTrue(frame.loc[0, 'safety_action_conditioned_feasible'])
        self.assertEqual(
            frame.loc[0, 'safety_projection_policy'], 'action_aware'
        )
        self.assertEqual(
            frame.loc[0, 'safety_projection_reason'],
            'positive_energy_saving',
        )
        self.assertAlmostEqual(
            frame.loc[0, 'safety_projected_energy_saving'], 20.0
        )
        self.assertAlmostEqual(frame.loc[0, 'raw_action_1'], 0.5)
        self.assertAlmostEqual(frame.loc[0, 'applied_action_1'], 0.0)
        self.assertAlmostEqual(frame.loc[0, 'action_delta_1'], -0.5)
        self.assertAlmostEqual(frame.loc[0, 'action_projection_l2'], 0.5)
        self.assertAlmostEqual(frame.loc[0, 'safety_projection_l2'], 0.5)
        self.assertEqual(
            frame.loc[0, 'scheduled_return_action_overridden'], False
        )
        self.assertEqual(frame.loc[0, 'energy_uncertainty_scale'], 1.0)
        self.assertEqual(frame.loc[0, 'E_uav_uncertainty_delta'], 0.0)
        self.assertAlmostEqual(frame.loc[0, 'energy_margin_ratio'], 0.31)
        self.assertAlmostEqual(frame.loc[0, 'cmdp_margin_target'], 0.42)
        self.assertAlmostEqual(frame.loc[0, 'cmdp_training_cost'], 0.55)
        self.assertEqual(
            frame.loc[0, 'cmdp_energy_cost_source'],
            'raw_action_counterfactual',
        )
        self.assertAlmostEqual(
            frame.loc[0, 'cmdp_pre_projection_action_cost'], 0.55
        )
        self.assertAlmostEqual(
            frame.loc[0, 'cmdp_applied_action_cost'], 0.30
        )
        self.assertAlmostEqual(
            frame.loc[0, 'cmdp_raw_action_margin_ratio'], 0.09
        )
        self.assertTrue(frame.loc[0, 'cmdp_cost_active'])


if __name__ == '__main__':
    unittest.main()
