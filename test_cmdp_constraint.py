import unittest
import tempfile
from pathlib import Path

import numpy as np

from agent.replay_buffer import PrioritizedReplayBuffer
from agent.sac import SAC
from baselines.sac_ablated import SACNoConstraint
from env.uav_env import UAVEnv
from utils.evaluation_metrics import annotate_cmdp_projection_risk


class CMDPConstraintTests(unittest.TestCase):
    def _agent_config(self, **overrides):
        config = {
            'num_users': 2,
            'batch_size': 2,
            'buffer_capacity': 16,
            'warmup_steps': 2,
            'actor_update_delay': 1,
            'replay_alpha': 0.0,
            'cmdp_enabled': True,
            'cmdp_cost_limit': 1.0,
            'lagrange_init': 0.1,
            'lagrange_lr': 1e-3,
            'lagrange_cost_ema_alpha': 1.0,
        }
        config.update(overrides)
        return config

    def test_stage_cost_is_separate_bounded_margin_signal(self):
        env = UAVEnv({
            'num_users': 1,
            'cmdp_margin_target': 0.20,
            'cmdp_cost_scale': 0.20,
            'cmdp_terminal_cost': 2.0,
        })
        self.assertEqual(env.cmdp_training_cost(0.20), (0.0, 0.0))
        total, margin = env.cmdp_training_cost(0.10)
        self.assertAlmostEqual(total, 0.5)
        self.assertAlmostEqual(margin, 0.5)
        self.assertEqual(env.cmdp_training_cost(-1.0), (1.0, 1.0))
        self.assertEqual(
            env.cmdp_training_cost(0.50, True), (2.0, 0.0)
        )

    def test_v3c_cost_activates_before_median_safety_projection(self):
        env = UAVEnv({
            'num_users': 1,
            'cmdp_margin_target': 0.42,
            'cmdp_cost_scale': 0.20,
        })
        self.assertAlmostEqual(env.cmdp_training_cost(0.401)[0], 0.095)
        self.assertAlmostEqual(env.cmdp_training_cost(0.323)[0], 0.485)
        self.assertAlmostEqual(env.cmdp_training_cost(0.245)[0], 0.875)

    def test_action_sensitive_cost_can_preserve_severity_above_one(self):
        env = UAVEnv({
            'num_users': 1,
            'cmdp_margin_target': 0.20,
            'cmdp_cost_scale': 0.20,
            'cmdp_energy_cost_max': 4.0,
        })
        total, margin = env.cmdp_training_cost(-0.40)
        self.assertAlmostEqual(total, 3.0)
        self.assertAlmostEqual(margin, 3.0)
        self.assertGreater(total, 1.0)

    def test_raw_action_counterfactual_cost_distinguishes_same_state_actions(self):
        base = {
            'num_users': 1,
            'policy_action_mode': 'offloading_only',
            'task_interval': 1000,
            'mission_return_enabled': False,
            'initial_energy_min': 0.8,
            'initial_energy_max': 0.8,
            'cmdp_margin_target': 0.20,
            'cmdp_cost_scale': 0.05,
            'cmdp_energy_cost_source': 'raw_action_counterfactual',
            'cmdp_energy_cost_max': 4.0,
        }

        def make_env():
            env = UAVEnv(base)
            env.reset(seed=81)
            env.set_energy_safety_supervisor(False)
            env.task_gen.users[0]['queue'] = [{
                'D': 2.0e6,
                'C': 2.0e9,
                'tau': 10.0,
                'arrival_time': env.step_count,
                'is_core': True,
                'decision_made': False,
            }]
            return env

        low_env = make_env()
        high_env = make_env()
        low_action = -np.ones(
            low_env.action_space.shape[0], dtype=np.float32
        )
        high_action = np.ones(
            high_env.action_space.shape[0], dtype=np.float32
        )
        low_context = low_env._action_conditioned_energy_context(
            low_env._expand_policy_action(low_action),
            low_env.get_energy_context(),
        )
        target_margin = 0.10
        adjustment = (
            target_margin - low_context['energy_margin_ratio']
        ) * low_env.energy.battery_capacity
        low_env.E_rem += adjustment
        high_env.E_rem += adjustment

        _, _, _, _, low_info = low_env.step(low_action)
        _, _, _, _, high_info = high_env.step(high_action)

        self.assertEqual(
            high_info['cmdp_energy_cost_source'],
            'raw_action_counterfactual',
        )
        self.assertLess(
            high_info['cmdp_raw_action_margin_ratio'],
            low_info['cmdp_raw_action_margin_ratio'],
        )
        self.assertGreater(
            high_info['cmdp_pre_projection_action_cost'],
            low_info['cmdp_pre_projection_action_cost'],
        )
        self.assertAlmostEqual(
            high_info['cmdp_training_cost'],
            high_info['cmdp_pre_projection_action_cost'],
        )

    def test_latency_overshoot_mode_is_independent_of_energy_margin(self):
        env = UAVEnv({
            'num_users': 1,
            'cmdp_cost_mode': 'latency_overshoot',
            'cmdp_latency_cost_scale': 0.10,
            'cmdp_margin_target': 0.20,
            'cmdp_cost_scale': 0.20,
            'cmdp_terminal_cost': 2.0,
        })
        components = env.cmdp_cost_components(
            -1.0,
            terminal_event=True,
            latency_overshoot_normalized=0.025,
        )
        self.assertAlmostEqual(components['margin_cost'], 1.0)
        self.assertAlmostEqual(components['latency_overshoot_cost'], 0.25)
        self.assertAlmostEqual(components['total_cost'], 0.25)
        total, margin = env.cmdp_training_cost(
            1.0, latency_overshoot_normalized=0.20
        )
        self.assertEqual(total, 1.0)
        self.assertEqual(margin, 0.0)

    def test_latency_violation_cost_includes_miss_event_and_overshoot(self):
        env = UAVEnv({
            'num_users': 5,
            'cmdp_cost_mode': 'latency_violation',
            'cmdp_latency_cost_scale': 0.10,
            'cmdp_latency_miss_weight': 1.0,
            'cmdp_latency_overshoot_weight': 1.0,
        })
        components = env.cmdp_cost_components(
            1.0,
            latency_overshoot_normalized=0.025,
            latency_miss_fraction=0.20,
        )
        self.assertAlmostEqual(components['latency_overshoot_cost'], 0.25)
        self.assertAlmostEqual(components['latency_miss_fraction'], 0.20)
        self.assertAlmostEqual(components['latency_violation_cost'], 0.45)
        self.assertAlmostEqual(components['total_cost'], 0.45)

    def test_latency_violation_rejects_zero_component_weights(self):
        with self.assertRaisesRegex(ValueError, 'positive miss or overshoot'):
            UAVEnv({
                'num_users': 1,
                'cmdp_cost_mode': 'latency_violation',
                'cmdp_latency_miss_weight': 0.0,
                'cmdp_latency_overshoot_weight': 0.0,
            })

    def test_invalid_cmdp_cost_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'cmdp_cost_mode'):
            UAVEnv({'num_users': 1, 'cmdp_cost_mode': 'unknown'})

    def test_physical_evaluation_constraint_ignores_prediction(self):
        env = UAVEnv({'num_users': 1, 'battery_capacity': 100000.0})
        env.reset(seed=23)
        before = env.get_physical_constraint_context()
        env.set_energy_prediction(
            np.full(env.horizon, 10000.0),
            residual_bias=5000.0,
            residual_uncertainty=2000.0,
        )
        after = env.get_physical_constraint_context()

        for key in (
            'physical_return_energy',
            'physical_energy_floor',
            'physical_required_energy',
            'physical_constraint_margin',
            'physical_constraint_margin_ratio',
            'physical_constraint_violation',
        ):
            self.assertEqual(after[key], before[key])
        self.assertTrue(after['physical_constraint_predictor_independent'])

        env.E_rem = before['physical_required_energy'] - 1.0
        self.assertTrue(
            env.get_physical_constraint_context()[
                'physical_constraint_violation'
            ]
        )
        env.E_rem = before['physical_required_energy'] + 1.0
        self.assertFalse(
            env.get_physical_constraint_context()[
                'physical_constraint_violation'
            ]
        )

    def test_step_reports_distinct_cmdp_and_physical_signals(self):
        env = UAVEnv({'num_users': 1, 'battery_capacity': 100000.0})
        env.reset(seed=29)
        env.set_energy_safety_supervisor(False)
        env.set_energy_prediction(np.full(env.horizon, 100000.0))

        _, _, _, _, info = env.step(
            np.zeros(env.action_space.shape[0], dtype=np.float32)
        )

        self.assertGreater(info['cmdp_training_cost'], 0.0)
        self.assertIn('cmdp_pre_projection_state_cost', info)
        self.assertEqual(
            info['cmdp_post_projection_state_cost'],
            info['cmdp_training_cost'],
        )
        self.assertAlmostEqual(
            info['cmdp_projection_state_cost_delta'],
            info['cmdp_post_projection_state_cost']
            - info['cmdp_pre_projection_state_cost'],
        )
        self.assertIn('physical_constraint_violation', info)
        self.assertNotIn('constraint_cost', info)

    def test_replay_buffer_preserves_nonnegative_constraint_cost(self):
        buffer = PrioritizedReplayBuffer(4, 3, 2, alpha=0.0)
        buffer.push(
            np.zeros(3), np.zeros(2), 1.0, np.ones(3), False,
            cost=0.75,
        )
        batch = buffer.sample(1)
        self.assertAlmostEqual(float(batch[3][0, 0]), 0.75)

    def test_replay_stores_requested_and_applied_actions_separately(self):
        buffer = PrioritizedReplayBuffer(4, 3, 2, alpha=0.0, seed=7)
        requested = np.array([0.8, -0.6], dtype=np.float32)
        applied = np.array([0.2, -0.1], dtype=np.float32)
        buffer.push(
            np.zeros(3), requested, 1.0, np.ones(3), False,
            cost=0.25, applied_action=applied,
        )

        np.testing.assert_allclose(buffer.actions[0], requested)
        np.testing.assert_allclose(buffer.applied_actions[0], applied)
        batch = buffer.sample(1)
        np.testing.assert_allclose(batch[1][0], requested)

    def test_cost_q_target_is_bounded_by_finite_horizon(self):
        agent = SAC(4, 7, self._agent_config(
            gamma=0.9,
            max_steps=3,
            cmdp_stage_cost_max=0.5,
            warmup_steps=2,
            actor_update_delay=1,
        ))
        expected_q_max = 0.5 * (1.0 + 0.9 + 0.9 ** 2)
        self.assertAlmostEqual(agent.cmdp_q_max, expected_q_max)
        for target in (
            agent.target_cost_critic1, agent.target_cost_critic2,
        ):
            target.net[-1].bias.data.fill_(1.0e6)
        for index in range(2):
            state = np.full(4, 0.1 * index, dtype=np.float32)
            agent.replay_buffer.push(
                state, np.zeros(7, dtype=np.float32), 0.0,
                state + 0.01, False, cost=5.0,
            )

        result = agent.update()

        self.assertTrue(np.isfinite(result['cost_critic_loss']))
        self.assertLessEqual(
            result['target_cost_q_max'], result['cmdp_q_max'] + 1e-6
        )

    def test_lagrange_multiplier_responds_to_episode_budget(self):
        high_cost_agent = SAC(4, 7, self._agent_config())
        initial_high = float(high_cost_agent.lagrange_multiplier.item())
        high_cost_agent.update_lagrange(3.0)
        self.assertGreater(
            float(high_cost_agent.lagrange_multiplier.item()), initial_high
        )

        low_cost_agent = SAC(4, 7, self._agent_config())
        initial_low = float(low_cost_agent.lagrange_multiplier.item())
        low_cost_agent.update_lagrange(0.0)
        self.assertLess(
            float(low_cost_agent.lagrange_multiplier.item()), initial_low
        )

    def test_projected_normalized_dual_step_uses_budget_relative_gap(self):
        agent = SAC(4, 7, self._agent_config(
            cmdp_cost_limit=10.0,
            lagrange_init=0.30,
            lagrange_lr=0.02,
            lagrange_update_mode='projected_normalized',
        ))
        result = agent.update_lagrange(15.0)
        self.assertAlmostEqual(result['lagrange_normalized_gap'], 0.5)
        self.assertAlmostEqual(result['lagrange_multiplier'], 0.31, places=6)
        self.assertAlmostEqual(result['lagrange_update_delta'], 0.01, places=6)
        self.assertEqual(result['lagrange_projected_update'], 1.0)

    def test_legacy_dual_update_remains_the_default(self):
        agent = SAC(4, 7, self._agent_config())
        self.assertEqual(agent.lagrange_update_mode, 'log_adam')
        result = agent.update_lagrange(3.0)
        self.assertEqual(result['lagrange_projected_update'], 0.0)

    def test_sparse_actor_gradient_diagnostics_are_finite(self):
        agent = SAC(4, 7, self._agent_config(
            cmdp_actor_penalty_warmup_updates=1,
            cmdp_actor_penalty_min_positive_samples=2,
            cmdp_gradient_diagnostic_interval=1,
        ))
        for index in range(2):
            state = np.full(4, index * 0.1, dtype=np.float32)
            agent.replay_buffer.push(
                state, np.zeros(7, dtype=np.float32), 0.0,
                state + 0.01, False, cost=0.25,
            )
        result = agent.update()
        self.assertTrue(np.isfinite(result['actor_reward_gradient_norm']))
        self.assertTrue(np.isfinite(result['actor_cost_gradient_norm']))
        self.assertTrue(np.isfinite(
            result['actor_cost_to_reward_gradient_ratio']
        ))
        self.assertGreaterEqual(result['policy_cost_q_std'], 0.0)

    def test_actor_penalty_waits_for_real_cost_and_critic_warmup(self):
        config = self._agent_config(
            warmup_steps=2,
            actor_update_delay=1,
            cmdp_actor_penalty_warmup_updates=2,
            cmdp_actor_penalty_min_positive_samples=2,
        )
        agent = SAC(4, 7, config)
        for index in range(2):
            state = np.full(4, index * 0.1, dtype=np.float32)
            agent.replay_buffer.push(
                state, np.zeros(7, dtype=np.float32), 0.0,
                state + 0.01, False, cost=0.25,
            )
        first = agent.update()
        self.assertEqual(first['cmdp_actor_penalty_active'], 0.0)
        second = agent.update()
        self.assertEqual(second['cmdp_actor_penalty_active'], 1.0)
        self.assertIn('cmdp_pre_projection_q_mean', second)
        self.assertIn('cmdp_post_projection_q_mean', second)

        zero_cost = SAC(4, 7, config)
        for index in range(2):
            state = np.full(4, index * 0.1, dtype=np.float32)
            zero_cost.replay_buffer.push(
                state, np.zeros(7, dtype=np.float32), 0.0,
                state + 0.01, False, cost=0.0,
            )
        zero_cost.update()
        result = zero_cost.update()
        self.assertEqual(result['cmdp_actor_penalty_active'], 0.0)
        self.assertEqual(result['constraint_term_mean'], 0.0)

    def test_actor_penalty_ramps_after_readiness_gates(self):
        config = self._agent_config(
            warmup_steps=2,
            actor_update_delay=1,
            cmdp_actor_penalty_warmup_updates=2,
            cmdp_actor_penalty_min_positive_samples=2,
            cmdp_actor_penalty_ramp_updates=4,
        )
        agent = SAC(4, 7, config)
        for index in range(2):
            state = np.full(4, index * 0.1, dtype=np.float32)
            agent.replay_buffer.push(
                state, np.zeros(7, dtype=np.float32), 0.0,
                state + 0.01, False, cost=0.25,
            )
        first = agent.update()
        second = agent.update()
        third = agent.update()
        self.assertEqual(first['cmdp_actor_penalty_scale'], 0.0)
        self.assertAlmostEqual(second['cmdp_actor_penalty_scale'], 0.25)
        self.assertAlmostEqual(third['cmdp_actor_penalty_scale'], 0.50)

    def test_exact_raw_and_applied_action_risk_is_paired(self):
        class DummyAgent:
            @staticmethod
            def estimate_cmdp_risk(state, action):
                return float(np.dot(state[:2], action[:2]))

        state = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        raw = np.array([0.5, 0.5], dtype=np.float32)
        applied = np.array([0.0, 0.25], dtype=np.float32)
        info = {'raw_action': raw, 'applied_action': applied}
        annotate_cmdp_projection_risk(DummyAgent(), state, info)
        self.assertAlmostEqual(
            info['cmdp_pre_projection_critic_risk'], 1.5
        )
        self.assertAlmostEqual(
            info['cmdp_post_projection_critic_risk'], 0.5
        )
        self.assertAlmostEqual(
            info['cmdp_projection_critic_risk_delta'], -1.0
        )

        nocmdp = SACNoConstraint(4, 7, self._agent_config())
        self.assertTrue(np.isnan(nocmdp.estimate_cmdp_risk(
            np.zeros(4, dtype=np.float32),
            np.zeros(7, dtype=np.float32),
        )))

    def test_no_constraint_ablation_disables_learned_constraint_only(self):
        agent = SACNoConstraint(4, 7, self._agent_config())
        self.assertFalse(agent.constraint_enabled)
        self.assertIsNone(agent.cost_critic1)
        metrics = agent.update_lagrange(3.0)
        self.assertEqual(metrics['lagrange_multiplier'], 0.0)

    def test_checkpoint_restores_cost_critics_and_dual_state(self):
        config = self._agent_config()
        source = SAC(4, 7, config)
        source.update_lagrange(3.0)
        expected_lambda = float(source.lagrange_multiplier.item())
        expected_ema = source.constraint_cost_ema
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'cmdp.pt'
            source.save(str(checkpoint))
            restored = SAC(4, 7, config)
            restored.load(str(checkpoint))
        self.assertAlmostEqual(
            float(restored.lagrange_multiplier.item()), expected_lambda
        )
        self.assertAlmostEqual(restored.constraint_cost_ema, expected_ema)
        for expected, actual in zip(
            source.cost_critic1.parameters(), restored.cost_critic1.parameters()
        ):
            np.testing.assert_allclose(
                expected.detach().cpu().numpy(), actual.detach().cpu().numpy()
            )


if __name__ == '__main__':
    unittest.main()
