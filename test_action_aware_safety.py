import copy
import unittest

import numpy as np

from env.uav_env import UAVEnv
from utils.config import load_config


class ActionAwareSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config('configs/task_offloading_prediction_v2.yaml')

    def make_env(self):
        config = dict(self.config)
        config.update({
            'num_users': 2,
            'task_interval': 1000,
            'mission_return_enabled': False,
            'initial_energy_min': 0.8,
            'initial_energy_max': 0.8,
        })
        env = UAVEnv(config)
        env.reset(seed=71)
        for user in env.task_gen.users:
            user['queue'] = []
        return env

    @staticmethod
    def add_undecided_task(env, user_index=0):
        env.task_gen.users[user_index]['queue'] = [{
            'D': 2.0e6,
            'C': 2.0e9,
            'tau': 10.0,
            'arrival_time': env.step_count,
            'is_core': True,
            'decision_made': False,
        }]

    @staticmethod
    def set_margin_ratio(env, ratio):
        context = env.get_energy_context()
        env.E_rem = (
            context['required_energy_with_reserve']
            + ratio * env.energy.battery_capacity
        )
        env.emergency_mode = 'normal'

    @staticmethod
    def set_action_conditioned_margin_ratio(env, action, ratio):
        full_action = env._expand_policy_action(action)
        energy_context = env.get_energy_context()
        conditioned = env._action_conditioned_energy_context(
            full_action, energy_context
        )
        env.E_rem += (
            float(ratio) - float(conditioned['energy_margin_ratio'])
        ) * env.energy.battery_capacity

    def test_counterfactual_estimate_does_not_mutate_environment(self):
        env = self.make_env()
        self.add_undecided_task(env)
        before_queues = copy.deepcopy([
            user['queue'] for user in env.task_gen.users
        ])
        before_pos = env.uav.pos.copy()
        before_vel = env.uav.vel.copy()
        action = np.ones(env.action_space.shape[0], dtype=np.float32)

        context = env.get_energy_context()
        full_action = env._expand_policy_action(action)
        estimate = env._action_conditioned_energy_context(
            full_action, context
        )

        self.assertGreater(estimate['estimated_uav_energy'], 0.0)
        self.assertEqual(
            [user['queue'] for user in env.task_gen.users], before_queues
        )
        np.testing.assert_allclose(env.uav.pos, before_pos)
        np.testing.assert_allclose(env.uav.vel, before_vel)

    def test_orange_conservative_action_is_advisory_only(self):
        env = self.make_env()
        self.set_margin_ratio(env, 0.28)
        # No offloading and minimum UAV CPU already satisfy every orange cap.
        action = -np.ones(env.action_space.shape[0], dtype=np.float32)

        _, _, _, _, info = env.step(action)

        self.assertEqual(info['autonomous_emergency_mode'], 'orange')
        self.assertEqual(info['safety_mode_applied'], 'normal')
        self.assertEqual(
            info['safety_projection_reason'],
            'advisory_only_no_material_energy_saving',
        )
        self.assertFalse(info['safety_action_overridden'])
        np.testing.assert_allclose(info['safety_applied_action'], action)

    def test_orange_high_cpu_action_is_projected_only_with_energy_saving(self):
        env = self.make_env()
        self.add_undecided_task(env)
        self.set_margin_ratio(env, 0.28)
        action = np.ones(env.action_space.shape[0], dtype=np.float32)

        _, _, _, _, info = env.step(action)

        self.assertEqual(info['safety_mode_applied'], 'orange')
        self.assertEqual(
            info['safety_projection_reason'], 'positive_energy_saving'
        )
        self.assertTrue(info['safety_action_overridden'])
        self.assertGreaterEqual(
            info['safety_projected_energy_saving'],
            env.safety_min_energy_saving,
        )
        self.assertLess(
            info['safety_projected_action_energy_estimate'],
            info['safety_raw_action_energy_estimate'],
        )
        # User transmit power is not UAV battery power and must not be capped.
        self.assertAlmostEqual(info['safety_applied_action'][-1], 1.0)

    def test_red_remains_unconditional_hard_takeover(self):
        env = self.make_env()
        self.set_margin_ratio(env, 0.20)
        action = -np.ones(env.action_space.shape[0], dtype=np.float32)

        _, _, _, _, info = env.step(action)

        self.assertEqual(info['safety_mode_applied'], 'red')
        self.assertEqual(info['safety_projection_reason'], 'hard_red_takeover')
        self.assertTrue(info['safety_action_overridden'])

    def test_feasible_legacy_red_warning_cannot_force_hard_return(self):
        config = dict(self.config)
        config['safety_red_requires_infeasibility'] = True
        env = self.make_env_from_config(config)
        action = -np.ones(env.action_space.shape[0], dtype=np.float32)
        self.set_action_conditioned_margin_ratio(env, action, 0.20)
        # Simulate the red request produced by the old positive threshold.
        env.emergency_mode = 'red'

        _, _, _, _, info = env.step(action)

        self.assertEqual(info['safety_candidate_mode_before_guard'], 'red')
        self.assertTrue(info['safety_red_feasibility_guard_applied'])
        self.assertTrue(info['safety_action_conditioned_feasible'])
        self.assertNotEqual(info['safety_candidate_mode'], 'red')
        self.assertNotEqual(info['safety_mode_applied'], 'red')

    def test_infeasible_action_conditioned_margin_still_forces_red(self):
        config = dict(self.config)
        config['safety_red_requires_infeasibility'] = True
        env = self.make_env_from_config(config)
        action = -np.ones(env.action_space.shape[0], dtype=np.float32)
        self.set_action_conditioned_margin_ratio(env, action, -0.001)
        env.emergency_mode = 'normal'

        _, _, _, _, info = env.step(action)

        self.assertFalse(info['safety_action_conditioned_feasible'])
        self.assertFalse(info['safety_red_feasibility_guard_applied'])
        self.assertEqual(info['safety_candidate_mode'], 'red')
        self.assertEqual(info['safety_mode_applied'], 'red')
        self.assertEqual(info['safety_projection_reason'], 'hard_red_takeover')

    def test_v3_thresholds_are_tied_to_residual_margin_feasibility(self):
        config = load_config('configs/task_offloading_prediction_v3.yaml')
        self.assertTrue(config['safety_red_requires_infeasibility'])
        self.assertEqual(config['threshold_red'], 0.0)
        self.assertEqual(config['threshold_orange'], 0.05)
        self.assertEqual(config['threshold_yellow'], 0.10)
        self.assertEqual(config['predictive_return_threshold'], 0.0)
        self.assertEqual(config['single_level_margin_threshold'], 0.10)

    def make_env_from_config(self, config):
        config = dict(config)
        config.update({
            'num_users': 2,
            'task_interval': 1000,
            'mission_return_enabled': False,
            'initial_energy_min': 0.8,
            'initial_energy_max': 0.8,
        })
        env = UAVEnv(config)
        env.reset(seed=71)
        for user in env.task_gen.users:
            user['queue'] = []
        return env


if __name__ == '__main__':
    unittest.main()
