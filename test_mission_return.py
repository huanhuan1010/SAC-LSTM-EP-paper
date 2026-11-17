import unittest

import numpy as np

from env.uav_env import UAVEnv
from agent.lstm_runtime import energy_margin_index, observation_for_agent
from utils.config import load_config
from utils.evaluation_metrics import update_mission_outcome
from utils.provenance import environment_fingerprint


def mission_config(**overrides):
    config = load_config('configs/default.yaml')
    config.update({
        'mission_return_enabled': True,
        'max_steps': 3,
        'mission_service_steps': 1,
        'mission_min_completed_tasks': 1,
        'wind_enabled': False,
    })
    config.update(overrides)
    return config


class MissionReturnTests(unittest.TestCase):
    def test_agent_observation_preserves_mission_phase_slots(self):
        config = mission_config()
        env = UAVEnv(config)
        state, _ = env.reset(seed=10)

        augmented, _ = observation_for_agent(
            state,
            env,
            lstm_predictor=None,
            config=config,
            use_lstm=False,
            emergency_mode='normal',
        )

        margin_index = energy_margin_index(env.num_users, env.horizon)
        expected_margin = np.clip(
            env.get_energy_context()['energy_margin_ratio'], -1.0, 1.0
        )
        self.assertAlmostEqual(augmented[margin_index], expected_margin)
        np.testing.assert_array_equal(augmented[-2:], state[-2:])

    def test_enabled_mission_adds_progress_and_phase_observations(self):
        nominal = UAVEnv(load_config('configs/default.yaml'))
        mission = UAVEnv(mission_config())
        state, _ = mission.reset(seed=11)

        self.assertEqual(
            mission.observation_space.shape[0],
            nominal.observation_space.shape[0] + 2,
        )
        self.assertEqual(state[-2], 0.0)
        self.assertEqual(state[-1], 0.0)

        action = -np.ones(mission.action_space.shape[0], dtype=np.float32)
        state, _, _, _, info = mission.step(action)
        self.assertEqual(info['mission_phase'], 'scheduled_return')
        self.assertAlmostEqual(state[-2], 1.0 / 3.0)
        self.assertEqual(state[-1], 1.0)

    def test_scheduled_return_is_common_and_not_an_emergency_trigger(self):
        env = UAVEnv(mission_config())
        env.set_energy_safety_supervisor(False)
        env.reset(seed=12)
        action = np.ones(env.action_space.shape[0], dtype=np.float32)

        env.step(action)  # final service step
        _, _, _, _, info = env.step(action)

        self.assertTrue(info['scheduled_return_active'])
        self.assertEqual(info['safety_mode_applied'], 'normal')
        self.assertFalse(info['safety_action_overridden'])
        self.assertTrue(info['scheduled_return_action_overridden'])
        np.testing.assert_allclose(
            info['safety_projection_delta'], 0.0, atol=1e-7
        )
        np.testing.assert_allclose(
            info['action_projection_delta'],
            info['safety_projection_delta']
            + info['scheduled_return_projection_delta'],
            atol=1e-7,
        )
        self.assertTrue(np.all(info['applied_action'][:env.num_users] == -1.0))
        expected_speed_action = np.clip(
            2.0 * env.config['return_speed'] / env.uav.v_max - 1.0,
            -1.0,
            1.0,
        )
        self.assertAlmostEqual(
            info['applied_action'][env.num_users + 2],
            expected_speed_action,
        )
        self.assertEqual(info['tasks_completed'], 0)
        self.assertEqual(info['task_progress']['tasks_in_service'], 0)

    def test_deadline_miss_is_terminal_but_not_a_reward_component(self):
        env = UAVEnv(mission_config(max_steps=2))
        env.set_energy_safety_supervisor(False)
        env.reset(seed=13)
        env.uav.pos = np.array([400.0, 400.0, 100.0])
        action = -np.ones(env.action_space.shape[0], dtype=np.float32)

        env.step(action)
        _, _, done, _, info = env.step(action)

        self.assertTrue(done)
        self.assertTrue(info['mission_deadline_missed'])
        self.assertFalse(info['mission_success'])
        self.assertNotIn(
            'mission_deadline_penalty', info['reward_breakdown']
        )

    def test_success_requires_service_and_on_time_safe_return(self):
        env = UAVEnv(mission_config())
        env.reset(seed=14)
        env.episode_tasks_completed = 1
        env.episode_tasks_generated = 1
        env.episode_core_tasks_completed = 1
        env.episode_core_tasks_generated = 1
        env.uav.pos = env.uav.base_pos.copy()
        env.uav.pos[2] = env.uav.h_min
        env.uav.vel[:] = 0.0
        env.uav.air_vel[:] = 0.0
        action = -np.ones(env.action_space.shape[0], dtype=np.float32)

        _, _, done, _, info = env.step(action)

        self.assertTrue(done)
        self.assertTrue(info['returned_safe'])
        self.assertTrue(info['mission_task_requirement_met'])
        self.assertTrue(info['mission_success'])
        self.assertEqual(info['mission_outcome'], 'success')
        self.assertEqual(info['mission_failure_reasons'], ())
        self.assertEqual(info['task_completion_ratio'], 1.0)
        self.assertEqual(info['core_task_completion_ratio'], 1.0)
        self.assertFalse(info['mission_deadline_missed'])

    def test_safe_abort_without_required_task_is_not_mission_success(self):
        env = UAVEnv(mission_config())
        env.reset(seed=15)
        env.uav.pos = env.uav.base_pos.copy()
        env.uav.pos[2] = env.uav.h_min
        env.uav.vel[:] = 0.0
        env.uav.air_vel[:] = 0.0
        action = -np.ones(env.action_space.shape[0], dtype=np.float32)

        _, _, _, _, info = env.step(action)

        self.assertTrue(info['returned_safe'])
        self.assertFalse(info['mission_success'])

    def test_early_safe_return_is_abort_not_mission_success(self):
        env = UAVEnv(mission_config(mission_service_steps=2))
        env.reset(seed=141)
        env.episode_tasks_completed = 5
        env.episode_tasks_generated = 5
        env.episode_core_tasks_completed = 2
        env.episode_core_tasks_generated = 2
        env.uav.pos = env.uav.base_pos.copy()
        env.uav.pos[2] = env.uav.h_min

        _, _, done, _, info = env.step(
            -np.ones(env.action_space.shape[0], dtype=np.float32)
        )

        self.assertTrue(done)
        self.assertTrue(info['returned_safe'])
        self.assertFalse(info['mission_full_service_window_met'])
        self.assertFalse(info['mission_success'])
        self.assertTrue(info['mission_safe_abort'])
        self.assertEqual(info['mission_outcome'], 'safe_abort')
        self.assertIn(
            'service_window_incomplete', info['mission_failure_reasons']
        )

    def test_overall_completion_ratio_is_required(self):
        env = UAVEnv(mission_config())
        env.reset(seed=142)
        env.episode_tasks_completed = 7
        env.episode_tasks_generated = 10
        env.episode_core_tasks_completed = 2
        env.episode_core_tasks_generated = 2
        env.uav.pos = env.uav.base_pos.copy()
        env.uav.pos[2] = env.uav.h_min

        _, _, _, _, info = env.step(
            -np.ones(env.action_space.shape[0], dtype=np.float32)
        )

        self.assertAlmostEqual(info['task_completion_ratio'], 0.7)
        self.assertFalse(info['mission_completion_ratio_met'])
        self.assertFalse(info['mission_success'])
        self.assertIn('completion_ratio_unmet', info['mission_failure_reasons'])

    def test_core_completion_ratio_is_required(self):
        env = UAVEnv(mission_config())
        env.reset(seed=143)
        env.episode_tasks_completed = 8
        env.episode_tasks_generated = 10
        env.episode_core_tasks_completed = 1
        env.episode_core_tasks_generated = 2
        env.uav.pos = env.uav.base_pos.copy()
        env.uav.pos[2] = env.uav.h_min

        _, _, _, _, info = env.step(
            -np.ones(env.action_space.shape[0], dtype=np.float32)
        )

        self.assertAlmostEqual(info['core_task_completion_ratio'], 0.5)
        self.assertFalse(info['mission_core_completion_ratio_met'])
        self.assertFalse(info['mission_success'])
        self.assertIn(
            'core_completion_ratio_unmet', info['mission_failure_reasons']
        )

    def test_prior_physical_violation_disqualifies_mission_success(self):
        env = UAVEnv(mission_config())
        env.reset(seed=144)
        env.episode_tasks_completed = 1
        env.episode_tasks_generated = 1
        env.episode_core_tasks_completed = 1
        env.episode_core_tasks_generated = 1
        env.episode_physical_constraint_violated = True
        env.uav.pos = env.uav.base_pos.copy()
        env.uav.pos[2] = env.uav.h_min

        _, _, _, _, info = env.step(
            -np.ones(env.action_space.shape[0], dtype=np.float32)
        )

        self.assertFalse(info['mission_energy_requirement_met'])
        self.assertFalse(info['mission_success'])
        self.assertIn(
            'physical_energy_infeasible', info['mission_failure_reasons']
        )

    def test_service_window_arrivals_are_counted(self):
        env = UAVEnv(mission_config(
            max_steps=4,
            mission_service_steps=3,
            task_interval=1,
        ))
        env.reset(seed=145)

        _, _, _, _, info = env.step(
            np.zeros(env.action_space.shape[0], dtype=np.float32)
        )

        self.assertEqual(info['episode_tasks_generated'], env.num_users)
        self.assertEqual(
            info['episode_core_tasks_generated'],
            sum(
                int(task.get('is_core', False))
                for user in env.task_gen.users for task in user['queue']
            ),
        )

    def test_mission_outcome_accumulates_across_steps(self):
        success, missed = update_mission_outcome(
            False, False, {'mission_success': True}
        )
        success, missed = update_mission_outcome(
            success, missed, {'mission_success': False}
        )
        self.assertTrue(success)
        self.assertFalse(missed)

    def test_mission_parameters_change_environment_fingerprint(self):
        first = mission_config(mission_service_steps=1)
        second = mission_config(mission_service_steps=2)
        self.assertNotEqual(
            environment_fingerprint(first), environment_fingerprint(second)
        )

    def test_invalid_service_horizon_is_rejected(self):
        with self.assertRaises(ValueError):
            UAVEnv(mission_config(mission_service_steps=3, max_steps=3))

    def test_invalid_completion_ratio_is_rejected(self):
        with self.assertRaises(ValueError):
            UAVEnv(mission_config(mission_min_completion_ratio=1.01))


if __name__ == '__main__':
    unittest.main()
