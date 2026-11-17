import unittest

import numpy as np

from agent.emergency import EmergencyManager
from env.uav_env import UAVEnv
from utils.ablation_support import (
    configure_profile_environment,
    profile_observation_controller,
    profile_safety_decision,
)
from utils.algorithm_profiles import algorithm_profile
from utils.config import load_config


class NativeAblationSupportTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config('configs/default.yaml')
        self.config.update({
            'num_users': 1,
            'max_steps': 2,
            'task_interval': 1000,
            'initial_energy_min': 0.19,
            'initial_energy_max': 0.19,
            'soc_return_threshold': 0.20,
        })

    def test_soc_threshold_controller_applies_red_return(self):
        env = UAVEnv(self.config)
        profile = algorithm_profile('SAC-LSTM-EP-SOC')
        configure_profile_environment(env, profile)
        env.reset(seed=11)
        mode, constraints = profile_safety_decision(
            profile,
            env,
            EmergencyManager(self.config),
            np.zeros(self.config['horizon']),
        )
        self.assertEqual(mode, 'red')
        self.assertEqual(constraints['heading_mode'], 'to_base')
        env.emergency_mode = mode
        _, _, _, _, info = env.step(
            np.ones(env.action_space.shape[0], dtype=np.float32)
        )
        self.assertEqual(info['energy_safety_controller'], 'soc_threshold')
        self.assertEqual(info['safety_mode_applied'], 'red')

    def test_soc_and_predictive_thresholds_use_the_same_red_action(self):
        env = UAVEnv(self.config)
        env.reset(seed=12)
        context = {'energy_margin_ratio': 0.19}
        env.get_energy_context = lambda: dict(context)

        env.E_rem = 0.50 * env.energy.battery_capacity
        env.set_energy_safety_supervisor(True, controller='soc_threshold')
        self.assertEqual(env._autonomous_safety_mode(context), 'normal')

        env.set_energy_safety_supervisor(True, controller='margin_threshold')
        self.assertEqual(env._autonomous_safety_mode(context), 'red')
        predictive_constraints = profile_safety_decision(
            algorithm_profile('SAC-LSTM-EP-PredictiveReturn'),
            env,
            EmergencyManager(self.config),
            np.zeros(self.config['horizon']),
        )[1]

        env.E_rem = 0.19 * env.energy.battery_capacity
        env.set_energy_safety_supervisor(True, controller='soc_threshold')
        self.assertEqual(env._autonomous_safety_mode(context), 'red')
        soc_constraints = profile_safety_decision(
            algorithm_profile('SAC-LSTM-EP-SOC'),
            env,
            EmergencyManager(self.config),
            np.zeros(self.config['horizon']),
        )[1]
        self.assertEqual(soc_constraints, predictive_constraints)

    def test_single_level_matches_graded_onset_but_not_its_severity(self):
        env = UAVEnv(self.config)
        context = {'energy_margin_ratio': 0.35}

        env.set_energy_safety_supervisor(True, controller='margin')
        self.assertEqual(env._autonomous_safety_mode(context), 'yellow')

        env.set_energy_safety_supervisor(
            True, controller='margin_single_level'
        )
        self.assertEqual(env._autonomous_safety_mode(context), 'red')

        env.set_energy_safety_supervisor(True, controller='margin_threshold')
        self.assertEqual(env._autonomous_safety_mode(context), 'normal')

    def test_observation_controller_preserves_matched_state_semantics(self):
        self.assertEqual(
            profile_observation_controller(
                algorithm_profile('SAC-LSTM-EP-NoSafety')
            ),
            'margin',
        )
        for name, controller in (
            ('SAC-LSTM-EP-SOC', 'soc_threshold'),
            ('SAC-LSTM-EP-PredictiveReturn', 'margin_threshold'),
            ('SAC-LSTM-EP-SingleLevel', 'margin_single_level'),
        ):
            self.assertEqual(
                profile_observation_controller(algorithm_profile(name)),
                controller,
            )

    def test_full_no_safety_disables_only_projection_controller(self):
        env = UAVEnv(self.config)
        profile = algorithm_profile('SAC-LSTM-EP-NoSafety')
        configure_profile_environment(env, profile)
        self.assertFalse(env.energy_safety_supervisor_enabled)
        self.assertEqual(env.energy_safety_controller, 'none')
        self.assertEqual(
            (env.reward_energy_weight, env.reward_delay_weight),
            (
                self.config['reward_energy_weight'],
                self.config['reward_delay_weight'],
            ),
        )

    def test_plain_sac_uses_fixed_reward_weights(self):
        env = UAVEnv(self.config)
        configure_profile_environment(env, algorithm_profile('SAC'))
        self.assertEqual(
            (env.reward_energy_weight, env.reward_delay_weight),
            (0.5, 0.5),
        )


if __name__ == '__main__':
    unittest.main()
