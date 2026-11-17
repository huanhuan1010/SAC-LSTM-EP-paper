import unittest

import numpy as np

from baselines.greedy import GreedyPolicy


class GreedyBaselineTests(unittest.TestCase):
    def _state(self, num_users=2):
        # 10个基础状态 + 信道N + 队列N + 任务进度5N + E_pred5 + 2
        state = np.zeros(10 + 7 * num_users + 5 + 2, dtype=np.float32)
        state[10:10 + num_users] = [-0.8, -0.9]
        state[10 + num_users:10 + 2 * num_users] = [0.2, 0.4]
        return state

    def test_configured_physical_speed_is_mapped_to_normalized_action(self):
        policy = GreedyPolicy(
            len(self._state()),
            7,
            {'num_users': 2, 'v_max': 20.0, 'greedy_speed': 10.0},
        )
        action = policy.select_action(self._state())
        self.assertAlmostEqual(float(action[4]), 0.0, places=6)

    def test_zero_physical_speed_maps_to_minus_one(self):
        policy = GreedyPolicy(
            len(self._state()),
            7,
            {'num_users': 2, 'v_max': 20.0, 'greedy_speed': 0.0},
        )
        action = policy.select_action(self._state())
        self.assertAlmostEqual(float(action[4]), -1.0, places=6)


if __name__ == '__main__':
    unittest.main()
