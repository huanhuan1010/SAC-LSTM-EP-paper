"""Regression tests for the generic action mode used by paper benchmarks."""
import unittest

import numpy as np

from agent.sac import SAC


class PaperActionModeTest(unittest.TestCase):
    def test_identity_projection_supports_four_actions(self):
        agent = SAC(20, 4, {
            "action_projection": "identity",
            "cmdp_enabled": False,
            "warmup_steps": 10,
            "buffer_capacity": 32,
            "batch_size": 4,
        })
        action = agent.select_action(np.zeros(20, dtype=np.float32))
        self.assertEqual(action.shape, (4,))
        self.assertTrue(np.all(action >= -1.0))
        self.assertTrue(np.all(action <= 1.0))

    def test_unknown_projection_is_rejected(self):
        with self.assertRaises(ValueError):
            SAC(20, 4, {
                "action_projection": "unknown",
                "cmdp_enabled": False,
            })


if __name__ == "__main__":
    unittest.main()

