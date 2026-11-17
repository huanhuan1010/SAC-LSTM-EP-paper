"""
随机策略基线: 完全随机动作
作为性能下界参考
"""
import numpy as np


class RandomPolicy:
    """随机策略: 从动作空间均匀采样"""

    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_users = config.get('num_users', 5)
        self.name = 'Random'

    def select_action(self, state, eval=False, **kwargs):
        """随机动作 ∈ [-1, 1]"""
        return np.random.uniform(-1, 1, size=self.action_dim).astype(np.float32)

    def update(self):
        """随机策略无更新"""
        return {}

    def save(self, path):
        pass

    def load(self, path):
        pass
