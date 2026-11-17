"""
优先级经验回放缓冲池 (PER) — 借鉴 SkyNetMEC 的 numpy 预分配存储
保留 PER 优先级采样 + 重要性采样权重
"""
import numpy as np


class PrioritizedReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim,
                 alpha=0.6, beta=0.4, beta_increment=0.001, seed=None,
                 positive_cost_epsilon=1e-8):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.rng = np.random.default_rng(seed)
        self.positive_cost_epsilon = float(positive_cost_epsilon)
        if self.positive_cost_epsilon < 0.0:
            raise ValueError('positive_cost_epsilon must be nonnegative')

        # 预分配 numpy 数组 (SkyNetMEC 风格)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        # ``actions`` always stores the policy-requested action.  The action
        # actually executed after the environment safety projection is kept
        # separately for diagnostics and shield-model research.  Critics must
        # learn Q(s, a_requested), because that is the action produced by the
        # actor at optimisation time.
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.applied_actions = np.zeros(
            (capacity, action_dim), dtype=np.float32
        )
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.costs = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.priorities = np.ones(capacity, dtype=np.float32)

        self.ptr = 0        # 当前写入位置
        self.size = 0       # 当前有效数据量
        self.positive_cost_count = 0

    def push(self, state, action, reward, next_state, done, cost=0.0,
             applied_action=None):
        """存储经验 — O(1) 数组写入"""
        max_priority = self.priorities[:self.size].max() if self.size > 0 else 1.0

        replacing_positive = bool(
            self.ptr < self.size
            and self.costs[self.ptr, 0] > self.positive_cost_epsilon
        )
        stored_cost = max(float(cost), 0.0)
        incoming_positive = stored_cost > self.positive_cost_epsilon

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.applied_actions[self.ptr] = (
            action if applied_action is None else applied_action
        )
        self.rewards[self.ptr, 0] = reward
        self.costs[self.ptr, 0] = stored_cost
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr, 0] = done
        self.priorities[self.ptr] = max_priority

        self.positive_cost_count += int(incoming_positive)
        self.positive_cost_count -= int(replacing_positive)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """按优先级采样 — 返回与旧接口兼容的元组"""
        if self.size < batch_size:
            return None

        # 计算采样概率
        probs = self.priorities[:self.size] ** self.alpha
        probs = np.nan_to_num(probs, nan=1.0/self.size)  # 防NaN崩掉
        probs /= probs.sum()

        # 采样索引 (replace=True: 200K buffer取512个, 重复概率<0.1%, 无视)
        indices = self.rng.choice(
            self.size, batch_size, p=probs, replace=True
        )

        # 重要性采样权重
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.costs[indices],
            self.next_states[indices],
            self.dones[indices],
            indices,
            weights.astype(np.float32).reshape(-1, 1),
        )

    def update_priorities(self, indices, td_errors):
        """更新优先级"""
        self.priorities[indices] = np.abs(td_errors) + 1e-6

    def __len__(self):
        return self.size
