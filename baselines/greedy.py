"""
贪心启发式策略基线: 基于规则的卸载+固定速度巡航
参考: P119-ACO启发式, 简单规则触发
"""
import numpy as np


class GreedyPolicy:
    """
    贪心策略规则:
    - 卸载比例: 根据信道质量与队列长度决定
    - 轨迹: 以显式配置的固定速度巡航, 低能量模式下返航
    - 计算频率与发射功率: 使用中档固定值
    """

    def __init__(self, state_dim, action_dim, config):
        self.num_users = config.get('num_users', 5)
        self.action_dim = int(action_dim)
        self.policy_action_mode = str(config.get(
            'policy_action_mode', 'joint_control'
        )).strip().lower()
        expected_action_dim = self.num_users + (
            2 if self.policy_action_mode == 'offloading_only' else 5
        )
        if self.action_dim != expected_action_dim:
            raise ValueError(
                f'Greedy action dimension mismatch: expected '
                f'{expected_action_dim}, got {self.action_dim}'
            )
        self.area_size = config.get('area_size', 500.0)
        self.v_max = config.get('v_max', 20.0)
        self.cruise_speed = float(config.get('greedy_speed', 10.0))
        if not 0.0 <= self.cruise_speed <= self.v_max:
            raise ValueError('greedy_speed must lie in [0, v_max]')
        self.name = 'Greedy'

        # 状态索引。新增任务进度特征位于Q_i之后，因此以下切片保持不变。
        self.E_idx = 0            # 剩余能量占比
        self.zone_idx = slice(1, 4)  # 能量区间 one-hot
        self.pos_idx = slice(4, 7)   # 归一化位置
        self.vel_idx = slice(7, 10)  # 归一化速度
        self.h_idx = slice(10, 10 + self.num_users)   # 信道增益
        self.Q_idx = slice(10 + self.num_users, 10 + 2 * self.num_users)  # 队列长度
        self.E_pred_idx = slice(10 + 7 * self.num_users,
                                10 + 7 * self.num_users + config.get('horizon', 5))

    def select_action(self, state, eval=False, energy_mode='normal',
                      lambda_max=1.0, speed_max=1.0, **kwargs):
        """
        根据当前状态决定动作
        """
        action = np.zeros(self.action_dim, dtype=np.float32)

        # 1. 卸载比例: 根据队列长度比例分配
        Q_i = state[self.Q_idx]
        total_Q = max(sum(Q_i), 1e-6)
        h_i = np.abs(state[self.h_idx])

        # 高信道增益 + 长队列 → 多卸载
        lam = h_i * Q_i / (h_i * Q_i).sum() if (h_i * Q_i).sum() > 0 else np.ones(self.num_users) / self.num_users
        lam = lam * lambda_max  # 受能量模式约束

        # 归一化到 [-1, 1] (env会映射回[0,1])
        action[:self.num_users] = lam * 2 - 1

        if self.policy_action_mode == 'offloading_only':
            # Mobility is supplied by the environment's shared deterministic
            # controller.  Greedy controls only CPU and transmit power here.
            action[self.num_users] = 0.0
            action[self.num_users + 1] = 0.0
            return np.clip(action, -1.0, 1.0).astype(np.float32)

        # 2. 轨迹: 固定速度缓慢绕飞。环境速度动作使用[-1,1]，必须
        # 显式反算；动作0对应0.5*v_max，而不是悬停。
        action[self.num_users] = np.sin(state[4] * 2 * np.pi) * 0.3  # 小角度偏航
        action[self.num_users + 1] = 0.0  # 俯仰: 保持水平
        action[self.num_users + 2] = (
            2.0 * self.cruise_speed / max(self.v_max, 1e-6) - 1.0
        )

        # 低电量时飞向基站
        E_rem = state[self.E_idx]
        if energy_mode in ('orange', 'red'):
            # 偏航指向基站 (此处简化: 假设基站在区域中心)
            pos = state[self.pos_idx] * self.area_size
            center = np.array([self.area_size / 2, self.area_size / 2])
            dx = center[0] - pos[0]
            dy = center[1] - pos[1]
            target_theta = np.arctan2(dy, dx)
            action[self.num_users] = target_theta / np.pi  # 归一化到[-1,1]
            action[self.num_users + 2] = speed_max * 2 - 1  # 返航速度

        # 3. CPU频率 & 发射功率: 中档
        action[self.num_users + 3] = 0.0  # [-1,1] → 映射到0.1~1.0 → 约0.55
        action[self.num_users + 4] = 0.0  # P_tx: [-1,1] → 映射到 [p_tx_min, p_tx_max] 中点

        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def update(self):
        return {}

    def save(self, path):
        pass

    def load(self, path):
        pass
