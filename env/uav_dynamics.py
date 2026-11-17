"""无人机运动学模型"""
import numpy as np


class UAVDynamics:
    def __init__(self, config):
        self.v_max = config.get('v_max', 20.0)          # 最大速度 (m/s)
        self.a_max = config.get('a_max', 5.0)           # 最大加速度 (m/s²)
        self.h_min = config.get('h_min', 30.0)          # 最低高度 (m)
        self.h_max = config.get('h_max', 300.0)         # 最高高度 (m)
        self.h_default = config.get('h_default', 100.0)  # 默认高度

        # 初始位置
        self.area_size = config.get('area_size', 500.0)
        self.base_pos = config.get('base_pos', np.array([250.0, 250.0, 0]))  # 基站位置
        self.rng = np.random.default_rng()

    def initialize(self, rng=None):
        """重置无人机初始状态"""
        if rng is not None:
            self.rng = rng
        self.pos = np.array([
            self.rng.uniform(self.area_size * 0.2, self.area_size * 0.8),
            self.rng.uniform(self.area_size * 0.2, self.area_size * 0.8),
            self.h_default,
        ])
        self.vel = np.zeros(3)
        # Ground velocity is exposed to the trajectory controller. Air-relative
        # velocity is used by the rotor propulsion model under wind.
        self.air_vel = np.zeros(3)
        return self.pos.copy(), self.vel.copy()

    def update(self, action, dt, wind_velocity=None):
        """
        更新无人机状态
        action: [θ(偏航角), φ(俯仰角), speed]
        θ: [-π, π]  水平方向
        φ: [-π/4, π/4]  垂直方向
        speed: [0, v_max]  速度大小
        """
        theta, phi, speed = action[0], action[1], np.clip(action[2], 0, self.v_max)
        wind = (
            np.zeros(3, dtype=np.float64)
            if wind_velocity is None
            else np.asarray(wind_velocity, dtype=np.float64).reshape(3)
        )

        # 目标速度向量
        v_target = np.array([
            speed * np.cos(phi) * np.cos(theta),
            speed * np.cos(phi) * np.sin(theta),
            speed * np.sin(phi),
        ])

        # 加速度限幅
        acc = (v_target - self.vel) / dt
        acc_norm = np.linalg.norm(acc)
        if acc_norm > self.a_max:
            acc = acc / acc_norm * self.a_max

        # 更新速度
        ground_candidate = self.vel + acc * dt
        required_air_velocity = ground_candidate - wind
        airspeed = np.linalg.norm(required_air_velocity)
        if airspeed > self.v_max:
            required_air_velocity = (
                required_air_velocity / airspeed * self.v_max
            )
        self.air_vel = required_air_velocity
        self.vel = self.air_vel + wind

        # 更新位置
        self.pos = self.pos + self.vel * dt

        # 高度约束
        self.pos[2] = np.clip(self.pos[2], self.h_min, self.h_max)
        if self.pos[2] <= self.h_min or self.pos[2] >= self.h_max:
            self.vel[2] = 0  # 碰到高度边界时停止垂直运动

        if self.pos[2] <= self.h_min or self.pos[2] >= self.h_max:
            self.air_vel[2] = 0
        return self.pos.copy(), self.vel.copy()

    def distance_to_base(self):
        """到基站的距离"""
        return np.linalg.norm(self.pos - self.base_pos)

    def horizontal_distance_to_base(self):
        """到基站的水平距离；用于返航时间与能耗估计。"""
        return np.linalg.norm(self.pos[:2] - self.base_pos[:2])

    def get_state(self):
        """返回状态向量"""
        return np.concatenate([self.pos.copy(), self.vel.copy()])
