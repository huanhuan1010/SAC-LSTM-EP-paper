"""通信模型: LoS信道 + 干扰模型"""
import numpy as np


class CommModel:
    def __init__(self, config):
        self.bandwidth = config.get('bandwidth', 1e6)        # 1MHz
        self.beta0 = config.get('beta0', -30)                # 参考距离信道增益 (dB)
        self.alpha = config.get('alpha', 2.5)                # 路径损耗指数
        self.noise_power = config.get('noise_power', 1e-13)  # 噪声功率 (W)
        self.f_c = config.get('f_c', 2.4e9)                  # 载频 2.4GHz

    def path_loss(self, distance):
        """
        线性信道增益: PL_linear = 10^(beta0/10) * d^(-alpha)
        beta0 = -30dB → 1m处增益 = 10^(-3) = 0.001
        d=200m时增益 ≈ 1.77e-7, d=50m时 ≈ 2.83e-6
        """
        pl_1m = 10 ** (self.beta0 / 10)  # -30dB → 0.001 (线性)
        return pl_1m * (distance ** (-self.alpha))

    def path_loss_db(self, distance):
        """路径损耗(dB), 供MDP状态使用"""
        return self.beta0 - 10 * self.alpha * np.log10(distance + 1e-10)

    def sinr(self, P_tx, distance, interference=0):
        """计算SINR (path_loss已返回线性增益, 不再需要dB转换)"""
        h = self.path_loss(distance)  # 线性增益
        signal = P_tx * h
        noise = self.noise_power * self.bandwidth
        return signal / (noise + interference + 1e-15)

    def data_rate(self, P_tx, distance, interference=0):
        """上行速率: R = B * log2(1 + SINR)"""
        sinr_val = self.sinr(P_tx, distance, interference)
        return self.bandwidth * np.log2(1 + sinr_val)

    def distance(self, uav_pos, user_pos):
        """计算无人机到用户的欧氏距离"""
        dx = uav_pos[0] - user_pos[0]
        dy = uav_pos[1] - user_pos[1]
        dz = uav_pos[2] - user_pos[2] if len(uav_pos) > 2 else uav_pos[2] if len(uav_pos) > 2 else 100
        return np.sqrt(dx**2 + dy**2 + dz**2)
