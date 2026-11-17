"""三级应急迫降管理模块"""
import numpy as np
from utils.safety_constraints import (
    mode_from_margin_ratio,
    safety_constraints,
    validate_safety_thresholds,
)


class EmergencyManager:
    def __init__(self, config):
        self.config = dict(config)
        self.battery_capacity = config.get('battery_capacity', 100.0)
        self.dt = float(config.get('dt', 0.5))
        thresholds = validate_safety_thresholds(config)
        self.threshold_yellow = thresholds['yellow']
        self.threshold_orange = thresholds['orange']
        self.threshold_red = thresholds['red']

        # 安全储备从剩余能量中显式扣除，不再将其混为“预测误差裕度”。
        self.safety_reserve_ratio = float(
            config.get('safety_reserve_ratio', config.get('prediction_margin', 0.05))
        )
        self.minimum_operating_ratio = float(config.get('E_min_ratio', 0.20))

    def check(self, E_rem, E_pred=None, return_energy=0.0, safety_reserve=None):
        """
        检查当前能量状态，返回 (mode, constraint_dict)
        E_rem: 当前剩余能量 (J)
        E_pred: LSTM预测的未来N步能耗序列 (J)
        return_energy: 从当前位置安全返航所需能量估计 (J)
        safety_reserve: 额外安全储备 (J)，默认由配置比例给出
        """
        pred = np.asarray(E_pred if E_pred is not None else [], dtype=np.float64)
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        predicted_energy = float(np.clip(pred, 0.0, None).sum())
        required_return = max(float(return_energy), 0.0)
        if safety_reserve is None:
            safety_reserve = (
                self.minimum_operating_ratio + self.safety_reserve_ratio
            ) * self.battery_capacity
        safety_reserve = max(float(safety_reserve), 0.0)

        # 预警依据为能量裕度，而非单独的电池百分比。
        margin = E_rem - predicted_energy - required_return - safety_reserve
        effective_ratio = margin / self.battery_capacity

        diagnostics = {
            'prediction_horizon_steps': int(pred.size),
            'prediction_horizon_seconds': int(pred.size) * self.dt,
            'energy_margin': margin,
            'energy_margin_ratio': effective_ratio,
            'short_horizon_energy_feasible': bool(margin >= 0.0),
            'energy_feasibility_scope': (
                'prediction_horizon_plus_current_position_return'
            ),
            'predicted_energy': predicted_energy,
            'return_energy': required_return,
            'safety_reserve': safety_reserve,
            'minimum_safe_energy': safety_reserve,
        }

        mode = mode_from_margin_ratio(effective_ratio, self.config)
        descriptions = {
            'red': '🔴 红色预警: 硬安全返航接管',
            'orange': '🟠 橙色预警: 暂停非核心UAV服务',
            'yellow': '🟡 黄色预警: 节能限幅',
            'normal': '✅ 正常模式',
        }
        return mode, {
            **safety_constraints(self.config, mode),
            'description': descriptions[mode],
            **diagnostics,
        }

    def get_safe_landing_trajectory(self, uav_pos, base_pos, h_min=5):
        """
        CBF安全迫降轨迹 (简化版本)
        返回: (目标位置, 目标速度)
        """
        direction = base_pos[:2] - uav_pos[:2]
        dist = np.linalg.norm(direction)
        if dist > 0:
            direction = direction / dist

        # 目标: 向基站飞 + 缓慢下降
        target_pos = np.array([
            uav_pos[0] + direction[0] * min(dist, 20),
            uav_pos[1] + direction[1] * min(dist, 20),
            max(uav_pos[2] - 2, h_min),  # 缓慢下降
        ])
        target_vel = direction * 5.0  # 5m/s 低速返航

        return target_pos, target_vel
