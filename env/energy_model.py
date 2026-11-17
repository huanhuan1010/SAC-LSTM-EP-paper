"""
无人机能耗模型
参考: P235 (TVT 2020) 旋翼飞行功耗闭合解
      P007 (TWC 2024) 高度-计算力退化模型
"""
import numpy as np


class EnergyModel:
    def __init__(self, config):
        # 旋翼参数 (P235: 诱导功率+型阻功率+寄生功率)
        self.P0 = float(config.get('P0', 100.0))  # blade profile power (W)
        self.P1 = float(config.get('P1', 90.0))   # hover induced power (W)
        self.v_tip = config.get('v_tip', 120.0) # 旋翼叶尖速度 (m/s)
        self.v0 = float(config.get('v0', 4.03)) # hover induced velocity (m/s)
        self.d0 = config.get('d0', 0.6)         # 机身阻力比
        self.rho = config.get('rho', 1.225)     # 空气密度 (kg/m^3)
        self.s = config.get('s', 0.05)          # 转子实度
        self.A = config.get('A', 0.79)          # 旋翼盘面积 (m^2)

        # 计算能耗参数
        self.kappa = float(config.get('kappa', 1e-28)) # 有效开关电容系数
        self.kappa_user = float(config.get('kappa_user', 1e-28))
        self.f0 = config.get('f0', 2.0e9)       # 地面计算频率 (Hz)
        self.f_uav_max = config.get('f_uav_max', 1.5e9)  # 机载最大频率
        self.f_user = config.get('f_user', 1e9)  # 用户设备固定频率 (地面, 与无人机高度无关)

        # 通信能耗参数 (A2修复: P_comm = P_tx/η + P_circuit)
        self.pa_efficiency = float(config.get('pa_efficiency', 0.3))  # 功放效率
        self.p_circuit = float(config.get('p_circuit', 5.0))          # 射频前端电路功耗 (W)
        self.p_rx_uav = float(config.get('p_rx_uav', 3.0))            # UAV接收机功耗 (W)
        if self.pa_efficiency <= 0.0:
            raise ValueError('pa_efficiency must be positive')

        # 电池参数
        self.battery_capacity = config.get('battery_capacity', 100.0)  # J
        self.E_min = config.get('E_min_ratio', 0.20) * self.battery_capacity # 最低安全能量
        self.safety_reserve_ratio = float(config.get('safety_reserve_ratio', 0.05))
        self.safety_power_gain = float(config.get('safety_power_gain', 0.5))
        self.safety_power_scale_max = float(
            config.get('safety_power_scale_max', 2.0)
        )
        if self.safety_power_scale_max < 1.0:
            raise ValueError('safety_power_scale_max must be >= 1')

        # 分段阈值是可配置超参数，应通过验证集搜索和敏感性实验确定。
        self.zone_threshold_high = float(config.get('zone_threshold_high', 0.60))
        self.zone_threshold_low = float(config.get('zone_threshold_low', 0.30))
        if not 0 <= self.zone_threshold_low < self.zone_threshold_high <= 1:
            raise ValueError('energy zone thresholds must satisfy 0 <= low < high <= 1')

        self.return_speed = float(config.get('return_speed', 10.0))
        self.descent_speed = float(config.get('return_descent_speed', 3.0))
        self.vertical_power_model = str(
            config.get('vertical_power_model', 'axial_momentum')
        )
        if self.P0 < 0.0 or self.P1 < 0.0:
            raise ValueError('P0 and P1 must be nonnegative')
        if self.v0 <= 0.0:
            raise ValueError('v0 must be positive')
        if self.return_speed <= 0.0 or self.descent_speed <= 0.0:
            raise ValueError('return and descent speeds must be positive')
        if self.vertical_power_model != 'axial_momentum':
            raise ValueError('vertical_power_model must be axial_momentum')

    def flight_power(self, v, h=100):
        """
        旋翼飞行功率 (P235 闭合模型)
        P_fly(v) = P0*(1+3v²/v_tip²)
                 + P1*√(√(1+v⁴/(4v0⁴)) - v²/(2v0²))
                 + 0.5*d0*ρ*s*A*v³
        """
        v_norm = np.linalg.norm(v) if isinstance(v, np.ndarray) else abs(v)

        # 桨叶剖面功率
        P_profile = self.P0 * (1 + 3 * v_norm**2 / self.v_tip**2)
        # 诱导功率
        inner = np.sqrt(1 + v_norm**4 / (4 * self.v0**4))
        P_induced = self.P1 * (
            np.sqrt(inner - v_norm**2 / (2 * self.v0**2))
        )
        # 寄生功率
        P_parasite = 0.5 * self.d0 * self.rho * self.s * self.A * v_norm**3

        return P_profile + np.nan_to_num(P_induced, nan=0.0) + P_parasite

    def vertical_flight_power(self, vertical_speed):
        """Return steady axial-flight power for signed vertical airspeed.

        ``vertical_speed`` is positive for climb and negative for descent.
        With hover induced velocity ``v0`` and induced hover power ``P1``,
        the equivalent steady thrust is ``T = P1 / v0``.  Axial momentum
        theory then gives

            P_z(V_z) = P0 + T/2 * (V_z + sqrt(V_z^2 + 4*v0^2)).

        This actuator-disk approximation does not resolve acceleration,
        ground effect, or vortex-ring/turbulent-wake descent aerodynamics.
        """
        vertical_speed = float(vertical_speed)
        equivalent_thrust = self.P1 / self.v0
        induced_and_climb = 0.5 * equivalent_thrust * (
            vertical_speed
            + np.sqrt(vertical_speed**2 + 4.0 * self.v0**2)
        )
        return float(max(self.P0 + induced_and_climb, 0.0))

    def propulsion_power(self, velocity, h=100):
        """Return steady 3-D propulsion power without double counting hover.

        ``P_3D = P_horizontal(||v_xy||) + P_vertical(v_z) - P_hover``.
        A scalar retains the historical interpretation of horizontal airspeed.
        """
        values = np.asarray(velocity, dtype=np.float64)
        if values.ndim == 0:
            return float(self.flight_power(float(values), h))
        flat = values.reshape(-1)
        if flat.size < 3:
            return float(self.flight_power(np.linalg.norm(flat), h))
        horizontal_speed = float(np.linalg.norm(flat[:2]))
        vertical_speed = float(flat[2])
        hover_power = float(self.flight_power(0.0, h))
        combined = (
            float(self.flight_power(horizontal_speed, h))
            + self.vertical_flight_power(vertical_speed)
            - hover_power
        )
        return float(max(combined, 0.0))

    def compute_power(self, f_uav):
        """计算能耗 P_comp = κ * f_uav^3"""
        return self.kappa * f_uav**3

    def compute_energy(self, f_uav, executed_cycles):
        """按实际执行周期计算UAV动态计算能耗: E=κ f² C_exec。"""
        frequency = max(float(f_uav), 0.0)
        cycles = max(float(executed_cycles), 0.0)
        return self.kappa * frequency**2 * cycles

    def local_compute_energy(self, executed_cycles):
        """用户本地计算能耗；仅计入系统能耗，不扣减UAV电池。"""
        cycles = max(float(executed_cycles), 0.0)
        return self.kappa_user * float(self.f_user)**2 * cycles

    def freq_at_altitude(self, h):
        """高度-计算频率退化 (P007)
        f_UAV(h) = f0 * (1 - 0.0012 * h)
        """
        return self.f0 * (1 - 0.0012 * h)

    def communication_power(self, P_tx):
        """用户上行发送功耗 = 功放功耗(P_tx/η) + 发射电路功耗。

        该功耗属于地面用户，不直接扣减UAV电池；UAV接收机功耗由
        ``p_rx_uav`` 单独建模。
        """
        return P_tx / self.pa_efficiency + self.p_circuit

    def step_energy(self, E_rem, v, h, f_uav, P_tx, dt, *,
                    uav_cycles_executed=0.0, uplink_time_sum=0.0,
                    uav_rx_active_time=0.0, local_cycles_executed=0.0,
                    uav_energy_scale=1.0):
        """根据实际工作量和外生倍率计算UAV电池与系统能耗。

        UAV电池只承担飞行、机载计算和接收机能耗。用户本地计算与上行
        发送能耗单独统计，二者相加得到系统总能耗。名义UAV能耗先由
        物理模型计算，再整体乘以 ``uav_energy_scale`` 得到实际扣电量。
        """
        uav_energy_scale = float(uav_energy_scale)
        if not np.isfinite(uav_energy_scale) or uav_energy_scale <= 0.0:
            raise ValueError('uav_energy_scale must be finite and positive')

        E_fly_nominal = self.propulsion_power(v, h) * dt
        E_comp_nominal = self.compute_energy(f_uav, uav_cycles_executed)
        E_comm_nominal = (
            self.p_rx_uav * max(float(uav_rx_active_time), 0.0)
        )
        E_uav_nominal = E_fly_nominal + E_comp_nominal + E_comm_nominal
        # The disturbance represents battery-side conversion, payload, and
        # unmodelled actuator/load effects.  It therefore scales all UAV
        # battery components while leaving user-device energy unchanged.
        E_fly = E_fly_nominal * uav_energy_scale
        E_comp = E_comp_nominal * uav_energy_scale
        E_comm = E_comm_nominal * uav_energy_scale
        E_uav_total = E_uav_nominal * uav_energy_scale

        E_user_local = self.local_compute_energy(local_cycles_executed)
        E_user_comm = (
            self.communication_power(P_tx)
            * max(float(uplink_time_sum), 0.0)
        )
        E_user_total = E_user_local + E_user_comm
        E_system_nominal = E_uav_nominal + E_user_total
        E_system_total = E_uav_total + E_user_total

        E_next = E_rem - E_uav_total
        done = E_next <= self.E_min
        breakdown = {
            'E_fly': E_fly,
            'E_comp': E_comp,
            'E_comm': E_comm,
            'E_uav_total': E_uav_total,
            'E_fly_nominal': E_fly_nominal,
            'E_comp_nominal': E_comp_nominal,
            'E_comm_nominal': E_comm_nominal,
            'E_uav_nominal': E_uav_nominal,
            'uav_energy_uncertainty_scale': uav_energy_scale,
            'E_uav_uncertainty_delta': E_uav_total - E_uav_nominal,
            'E_user_local': E_user_local,
            'E_user_comm': E_user_comm,
            'E_user_total': E_user_total,
            'E_system_nominal': E_system_nominal,
            'E_system_total': E_system_total,
        }
        return E_next, E_uav_total, breakdown, done

    def safety_reserve(self, average_power=None, uncertainty_energy=0.0):
        """返回负载感知的最低安全能量下界 (J)。

        真实系统可根据端电压、内阻和推力能力辨识最低可用 SoC；当前仿真
        缺少这些测量量，因此采用可解释的代理模型：最低运行能量 E_min 加上
        随平均功率增加的基础储备，并叠加近期预测绝对残差对应的不确定性能量。
        该值是能量裕度公式中的 E_safe，而不只是额外的固定百分比。
        """
        nominal_buffer = self.safety_reserve_ratio * self.battery_capacity
        load_scale = 1.0
        if average_power is not None:
            hover_power = max(float(self.flight_power(0.0)), 1e-6)
            load_ratio = max(float(average_power), 0.0) / hover_power
            load_scale = 1.0 + self.safety_power_gain * max(load_ratio - 1.0, 0.0)
            load_scale = float(np.clip(
                load_scale, 1.0, self.safety_power_scale_max
            ))

        reserve = (
            self.E_min
            + nominal_buffer * load_scale
            + max(float(uncertainty_energy), 0.0)
        )
        return min(reserve, self.battery_capacity)

    def energy_margin(self, E_rem, predicted_energy=0.0, return_energy=0.0,
                      safety_reserve=None):
        """返回可用于新任务的能量裕度及其归一化值。

        margin = E_rem - E_pred - E_return - E_safe
        其中预测、返航和安全储备均为非负能量（J）。
        """
        predicted = max(float(predicted_energy), 0.0)
        required_return = max(float(return_energy), 0.0)
        reserve = self.safety_reserve() if safety_reserve is None else max(
            float(safety_reserve), 0.0
        )
        margin = float(E_rem) - predicted - required_return - reserve
        return margin, margin / self.battery_capacity

    @staticmethod
    def energy_margin_time(energy_margin, average_power):
        """按当前平均功率把正能量裕度换算为等效持续时间 (s)。

        该值只是 ``max(E_margin, 0) / P_avg``，不是完整任务的剩余飞行
        时间，也不包含未来轨迹、风场或负载变化。能量条件不满足时返回0，
        判定时必须同时读取有符号的 ``energy_margin``。
        """
        power = max(float(average_power), 1e-6)
        return max(float(energy_margin), 0.0) / power

    def get_energy_zone(self, E_rem, predicted_energy=0.0, return_energy=0.0,
                        safety_reserve=None):
        """按归一化能量裕度划分状态区间，仅返回区间名称。

        区间用于观测编码和诊断，不再携带或改变奖励权重。
        """
        _, ratio = self.energy_margin(
            E_rem, predicted_energy, return_energy, safety_reserve
        )
        if ratio > self.zone_threshold_high:
            return 'high'
        elif ratio > self.zone_threshold_low:
            return 'medium'
        return 'low'

    def return_energy(self, horizontal_distance, altitude=None,
                      landing_altitude=None, return_direction=None,
                      wind_velocity=None, uav_energy_scale=1.0):
        """估算水平返航与稳态轴向下降所需的飞行能量 (J)。

        下降阶段调用独立的垂直飞行功率模型，不再使用悬停功率近似。模型仍
        假设匀速稳态轴向飞行，不包含加减速、地面效应以及涡环/湍流尾迹效应。
        ``uav_energy_scale`` 可用于统一施加声明的有界能耗不确定性上界。
        """
        uav_energy_scale = float(uav_energy_scale)
        if not np.isfinite(uav_energy_scale) or uav_energy_scale <= 0.0:
            raise ValueError('uav_energy_scale must be finite and positive')
        horizontal_distance = max(float(horizontal_distance), 0.0)
        horizontal_time = horizontal_distance / max(self.return_speed, 1e-6)
        return_airspeed = self.return_speed
        if return_direction is not None and wind_velocity is not None:
            direction = np.asarray(
                return_direction, dtype=np.float64
            ).reshape(-1)[:2]
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 1e-9:
                direction = direction / direction_norm
                wind = np.asarray(
                    wind_velocity, dtype=np.float64
                ).reshape(-1)[:2]
                desired_ground_velocity = self.return_speed * direction
                return_airspeed = float(np.linalg.norm(
                    desired_ground_velocity - wind
                ))
        energy = self.flight_power(return_airspeed) * horizontal_time

        if altitude is not None and landing_altitude is not None:
            descent_distance = max(float(altitude) - float(landing_altitude), 0.0)
            descent_time = descent_distance / max(self.descent_speed, 1e-6)
            descent_power = self.vertical_flight_power(-self.descent_speed)
            energy += descent_power * descent_time
        return energy * uav_energy_scale
