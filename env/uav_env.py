"""
UAV-MEC Gym环境 (SAC-LSTM-EP)
MDP五元组: (S, A, P, R, γ)
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .energy_model import EnergyModel
from .energy_uncertainty import EnergyUncertaintyModel
from .comm_model import CommModel
from .task_generator import TaskGenerator
from .uav_dynamics import UAVDynamics
from .wind_model import WindModel
from utils.safety_constraints import (
    mode_from_margin_ratio,
    project_normalized_action,
    safety_constraints,
    stricter_mode,
    validate_safety_thresholds,
)


class UAVEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config):
        super().__init__()
        self.config = dict(config)

        # 子模块
        self.energy = EnergyModel(config)
        self.energy_uncertainty = EnergyUncertaintyModel(config)
        self.comm = CommModel(config)
        self.task_gen = TaskGenerator(config)
        self.uav = UAVDynamics(config)
        self.wind = WindModel(config)

        # 时间参数
        self.dt = config.get('dt', 0.5)          # 决策间隔 (s)
        self.max_steps = config.get('max_steps', 200)
        self.mission_return_enabled = bool(
            config.get('mission_return_enabled', False)
        )
        self.mission_service_steps = int(
            config.get('mission_service_steps', max(self.max_steps - 40, 1))
        )
        self.mission_min_completed_tasks = int(
            config.get('mission_min_completed_tasks', 1)
        )
        self.mission_min_completion_ratio = float(
            config.get('mission_min_completion_ratio', 0.80)
        )
        self.mission_min_core_completion_ratio = float(
            config.get('mission_min_core_completion_ratio', 0.80)
        )
        self.mission_require_full_service_window = bool(
            config.get('mission_require_full_service_window', True)
        )
        self.mission_require_energy_feasible = bool(
            config.get('mission_require_energy_feasible', True)
        )
        if self.mission_return_enabled and not (
                1 <= self.mission_service_steps < self.max_steps):
            raise ValueError(
                'mission_service_steps must lie in [1, max_steps) when '
                'mission_return_enabled is true'
            )
        if self.mission_min_completed_tasks < 0:
            raise ValueError('mission_min_completed_tasks must be nonnegative')
        for name, value in (
            ('mission_min_completion_ratio',
             self.mission_min_completion_ratio),
            ('mission_min_core_completion_ratio',
             self.mission_min_core_completion_ratio),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f'{name} must lie in [0, 1]')
        self.lookback = config.get('lookback', 10)   # LSTM回溯步数
        self.horizon = config.get('horizon', 5)      # LSTM预测步数
        # A2: 发射功率边界 (供step()映射)
        self.config_p_tx_min = config.get('p_tx_min', 0.1)
        self.config_p_tx_max = config.get('p_tx_max', 5.0)
        self.initial_energy_min = float(config.get('initial_energy_min', 0.80))
        self.initial_energy_max = float(config.get('initial_energy_max', 1.00))
        if not 0 < self.initial_energy_min <= self.initial_energy_max <= 1:
            raise ValueError('initial energy range must lie in (0, 1]')

        # 所有算法共享的固定基础奖励权重，不依赖安全状态或预测裕度。
        self.energy_penalty_scale = float(config.get('energy_penalty_scale', 5.0))
        self.energy_normalizer = float(config.get('energy_normalizer', 100.0))
        self.delay_penalty_scale = float(config.get('delay_penalty_scale', 2.0))
        self.reward_energy_weight = float(
            config.get('reward_energy_weight', 0.5)
        )
        self.reward_delay_weight = float(
            config.get('reward_delay_weight', 0.5)
        )
        self.reward_user_energy_weight = float(
            config.get('reward_user_energy_weight', 1.0)
        )
        if self.reward_energy_weight < 0.0 or self.reward_delay_weight < 0.0:
            raise ValueError('fixed reward weights must be nonnegative')
        if self.reward_energy_weight + self.reward_delay_weight <= 0.0:
            raise ValueError('at least one fixed reward weight must be positive')
        if self.reward_user_energy_weight < 0.0:
            raise ValueError('reward_user_energy_weight must be nonnegative')
        self.prediction_uncertainty_gain = float(
            config.get('prediction_uncertainty_gain', 1.0)
        )
        thresholds = validate_safety_thresholds(config)
        self.threshold_yellow = thresholds['yellow']
        self.threshold_orange = thresholds['orange']
        self.threshold_red = thresholds['red']
        self.soc_return_threshold = float(
            config.get('soc_return_threshold', 0.20)
        )
        self.predictive_return_threshold = float(
            config.get('predictive_return_threshold', self.threshold_red)
        )
        self.single_level_margin_threshold = float(
            config.get('single_level_margin_threshold', self.threshold_yellow)
        )
        for name, value in (
            ('soc_return_threshold', self.soc_return_threshold),
            ('predictive_return_threshold', self.predictive_return_threshold),
            ('single_level_margin_threshold',
             self.single_level_margin_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f'{name} must lie in [0, 1]')
        # CMDP training cost.  The proposed method keeps the original
        # predictor-dependent energy-margin signal.  The adapted C-SAC
        # baseline can instead use the Alvi-style latency signal.  The
        # ``latency_violation`` variant adds an explicit deadline-miss event
        # term to the overshoot severity so that the training cost is aligned
        # with the independently reported task-deadline violation rate.
        # Evaluation metrics remain independent of every training cost.
        self.cmdp_cost_mode = str(
            config.get('cmdp_cost_mode', 'energy_margin')
        ).strip().lower()
        if self.cmdp_cost_mode not in (
                'energy_margin', 'latency_overshoot', 'latency_violation'):
            raise ValueError(
                'cmdp_cost_mode must be energy_margin, latency_overshoot, '
                'or latency_violation'
            )
        self.cmdp_margin_target = float(config.get(
            'cmdp_margin_target',
            config.get('constraint_margin_target', self.threshold_red),
        ))
        self.cmdp_cost_scale = float(config.get(
            'cmdp_cost_scale', config.get('constraint_cost_scale', 0.20)
        ))
        self.cmdp_energy_cost_source = str(config.get(
            'cmdp_energy_cost_source', 'post_transition_state'
        )).strip().lower()
        if self.cmdp_energy_cost_source not in (
                'post_transition_state', 'raw_action_counterfactual'):
            raise ValueError(
                'cmdp_energy_cost_source must be post_transition_state or '
                'raw_action_counterfactual'
            )
        self.cmdp_action_forecast_replacement_steps = int(config.get(
            'cmdp_action_forecast_replacement_steps', 1
        ))
        if not 1 <= self.cmdp_action_forecast_replacement_steps <= self.horizon:
            raise ValueError(
                'cmdp_action_forecast_replacement_steps must lie in '
                '[1, horizon]'
            )
        self.cmdp_energy_cost_max = float(config.get(
            'cmdp_energy_cost_max', 1.0
        ))
        self.cmdp_terminal_cost = float(config.get(
            'cmdp_terminal_cost',
            config.get('constraint_terminal_cost', 1.0),
        ))
        self.cmdp_latency_cost_scale = float(config.get(
            'cmdp_latency_cost_scale', 0.05
        ))
        self.cmdp_latency_miss_weight = float(config.get(
            'cmdp_latency_miss_weight', 1.0
        ))
        self.cmdp_latency_overshoot_weight = float(config.get(
            'cmdp_latency_overshoot_weight', 1.0
        ))
        if not 0.0 <= self.cmdp_margin_target <= 1.0:
            raise ValueError('cmdp_margin_target must lie in [0, 1]')
        if self.cmdp_cost_scale <= 0.0:
            raise ValueError('cmdp_cost_scale must be positive')
        if self.cmdp_energy_cost_max <= 0.0:
            raise ValueError('cmdp_energy_cost_max must be positive')
        if self.cmdp_terminal_cost < 0.0:
            raise ValueError('cmdp_terminal_cost must be nonnegative')
        self.cmdp_cost_saturation_threshold = max(
            self.cmdp_energy_cost_max,
            self.cmdp_terminal_cost,
            1.0,
        )
        if self.cmdp_latency_cost_scale <= 0.0:
            raise ValueError('cmdp_latency_cost_scale must be positive')
        if self.cmdp_latency_miss_weight < 0.0:
            raise ValueError('cmdp_latency_miss_weight must be nonnegative')
        if self.cmdp_latency_overshoot_weight < 0.0:
            raise ValueError(
                'cmdp_latency_overshoot_weight must be nonnegative'
            )
        if (
            self.cmdp_cost_mode == 'latency_violation'
            and self.cmdp_latency_miss_weight == 0.0
            and self.cmdp_latency_overshoot_weight == 0.0
        ):
            raise ValueError(
                'latency_violation requires a positive miss or overshoot weight'
            )
        self.channel_db_normalizer = float(
            config.get('channel_db_normalizer', 100.0)
        )
        self.queue_length_normalizer = float(
            config.get('queue_length_normalizer', 5.0)
        )
        if self.channel_db_normalizer <= 0.0:
            raise ValueError('channel_db_normalizer must be positive')
        if self.queue_length_normalizer <= 0.0:
            raise ValueError('queue_length_normalizer must be positive')

        # === 状态空间 ===
        # [E_rem, E_zone(one-hot:3), pos_x, pos_y, pos_z,
        #  v_x, v_y, v_z,
        #  h_ij (:num_users), Q_i (:num_users),
        #  head_task_progress (:5*num_users),
        #  E_pred(:horizon), task_priority, energy_margin_ratio,
        #  optional mission_progress and scheduled_return_phase]
        self.num_users = config.get('num_users', 5)
        self.task_progress_features_per_user = 5
        dim_E_pred = self.horizon
        dim_mission = 2 if self.mission_return_enabled else 0
        dim_obs = (1 + 3 + 3 + 3 + self.num_users + self.num_users
                   + self.task_progress_features_per_user * self.num_users
                   + dim_E_pred + 1 + 1 + dim_mission)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(dim_obs,), dtype=np.float32
        )

        # === 动作空间 ===
        # joint_control: [λ_1..λ_N, θ, φ, speed, f_uav, P_tx]
        # offloading_only: [λ_1..λ_N, f_uav, P_tx].  In the latter mode a
        # deterministic mobility controller supplies θ, φ and speed equally
        # for every algorithm, so policy comparisons isolate task offloading.
        # λ: 卸载比例 ∈ [0,1]
        # θ: 偏航角 ∈ [-π, π]
        # φ: 俯仰角 ∈ [-π/4, π/4]
        # speed: 速度 ∈ [0, v_max]
        # f_uav: CPU频率 ∈ [0.1, 1.0] (归一化)
        # P_tx: 发射功率 ∈ [p_tx_min, p_tx_max] = [0.1, 5.0] W (A2: 原最大1W无能耗代价)
        self.policy_action_mode = str(config.get(
            'policy_action_mode', 'joint_control'
        )).strip().lower()
        if self.policy_action_mode not in ('joint_control', 'offloading_only'):
            raise ValueError(
                'policy_action_mode must be joint_control or offloading_only'
            )
        self.common_mobility_speed = float(config.get(
            'common_mobility_speed', 0.0
        ))
        self.common_mobility_heading_deg = float(config.get(
            'common_mobility_heading_deg', 0.0
        ))
        self.common_mobility_pitch_deg = float(config.get(
            'common_mobility_pitch_deg', 0.0
        ))
        if not 0.0 <= self.common_mobility_speed <= self.uav.v_max:
            raise ValueError('common_mobility_speed must lie in [0, v_max]')
        if not -180.0 <= self.common_mobility_heading_deg <= 180.0:
            raise ValueError(
                'common_mobility_heading_deg must lie in [-180, 180]'
            )
        if not -45.0 <= self.common_mobility_pitch_deg <= 45.0:
            raise ValueError(
                'common_mobility_pitch_deg must lie in [-45, 45]'
            )
        dim_action = (
            self.num_users + 2
            if self.policy_action_mode == 'offloading_only'
            else self.num_users + 5
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(dim_action,), dtype=np.float32
        )

        # 状态记录 (用于LSTM)
        self.energy_history = []  # 最近N步能量消耗记录
        self.predicted_horizon_energy = 0.0
        self.raw_predicted_horizon_energy = 0.0
        self.ungated_predicted_horizon_energy = 0.0
        self.predicted_energy_vector = np.zeros(
            self.horizon, dtype=np.float32
        )
        self.raw_predicted_energy_vector = np.zeros(
            self.horizon, dtype=np.float32
        )
        self.ungated_predicted_energy_vector = np.zeros(
            self.horizon, dtype=np.float32
        )
        self.prediction_history_gate_weight = 0.0
        self.prediction_residual_bias = 0.0
        self.prediction_residual_uncertainty = 0.0
        self.prediction_source = 'uninitialized'

        # 紧急迫降状态
        self.emergency_mode = 'normal'
        self.energy_safety_supervisor_enabled = True
        self.energy_safety_controller = 'margin'
        self.safety_projection_policy = str(config.get(
            'safety_projection_policy', 'state_threshold'
        )).strip().lower()
        if self.safety_projection_policy not in (
                'state_threshold', 'action_aware'):
            raise ValueError(
                'safety_projection_policy must be state_threshold or '
                'action_aware'
            )
        self.safety_min_energy_saving = float(config.get(
            'safety_min_energy_saving', 1.0
        ))
        if self.safety_min_energy_saving < 0.0:
            raise ValueError('safety_min_energy_saving must be nonnegative')
        # A positive predictive margin already includes the H-step forecast,
        # return energy and the dynamic reserve.  When this guard is enabled,
        # a positive-margin warning may still request a mild projection, but
        # it cannot be promoted to the hard return takeover merely because a
        # legacy warning threshold is positive.
        self.safety_red_requires_infeasibility = bool(config.get(
            'safety_red_requires_infeasibility', False
        ))
        self.last_reward_breakdown = {}

    @property
    def full_action_dim(self):
        return self.num_users + 5

    def _expand_policy_action(self, action):
        """Map a policy action to the full physical-control action layout."""
        policy_action = np.asarray(
            action, dtype=np.float32
        ).reshape(-1).copy()
        expected = int(self.action_space.shape[0])
        if policy_action.size != expected:
            raise ValueError(
                f'policy action dimension mismatch: expected {expected}, '
                f'got {policy_action.size}'
            )
        if not np.all(np.isfinite(policy_action)):
            raise ValueError('policy action must contain only finite values')
        policy_action = np.clip(policy_action, -1.0, 1.0)
        if self.policy_action_mode == 'joint_control':
            return policy_action

        n = self.num_users
        full_action = np.zeros(self.full_action_dim, dtype=np.float32)
        full_action[:n] = policy_action[:n]
        full_action[n] = self.common_mobility_heading_deg / 180.0
        full_action[n + 1] = self.common_mobility_pitch_deg / 45.0
        full_action[n + 2] = np.clip(
            2.0 * self.common_mobility_speed
            / max(self.uav.v_max, 1e-9) - 1.0,
            -1.0,
            1.0,
        )
        full_action[n + 3] = policy_action[n]
        full_action[n + 4] = policy_action[n + 1]
        return full_action

    def _compact_full_action(self, full_action):
        """Return the action coordinates controlled by the policy."""
        full_action = np.asarray(
            full_action, dtype=np.float32
        ).reshape(-1)
        if full_action.size != self.full_action_dim:
            raise ValueError(
                f'full action dimension mismatch: expected '
                f'{self.full_action_dim}, got {full_action.size}'
            )
        if self.policy_action_mode == 'joint_control':
            return full_action.copy()
        n = self.num_users
        return np.concatenate((
            full_action[:n], full_action[n + 3:n + 5]
        )).astype(np.float32, copy=False)

    def set_energy_safety_supervisor(self, enabled, controller='margin'):
        """Configure intervention while retaining physical failure checks.

        ``margin`` is the proposed predictive multi-level controller.
        ``soc_threshold`` and ``margin_threshold`` are the matched
        single-threshold SOC and predictive-margin return baselines.
        ``margin_single_level`` replaces all graded interventions with one
        red return action at the same onset threshold as the graded controller.
        ``none`` disables action projection and is allowed only in simulation.
        """
        controller = str(controller).strip().lower()
        if controller not in (
            'margin',
            'soc_threshold',
            'margin_threshold',
            'margin_single_level',
            'none',
        ):
            raise ValueError(f'unsupported energy safety controller: {controller}')
        self.energy_safety_supervisor_enabled = bool(enabled)
        self.energy_safety_controller = (
            controller if self.energy_safety_supervisor_enabled else 'none'
        )
        if not self.energy_safety_supervisor_enabled:
            self.emergency_mode = 'normal'

    def _autonomous_safety_mode(self, energy_context):
        if not self.energy_safety_supervisor_enabled:
            return 'normal'
        if self.energy_safety_controller == 'soc_threshold':
            soc = self.E_rem / max(self.energy.battery_capacity, 1e-9)
            return (
                'red' if soc < self.soc_return_threshold else 'normal'
            )
        if self.energy_safety_controller == 'margin_threshold':
            return (
                'red'
                if energy_context['energy_margin_ratio']
                < self.predictive_return_threshold
                else 'normal'
            )
        if self.energy_safety_controller == 'margin_single_level':
            return (
                'red'
                if energy_context['energy_margin_ratio']
                < self.single_level_margin_threshold
                else 'normal'
            )
        return mode_from_margin_ratio(
            energy_context['energy_margin_ratio'], self.config
        )

    def get_autonomous_safety_mode(self):
        """Expose the configured controller mode for experiment drivers."""
        return self._autonomous_safety_mode(self.get_energy_context())

    def _estimate_action_uav_energy(self, full_action):
        """Estimate next-slot UAV battery use without mutating the environment.

        The estimate is deliberately limited to quantities that are available
        before execution: acceleration-limited propulsion, the maximum amount
        of queued UAV work that can be served in one slot, and receiver
        activity when an upload is pending. It is used only to decide whether
        a yellow/orange projection has a physically meaningful battery saving;
        measured energy after ``step`` remains authoritative.
        """
        action = np.clip(
            np.asarray(full_action, dtype=np.float64).reshape(-1), -1.0, 1.0
        )
        if action.size != self.full_action_dim:
            raise ValueError(
                f'full action dimension mismatch: expected '
                f'{self.full_action_dim}, got {action.size}'
            )

        n = self.num_users
        theta = action[n] * np.pi
        phi = action[n + 1] * (np.pi / 4.0)
        speed = (action[n + 2] + 1.0) * 0.5 * self.uav.v_max
        target_velocity = np.array([
            speed * np.cos(phi) * np.cos(theta),
            speed * np.cos(phi) * np.sin(theta),
            speed * np.sin(phi),
        ], dtype=np.float64)
        acceleration = (target_velocity - self.uav.vel) / max(self.dt, 1e-9)
        acceleration_norm = float(np.linalg.norm(acceleration))
        if acceleration_norm > self.uav.a_max:
            acceleration *= self.uav.a_max / acceleration_norm
        ground_velocity = self.uav.vel + acceleration * self.dt
        air_velocity = ground_velocity - self.wind.velocity
        airspeed = float(np.linalg.norm(air_velocity))
        if airspeed > self.uav.v_max:
            air_velocity *= self.uav.v_max / airspeed

        frequency_fraction = 0.1 + 0.9 * (action[n + 3] + 1.0) * 0.5
        frequency = frequency_fraction * self.energy.freq_at_altitude(
            self.uav.pos[2]
        )
        offload_ratios = (action[:n] + 1.0) * 0.5
        pending_uav_cycles = 0.0
        upload_pending = False
        for user_index, user in enumerate(self.task_gen.users):
            if not user['queue']:
                continue
            task = user['queue'][0]
            if task.get('decision_made', False):
                pending_uav_cycles += max(
                    float(task.get('uav_cycles_remaining', 0.0)), 0.0
                )
                upload_pending = bool(
                    upload_pending
                    or float(task.get('upload_bits_remaining', 0.0)) > 0.0
                )
            else:
                ratio = float(np.clip(offload_ratios[user_index], 0.0, 1.0))
                pending_uav_cycles += max(float(task.get('C', 0.0)), 0.0) * ratio
                upload_pending = bool(
                    upload_pending
                    or max(float(task.get('D', 0.0)), 0.0) * ratio > 0.0
                )

        executable_cycles = min(
            pending_uav_cycles, max(frequency, 0.0) * self.dt
        )
        flight_energy = self.energy.propulsion_power(
            air_velocity, self.uav.pos[2]
        ) * self.dt
        compute_energy = self.energy.compute_energy(
            frequency, executable_cycles
        )
        receiver_energy = (
            self.energy.p_rx_uav * self.dt if upload_pending else 0.0
        )
        nominal = flight_energy + compute_energy + receiver_energy
        uncertainty_scale = float(self.energy_uncertainty.scale)
        return {
            'estimated_uav_energy': float(nominal * uncertainty_scale),
            'estimated_flight_energy': float(flight_energy * uncertainty_scale),
            'estimated_compute_energy': float(compute_energy * uncertainty_scale),
            'estimated_receiver_energy': float(receiver_energy * uncertainty_scale),
            'estimated_uav_cycles': float(executable_cycles),
        }

    def _action_conditioned_energy_context(
            self, full_action, energy_context, replacement_steps=1):
        """Condition a finite-horizon forecast on a requested action.

        ``replacement_steps=1`` preserves the runtime safety supervisor's
        one-step estimate.  CMDP training may separately request a short
        action-persistence counterfactual so that the cost critic can identify
        how the actor request changes near-horizon energy risk without changing
        the action that is actually executed by the environment.
        """
        replacement_steps = int(replacement_steps)
        if not 1 <= replacement_steps <= self.horizon:
            raise ValueError('replacement_steps must lie in [1, horizon]')
        estimate = self._estimate_action_uav_energy(full_action)
        vector = np.asarray(
            energy_context.get('predicted_energy_vector', []),
            dtype=np.float64,
        ).reshape(-1)
        if vector.size:
            available_steps = min(replacement_steps, int(vector.size))
            replaced_forecast_energy = float(np.maximum(
                vector[:available_steps], 0.0
            ).sum())
        else:
            available_steps = replacement_steps
            first_forecast = max(
                float(energy_context.get('predicted_energy', 0.0))
                / max(self.horizon, 1),
                0.0,
            )
            replaced_forecast_energy = first_forecast * available_steps
        predicted_energy = max(
            float(energy_context.get('predicted_energy', 0.0))
            - replaced_forecast_energy
            + estimate['estimated_uav_energy'] * available_steps,
            0.0,
        )
        margin = (
            self.E_rem
            - predicted_energy
            - float(energy_context['return_energy'])
            - float(energy_context['safety_reserve'])
        )
        context = dict(energy_context)
        context.update(estimate)
        context.update({
            'replaced_first_forecast_energy': (
                replaced_forecast_energy / max(available_steps, 1)
            ),
            'replaced_forecast_energy': replaced_forecast_energy,
            'action_forecast_replacement_steps': available_steps,
            'predicted_energy': predicted_energy,
            'energy_margin': margin,
            'energy_margin_ratio': (
                margin / max(self.energy.battery_capacity, 1e-9)
            ),
            'short_horizon_energy_feasible': bool(margin >= 0.0),
        })
        return context

    def set_energy_prediction(self, prediction, raw_prediction=None,
                              ungated_prediction=None,
                              history_gate_weight=1.0,
                              residual_bias=0.0, residual_uncertainty=0.0,
                              prediction_source='external'):
        """设置未来有限时域能耗预测及在线残差诊断量。"""
        def normalized_vector(value):
            vector = np.asarray(
                value if value is not None else [], dtype=np.float64
            ).reshape(-1)
            vector = np.nan_to_num(
                vector, nan=0.0, posinf=0.0, neginf=0.0
            )
            vector = np.clip(vector, 0.0, None)
            if vector.size < self.horizon:
                vector = np.pad(vector, (0, self.horizon - vector.size))
            return vector[:self.horizon].astype(np.float32, copy=False)

        pred = normalized_vector(prediction)
        raw = normalized_vector(
            raw_prediction if raw_prediction is not None else pred
        )
        ungated = normalized_vector(
            ungated_prediction if ungated_prediction is not None else raw
        )
        self.predicted_energy_vector = pred.copy()
        self.raw_predicted_energy_vector = raw.copy()
        self.ungated_predicted_energy_vector = ungated.copy()
        self.predicted_horizon_energy = float(pred.sum())
        self.raw_predicted_horizon_energy = float(raw.sum())
        self.ungated_predicted_horizon_energy = float(
            ungated.sum()
        )
        self.prediction_history_gate_weight = float(np.clip(
            history_gate_weight, 0.0, 1.0
        ))
        self.prediction_residual_bias = float(residual_bias)
        self.prediction_residual_uncertainty = max(
            float(residual_uncertainty), 0.0
        )
        self.prediction_source = str(prediction_source)

    def mission_phase(self):
        """Return the phase represented by the current observation."""
        if not self.mission_return_enabled:
            return 'unconstrained'
        if self.step_count >= self.mission_service_steps:
            return 'scheduled_return'
        return 'service'

    def get_energy_context(self):
        """返回短时域能量条件，所有能量单位均为J。

        判定范围仅为：LSTM预测窗口能耗 + 从当前位置估算的直线返航能耗
        + 安全储备。它不等价于完整任务、未来规划轨迹或全航程可行性。
        """
        horizontal_distance = self.uav.horizontal_distance_to_base()
        return_direction = self.uav.base_pos[:2] - self.uav.pos[:2]
        return_uncertainty_scale = (
            self.energy_uncertainty.return_energy_scale
        )
        return_energy = self.energy.return_energy(
            horizontal_distance,
            altitude=self.uav.pos[2],
            landing_altitude=self.uav.h_min,
            return_direction=return_direction,
            wind_velocity=self.wind.velocity,
            uav_energy_scale=return_uncertainty_scale,
        )
        if self.predicted_horizon_energy > 0.0:
            average_power = self.predicted_horizon_energy / max(
                self.horizon * self.dt, 1e-6
            )
        elif self.energy_history:
            average_power = float(np.mean(self.energy_history)) / max(self.dt, 1e-6)
        else:
            average_power = float(self.energy.flight_power(0.0))

        uncertainty_energy = (
            self.prediction_residual_uncertainty
            * self.horizon
            * self.prediction_uncertainty_gain
        )
        safety_reserve = self.energy.safety_reserve(
            average_power=average_power,
            uncertainty_energy=uncertainty_energy,
        )
        margin, margin_ratio = self.energy.energy_margin(
            self.E_rem,
            predicted_energy=self.predicted_horizon_energy,
            return_energy=return_energy,
            safety_reserve=safety_reserve,
        )
        energy_after_predicted_load_and_return = (
            self.E_rem - self.predicted_horizon_energy - return_energy
        )
        required_energy_with_reserve = (
            self.predicted_horizon_energy + return_energy + safety_reserve
        )
        return {
            'prediction_horizon_steps': self.horizon,
            'prediction_horizon_seconds': self.horizon * self.dt,
            'predicted_energy': self.predicted_horizon_energy,
            'predicted_energy_vector': self.predicted_energy_vector.copy(),
            'raw_predicted_energy': self.raw_predicted_horizon_energy,
            'raw_predicted_energy_vector': (
                self.raw_predicted_energy_vector.copy()
            ),
            'ungated_predicted_energy': (
                self.ungated_predicted_horizon_energy
            ),
            'ungated_predicted_energy_vector': (
                self.ungated_predicted_energy_vector.copy()
            ),
            'prediction_history_gate_weight': (
                self.prediction_history_gate_weight
            ),
            'prediction_residual_correction': (
                self.predicted_horizon_energy
                - self.raw_predicted_horizon_energy
            ),
            'prediction_residual_bias': self.prediction_residual_bias,
            'prediction_source': self.prediction_source,
            'prediction_uncertainty_energy': uncertainty_energy,
            'return_energy': return_energy,
            'return_energy_uncertainty_scale': return_uncertainty_scale,
            'return_energy_uses_uncertainty_bound': bool(
                self.energy_uncertainty.enabled
            ),
            'return_energy_from_current_position': True,
            'safety_reserve': safety_reserve,
            'minimum_safe_energy': safety_reserve,
            'required_energy_with_reserve': required_energy_with_reserve,
            'energy_after_predicted_load_and_return': (
                energy_after_predicted_load_and_return
            ),
            'energy_margin': margin,
            'energy_margin_ratio': margin_ratio,
            'short_horizon_energy_feasible': bool(margin >= 0.0),
            'energy_feasibility_scope': (
                'prediction_horizon_plus_current_position_return'
            ),
            'average_power_estimate': average_power,
            'estimated_energy_margin_time': self.energy.energy_margin_time(
                margin, average_power
            ),
        }

    def get_physical_constraint_context(self):
        """Return the predictor-independent physical evaluation condition.

        Every algorithm is evaluated against the same post-transition rule:
        the remaining UAV battery must cover a direct return from the current
        physical state under the current wind and declared bounded energy
        uncertainty while retaining the physical minimum operating energy
        ``E_min``.  No predicted horizon energy, prediction residual,
        configurable safety buffer, CMDP target, or safety-layer threshold
        enters this calculation.
        """
        horizontal_distance = self.uav.horizontal_distance_to_base()
        return_direction = self.uav.base_pos[:2] - self.uav.pos[:2]
        return_uncertainty_scale = (
            self.energy_uncertainty.return_energy_scale
        )
        return_energy = self.energy.return_energy(
            horizontal_distance,
            altitude=self.uav.pos[2],
            landing_altitude=self.uav.h_min,
            return_direction=return_direction,
            wind_velocity=self.wind.velocity,
            uav_energy_scale=return_uncertainty_scale,
        )
        energy_floor = float(self.energy.E_min)
        margin, margin_ratio = self.energy.energy_margin(
            self.E_rem,
            predicted_energy=0.0,
            return_energy=return_energy,
            safety_reserve=energy_floor,
        )
        return {
            'physical_return_energy': float(return_energy),
            'physical_return_energy_uncertainty_scale': float(
                return_uncertainty_scale
            ),
            'physical_return_energy_uses_uncertainty_bound': bool(
                self.energy_uncertainty.enabled
            ),
            'physical_energy_floor': energy_floor,
            'physical_required_energy': float(
                return_energy + energy_floor
            ),
            'physical_constraint_margin': float(margin),
            'physical_constraint_margin_ratio': float(margin_ratio),
            'physical_constraint_violation': bool(margin < 0.0),
            'physical_constraint_predictor_independent': True,
            'physical_constraint_scope': (
                'current_state_robust_return_plus_energy_floor'
            ),
        }

    def _get_obs(self):
        """构建观测向量"""
        uav_pos, uav_vel = self.uav.pos, self.uav.vel

        # 1. 剩余能量 (归一化)
        E_rem_norm = self.E_rem / self.energy.battery_capacity

        # 2. 能量区间 (one-hot)
        energy_ctx = self.get_energy_context()
        zone = self.energy.get_energy_zone(
            self.E_rem,
            energy_ctx['predicted_energy'],
            energy_ctx['return_energy'],
            energy_ctx['safety_reserve'],
        )
        zone_onehot = {
            'high': [1, 0, 0],
            'medium': [0, 1, 0],
            'low': [0, 0, 1],
        }[zone]

        # 3. 位置和速度
        pos_norm = np.array([
            uav_pos[0] / self.uav.area_size,
            uav_pos[1] / self.uav.area_size,
            uav_pos[2] / self.uav.h_max,
        ], dtype=np.float64)
        vel_norm = uav_vel / self.uav.v_max

        # 4. 信道增益和队列长度
        h_ij = []
        Q_i = []
        for user in self.task_gen.users:
            d = self.comm.distance(uav_pos, user['pos'])
            # dB值通常约为-70~-100；直接与0~1特征拼接会支配首层网络。
            # 除以固定尺度后保留相对信道差异，并约束到稳定范围。
            h_ij.append(np.clip(
                self.comm.path_loss_db(d) / self.channel_db_normalizer,
                -2.0,
                0.0,
            ))
            Q_i.append(np.clip(
                len(user['queue']) / self.queue_length_normalizer,
                0.0,
                1.0,
            ))

        # 5. 各用户队首任务的跨时隙执行状态
        task_progress = []
        for user in self.task_gen.users:
            task_progress.extend(self._head_task_progress_features(user))

        # 6. LSTM预测能量序列 (填充0，实际由外部agent提供)
        E_pred = np.zeros(self.horizon)

        # 7. 任务优先级和显式动态能量裕度
        core_tasks = sum(1 for u in self.task_gen.users
                         for t in u['queue'] if t.get('is_core'))
        total_tasks = sum(len(u['queue']) for u in self.task_gen.users)
        task_priority = core_tasks / (total_tasks + 1)
        margin_feature = np.clip(energy_ctx['energy_margin_ratio'], -1.0, 1.0)

        obs = np.concatenate([
            [E_rem_norm],
            zone_onehot,
            pos_norm,
            vel_norm,
            h_ij,
            Q_i,
            task_progress,
            E_pred,
            [task_priority],
            [margin_feature],
            *([
                [np.clip(self.step_count / max(self.max_steps, 1), 0.0, 1.0)],
                [float(self.mission_phase() == 'scheduled_return')],
            ] if self.mission_return_enabled else []),
        ]).astype(np.float32)

        return obs

    def _head_task_progress_features(self, user):
        """返回队首任务的归一化剩余工作量，降低跨时隙部分可观测性。"""
        if not user['queue']:
            return [0.0] * self.task_progress_features_per_user

        task = user['queue'][0]
        data_size = max(float(task.get('D', 0.0)), 1e-6)
        cycles = max(float(task.get('C', 0.0)), 1e-6)
        deadline = max(float(task.get('tau', 0.0)), 1e-6)
        elapsed = max(
            (self.step_count - float(task.get('arrival_time', self.step_count)))
            * self.dt,
            0.0,
        )
        deadline_ratio = np.clip((deadline - elapsed) / deadline, 0.0, 1.0)

        if not task.get('decision_made', False):
            # -1 表示尚未确定卸载比例；其余三个1表示完整工作量仍待决策。
            return [-1.0, 1.0, 1.0, 1.0, float(deadline_ratio)]

        return [
            float(np.clip(task.get('offload_ratio', 0.0), 0.0, 1.0)),
            float(np.clip(
                task.get('upload_bits_remaining', 0.0) / data_size, 0.0, 1.0
            )),
            float(np.clip(
                task.get('local_cycles_remaining', 0.0) / cycles, 0.0, 1.0
            )),
            float(np.clip(
                task.get('uav_cycles_remaining', 0.0) / cycles, 0.0, 1.0
            )),
            float(deadline_ratio),
        ]

    @staticmethod
    def _initialize_task_progress(task, offload_ratio):
        """在任务首次获得服务时固定卸载比例并建立剩余工作量。"""
        ratio = float(np.clip(offload_ratio, 0.0, 1.0))
        data_size = max(float(task.get('D', 0.0)), 0.0)
        cycles = max(float(task.get('C', 0.0)), 0.0)
        task.update({
            'decision_made': True,
            'offload_ratio': ratio,
            'upload_bits_remaining': data_size * ratio,
            'local_cycles_remaining': cycles * (1.0 - ratio),
            'uav_cycles_remaining': cycles * ratio,
        })

    @staticmethod
    def _task_is_complete(task):
        """使用相对容差判断本地、上传和UAV计算三部分是否均完成。"""
        data_tol = max(float(task.get('D', 0.0)), 1.0) * 1e-9
        cycle_tol = max(float(task.get('C', 0.0)), 1.0) * 1e-9
        return (
            float(task.get('upload_bits_remaining', 0.0)) <= data_tol
            and float(task.get('local_cycles_remaining', 0.0)) <= cycle_tol
            and float(task.get('uav_cycles_remaining', 0.0)) <= cycle_tol
        )

    @staticmethod
    def _shared_cpu_execution(service_records, cpu_frequency):
        """按处理器共享规则推进UAV计算，严格限制任一时刻的总CPU频率。"""
        if not service_records or cpu_frequency <= 0.0:
            return np.zeros(len(service_records), dtype=np.float64)

        remaining = np.array([
            max(float(record['task']['uav_cycles_remaining']), 0.0)
            for record in service_records
        ], dtype=np.float64)
        executed = np.zeros_like(remaining)
        releases = np.array([
            float(record['release_time']) for record in service_records
        ])
        ends = np.array([
            float(record['end_time']) for record in service_records
        ])
        current_time = 0.0
        final_time = float(np.max(ends))
        tolerance = 1e-12

        while current_time < final_time - tolerance:
            active = np.flatnonzero(
                (releases <= current_time + tolerance)
                & (ends > current_time + tolerance)
                & (remaining > tolerance)
            )
            future_releases = releases[releases > current_time + tolerance]

            if active.size == 0:
                if future_releases.size == 0:
                    break
                current_time = float(np.min(future_releases))
                continue

            frequency_share = float(cpu_frequency) / active.size
            completion_times = remaining[active] / max(
                frequency_share, tolerance
            )
            # A floating-point residual can require far less than ``tolerance``
            # seconds of CPU time. Advancing time by ``tolerance`` without
            # clearing that residual leaves the same task active and can turn
            # this event loop into hundreds of billions of iterations.
            near_complete = active[completion_times <= tolerance]
            if near_complete.size:
                executed[near_complete] += remaining[near_complete]
                remaining[near_complete] = 0.0
                continue

            next_release = (
                float(np.min(future_releases))
                if future_releases.size else float('inf')
            )
            next_end = float(np.min(ends[active]))
            time_to_completion = float(np.min(completion_times))
            delta = min(
                next_release - current_time,
                next_end - current_time,
                time_to_completion,
            )
            if delta <= tolerance:
                current_time += tolerance
                continue

            cycles_each = frequency_share * delta
            remaining[active] = np.maximum(remaining[active] - cycles_each, 0.0)
            executed[active] += cycles_each
            current_time += delta

        return executed

    def _process_tasks(self, offload_ratios, pos, f_uav_actual, p_tx,
                       pause_noncore_uav=False):
        """按一个时隙推进队首任务，返回完成/失败数及实际工作量。

        每个用户仅处理其队首任务；本地计算与上行传输可并行。卸载分支必须
        先完成上传，随后才能使用共享的UAV CPU。卸载比例在首次调度时确定，
        后续时隙只推进剩余工作量，不再重复拆分整个任务。
        """
        service_records = []
        tasks_in_service = 0
        local_cycles_executed = 0.0
        uploaded_bits = 0.0
        uplink_time_sum = 0.0
        uav_rx_active_time = 0.0
        noncore_uav_paused = 0

        # 第一阶段：各用户独立推进本地计算和上行传输。
        for user_index, user in enumerate(self.task_gen.users):
            if not user['queue']:
                continue

            task = user['queue'][0]
            tasks_in_service += 1
            pause_this_task = bool(
                pause_noncore_uav and not task.get('is_core', False)
            )
            if not task.get('decision_made', False):
                initial_ratio = 0.0 if pause_this_task else offload_ratios[user_index]
                self._initialize_task_progress(task, initial_ratio)

            arrival = float(task.get('arrival_time', self.step_count))
            deadline = max(float(task.get('tau', 0.0)), 0.0)
            elapsed_before = max((self.step_count - 1 - arrival) * self.dt, 0.0)
            service_time = min(self.dt, max(deadline - elapsed_before, 0.0))
            if service_time <= 0.0:
                continue

            local_before = max(float(task['local_cycles_remaining']), 0.0)
            local_done = min(local_before, self.energy.f_user * service_time)
            task['local_cycles_remaining'] = max(local_before - local_done, 0.0)
            local_cycles_executed += local_done

            if pause_this_task:
                noncore_uav_paused += 1
                continue

            upload_before = max(float(task['upload_bits_remaining']), 0.0)
            upload_time = 0.0
            if upload_before > 0.0:
                distance = self.comm.distance(pos, user['pos'])
                rate = max(float(self.comm.data_rate(p_tx, distance)), 0.0)
                if rate > 0.0:
                    upload_time = min(service_time, upload_before / rate)
                    bits_done = min(upload_before, rate * upload_time)
                    task['upload_bits_remaining'] = max(
                        upload_before - bits_done, 0.0
                    )
                    uploaded_bits += bits_done
                    uplink_time_sum += upload_time
                    uav_rx_active_time = max(uav_rx_active_time, upload_time)

            if float(task['upload_bits_remaining']) <= max(task['D'], 1.0) * 1e-9:
                task['upload_bits_remaining'] = 0.0
                compute_window = max(service_time - upload_time, 0.0)
            else:
                compute_window = 0.0

            if task['uav_cycles_remaining'] > 0.0 and compute_window > 0.0:
                service_records.append({
                    'task': task,
                    'release_time': service_time - compute_window,
                    'end_time': service_time,
                })

        # 第二阶段：所有已上传的队首任务共享同一台UAV处理器。
        uav_cycles_executed = 0.0
        if service_records and f_uav_actual > 0.0:
            allocations = self._shared_cpu_execution(
                service_records, f_uav_actual
            )
            for record, cycles_done in zip(service_records, allocations):
                task = record['task']
                task['uav_cycles_remaining'] = max(
                    float(task['uav_cycles_remaining']) - float(cycles_done), 0.0
                )
                uav_cycles_executed += float(cycles_done)

        # 第三阶段：累计排队时延，并在时隙末判定完成或超时。
        tasks_completed = 0
        tasks_failed = 0
        core_tasks_completed = 0
        core_tasks_failed = 0
        delay_cost = 0.0
        completed_task_delays = []
        failed_task_delays = []
        deadline_overshoots_s = []
        deadline_overshoots_normalized = []
        for user in self.task_gen.users:
            head_task = user['queue'][0] if user['queue'] else None
            remaining_tasks = []
            for task in user['queue']:
                arrival = float(task.get('arrival_time', self.step_count))
                deadline = max(float(task.get('tau', 0.0)), 1e-6)
                elapsed_before = max((self.step_count - 1 - arrival) * self.dt, 0.0)
                elapsed_after = max((self.step_count - arrival) * self.dt, 0.0)
                active_time = min(self.dt, max(deadline - elapsed_before, 0.0))
                delay_cost += active_time / deadline

                if task is head_task and self._task_is_complete(task):
                    tasks_completed += 1
                    core_tasks_completed += int(task.get('is_core', False))
                    completed_task_delays.append(float(elapsed_after))
                elif elapsed_after >= deadline - 1e-9:
                    tasks_failed += 1
                    core_tasks_failed += int(task.get('is_core', False))
                    failed_task_delays.append(float(elapsed_after))
                    overshoot_s = max(float(elapsed_after - deadline), 0.0)
                    deadline_overshoots_s.append(overshoot_s)
                    deadline_overshoots_normalized.append(
                        overshoot_s / deadline
                    )
                else:
                    remaining_tasks.append(task)
            user['queue'] = remaining_tasks

        progress = {
            'tasks_in_service': tasks_in_service,
            'local_cycles_executed': local_cycles_executed,
            'uploaded_bits': uploaded_bits,
            'uplink_time_sum': uplink_time_sum,
            'uav_rx_active_time': uav_rx_active_time,
            'uav_cycles_executed': uav_cycles_executed,
            'noncore_uav_paused': noncore_uav_paused,
            'completed_task_delays': completed_task_delays,
            'failed_task_delays': failed_task_delays,
            'deadline_miss_count': tasks_failed,
            'deadline_overshoots_s': deadline_overshoots_s,
            'deadline_overshoot_s_sum': float(sum(deadline_overshoots_s)),
            'deadline_overshoots_normalized': (
                deadline_overshoots_normalized
            ),
            'deadline_overshoot_normalized_sum': float(
                sum(deadline_overshoots_normalized)
            ),
            'core_tasks_completed': core_tasks_completed,
            'core_tasks_failed': core_tasks_failed,
        }
        return tasks_completed, tasks_failed, delay_cost, progress

    def cmdp_cost_components(self, energy_margin_ratio,
                             terminal_event=False,
                             latency_overshoot_normalized=0.0,
                             latency_miss_fraction=0.0):
        """Return explicit components of the configured CMDP stage cost.

        ``energy_margin`` exactly preserves the proposed method's original
        predictor-dependent cost.  ``latency_overshoot`` retains the severity-
        only diagnostic.  ``latency_violation`` combines a normalized miss
        event with overshoot severity; this prevents a policy from reducing
        only overshoot magnitude while increasing the fraction of missed
        tasks.  Energy failure is deliberately not folded into either
        latency-only baseline constraint.
        """
        margin_deficit = max(
            self.cmdp_margin_target - float(energy_margin_ratio), 0.0
        )
        margin_cost_unclipped = float(
            margin_deficit / self.cmdp_cost_scale
        )
        margin_cost = float(np.clip(
            margin_cost_unclipped, 0.0, self.cmdp_energy_cost_max
        ))
        latency_signal = max(float(latency_overshoot_normalized), 0.0)
        latency_cost = float(np.clip(
            latency_signal / self.cmdp_latency_cost_scale, 0.0, 1.0
        ))
        miss_fraction = float(np.clip(
            max(float(latency_miss_fraction), 0.0), 0.0, 1.0
        ))
        latency_violation_cost = float(np.clip(
            self.cmdp_latency_miss_weight * miss_fraction
            + self.cmdp_latency_overshoot_weight * latency_cost,
            0.0,
            1.0,
        ))
        if self.cmdp_cost_mode == 'latency_overshoot':
            total_cost = latency_cost
        elif self.cmdp_cost_mode == 'latency_violation':
            total_cost = latency_violation_cost
        else:
            total_cost = max(
                margin_cost,
                self.cmdp_terminal_cost if terminal_event else 0.0,
            )
        return {
            'total_cost': float(total_cost),
            'margin_cost': margin_cost,
            'margin_cost_unclipped': margin_cost_unclipped,
            'latency_overshoot_cost': latency_cost,
            'latency_overshoot_normalized': latency_signal,
            'latency_miss_fraction': miss_fraction,
            'latency_violation_cost': latency_violation_cost,
        }

    def cmdp_training_cost(self, energy_margin_ratio, terminal_event=False,
                           latency_overshoot_normalized=0.0,
                           latency_miss_fraction=0.0):
        """Return training cost and energy-margin diagnostic cost.

        The two-value return is retained for backward compatibility.  Use
        :meth:`cmdp_cost_components` when the latency component is needed.
        """
        components = self.cmdp_cost_components(
            energy_margin_ratio,
            terminal_event=terminal_event,
            latency_overshoot_normalized=latency_overshoot_normalized,
            latency_miss_fraction=latency_miss_fraction,
        )
        return components['total_cost'], components['margin_cost']

    def constraint_cost(self, energy_margin_ratio, terminal_event=False):
        """Deprecated compatibility alias for :meth:`cmdp_training_cost`."""
        return self.cmdp_training_cost(energy_margin_ratio, terminal_event)

    def _get_reward(self, tasks_completed, tasks_failed, delay_cost,
                    energy_objective):
        """Return the safety-independent task-performance reward.

        Safety outcomes, predicted energy margin, emergency modes, return
        direction, and mission-return status are intentionally excluded. They
        are handled by CMDP cost, the runtime safety layer, and evaluation
        metrics instead of being hidden in the scalar performance objective.
        """

        # 能耗惩罚使用UAV能耗 + 可配置权重的用户侧能耗。
        penalty_energy = (self.energy_penalty_scale * self.reward_energy_weight
                          * energy_objective
                          / max(self.energy_normalizer, 1e-6))

        # 时延和能耗使用固定、全算法共享的性能权重。
        penalty_delay = (
            self.delay_penalty_scale * self.reward_delay_weight * delay_cost
        )

        # 距离奖励: 鼓励靠近用户 (每步 ~0.05~1.25, 单集~10~250)
        reward_proximity = 0.0
        uav_xy = self.uav.pos[:2]
        for user in self.task_gen.users:
            d = np.linalg.norm(uav_xy - user['pos'][:2])
            reward_proximity += 0.05 / (d / 100 + 0.1)

        # 任务奖励: 10:1 完成/失败比 (修正:原2:1, Agent做到死也是负分)
        reward_task = 10.0 * tasks_completed - 1.0 * tasks_failed

        reward = (
            reward_task - penalty_energy - penalty_delay + reward_proximity
        )
        self.last_reward_breakdown = {
            'task_reward': float(reward_task),
            'energy_penalty': float(penalty_energy),
            'delay_penalty': float(penalty_delay),
            'proximity_reward': float(reward_proximity),
            'reward_energy_weight': self.reward_energy_weight,
            'reward_delay_weight': self.reward_delay_weight,
            'total_reward': float(reward),
        }
        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
            self.observation_space.seed(seed)
        # 主Generator控制任务、位置和初始电量；风场与能耗不确定性使用由
        # 同一环境seed派生的独立随机流，避免新增扰动改变既有任务样本。
        self.task_gen.reset(self.np_random)
        self.uav.initialize(self.np_random)
        self.wind.reset(environment_seed=seed)
        self.energy_uncertainty.reset(environment_seed=seed)
        self.step_count = 0
        self.episode_tasks_completed = 0
        self.episode_tasks_failed = 0
        self.episode_tasks_generated = 0
        self.episode_core_tasks_completed = 0
        self.episode_core_tasks_failed = 0
        self.episode_core_tasks_generated = 0
        self.episode_physical_constraint_violated = False

        # 初始能量 (80%-100%)
        self.E_rem = self.np_random.uniform(
            self.initial_energy_min, self.initial_energy_max
        ) * self.energy.battery_capacity

        # 重置能量历史
        self.energy_history = []
        self.predicted_horizon_energy = 0.0
        self.raw_predicted_horizon_energy = 0.0
        self.ungated_predicted_horizon_energy = 0.0
        self.predicted_energy_vector = np.zeros(
            self.horizon, dtype=np.float32
        )
        self.raw_predicted_energy_vector = np.zeros(
            self.horizon, dtype=np.float32
        )
        self.ungated_predicted_energy_vector = np.zeros(
            self.horizon, dtype=np.float32
        )
        self.prediction_history_gate_weight = 0.0
        self.prediction_residual_bias = 0.0
        self.prediction_residual_uncertainty = 0.0
        self.prediction_source = 'uninitialized'
        self.emergency_mode = 'normal'
        self.last_reward_breakdown = {}

        obs = self._get_obs()
        energy_ctx = self.get_energy_context()
        initial_cmdp_cost, initial_margin_cost = self.cmdp_training_cost(
            energy_ctx['energy_margin_ratio'], terminal_event=False
        )
        physical_ctx = self.get_physical_constraint_context()
        info = {
            'emergency_mode': 'normal',
            'mission_phase': self.mission_phase(),
            'mission_return_enabled': self.mission_return_enabled,
            'mission_success': False,
            'mission_safe_abort': False,
            'mission_deadline_missed': False,
            'mission_outcome': (
                'in_progress' if self.mission_return_enabled else 'disabled'
            ),
            'mission_failure_reasons': (),
            'episode_tasks_completed': 0,
            'episode_tasks_failed': 0,
            'episode_tasks_generated': 0,
            'episode_core_tasks_completed': 0,
            'episode_core_tasks_failed': 0,
            'episode_core_tasks_generated': 0,
            'task_completion_ratio': 0.0,
            'core_task_completion_ratio': 1.0,
            'cmdp_training_cost': float(initial_cmdp_cost),
            'cmdp_margin_cost': float(initial_margin_cost),
            'cmdp_latency_overshoot_cost': 0.0,
            'cmdp_latency_overshoot_normalized': 0.0,
            'cmdp_latency_miss_fraction': 0.0,
            'cmdp_latency_violation_cost': 0.0,
            'cmdp_cost_active': bool(initial_cmdp_cost > 0.0),
            'cmdp_energy_cost_source': self.cmdp_energy_cost_source,
            'cmdp_energy_cost_max': self.cmdp_energy_cost_max,
            'cmdp_cost_saturation_threshold': (
                self.cmdp_cost_saturation_threshold
            ),
            'cmdp_pre_projection_action_cost': float(initial_cmdp_cost),
            'cmdp_applied_action_cost': float(initial_cmdp_cost),
            'cmdp_action_cost_delta': 0.0,
            'cmdp_pre_projection_state_cost': float(initial_cmdp_cost),
            'cmdp_post_projection_state_cost': float(initial_cmdp_cost),
            'cmdp_projection_state_cost_delta': 0.0,
            'cmdp_projection_cost_comparison_available': bool(
                self.cmdp_cost_mode == 'energy_margin'
            ),
            'cmdp_pre_projection_margin_ratio': float(
                energy_ctx['energy_margin_ratio']
            ),
            'cmdp_post_projection_margin_ratio': float(
                energy_ctx['energy_margin_ratio']
            ),
            'cmdp_raw_action_margin_ratio': float(
                energy_ctx['energy_margin_ratio']
            ),
            'cmdp_applied_action_margin_ratio': float(
                energy_ctx['energy_margin_ratio']
            ),
            'cmdp_terminal_event': False,
            'cmdp_cost_mode': self.cmdp_cost_mode,
            'cmdp_margin_target': self.cmdp_margin_target,
            'cmdp_cost_scale': self.cmdp_cost_scale,
            'cmdp_latency_cost_scale': self.cmdp_latency_cost_scale,
            'energy_uncertainty_enabled': bool(
                self.energy_uncertainty.enabled
            ),
            'energy_uncertainty_scale': float(
                self.energy_uncertainty.scale
            ),
            **physical_ctx,
            'wind_velocity': self.wind.velocity.copy(),
            'wind_speed': self.wind.speed,
            'wind_enabled': bool(self.wind.enabled),
        }
        return obs, info

    def step(self, action):
        """执行一步决策"""
        self.step_count += 1
        scheduled_return_active = bool(
            self.mission_return_enabled
            and self.step_count > self.mission_service_steps
        )
        applied_wind_velocity = self.wind.velocity.copy()

        # 1. 环境根据当前能量裕度独立计算权威模式，并与外部请求模式取更
        #    严格者。随后投影完整动作，任何策略或基线都不能绕过硬约束。
        raw_action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        full_raw_action = self._expand_policy_action(raw_action)
        pre_step_energy_ctx = self.get_energy_context()
        (
            cmdp_pre_projection_state_cost,
            cmdp_pre_projection_margin_cost,
        ) = self.cmdp_training_cost(
            pre_step_energy_ctx['energy_margin_ratio'], terminal_event=False
        )
        raw_action_energy_ctx = self._action_conditioned_energy_context(
            full_raw_action, pre_step_energy_ctx
        )
        cmdp_raw_action_energy_ctx = self._action_conditioned_energy_context(
            full_raw_action,
            pre_step_energy_ctx,
            replacement_steps=self.cmdp_action_forecast_replacement_steps,
        )
        action_energy_ctx = dict(raw_action_energy_ctx)
        action_conditioned_mode = 'normal'
        candidate_mode_before_guard = 'normal'
        candidate_mode = 'normal'
        red_feasibility_guard_applied = False
        safety_projection_reason = 'supervisor_disabled'
        raw_action_energy_estimate = float(
            raw_action_energy_ctx['estimated_uav_energy']
        )
        projected_action_energy_estimate = float('nan')
        projected_energy_saving = 0.0
        if self.energy_safety_supervisor_enabled:
            autonomous_mode = self._autonomous_safety_mode(pre_step_energy_ctx)
            requested_mode = (
                self.emergency_mode
                if self.emergency_mode in ('normal', 'yellow', 'orange', 'red')
                else 'normal'
            )
            if self.safety_projection_policy == 'action_aware':
                action_conditioned_mode = self._autonomous_safety_mode(
                    action_energy_ctx
                )
                candidate_mode = stricter_mode(
                    requested_mode, action_conditioned_mode
                )
                candidate_mode_before_guard = candidate_mode
                if (
                    self.safety_red_requires_infeasibility
                    and candidate_mode == 'red'
                    and action_energy_ctx['short_horizon_energy_feasible']
                ):
                    # Keep the graded warning information, but reserve the
                    # hard red return for a non-positive action-conditioned
                    # predictive margin.  Re-resolve the feasible request
                    # against only the non-return tiers so a stale external
                    # red request cannot force an unnecessary projection.
                    action_margin_ratio = float(
                        action_energy_ctx['energy_margin_ratio']
                    )
                    if action_margin_ratio < self.threshold_orange:
                        candidate_mode = 'orange'
                    elif action_margin_ratio < self.threshold_yellow:
                        candidate_mode = 'yellow'
                    else:
                        candidate_mode = 'normal'
                    red_feasibility_guard_applied = True
            else:
                action_conditioned_mode = autonomous_mode
                candidate_mode = stricter_mode(
                    requested_mode, autonomous_mode
                )
                candidate_mode_before_guard = candidate_mode

            candidate_constraints = safety_constraints(
                self.config, candidate_mode
            )
            candidate_action = project_normalized_action(
                full_raw_action, self.num_users, candidate_constraints
            )
            if self.safety_projection_policy == 'action_aware':
                raw_action_energy_estimate = float(
                    action_energy_ctx['estimated_uav_energy']
                )
                projected_energy_ctx = self._action_conditioned_energy_context(
                    candidate_action, pre_step_energy_ctx
                )
                projected_action_energy_estimate = float(
                    projected_energy_ctx['estimated_uav_energy']
                )
                projected_energy_saving = max(
                    raw_action_energy_estimate
                    - projected_action_energy_estimate,
                    0.0,
                )
                if candidate_mode == 'red':
                    safety_mode_applied = 'red'
                    applied_constraints = candidate_constraints
                    applied_action = candidate_action
                    safety_projection_reason = 'hard_red_takeover'
                elif candidate_mode in ('yellow', 'orange') and (
                    projected_energy_saving
                    >= self.safety_min_energy_saving
                ):
                    safety_mode_applied = candidate_mode
                    applied_constraints = candidate_constraints
                    applied_action = candidate_action
                    safety_projection_reason = (
                        'feasible_red_warning_soft_projection'
                        if red_feasibility_guard_applied
                        else 'positive_energy_saving'
                    )
                else:
                    safety_mode_applied = 'normal'
                    applied_constraints = safety_constraints(
                        self.config, 'normal'
                    )
                    applied_action = np.clip(
                        full_raw_action, -1.0, 1.0
                    ).astype(np.float32, copy=False)
                    safety_projection_reason = (
                        'feasible_red_warning_advisory'
                        if red_feasibility_guard_applied
                        else (
                            'advisory_only_no_material_energy_saving'
                            if candidate_mode in ('yellow', 'orange')
                            else 'normal_margin'
                        )
                    )
            else:
                safety_mode_applied = candidate_mode
                applied_constraints = candidate_constraints
                applied_action = candidate_action
                safety_projection_reason = 'state_threshold_projection'
        else:
            autonomous_mode = 'normal'
            requested_mode = 'normal'
            safety_mode_applied = 'normal'
            applied_constraints = safety_constraints(self.config, 'normal')
            applied_action = project_normalized_action(
                full_raw_action, self.num_users, applied_constraints
            )
        # Preserve the action produced by the energy-safety projection before
        # the common mission-return controller makes any further changes.
        # This keeps safety intervention statistics separate from scheduled
        # return, while ``applied_action`` remains the action executed by the
        # dynamics.
        safety_applied_action = applied_action.copy()
        safety_applied_energy_ctx = self._action_conditioned_energy_context(
            safety_applied_action, pre_step_energy_ctx
        )

        # The scheduled-return controller is a common mission rule applied to
        # every algorithm.  It is deliberately distinct from the proposed
        # energy-emergency supervisor and therefore does not change
        # ``safety_mode_applied`` or the emergency trigger statistics.
        if scheduled_return_active:
            applied_action[:self.num_users] = -1.0
            return_speed = float(self.config.get('return_speed', 10.0))
            applied_action[self.num_users + 2] = np.clip(
                2.0 * return_speed / max(self.uav.v_max, 1e-6) - 1.0,
                -1.0,
                1.0,
            )
            applied_action[self.num_users + 3] = -1.0
            applied_action[self.num_users + 4] = -1.0

        executed_action_energy_ctx = self._action_conditioned_energy_context(
            applied_action, pre_step_energy_ctx
        )

        # 2. 解包投影后的动作：[-1,1]映射到实际物理范围。
        lam = (applied_action[:self.num_users] + 1) / 2
        lam = np.clip(lam, 0, 1)
        self.lam = lam
        theta = applied_action[self.num_users] * np.pi
        phi = applied_action[self.num_users + 1] * (np.pi / 4)
        speed = ((applied_action[self.num_users + 2] + 1) / 2
                 * self.uav.v_max)
        f_uav_norm = (applied_action[self.num_users + 3] + 1) / 2
        P_tx_norm = (applied_action[self.num_users + 4] + 1) / 2

        f_uav = 0.1 + f_uav_norm * 0.9  # 归一化频率
        # A2修复: 发射功率范围 [p_tx_min, p_tx_max], 配合P_comm模型形成真实权衡
        p_tx_min = self.config_p_tx_min
        p_tx_max = self.config_p_tx_max
        P_tx = p_tx_min + P_tx_norm * (p_tx_max - p_tx_min)

        # 3. 红色预警强制接管航向和俯仰；速度已被动作投影固定为返航速度。
        if safety_mode_applied == 'red' or scheduled_return_active:
            to_base = self.uav.base_pos[:2] - self.uav.pos[:2]
            horizontal_distance = np.linalg.norm(to_base)
            if horizontal_distance > 1e-6:
                theta = np.arctan2(to_base[1], to_base[0])
            altitude_delta = self.uav.h_min - self.uav.pos[2]
            phi = np.clip(
                np.arctan2(altitude_delta, max(horizontal_distance, 1.0)),
                -np.pi / 4, 0.0,
            )
            applied_action[self.num_users] = theta / np.pi
            applied_action[self.num_users + 1] = phi / (np.pi / 4)
            if safety_mode_applied == 'red':
                safety_applied_action[self.num_users] = theta / np.pi
                safety_applied_action[self.num_users + 1] = (
                    phi / (np.pi / 4)
                )

        # 4. 更新无人机位置
        pos, vel = self.uav.update(
            [theta, phi, speed],
            self.dt,
            wind_velocity=applied_wind_velocity,
        )
        # 使用本时隙更新后的高度计算可用CPU频率。
        f_uav_actual = f_uav * self.energy.freq_at_altitude(pos[2])

        # 5. 按实际时隙容量推进任务。任务不会再根据一次公式计算直接完成，
        #    而是持续保存待上传数据量、本地周期数和UAV周期数。
        if scheduled_return_active:
            # Service accounting stops at the common service cutoff.  This
            # prevents a policy that takes longer to return from accumulating
            # extra completed tasks during the return-only phase.
            tasks_completed, tasks_failed, delay_cost = 0, 0, 0.0
            task_progress = {
                'tasks_in_service': 0,
                'local_cycles_executed': 0.0,
                'uploaded_bits': 0.0,
                'uplink_time_sum': 0.0,
                'uav_rx_active_time': 0.0,
                'uav_cycles_executed': 0.0,
                'noncore_uav_paused': 0,
                'completed_task_delays': [],
                'failed_task_delays': [],
                'deadline_miss_count': 0,
                'deadline_overshoots_s': [],
                'deadline_overshoot_s_sum': 0.0,
                'deadline_overshoots_normalized': [],
                'deadline_overshoot_normalized_sum': 0.0,
                'core_tasks_completed': 0,
                'core_tasks_failed': 0,
            }
        else:
            tasks_completed, tasks_failed, delay_cost, task_progress = (
                self._process_tasks(
                    lam,
                    pos,
                    f_uav_actual,
                    P_tx,
                    pause_noncore_uav=applied_constraints['pause_noncore_uav'],
                )
            )
        self.episode_tasks_completed += int(tasks_completed)
        self.episode_tasks_failed += int(tasks_failed)
        self.episode_core_tasks_completed += int(
            task_progress.get('core_tasks_completed', 0)
        )
        self.episode_core_tasks_failed += int(
            task_progress.get('core_tasks_failed', 0)
        )

        # 6. 能耗按本时隙真实执行量计算。E_total仅表示UAV电池能耗；
        #    用户本地计算和上行发送能耗保存在energy_breakdown中。
        applied_energy_uncertainty_scale = float(
            self.energy_uncertainty.scale
        )
        self.E_rem, E_total, energy_breakdown, energy_done = (
            self.energy.step_energy(
                self.E_rem,
                self.uav.air_vel,
                pos[2],
                f_uav_actual,
                P_tx,
                self.dt,
                uav_cycles_executed=task_progress['uav_cycles_executed'],
                uplink_time_sum=task_progress['uplink_time_sum'],
                uav_rx_active_time=task_progress['uav_rx_active_time'],
                local_cycles_executed=task_progress['local_cycles_executed'],
                uav_energy_scale=applied_energy_uncertainty_scale,
            )
        )
        E_objective = (
            E_total
            + self.reward_user_energy_weight * energy_breakdown['E_user_total']
        )

        # LSTM只预测无人机电池能耗，与能量裕度的物理口径一致。
        self.energy_history.append(E_total)
        if len(self.energy_history) > self.lookback:
            self.energy_history.pop(0)

        # 7. 生成新任务
        if (not self.mission_return_enabled
                or self.step_count < self.mission_service_steps):
            new_tasks = self.task_gen.generate_task(self.step_count)
        else:
            new_tasks = []
        self.episode_tasks_generated += len(new_tasks)
        self.episode_core_tasks_generated += sum(
            int(task.get('is_core', False)) for task in new_tasks
        )

        # 8. 检查结束条件
        dist_to_base = self.uav.distance_to_base()
        horizontal_dist_to_base = self.uav.horizontal_distance_to_base()
        energy_violation = energy_done
        # This is battery exhaustion, not a physical collision.  The old
        # ``crashed`` name caused a persistent 0% value to be misreported as a
        # collision rate even though the environment normally terminates at
        # the reserve-energy threshold before the battery reaches zero.
        battery_depleted = self.E_rem <= 0.0
        energy_ctx = self.get_energy_context()
        zone = self.energy.get_energy_zone(
            self.E_rem,
            energy_ctx['predicted_energy'],
            energy_ctx['return_energy'],
            energy_ctx['safety_reserve'],
        )

        # 安全返航检测必须使用水平距离；三维距离至少包含h_min，原实现永远无法<15m。
        if horizontal_dist_to_base < 15 and self.uav.pos[2] <= self.uav.h_min + 2:
            returned_safe = True
            done = True
        else:
            returned_safe = False
            done = energy_violation or (self.step_count >= self.max_steps)

        # Predictor-dependent cost used only by CMDP training.  It is kept
        # separate from both the scalar reward and the common physical
        # evaluation constraint below.
        cmdp_terminal_event = bool(energy_violation or battery_depleted)
        latency_overshoot_normalized = (
            float(task_progress.get(
                'deadline_overshoot_normalized_sum', 0.0
            )) / max(self.num_users, 1)
        )
        latency_miss_fraction = (
            float(task_progress.get('deadline_miss_count', 0))
            / max(self.num_users, 1)
        )
        post_transition_cmdp_components = self.cmdp_cost_components(
            energy_ctx['energy_margin_ratio'],
            terminal_event=cmdp_terminal_event,
            latency_overshoot_normalized=latency_overshoot_normalized,
            latency_miss_fraction=latency_miss_fraction,
        )
        raw_action_cmdp_components = self.cmdp_cost_components(
            cmdp_raw_action_energy_ctx['energy_margin_ratio'],
            terminal_event=cmdp_terminal_event,
            latency_overshoot_normalized=latency_overshoot_normalized,
            latency_miss_fraction=latency_miss_fraction,
        )
        safety_applied_cmdp_components = self.cmdp_cost_components(
            safety_applied_energy_ctx['energy_margin_ratio'],
            terminal_event=cmdp_terminal_event,
            latency_overshoot_normalized=latency_overshoot_normalized,
            latency_miss_fraction=latency_miss_fraction,
        )
        applied_action_cmdp_components = self.cmdp_cost_components(
            executed_action_energy_ctx['energy_margin_ratio'],
            terminal_event=cmdp_terminal_event,
            latency_overshoot_normalized=latency_overshoot_normalized,
            latency_miss_fraction=latency_miss_fraction,
        )
        use_raw_action_cost = bool(
            self.cmdp_cost_mode == 'energy_margin'
            and self.cmdp_energy_cost_source == 'raw_action_counterfactual'
        )
        cmdp_components = (
            raw_action_cmdp_components
            if use_raw_action_cost else post_transition_cmdp_components
        )
        cmdp_training_cost = cmdp_components['total_cost']
        cmdp_margin_cost = cmdp_components['margin_cost']
        cmdp_latency_overshoot_cost = cmdp_components[
            'latency_overshoot_cost'
        ]
        cmdp_latency_violation_cost = cmdp_components[
            'latency_violation_cost'
        ]
        cmdp_projection_cost_comparison_available = bool(
            self.cmdp_cost_mode == 'energy_margin'
        )
        cmdp_projection_state_cost_delta = (
            post_transition_cmdp_components['total_cost']
            - cmdp_pre_projection_state_cost
            if cmdp_projection_cost_comparison_available else float('nan')
        )

        # Common post-transition physical criterion for every algorithm.  Its
        # return requirement and physical energy floor contain no LSTM
        # prediction, residual correction, configurable safety buffer, CMDP
        # target, or safety-controller threshold.
        physical_constraint_ctx = self.get_physical_constraint_context()
        physical_constraint_ctx['physical_constraint_violation'] = bool(
            physical_constraint_ctx['physical_constraint_violation']
            or energy_violation
            or battery_depleted
        )
        self.episode_physical_constraint_violated = bool(
            self.episode_physical_constraint_violated
            or physical_constraint_ctx['physical_constraint_violation']
        )

        # A complete mission is a joint service-and-safety event.  Counting a
        # single completed task followed by an early return as success makes
        # conservative controllers look artificially strong, so the service
        # horizon, aggregate/core completion ratios, and physical feasibility
        # are explicit requirements.
        task_completion_ratio = (
            self.episode_tasks_completed / self.episode_tasks_generated
            if self.episode_tasks_generated > 0 else 0.0
        )
        core_task_completion_ratio = (
            self.episode_core_tasks_completed
            / self.episode_core_tasks_generated
            if self.episode_core_tasks_generated > 0 else 1.0
        )
        mission_min_count_met = bool(
            self.episode_tasks_completed >= self.mission_min_completed_tasks
        )
        mission_completion_ratio_met = bool(
            task_completion_ratio + 1e-12
            >= self.mission_min_completion_ratio
        )
        mission_core_completion_ratio_met = bool(
            core_task_completion_ratio + 1e-12
            >= self.mission_min_core_completion_ratio
        )
        mission_task_requirement_met = bool(
            mission_min_count_met
            and mission_completion_ratio_met
            and mission_core_completion_ratio_met
        )
        mission_full_service_window_met = bool(
            not self.mission_require_full_service_window
            or self.step_count >= self.mission_service_steps
        )
        mission_energy_requirement_met = bool(
            not self.mission_require_energy_feasible
            or not self.episode_physical_constraint_violated
        )
        mission_deadline_missed = bool(
            self.mission_return_enabled
            and self.step_count >= self.max_steps
            and not returned_safe
        )
        mission_success = bool(
            self.mission_return_enabled
            and returned_safe
            and mission_full_service_window_met
            and mission_task_requirement_met
            and mission_energy_requirement_met
        )
        mission_safe_abort = bool(
            self.mission_return_enabled and returned_safe and not mission_success
        )
        mission_failure_reasons = []
        if self.mission_return_enabled and done and not mission_success:
            if not returned_safe:
                mission_failure_reasons.append('safe_return_unmet')
            if mission_deadline_missed:
                mission_failure_reasons.append('return_deadline_missed')
            if not mission_full_service_window_met:
                mission_failure_reasons.append('service_window_incomplete')
            if not mission_min_count_met:
                mission_failure_reasons.append(
                    'minimum_completed_tasks_unmet'
                )
            if not mission_completion_ratio_met:
                mission_failure_reasons.append('completion_ratio_unmet')
            if not mission_core_completion_ratio_met:
                mission_failure_reasons.append(
                    'core_completion_ratio_unmet'
                )
            if not mission_energy_requirement_met:
                mission_failure_reasons.append('physical_energy_infeasible')
        if not self.mission_return_enabled:
            mission_outcome = 'disabled'
        elif not done:
            mission_outcome = 'in_progress'
        elif mission_success:
            mission_outcome = 'success'
        elif returned_safe:
            mission_outcome = 'safe_abort'
        elif mission_deadline_missed:
            mission_outcome = 'return_deadline_missed'
        elif energy_violation or battery_depleted:
            mission_outcome = 'energy_failure'
        else:
            mission_outcome = 'return_failure'

        # 9. 计算奖励
        reward = self._get_reward(
            tasks_completed, tasks_failed, delay_cost, E_objective
        )

        # 10. 根据本步结束后的能量裕度更新下一时隙模式。
        self.emergency_mode = (
            self._autonomous_safety_mode(energy_ctx)
            if self.energy_safety_supervisor_enabled
            else 'normal'
        )

        next_wind_velocity = self.wind.advance()
        next_energy_uncertainty_scale = self.energy_uncertainty.advance()
        obs = self._get_obs()
        policy_safety_applied_action = self._compact_full_action(
            safety_applied_action
        )
        policy_applied_action = self._compact_full_action(applied_action)
        action_projection_delta = policy_applied_action - raw_action
        action_projection_abs = np.abs(action_projection_delta)
        safety_projection_delta = (
            policy_safety_applied_action - raw_action
        )
        safety_projection_abs = np.abs(safety_projection_delta)
        scheduled_return_projection_delta = (
            policy_applied_action - policy_safety_applied_action
        )
        scheduled_return_projection_abs = np.abs(
            scheduled_return_projection_delta
        )
        full_action_projection_delta = applied_action - full_raw_action
        full_safety_projection_delta = (
            safety_applied_action - full_raw_action
        )
        full_scheduled_return_projection_delta = (
            applied_action - safety_applied_action
        )
        info = {
            'E_rem': self.E_rem,
            'E_total': E_total,
            'E_uav_total': E_total,
            'E_uav_nominal': energy_breakdown['E_uav_nominal'],
            'E_uav_uncertainty_delta': energy_breakdown[
                'E_uav_uncertainty_delta'
            ],
            'E_system_total': energy_breakdown['E_system_total'],
            'E_objective': E_objective,
            'tasks_completed': tasks_completed,
            'tasks_failed': tasks_failed,
            'emergency_mode': self.emergency_mode,
            'requested_emergency_mode': requested_mode,
            'autonomous_emergency_mode': autonomous_mode,
            'action_conditioned_emergency_mode': action_conditioned_mode,
            'safety_candidate_mode_before_guard': (
                candidate_mode_before_guard
            ),
            'safety_candidate_mode': candidate_mode,
            'safety_red_requires_infeasibility': bool(
                self.safety_red_requires_infeasibility
            ),
            'safety_red_feasibility_guard_applied': bool(
                red_feasibility_guard_applied
            ),
            'safety_action_conditioned_feasible': bool(
                action_energy_ctx['short_horizon_energy_feasible']
            ),
            'safety_mode_applied': safety_mode_applied,
            'safety_projection_policy': self.safety_projection_policy,
            'safety_projection_reason': safety_projection_reason,
            'safety_raw_action_energy_estimate': raw_action_energy_estimate,
            'safety_projected_action_energy_estimate': (
                projected_action_energy_estimate
            ),
            'safety_projected_energy_saving': projected_energy_saving,
            'safety_action_conditioned_margin_ratio': float(
                action_energy_ctx['energy_margin_ratio']
            ),
            'energy_safety_supervisor_enabled': bool(
                self.energy_safety_supervisor_enabled
            ),
            'energy_safety_controller': self.energy_safety_controller,
            'safety_constraints_applied': dict(applied_constraints),
            'policy_action_mode': self.policy_action_mode,
            'raw_action': raw_action.copy(),
            'safety_applied_action': policy_safety_applied_action.copy(),
            'applied_action': policy_applied_action.copy(),
            'full_raw_action': full_raw_action.copy(),
            'full_safety_applied_action': safety_applied_action.copy(),
            'full_applied_action': applied_action.copy(),
            'action_projection_delta': action_projection_delta.copy(),
            'safety_projection_delta': safety_projection_delta.copy(),
            'scheduled_return_projection_delta': (
                scheduled_return_projection_delta.copy()
            ),
            'action_projection_l1': float(np.sum(action_projection_abs)),
            'action_projection_l2': float(
                np.linalg.norm(action_projection_delta)
            ),
            'action_projection_linf': float(
                np.max(action_projection_abs, initial=0.0)
            ),
            'action_projection_changed_dimensions': int(
                np.count_nonzero(action_projection_abs > 1e-7)
            ),
            'safety_action_overridden': bool(
                np.any(np.abs(full_safety_projection_delta) > 1e-7)
            ),
            'safety_projection_l1': float(
                np.sum(safety_projection_abs)
            ),
            'safety_projection_l2': float(
                np.linalg.norm(safety_projection_delta)
            ),
            'safety_projection_linf': float(
                np.max(safety_projection_abs, initial=0.0)
            ),
            'scheduled_return_action_overridden': bool(
                np.any(
                    np.abs(full_scheduled_return_projection_delta) > 1e-7
                )
            ),
            'scheduled_return_projection_l1': float(
                np.sum(scheduled_return_projection_abs)
            ),
            'scheduled_return_projection_l2': float(
                np.linalg.norm(scheduled_return_projection_delta)
            ),
            'scheduled_return_projection_linf': float(
                np.max(scheduled_return_projection_abs, initial=0.0)
            ),
            'full_action_projection_l1': float(
                np.sum(np.abs(full_action_projection_delta))
            ),
            'full_action_projection_l2': float(
                np.linalg.norm(full_action_projection_delta)
            ),
            'full_safety_projection_l2': float(
                np.linalg.norm(full_safety_projection_delta)
            ),
            'full_scheduled_return_projection_l2': float(
                np.linalg.norm(full_scheduled_return_projection_delta)
            ),
            'battery_depleted': battery_depleted,
            # Deprecated compatibility alias for old result readers.  New
            # reports must use ``battery_depleted``/``battery_depletion_rate``.
            'crashed': battery_depleted,
            'energy_violation': energy_violation,
            'returned_safe': returned_safe,
            'mission_phase': self.mission_phase(),
            'mission_return_enabled': self.mission_return_enabled,
            'scheduled_return_active': scheduled_return_active,
            'mission_success': mission_success,
            'mission_safe_abort': mission_safe_abort,
            'mission_deadline_missed': mission_deadline_missed,
            'mission_task_requirement_met': mission_task_requirement_met,
            'mission_min_count_met': mission_min_count_met,
            'mission_completion_ratio_met': mission_completion_ratio_met,
            'mission_core_completion_ratio_met': (
                mission_core_completion_ratio_met
            ),
            'mission_full_service_window_met': (
                mission_full_service_window_met
            ),
            'mission_energy_requirement_met': (
                mission_energy_requirement_met
            ),
            'mission_outcome': mission_outcome,
            'mission_failure_reasons': tuple(mission_failure_reasons),
            'mission_min_completed_tasks': self.mission_min_completed_tasks,
            'mission_min_completion_ratio': self.mission_min_completion_ratio,
            'mission_min_core_completion_ratio': (
                self.mission_min_core_completion_ratio
            ),
            'episode_tasks_completed': self.episode_tasks_completed,
            'episode_tasks_failed': self.episode_tasks_failed,
            'episode_tasks_generated': self.episode_tasks_generated,
            'episode_core_tasks_completed': (
                self.episode_core_tasks_completed
            ),
            'episode_core_tasks_failed': self.episode_core_tasks_failed,
            'episode_core_tasks_generated': (
                self.episode_core_tasks_generated
            ),
            'task_completion_ratio': float(task_completion_ratio),
            'core_task_completion_ratio': float(
                core_task_completion_ratio
            ),
            'episode_physical_constraint_violated': bool(
                self.episode_physical_constraint_violated
            ),
            'cmdp_training_cost': float(cmdp_training_cost),
            'cmdp_margin_cost': cmdp_margin_cost,
            'cmdp_latency_overshoot_cost': float(
                cmdp_latency_overshoot_cost
            ),
            'cmdp_latency_overshoot_normalized': float(
                latency_overshoot_normalized
            ),
            'cmdp_latency_miss_fraction': float(latency_miss_fraction),
            'cmdp_latency_violation_cost': float(
                cmdp_latency_violation_cost
            ),
            'cmdp_cost_active': bool(cmdp_training_cost > 0.0),
            'cmdp_energy_cost_source': self.cmdp_energy_cost_source,
            'cmdp_action_forecast_replacement_steps': (
                self.cmdp_action_forecast_replacement_steps
            ),
            'cmdp_energy_cost_max': self.cmdp_energy_cost_max,
            'cmdp_cost_saturation_threshold': (
                self.cmdp_cost_saturation_threshold
            ),
            'cmdp_pre_projection_action_cost': float(
                raw_action_cmdp_components['total_cost']
                if cmdp_projection_cost_comparison_available
                else float('nan')
            ),
            'cmdp_safety_applied_action_cost': float(
                safety_applied_cmdp_components['total_cost']
                if cmdp_projection_cost_comparison_available
                else float('nan')
            ),
            'cmdp_applied_action_cost': float(
                applied_action_cmdp_components['total_cost']
                if cmdp_projection_cost_comparison_available
                else float('nan')
            ),
            'cmdp_action_cost_delta': float(
                applied_action_cmdp_components['total_cost']
                - raw_action_cmdp_components['total_cost']
                if cmdp_projection_cost_comparison_available
                else float('nan')
            ),
            # These are state-risk snapshots around the shielded transition,
            # not a counterfactual simulation of executing the raw action.
            # Action-specific intervention remains available through the raw,
            # safety-applied, and applied action/projection fields above.
            'cmdp_pre_projection_state_cost': float(
                cmdp_pre_projection_state_cost
                if cmdp_projection_cost_comparison_available
                else float('nan')
            ),
            'cmdp_pre_projection_margin_cost': float(
                cmdp_pre_projection_margin_cost
            ),
            'cmdp_post_projection_state_cost': float(
                post_transition_cmdp_components['total_cost']
                if cmdp_projection_cost_comparison_available
                else float('nan')
            ),
            'cmdp_projection_state_cost_delta': float(
                cmdp_projection_state_cost_delta
            ),
            'cmdp_projection_cost_comparison_available': (
                cmdp_projection_cost_comparison_available
            ),
            'cmdp_pre_projection_margin_ratio': float(
                pre_step_energy_ctx['energy_margin_ratio']
            ),
            'cmdp_post_projection_margin_ratio': float(
                energy_ctx['energy_margin_ratio']
            ),
            'cmdp_raw_action_margin_ratio': float(
                cmdp_raw_action_energy_ctx['energy_margin_ratio']
            ),
            'cmdp_safety_applied_action_margin_ratio': float(
                safety_applied_energy_ctx['energy_margin_ratio']
            ),
            'cmdp_applied_action_margin_ratio': float(
                executed_action_energy_ctx['energy_margin_ratio']
            ),
            'cmdp_terminal_event': cmdp_terminal_event,
            'cmdp_cost_mode': self.cmdp_cost_mode,
            'cmdp_margin_target': self.cmdp_margin_target,
            'cmdp_cost_scale': self.cmdp_cost_scale,
            'cmdp_margin_cost_unclipped': float(
                cmdp_components['margin_cost_unclipped']
            ),
            'cmdp_latency_cost_scale': self.cmdp_latency_cost_scale,
            **physical_constraint_ctx,
            'dist_to_base': dist_to_base,
            'horizontal_dist_to_base': horizontal_dist_to_base,
            'delay_cost': delay_cost,
            'reward_breakdown': dict(self.last_reward_breakdown),
            'task_progress': task_progress,
            'energy_zone': zone,
            **energy_ctx,
            'energy_breakdown': energy_breakdown,
            'pos': pos.copy(),
            'ground_velocity': vel.copy(),
            'air_velocity': self.uav.air_vel.copy(),
            'wind_velocity': applied_wind_velocity,
            'wind_speed': float(np.linalg.norm(applied_wind_velocity[:2])),
            'next_wind_velocity': next_wind_velocity,
            'wind_enabled': bool(self.wind.enabled),
            'energy_uncertainty_enabled': bool(
                self.energy_uncertainty.enabled
            ),
            'energy_uncertainty_scale': (
                applied_energy_uncertainty_scale
            ),
            'next_energy_uncertainty_scale': float(
                next_energy_uncertainty_scale
            ),
            'energy_uncertainty_observed_by_policy': False,
        }

        return obs, reward, done, False, info
