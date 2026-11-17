"""日志和数据记录"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict


class Logger:
    def __init__(self, config):
        self.data = defaultdict(list)
        self.action_logging_enabled = bool(
            config.get('log_action_steps', True)
        )
        expected_rows = (
            max(int(config.get('max_episodes', 1)), 1)
            * max(int(config.get('max_steps', 1)), 1)
        )
        self._action_capacity = expected_rows
        self._action_count = 0
        self._action_dim = None
        self._action_arrays = None
        self._prediction_horizon = max(int(config.get('horizon', 5)), 1)

    def log_episode(self, episode, metrics):
        self.data['episode'].append(episode)
        for k, v in metrics.items():
            self.data[k].append(v)

    def _allocate_action_arrays(self, action_dim, capacity):
        self._action_dim = int(action_dim)
        self._action_capacity = max(int(capacity), 1)
        shape = (self._action_capacity, self._action_dim)
        prediction_shape = (
            self._action_capacity, self._prediction_horizon
        )
        self._action_arrays = {
            'episode': np.empty(self._action_capacity, dtype=np.int32),
            'step': np.empty(self._action_capacity, dtype=np.int32),
            'safety_mode_applied': np.empty(
                self._action_capacity, dtype='<U8'
            ),
            'action_conditioned_emergency_mode': np.empty(
                self._action_capacity, dtype='<U8'
            ),
            'safety_candidate_mode_before_guard': np.empty(
                self._action_capacity, dtype='<U8'
            ),
            'safety_candidate_mode': np.empty(
                self._action_capacity, dtype='<U8'
            ),
            'safety_projection_policy': np.empty(
                self._action_capacity, dtype='<U24'
            ),
            'safety_projection_reason': np.empty(
                self._action_capacity, dtype='<U64'
            ),
            'prediction_source': np.empty(
                self._action_capacity, dtype='<U48'
            ),
            'cmdp_cost_mode': np.empty(
                self._action_capacity, dtype='<U24'
            ),
            'cmdp_energy_cost_source': np.empty(
                self._action_capacity, dtype='<U32'
            ),
            'safety_action_overridden': np.empty(
                self._action_capacity, dtype=np.bool_
            ),
            'safety_red_feasibility_guard_applied': np.empty(
                self._action_capacity, dtype=np.bool_
            ),
            'safety_action_conditioned_feasible': np.empty(
                self._action_capacity, dtype=np.bool_
            ),
            'scheduled_return_active': np.empty(
                self._action_capacity, dtype=np.bool_
            ),
            'scheduled_return_action_overridden': np.empty(
                self._action_capacity, dtype=np.bool_
            ),
            'action_projection_l1': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'action_projection_l2': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'action_projection_linf': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'safety_projection_l1': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'safety_projection_l2': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'safety_projection_linf': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'scheduled_return_projection_l1': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'scheduled_return_projection_l2': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'scheduled_return_projection_linf': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'energy_uncertainty_scale': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'E_uav_nominal': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'E_uav_uncertainty_delta': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'energy_margin_ratio': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'safety_action_conditioned_margin_ratio': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'safety_raw_action_energy_estimate': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'safety_projected_action_energy_estimate': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'safety_projected_energy_saving': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'E_rem': np.empty(self._action_capacity, dtype=np.float32),
            'E_total': np.empty(self._action_capacity, dtype=np.float32),
            'predicted_energy': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'raw_predicted_energy': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'return_energy': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'safety_reserve': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'energy_margin': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'predicted_energy_vector': np.empty(
                prediction_shape, dtype=np.float32
            ),
            'raw_predicted_energy_vector': np.empty(
                prediction_shape, dtype=np.float32
            ),
            'cmdp_margin_target': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'cmdp_cost_scale': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'cmdp_energy_cost_max': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'cmdp_latency_cost_scale': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'cmdp_margin_cost': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'cmdp_margin_cost_unclipped': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'cmdp_latency_overshoot_cost': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'cmdp_latency_overshoot_normalized': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'cmdp_latency_miss_fraction': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'cmdp_latency_violation_cost': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'cmdp_training_cost': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'cmdp_cost_active': np.empty(
                self._action_capacity, dtype=np.bool_
            ),
            'cmdp_pre_projection_state_cost': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'cmdp_pre_projection_action_cost': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'cmdp_safety_applied_action_cost': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'cmdp_applied_action_cost': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'cmdp_action_cost_delta': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'cmdp_post_projection_state_cost': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'cmdp_projection_state_cost_delta': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'cmdp_projection_cost_comparison_available': np.empty(
                self._action_capacity, dtype=np.bool_
            ),
            'deadline_miss_count': np.empty(
                self._action_capacity, dtype=np.int32
            ),
            'deadline_overshoot_s_sum': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'cmdp_pre_projection_margin_ratio': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'cmdp_post_projection_margin_ratio': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'cmdp_raw_action_margin_ratio': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'cmdp_safety_applied_action_margin_ratio': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'cmdp_applied_action_margin_ratio': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'cmdp_pre_projection_critic_risk': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'cmdp_post_projection_critic_risk': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'cmdp_projection_critic_risk_delta': np.empty(
                self._action_capacity, dtype=np.float32
            ),
            'raw_action': np.empty(shape, dtype=np.float32),
            'safety_applied_action': np.empty(shape, dtype=np.float32),
            'applied_action': np.empty(shape, dtype=np.float32),
            'safety_action_delta': np.empty(shape, dtype=np.float32),
            'scheduled_return_action_delta': np.empty(
                shape, dtype=np.float32
            ),
            'action_delta': np.empty(shape, dtype=np.float32),
        }

    def _grow_action_arrays(self):
        new_capacity = max(self._action_capacity * 2, self._action_count + 1)
        for key, current in self._action_arrays.items():
            expanded = np.empty(
                (new_capacity,) + current.shape[1:], dtype=current.dtype
            )
            expanded[:self._action_count] = current[:self._action_count]
            self._action_arrays[key] = expanded
        self._action_capacity = new_capacity

    def log_action(self, episode, step, info):
        """Record requested/executed normalized actions for one environment step."""
        if not self.action_logging_enabled:
            return
        if 'raw_action' not in info or 'applied_action' not in info:
            raise KeyError(
                'action logging requires raw_action and applied_action in info'
            )
        raw_action = np.asarray(
            info['raw_action'], dtype=np.float32
        ).reshape(-1)
        applied_action = np.asarray(
            info['applied_action'], dtype=np.float32
        ).reshape(-1)
        safety_applied_action = np.asarray(
            info.get('safety_applied_action', applied_action),
            dtype=np.float32,
        ).reshape(-1)
        if not (
            raw_action.shape == safety_applied_action.shape
            == applied_action.shape
        ):
            raise ValueError(
                'raw, safety-applied, and applied action shapes must match'
            )
        if not (
            np.all(np.isfinite(raw_action))
            and np.all(np.isfinite(safety_applied_action))
            and np.all(np.isfinite(applied_action))
        ):
            raise ValueError('logged actions must contain only finite values')

        action_delta = applied_action - raw_action
        safety_action_delta = safety_applied_action - raw_action
        scheduled_return_action_delta = (
            applied_action - safety_applied_action
        )
        supplied_delta = info.get('action_projection_delta')
        if supplied_delta is not None and not np.allclose(
                np.asarray(supplied_delta, dtype=np.float32).reshape(-1),
                action_delta, atol=1e-7):
            raise ValueError(
                'action_projection_delta is inconsistent with applied - raw'
            )
        supplied_safety_delta = info.get('safety_projection_delta')
        if supplied_safety_delta is not None and not np.allclose(
                np.asarray(
                    supplied_safety_delta, dtype=np.float32
                ).reshape(-1), safety_action_delta, atol=1e-7):
            raise ValueError(
                'safety_projection_delta is inconsistent with '
                'safety-applied - raw'
            )
        supplied_return_delta = info.get(
            'scheduled_return_projection_delta'
        )
        if supplied_return_delta is not None and not np.allclose(
                np.asarray(
                    supplied_return_delta, dtype=np.float32
                ).reshape(-1), scheduled_return_action_delta, atol=1e-7):
            raise ValueError(
                'scheduled_return_projection_delta is inconsistent with '
                'applied - safety-applied'
            )

        if self._action_arrays is None:
            self._allocate_action_arrays(
                raw_action.size, self._action_capacity
            )
        elif raw_action.size != self._action_dim:
            raise ValueError(
                f'action dimension changed from {self._action_dim} '
                f'to {raw_action.size}'
            )
        if self._action_count >= self._action_capacity:
            self._grow_action_arrays()

        index = self._action_count
        arrays = self._action_arrays
        arrays['episode'][index] = int(episode)
        arrays['step'][index] = int(step)
        arrays['safety_mode_applied'][index] = str(
            info.get('safety_mode_applied', 'normal')
        )
        arrays['action_conditioned_emergency_mode'][index] = str(
            info.get('action_conditioned_emergency_mode', 'normal')
        )
        arrays['safety_candidate_mode_before_guard'][index] = str(
            info.get('safety_candidate_mode_before_guard', 'normal')
        )
        arrays['safety_candidate_mode'][index] = str(
            info.get('safety_candidate_mode', 'normal')
        )
        arrays['safety_projection_policy'][index] = str(
            info.get('safety_projection_policy', 'state_threshold')
        )
        arrays['safety_projection_reason'][index] = str(
            info.get('safety_projection_reason', 'unknown')
        )
        arrays['prediction_source'][index] = str(
            info.get('prediction_source', 'unknown')
        )
        arrays['safety_action_overridden'][index] = bool(
            info.get(
                'safety_action_overridden',
                np.any(np.abs(safety_action_delta) > 1e-7),
            )
        )
        arrays['safety_red_feasibility_guard_applied'][index] = bool(
            info.get('safety_red_feasibility_guard_applied', False)
        )
        arrays['safety_action_conditioned_feasible'][index] = bool(
            info.get('safety_action_conditioned_feasible', True)
        )
        arrays['scheduled_return_active'][index] = bool(
            info.get('scheduled_return_active', False)
        )
        arrays['scheduled_return_action_overridden'][index] = bool(
            info.get(
                'scheduled_return_action_overridden',
                np.any(np.abs(scheduled_return_action_delta) > 1e-7),
            )
        )
        arrays['energy_margin_ratio'][index] = float(
            info.get('energy_margin_ratio', np.nan)
        )
        arrays['safety_action_conditioned_margin_ratio'][index] = float(
            info.get('safety_action_conditioned_margin_ratio', np.nan)
        )
        arrays['safety_raw_action_energy_estimate'][index] = float(
            info.get('safety_raw_action_energy_estimate', np.nan)
        )
        arrays['safety_projected_action_energy_estimate'][index] = float(
            info.get('safety_projected_action_energy_estimate', np.nan)
        )
        arrays['safety_projected_energy_saving'][index] = float(
            info.get('safety_projected_energy_saving', 0.0)
        )
        arrays['E_rem'][index] = float(info.get('E_rem', np.nan))
        arrays['E_total'][index] = float(info.get('E_total', np.nan))
        arrays['predicted_energy'][index] = float(
            info.get('predicted_energy', np.nan)
        )
        arrays['raw_predicted_energy'][index] = float(
            info.get('raw_predicted_energy', np.nan)
        )
        arrays['return_energy'][index] = float(
            info.get('return_energy', np.nan)
        )
        arrays['safety_reserve'][index] = float(
            info.get('safety_reserve', np.nan)
        )
        arrays['energy_margin'][index] = float(
            info.get('energy_margin', np.nan)
        )
        for key in (
            'predicted_energy_vector', 'raw_predicted_energy_vector'
        ):
            vector = np.asarray(
                info.get(key, np.zeros(self._prediction_horizon)),
                dtype=np.float32,
            ).reshape(-1)
            if vector.size < self._prediction_horizon:
                vector = np.pad(
                    vector, (0, self._prediction_horizon - vector.size)
                )
            arrays[key][index] = vector[:self._prediction_horizon]
        arrays['cmdp_margin_target'][index] = float(
            info.get('cmdp_margin_target', np.nan)
        )
        arrays['cmdp_cost_mode'][index] = str(
            info.get('cmdp_cost_mode', 'energy_margin')
        )
        arrays['cmdp_energy_cost_source'][index] = str(
            info.get('cmdp_energy_cost_source', 'post_transition_state')
        )
        arrays['cmdp_cost_scale'][index] = float(
            info.get('cmdp_cost_scale', np.nan)
        )
        arrays['cmdp_energy_cost_max'][index] = float(
            info.get('cmdp_energy_cost_max', 1.0)
        )
        arrays['cmdp_latency_cost_scale'][index] = float(
            info.get('cmdp_latency_cost_scale', np.nan)
        )
        arrays['cmdp_margin_cost'][index] = float(
            info.get('cmdp_margin_cost', 0.0)
        )
        arrays['cmdp_margin_cost_unclipped'][index] = float(
            info.get(
                'cmdp_margin_cost_unclipped',
                info.get('cmdp_margin_cost', 0.0),
            )
        )
        arrays['cmdp_latency_overshoot_cost'][index] = float(
            info.get('cmdp_latency_overshoot_cost', 0.0)
        )
        arrays['cmdp_latency_overshoot_normalized'][index] = float(
            info.get('cmdp_latency_overshoot_normalized', 0.0)
        )
        arrays['cmdp_latency_miss_fraction'][index] = float(
            info.get('cmdp_latency_miss_fraction', 0.0)
        )
        arrays['cmdp_latency_violation_cost'][index] = float(
            info.get('cmdp_latency_violation_cost', 0.0)
        )
        arrays['cmdp_training_cost'][index] = float(
            info.get('cmdp_training_cost', 0.0)
        )
        arrays['cmdp_cost_active'][index] = bool(
            info.get(
                'cmdp_cost_active',
                float(info.get('cmdp_training_cost', 0.0)) > 0.0,
            )
        )
        arrays['cmdp_pre_projection_state_cost'][index] = float(
            info.get('cmdp_pre_projection_state_cost', 0.0)
        )
        arrays['cmdp_pre_projection_action_cost'][index] = float(
            info.get('cmdp_pre_projection_action_cost', np.nan)
        )
        arrays['cmdp_safety_applied_action_cost'][index] = float(
            info.get('cmdp_safety_applied_action_cost', np.nan)
        )
        arrays['cmdp_applied_action_cost'][index] = float(
            info.get('cmdp_applied_action_cost', np.nan)
        )
        arrays['cmdp_action_cost_delta'][index] = float(
            info.get('cmdp_action_cost_delta', np.nan)
        )
        arrays['cmdp_post_projection_state_cost'][index] = float(
            info.get(
                'cmdp_post_projection_state_cost',
                info.get('cmdp_training_cost', 0.0),
            )
        )
        arrays['cmdp_projection_state_cost_delta'][index] = float(
            info.get('cmdp_projection_state_cost_delta', 0.0)
        )
        arrays['cmdp_projection_cost_comparison_available'][index] = bool(
            info.get('cmdp_projection_cost_comparison_available', True)
        )
        task_progress = info.get('task_progress', {})
        arrays['deadline_miss_count'][index] = int(
            task_progress.get('deadline_miss_count', 0)
        )
        arrays['deadline_overshoot_s_sum'][index] = float(
            task_progress.get('deadline_overshoot_s_sum', 0.0)
        )
        arrays['cmdp_pre_projection_margin_ratio'][index] = float(
            info.get('cmdp_pre_projection_margin_ratio', np.nan)
        )
        arrays['cmdp_post_projection_margin_ratio'][index] = float(
            info.get(
                'cmdp_post_projection_margin_ratio',
                info.get('energy_margin_ratio', np.nan),
            )
        )
        arrays['cmdp_raw_action_margin_ratio'][index] = float(
            info.get('cmdp_raw_action_margin_ratio', np.nan)
        )
        arrays['cmdp_safety_applied_action_margin_ratio'][index] = float(
            info.get('cmdp_safety_applied_action_margin_ratio', np.nan)
        )
        arrays['cmdp_applied_action_margin_ratio'][index] = float(
            info.get('cmdp_applied_action_margin_ratio', np.nan)
        )
        arrays['cmdp_pre_projection_critic_risk'][index] = float(
            info.get('cmdp_pre_projection_critic_risk', np.nan)
        )
        arrays['cmdp_post_projection_critic_risk'][index] = float(
            info.get('cmdp_post_projection_critic_risk', np.nan)
        )
        arrays['cmdp_projection_critic_risk_delta'][index] = float(
            info.get('cmdp_projection_critic_risk_delta', np.nan)
        )
        absolute_delta = np.abs(action_delta)
        arrays['action_projection_l1'][index] = float(
            np.sum(absolute_delta)
        )
        arrays['action_projection_l2'][index] = float(
            np.linalg.norm(action_delta)
        )
        arrays['action_projection_linf'][index] = float(
            np.max(absolute_delta, initial=0.0)
        )
        safety_absolute_delta = np.abs(safety_action_delta)
        arrays['safety_projection_l1'][index] = float(
            np.sum(safety_absolute_delta)
        )
        arrays['safety_projection_l2'][index] = float(
            np.linalg.norm(safety_action_delta)
        )
        arrays['safety_projection_linf'][index] = float(
            np.max(safety_absolute_delta, initial=0.0)
        )
        return_absolute_delta = np.abs(scheduled_return_action_delta)
        arrays['scheduled_return_projection_l1'][index] = float(
            np.sum(return_absolute_delta)
        )
        arrays['scheduled_return_projection_l2'][index] = float(
            np.linalg.norm(scheduled_return_action_delta)
        )
        arrays['scheduled_return_projection_linf'][index] = float(
            np.max(return_absolute_delta, initial=0.0)
        )
        arrays['energy_uncertainty_scale'][index] = float(
            info.get('energy_uncertainty_scale', 1.0)
        )
        arrays['E_uav_nominal'][index] = float(
            info.get('E_uav_nominal', info.get('E_uav_total', 0.0))
        )
        arrays['E_uav_uncertainty_delta'][index] = float(
            info.get('E_uav_uncertainty_delta', 0.0)
        )
        arrays['raw_action'][index] = raw_action
        arrays['safety_applied_action'][index] = safety_applied_action
        arrays['applied_action'][index] = applied_action
        arrays['safety_action_delta'][index] = safety_action_delta
        arrays['scheduled_return_action_delta'][index] = (
            scheduled_return_action_delta
        )
        arrays['action_delta'][index] = action_delta
        self._action_count += 1

    def action_frame(self):
        """Return the step-level action trace as a flat tabular data frame."""
        if self._action_arrays is None:
            return pd.DataFrame()
        count = self._action_count
        arrays = self._action_arrays
        columns = {
            'episode': arrays['episode'][:count],
            'step': arrays['step'][:count],
            'safety_mode_applied': arrays[
                'safety_mode_applied'
            ][:count],
            'action_conditioned_emergency_mode': arrays[
                'action_conditioned_emergency_mode'
            ][:count],
            'safety_candidate_mode_before_guard': arrays[
                'safety_candidate_mode_before_guard'
            ][:count],
            'safety_candidate_mode': arrays[
                'safety_candidate_mode'
            ][:count],
            'safety_projection_policy': arrays[
                'safety_projection_policy'
            ][:count],
            'safety_projection_reason': arrays[
                'safety_projection_reason'
            ][:count],
            'prediction_source': arrays['prediction_source'][:count],
            'cmdp_cost_mode': arrays['cmdp_cost_mode'][:count],
            'safety_action_overridden': arrays[
                'safety_action_overridden'
            ][:count],
            'safety_red_feasibility_guard_applied': arrays[
                'safety_red_feasibility_guard_applied'
            ][:count],
            'safety_action_conditioned_feasible': arrays[
                'safety_action_conditioned_feasible'
            ][:count],
            'scheduled_return_active': arrays[
                'scheduled_return_active'
            ][:count],
            'scheduled_return_action_overridden': arrays[
                'scheduled_return_action_overridden'
            ][:count],
            'action_projection_l1': arrays[
                'action_projection_l1'
            ][:count],
            'action_projection_l2': arrays[
                'action_projection_l2'
            ][:count],
            'action_projection_linf': arrays[
                'action_projection_linf'
            ][:count],
            'safety_projection_l1': arrays['safety_projection_l1'][:count],
            'safety_projection_l2': arrays['safety_projection_l2'][:count],
            'safety_projection_linf': arrays[
                'safety_projection_linf'
            ][:count],
            'scheduled_return_projection_l1': arrays[
                'scheduled_return_projection_l1'
            ][:count],
            'scheduled_return_projection_l2': arrays[
                'scheduled_return_projection_l2'
            ][:count],
            'scheduled_return_projection_linf': arrays[
                'scheduled_return_projection_linf'
            ][:count],
            'energy_uncertainty_scale': arrays[
                'energy_uncertainty_scale'
            ][:count],
            'E_uav_nominal': arrays['E_uav_nominal'][:count],
            'E_uav_uncertainty_delta': arrays[
                'E_uav_uncertainty_delta'
            ][:count],
            'energy_margin_ratio': arrays['energy_margin_ratio'][:count],
            'safety_action_conditioned_margin_ratio': arrays[
                'safety_action_conditioned_margin_ratio'
            ][:count],
            'safety_raw_action_energy_estimate': arrays[
                'safety_raw_action_energy_estimate'
            ][:count],
            'safety_projected_action_energy_estimate': arrays[
                'safety_projected_action_energy_estimate'
            ][:count],
            'safety_projected_energy_saving': arrays[
                'safety_projected_energy_saving'
            ][:count],
            'E_rem': arrays['E_rem'][:count],
            'E_total': arrays['E_total'][:count],
            'predicted_energy': arrays['predicted_energy'][:count],
            'raw_predicted_energy': arrays[
                'raw_predicted_energy'
            ][:count],
            'return_energy': arrays['return_energy'][:count],
            'safety_reserve': arrays['safety_reserve'][:count],
            'energy_margin': arrays['energy_margin'][:count],
            'cmdp_margin_target': arrays['cmdp_margin_target'][:count],
            'cmdp_energy_cost_source': arrays[
                'cmdp_energy_cost_source'
            ][:count],
            'cmdp_cost_scale': arrays['cmdp_cost_scale'][:count],
            'cmdp_energy_cost_max': arrays[
                'cmdp_energy_cost_max'
            ][:count],
            'cmdp_latency_cost_scale': arrays[
                'cmdp_latency_cost_scale'
            ][:count],
            'cmdp_margin_cost': arrays['cmdp_margin_cost'][:count],
            'cmdp_margin_cost_unclipped': arrays[
                'cmdp_margin_cost_unclipped'
            ][:count],
            'cmdp_latency_overshoot_cost': arrays[
                'cmdp_latency_overshoot_cost'
            ][:count],
            'cmdp_latency_overshoot_normalized': arrays[
                'cmdp_latency_overshoot_normalized'
            ][:count],
            'cmdp_latency_miss_fraction': arrays[
                'cmdp_latency_miss_fraction'
            ][:count],
            'cmdp_latency_violation_cost': arrays[
                'cmdp_latency_violation_cost'
            ][:count],
            'cmdp_training_cost': arrays['cmdp_training_cost'][:count],
            'cmdp_cost_active': arrays['cmdp_cost_active'][:count],
            'cmdp_pre_projection_state_cost': arrays[
                'cmdp_pre_projection_state_cost'
            ][:count],
            'cmdp_pre_projection_action_cost': arrays[
                'cmdp_pre_projection_action_cost'
            ][:count],
            'cmdp_safety_applied_action_cost': arrays[
                'cmdp_safety_applied_action_cost'
            ][:count],
            'cmdp_applied_action_cost': arrays[
                'cmdp_applied_action_cost'
            ][:count],
            'cmdp_action_cost_delta': arrays[
                'cmdp_action_cost_delta'
            ][:count],
            'cmdp_post_projection_state_cost': arrays[
                'cmdp_post_projection_state_cost'
            ][:count],
            'cmdp_projection_state_cost_delta': arrays[
                'cmdp_projection_state_cost_delta'
            ][:count],
            'cmdp_projection_cost_comparison_available': arrays[
                'cmdp_projection_cost_comparison_available'
            ][:count],
            'deadline_miss_count': arrays['deadline_miss_count'][:count],
            'deadline_overshoot_s_sum': arrays[
                'deadline_overshoot_s_sum'
            ][:count],
            'cmdp_pre_projection_margin_ratio': arrays[
                'cmdp_pre_projection_margin_ratio'
            ][:count],
            'cmdp_post_projection_margin_ratio': arrays[
                'cmdp_post_projection_margin_ratio'
            ][:count],
            'cmdp_raw_action_margin_ratio': arrays[
                'cmdp_raw_action_margin_ratio'
            ][:count],
            'cmdp_safety_applied_action_margin_ratio': arrays[
                'cmdp_safety_applied_action_margin_ratio'
            ][:count],
            'cmdp_applied_action_margin_ratio': arrays[
                'cmdp_applied_action_margin_ratio'
            ][:count],
            'cmdp_pre_projection_critic_risk': arrays[
                'cmdp_pre_projection_critic_risk'
            ][:count],
            'cmdp_post_projection_critic_risk': arrays[
                'cmdp_post_projection_critic_risk'
            ][:count],
            'cmdp_projection_critic_risk_delta': arrays[
                'cmdp_projection_critic_risk_delta'
            ][:count],
        }
        for action_index in range(self._action_dim):
            columns[f'raw_action_{action_index}'] = arrays[
                'raw_action'
            ][:count, action_index]
            columns[f'applied_action_{action_index}'] = arrays[
                'applied_action'
            ][:count, action_index]
            columns[f'safety_applied_action_{action_index}'] = arrays[
                'safety_applied_action'
            ][:count, action_index]
            columns[f'safety_action_delta_{action_index}'] = arrays[
                'safety_action_delta'
            ][:count, action_index]
            columns[
                f'scheduled_return_action_delta_{action_index}'
            ] = arrays['scheduled_return_action_delta'][:count, action_index]
            columns[f'action_delta_{action_index}'] = arrays[
                'action_delta'
            ][:count, action_index]
        for horizon_index in range(self._prediction_horizon):
            step = horizon_index + 1
            columns[f'predicted_energy_step_{step}'] = arrays[
                'predicted_energy_vector'
            ][:count, horizon_index]
            columns[f'raw_predicted_energy_step_{step}'] = arrays[
                'raw_predicted_energy_vector'
            ][:count, horizon_index]

        frame = pd.DataFrame(columns)
        # Predictions are produced immediately before executing the logged
        # action, so forecast step 1 aligns with this row's measured E_total.
        grouped_energy = frame.groupby('episode', sort=False)['E_total']
        actual_columns = []
        for horizon_index in range(self._prediction_horizon):
            step = horizon_index + 1
            name = f'actual_energy_step_{step}'
            frame[name] = grouped_energy.shift(-horizon_index)
            actual_columns.append(name)
        frame['actual_horizon_energy'] = frame[actual_columns].sum(
            axis=1, min_count=self._prediction_horizon
        )
        frame['prediction_error_horizon'] = (
            frame['predicted_energy'] - frame['actual_horizon_energy']
        )
        return frame

    def save_actions(self, path):
        """Save the step-level action trace to a dedicated CSV file."""
        output_dir = os.path.dirname(os.path.abspath(path))
        os.makedirs(output_dir, exist_ok=True)
        frame = self.action_frame()
        frame.to_csv(path, index=False)
        return frame

    def save(self, path):
        output_dir = os.path.dirname(os.path.abspath(path))
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame(self.data)
        df.to_csv(path, index=False)
        return df

    def plot(self, path):
        df = pd.DataFrame(self.data)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # 绘制各指标曲线 (滑动平均)
        window = 10
        for ax, col, title in zip(
            axes.flat[:5],
            ['reward', 'energy', 'tasks_completed', 'battery_depleted',
             'returned_safe'],
            ['Episode Reward', 'UAV Energy Consumption', 'Tasks Completed',
             'Battery Depletion Rate (100-ep avg)',
             'Conditional Safe Return Rate (100-ep avg)']
        ):
            if col in df.columns and len(df[col]) > 0:
                values = df[col].values
                if col == 'battery_depleted' or col == 'returned_safe':
                    values = values.astype(float)
                if col == 'returned_safe' and 'return_triggered' in df.columns:
                    triggered = df['return_triggered'].astype(float).values
                    successful_sum = np.convolve(
                        values, np.ones(window), mode='valid'
                    )
                    triggered_sum = np.convolve(
                        triggered, np.ones(window), mode='valid'
                    )
                    smooth = np.divide(
                        successful_sum,
                        triggered_sum,
                        out=np.zeros_like(successful_sum),
                        where=triggered_sum > 0,
                    )
                else:
                    smooth = np.convolve(
                        values, np.ones(window) / window, mode='valid'
                    )
                ax.plot(smooth)
                ax.set_title(title)
                ax.set_xlabel('Episode')

        axes[1, 2].axis('off')
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
