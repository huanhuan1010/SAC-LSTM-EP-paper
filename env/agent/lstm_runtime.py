"""LSTM 历史维护、在线微调、将 E_pred 写入 SAC 观测。"""
import numpy as np
import torch

from agent.emergency import EmergencyManager
from agent.lstm_features import LEGACY_SCHEMA, lstm_state_features


def epred_index(num_users):
    """观测向量中 E_pred(horizon) 的起始下标。"""
    # 基础状态10维 + 信道N维 + 队列N维 + 队首任务进度5N维。
    return 10 + 7 * num_users


def energy_margin_index(num_users, horizon):
    """观测向量中显式能量裕度特征的下标。

    E_pred 后依次是 task_priority 和 energy_margin_ratio。任务返航模式会
    在它们后面追加 mission_progress 与 scheduled_return_phase，因此不能
    使用 ``-1`` 定位能量裕度。
    """
    return epred_index(num_users) + int(horizon) + 1


def augment_state_with_epred(state, E_pred, num_users, horizon, battery_capacity):
    """把逐步能耗预测 (J) 归一化为占电池比例，写入 state 的 E_pred 槽。"""
    s = np.asarray(state, dtype=np.float32).copy()
    start = epred_index(num_users)
    pred = np.asarray(E_pred, dtype=np.float32).reshape(-1)
    if pred.size < horizon:
        pred = np.pad(pred, (0, horizon - pred.size))
    pred = pred[:horizon]
    s[start:start + horizon] = pred / float(battery_capacity)
    return s


def append_lstm_history(lstm_predictor, env, info):
    # 用本步真实总能耗修正上一步生成的短期预测，再写入历史窗口。
    # 这对应“实测功率 - 预测功率”的在线残差校正思想。
    actual_energy = info.get('E_total', 0.0)
    lstm_predictor.observe_actual_energy(actual_energy)
    state_features = lstm_state_features(
        env, info, lstm_predictor.feature_schema
    )
    lstm_predictor.history.append((actual_energy, *state_features.tolist()))
    lstm_predictor.history_observation_count += 1
    # 与 state_frames 保持同一滑动窗口 (lookback+horizon)，
    # 防止 history 无界增长导致两数组时间轴错位 (在线微调索引 bug)。
    window = lstm_predictor.lookback + lstm_predictor.horizon
    while len(lstm_predictor.history) > window:
        lstm_predictor.history.pop(0)


def append_state_frame(state_frames, env, max_len, info=None,
                       feature_schema=LEGACY_SCHEMA):
    """Append the same post-step feature vector used by runtime inference."""
    state_frames.append(lstm_state_features(env, info, feature_schema))
    while len(state_frames) > max_len:
        state_frames.pop(0)


def maybe_update_lstm(lstm_predictor, state_frames, step, update_interval=10):
    """按配置间隔使用滑动窗口在线微调 LSTM。"""
    update_interval = max(1, int(update_interval))
    if step % update_interval != 0:
        return
    lookback = lstm_predictor.lookback
    horizon = lstm_predictor.horizon
    window = lookback + horizon
    # history 与 state_frames 均被裁剪为最近 window 帧，统一取尾部窗口即可对齐时间轴。
    # 不能再用 history 的绝对索引 (len(history)-window)：history 被裁剪后长度恒为 window，
    # 但更早的版本中 state_frames 容量只有 window，绝对索引会随步数增长而越界，
    # 导致 idx+lookback > len(state_frames)，在线微调从未真正执行。
    if len(lstm_predictor.history) < window or len(state_frames) < window:
        return
    hist_energy = [h[0] for h in lstm_predictor.history[-window:-horizon]]
    state_slice = state_frames[-window:-horizon]
    target = np.array([
        h[0] for h in lstm_predictor.history[-horizon:]
    ], dtype=np.float32)
    input_tensor = lstm_predictor.build_input(hist_energy, state_slice)
    target_tensor = torch.FloatTensor(target).unsqueeze(0)
    lstm_predictor.update(input_tensor, target_tensor)


def emergency_zone_encoding(mode):
    """紧急模式 → 3维 zone 编码 (写入观测槽位 1:4)。

    状态只有 3 个 zone 槽, 但紧急模式有 4 档 (normal/yellow/orange/red),
    采用温度计(severity)编码, 数值单调递增且 4 档全部可区分:
      normal → [0,0,0]     (无约束)
      yellow → [1,0,0]     (节能模式, λ≤0.7 / v≤0.7)
      orange → [1,1,0]     (暂停非核心, λ≤0.5 / v≤0.6)
      red    → [1,1,1]     (CBF迫降接管, λ≤0.2 / v≤0.25)

    取代 env 原始的 energy_zone (high/medium/low, 阈值 0.60/0.30):
    energy_zone 只由当前 E_rem 决定, 与 EmergencyManager 实际生效的约束档位
    (阈值 0.40/0.30/0.20 + LSTM 预测投影) 不一致, 导致 agent 无法从状态中
    观测到当前约束, MDP 部分可观测 (#3)。
    """
    return {
        'normal': np.array([0.0, 0.0, 0.0], dtype=np.float32),
        'yellow': np.array([1.0, 0.0, 0.0], dtype=np.float32),
        'orange': np.array([1.0, 1.0, 0.0], dtype=np.float32),
        'red':    np.array([1.0, 1.0, 1.0], dtype=np.float32),
    }.get(mode, np.array([0.0, 0.0, 0.0], dtype=np.float32))


def observation_for_agent(state, env, lstm_predictor, config, use_lstm,
                          emergency_mode=None, safety_controller=None):
    """返回 (state_aug, E_pred_J) 供 SAC 与 Emergency 使用。

    emergency_mode: 显式指定当前紧急模式并写入 state[1:4]。
    safety_controller: 在更新本步预测后解析控制器模式。实验驱动应优先
      传入该参数，避免单阈值控制器使用上一时隙预测。为兼容旧调用，两者
      均为 None 时仍按分级预测裕度控制器推算模式。
    """
    horizon = config.get('horizon', 5)
    if use_lstm and lstm_predictor is not None:
        history_ready = (
            len(lstm_predictor.history) >= lstm_predictor.lookback
        )
        if history_ready:
            E_pred = lstm_predictor.get_energy_features()
            raw_prediction = lstm_predictor.last_raw_prediction_joule
            ungated_prediction = (
                lstm_predictor.last_ungated_prediction_joule
            )
            history_gate_weight = (
                lstm_predictor.last_history_gate_weight
            )
            residual_bias = lstm_predictor.residual_bias_joule
            residual_uncertainty = lstm_predictor.residual_abs_ema_joule
            prediction_source = 'lstm'
        else:
            # A newly reset episode has not accumulated ``lookback`` samples.
            # Treating the missing sequence as zero energy makes the predictive
            # margin optimistically unsafe.  Bootstrap only this cold-start
            # interval with a physically meaningful per-slot estimate; once
            # enough history exists the actual LSTM output takes over.
            if env.energy_history:
                per_step_energy = float(np.mean(env.energy_history))
                prediction_source = 'history_bootstrap'
            else:
                per_step_energy = float(
                    env.energy.flight_power(0.0) * env.dt
                )
                prediction_source = 'hover_bootstrap'
            E_pred = np.full(
                horizon, max(per_step_energy, 0.0), dtype=np.float32
            )
            raw_prediction = E_pred
            ungated_prediction = E_pred
            history_gate_weight = 0.0
            residual_bias = 0.0
            residual_uncertainty = 0.0
        state_aug = augment_state_with_epred(
            state, E_pred, env.num_users, horizon, env.energy.battery_capacity,
        )
    else:
        E_pred = np.zeros(horizon, dtype=np.float32)
        state_aug = np.asarray(state, dtype=np.float32).copy()
        raw_prediction = E_pred
        ungated_prediction = E_pred
        history_gate_weight = 0.0
        residual_bias = 0.0
        residual_uncertainty = 0.0
        prediction_source = 'disabled'

    # 状态分区、CMDP代价与安全管理器共享同一份短期能耗预测；基础奖励
    # 已与预测和安全状态解耦。
    if use_lstm and lstm_predictor is not None:
        env.set_energy_prediction(
            E_pred,
            raw_prediction=raw_prediction,
            ungated_prediction=ungated_prediction,
            history_gate_weight=history_gate_weight,
            residual_bias=residual_bias,
            residual_uncertainty=residual_uncertainty,
            prediction_source=prediction_source,
        )
    else:
        env.set_energy_prediction(
            E_pred, prediction_source=prediction_source
        )

    if safety_controller is not None:
        controller = str(safety_controller).strip().lower()
        if controller == 'none':
            emergency_mode = 'normal'
        elif controller in (
            'soc_threshold', 'margin_threshold', 'margin_single_level'
        ):
            emergency_mode = env.get_autonomous_safety_mode()
        elif controller == 'margin':
            emergency_mode = None
        else:
            raise ValueError(
                f'unsupported observation safety controller: {controller}'
            )

    # #3修复: zone槽位(1:4)写入紧急模式编码, 替代env的energy_zone(high/medium/low)
    if emergency_mode is None:
        energy_ctx = env.get_energy_context()
        emergency_mode = EmergencyManager(config).check(
            env.E_rem,
            E_pred,
            return_energy=energy_ctx['return_energy'],
            safety_reserve=energy_ctx['safety_reserve'],
        )[0]
    state_aug[1:4] = emergency_zone_encoding(emergency_mode)
    # 显式写入能量裕度槽。mission模式会在该槽之后追加两个任务阶段特征，
    # 因而不能用 state_aug[-1]，否则会覆盖 scheduled_return_phase。
    margin_index = energy_margin_index(env.num_users, horizon)
    state_aug[margin_index] = np.clip(
        env.get_energy_context()['energy_margin_ratio'], -1.0, 1.0
    )
    return state_aug, E_pred
