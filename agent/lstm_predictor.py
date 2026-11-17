"""LSTM能量预测模块"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

from utils.seeding import derive_seed, isolated_torch_seed, normalize_seed

from agent.lstm_features import (
    LEGACY_SCHEMA,
    lstm_feature_input_dim,
    lstm_feature_names,
    lstm_history_gate_weight,
    normalize_lstm_feature_schema,
)


class LSTMEnergyPredictor(nn.Module):
    """
    LSTM能量预测 (输入/输出均归一化为电池占比量级, 对外接口返回 J)
    输入列由 feature_schema 固定；第0列始终是 E_step_norm。
    输出: 未来 horizon 步的能量值 (J, 内部×battery_capacity 反归一化)
    """

    def __init__(self, input_dim=6, hidden_dim=64, num_layers=2,
                 lookback=10, horizon=5, lr=1e-3, battery_capacity=100000.0,
                 prediction_mode='absolute',
                 residual_correction=True, residual_ema_alpha=0.2,
                 residual_clip_ratio=0.01,
                 feature_schema=LEGACY_SCHEMA,
                 history_gate_warmup_steps=0,
                 rng_seed=None):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.feature_schema = normalize_lstm_feature_schema(feature_schema)
        expected_input_dim = lstm_feature_input_dim(self.feature_schema)
        if int(input_dim) != expected_input_dim:
            raise ValueError(
                f'input_dim={input_dim} conflicts with feature_schema='
                f'{self.feature_schema} (expected {expected_input_dim})'
            )
        self.input_dim = expected_input_dim
        self.feature_names = lstm_feature_names(self.feature_schema)
        self.hidden_dim = hidden_dim
        # LSTM dropout during online updates must not consume the actor or
        # cost-network RNG streams.  When no explicit seed is supplied, derive
        # one from the construction seed visible inside isolated_torch_seed().
        seed_basis = torch.initial_seed() if rng_seed is None else rng_seed
        self.rng_seed = normalize_seed(seed_basis)
        self.rng_update_count = 0
        self.history_gate_warmup_steps = int(history_gate_warmup_steps)
        if self.history_gate_warmup_steps < 0:
            raise ValueError('history_gate_warmup_steps must be nonnegative')
        # #1修复: 输入/目标统一除以 battery_capacity 归一化到 [0,1] 量级.
        # 否则 E_step(~100J) 与其他特征(0~1)尺度差100倍, 在线微调几乎学不动
        # (预测卡在~0, loss≈E[E²]=10000 且不下降); 输出侧再反乘回 J.
        self.battery_capacity = battery_capacity
        self.prediction_mode = str(prediction_mode)
        if self.prediction_mode not in ('absolute', 'persistence_residual'):
            raise ValueError(
                'prediction_mode must be absolute or persistence_residual'
            )
        self.residual_correction = bool(residual_correction)
        self.residual_ema_alpha = float(residual_ema_alpha)
        if not 0.0 < self.residual_ema_alpha <= 1.0:
            raise ValueError('residual_ema_alpha must lie in (0, 1]')
        self.residual_clip_energy = (
            max(float(residual_clip_ratio), 0.0) * self.battery_capacity
        )

        self.lstm = nn.LSTM(self.input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, horizon)
        if self.prediction_mode == 'persistence_residual':
            # 残差模式从严格的 persistence 基线开始，再学习每个预测步的
            # 有符号对数比例修正。零初始化不会改变旧 absolute 模式语义。
            nn.init.zeros_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)
        else:
            # softplus(约-7)≈9e-4；按100 kJ电池折算约90 J/步，避免未训练
            # 模型一开始输出不合理的大能耗，同时保证预测始终非负。
            nn.init.constant_(self.fc.bias, -7.0)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.history = []  # 缓存最近N步用于预测
        self.reset_runtime_state(clear_history=False)

    def reset_runtime_state(self, clear_history=True):
        """重置单个 episode 的预测残差状态。

        有符号残差均值用于修正系统性偏差，绝对残差均值用于量化近期预测
        不确定性。二者均为逐步能耗（J），不写入模型权重。
        """
        if clear_history:
            self.history.clear()
            self.history_observation_count = 0
        elif not hasattr(self, 'history_observation_count'):
            self.history_observation_count = len(self.history)
        self.residual_bias_joule = 0.0
        self.residual_abs_ema_joule = 0.0
        self.residual_count = 0
        self._pending_raw_prediction_joule = None
        self.last_raw_prediction_joule = np.zeros(
            self.horizon, dtype=np.float32
        )
        self.last_ungated_prediction_joule = np.zeros(
            self.horizon, dtype=np.float32
        )
        self.last_corrected_prediction_joule = np.zeros(
            self.horizon, dtype=np.float32
        )
        self.last_history_gate_weight = 0.0

    def observe_actual_energy(self, actual_energy):
        """用最新实测单步能耗更新指数移动平均预测残差。

        residual = E_actual - E_pred_raw。正值表示模型近期低估能耗，后续
        预测应上调；负值表示模型近期高估能耗。只有存在上一时刻的有效
        原始预测时才更新，避免历史不足阶段用零预测产生虚假大残差。
        """
        if self._pending_raw_prediction_joule is None:
            return None

        actual = max(float(actual_energy), 0.0)
        residual = actual - float(self._pending_raw_prediction_joule)
        if self.residual_clip_energy > 0.0:
            residual = float(np.clip(
                residual,
                -self.residual_clip_energy,
                self.residual_clip_energy,
            ))

        alpha = self.residual_ema_alpha
        if self.residual_count == 0:
            self.residual_bias_joule = residual
            self.residual_abs_ema_joule = abs(residual)
        else:
            self.residual_bias_joule = (
                (1.0 - alpha) * self.residual_bias_joule + alpha * residual
            )
            self.residual_abs_ema_joule = (
                (1.0 - alpha) * self.residual_abs_ema_joule
                + alpha * abs(residual)
            )
        self.residual_count += 1
        self._pending_raw_prediction_joule = None
        return residual

    def forward(self, x):
        """
        x: (batch, lookback, input_dim) 或 (lookback, input_dim)
        返回: (batch, horizon) 或 (horizon,)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # 增加batch维度
        lstm_out, _ = self.lstm(x)
        raw_prediction = self.fc(lstm_out[:, -1, :])  # 取最后一步输出
        if self.prediction_mode == 'persistence_residual':
            persistence = x[:, -1, 0:1].expand(-1, self.horizon)
            # 直接相加时，约 1e-3 的 Adam 首次更新可能大于约 7e-4 的
            # 归一化单步能耗，使 ReLU/clip 落入零梯度区。改为学习有界
            # log-ratio：初始值仍严格等于 persistence，且全程保持正值和
            # 有效梯度。
            log_ratio = torch.clamp(raw_prediction, min=-2.0, max=2.0)
            prediction = persistence * torch.exp(log_ratio)
        else:
            prediction = F.softplus(raw_prediction)   # 物理约束: 能耗不能为负
        if x.size(0) == 1:
            return prediction.squeeze(0)
        return prediction

    def build_input(self, energy_hist, state_features):
        """
        构建LSTM输入
        energy_hist: 最近N步的能耗值列表 [E_step_0, ..., E_step_N]
        state_features: 对应步、按 feature_schema 排列且不含 E_step 的特征
        输出: (lookback, input_dim)
        """
        if len(energy_hist) != self.lookback:
            raise ValueError(
                f'energy history length must be {self.lookback}'
            )
        if len(state_features) != self.lookback:
            raise ValueError(
                f'state feature history length must be {self.lookback}'
            )
        features = []
        state_dim = self.input_dim - 1
        for i in range(self.lookback):
            E_step = energy_hist[i]
            s = np.asarray(state_features[i], dtype=np.float32).reshape(-1)
            if s.size != state_dim:
                raise ValueError(
                    f'state feature row has {s.size} values; expected '
                    f'{state_dim} for {self.feature_schema}'
                )
            # #1修复: E_step 归一化到 [0,1] 量级 (与 E_rem/vx/vy/vz/h 一致)
            E_step_norm = E_step / self.battery_capacity
            feat = np.concatenate([[E_step_norm], s])
            features.append(feat)

        return torch.FloatTensor(np.array(features, dtype=np.float32))

    def update(self, history, target):
        """在线微调 (target 为 J 原始量级, 内部归一化到与输入一致)"""
        if history.dim() == 2:
            history = history.unsqueeze(0)
        if target.dim() == 1:
            target = target.unsqueeze(0)
        target = torch.clamp(target / self.battery_capacity, min=0.0)

        update_seed = derive_seed(
            self.rng_seed, f'lstm_online_update:{self.rng_update_count}'
        )
        with isolated_torch_seed(update_seed):
            if self.prediction_mode == 'persistence_residual':
                # 直接用约 1e-8 的绝对归一化 MSE 训练残差头时，网络容易停在
                # persistence 初值。残差模式改为预测未来/当前能耗的 log-ratio，
                # 使学习目标处于 O(1) 量级；推理接口仍返回绝对能耗。
                lstm_out, _ = self.lstm(history)
                predicted_log_ratio = self.fc(lstm_out[:, -1, :])
                persistence = torch.clamp(history[:, -1, 0:1], min=1e-8)
                target_log_ratio = torch.log(torch.clamp(
                    target / persistence, min=np.exp(-2.0), max=np.exp(2.0)
                ))
                loss = self.loss_fn(predicted_log_ratio, target_log_ratio)
            else:
                pred = self.forward(history)
                if pred.dim() == 1:
                    pred = pred.unsqueeze(0)
                loss = self.loss_fn(pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        self.rng_update_count += 1
        return loss.item()

    def get_energy_features(self):
        """从历史记录生成经近期残差校正的预测特征（供MDP状态使用）。

        校正仅作用于推理输出，不替代 LSTM 在线微调：在线微调学习长期模型，
        指数移动平均残差补偿短期工况变化。

        history格式: [(E_step, *feature_schema对应的状态/控制特征), ...]
        """
        if len(self.history) < self.lookback:
            self._pending_raw_prediction_joule = None
            self.last_raw_prediction_joule.fill(0.0)
            self.last_ungated_prediction_joule.fill(0.0)
            self.last_corrected_prediction_joule.fill(0.0)
            self.last_history_gate_weight = 0.0
            return self.last_corrected_prediction_joule.copy()

        hist_tuples = self.history[-self.lookback:]
        energy_vals = [h[0] for h in hist_tuples]         # [E_step]
        state_feats = [list(h[1:self.input_dim]) for h in hist_tuples]
        input_tensor = self.build_input(energy_vals, state_feats)
        was_training = self.training
        self.eval()
        with torch.no_grad():
            pred = self.forward(input_tensor).cpu().numpy()
        if was_training:
            self.train()

        # #1修复: 反归一化回 J (augment_state_with_epred 会再除以 battery_capacity;
        # EmergencyManager 的预测投影也需要 J 量级)
        ungated_joule = np.nan_to_num(
            pred * self.battery_capacity,
            nan=0.0, posinf=self.battery_capacity, neginf=0.0,
        )
        ungated_joule = np.clip(
            ungated_joule, 0.0, self.battery_capacity
        ).astype(np.float32)
        gate_weight = lstm_history_gate_weight(
            self.history_observation_count,
            self.lookback,
            self.history_gate_warmup_steps,
        )
        persistence_joule = float(energy_vals[-1])
        raw_joule = np.clip(
            persistence_joule
            + gate_weight * (ungated_joule - persistence_joule),
            0.0,
            self.battery_capacity,
        ).astype(np.float32)
        correction = self.residual_bias_joule if self.residual_correction else 0.0
        corrected_joule = np.clip(
            raw_joule + correction, 0.0, self.battery_capacity
        ).astype(np.float32)

        self.last_ungated_prediction_joule = ungated_joule.copy()
        self.last_raw_prediction_joule = raw_joule.copy()
        self.last_corrected_prediction_joule = corrected_joule.copy()
        self.last_history_gate_weight = float(gate_weight)
        self._pending_raw_prediction_joule = float(raw_joule[0])
        return corrected_joule

    def save(self, path):
        """保存能耗预测器；SAC与LSTM必须成对保存和评估。"""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path, map_location='cpu'):
        """加载能耗预测器参数。"""
        try:
            state_dict = torch.load(
                path, map_location=map_location, weights_only=True
            )
        except TypeError:  # 兼容旧版PyTorch
            state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)
        return self
