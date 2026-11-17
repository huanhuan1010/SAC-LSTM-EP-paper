"""
消融实验基线: SAC-NoLSTM / SAC-NoEmergency
通过修改SAC-LSTM-EP框架的组件实现
"""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent.sac import SAC
from agent.emergency import EmergencyManager


class SACNoLSTM(SAC):
    """
    SAC (去掉LSTM能量预测):
    - 状态空间回退到不含预测能量
    - 相同的SAC算法, 但less state info
    - EnergyManager使用纯当前能量阈值(无预测)
    """

    def __init__(self, state_dim, action_dim, config):
        # state_dim由环境动态提供，包含队首任务的跨时隙进度特征。
        # LSTM去掉只是预测部分不再使用, SAC本身不受影响
        super().__init__(state_dim, action_dim, config)
        self.name = 'SAC-NoLSTM'

    def select_action(self, state, eval=False, energy_mode='normal',
                      lambda_max=1.0, speed_max=1.0, **kwargs):
        # 完全相同的SAC策略, 但LSTM预测模块被移除
        # 外部不会再提供E_pred到MDP
        return super().select_action(
            state, eval, energy_mode, lambda_max, speed_max, **kwargs
        )


class SACNoEmergency(SAC):
    """
    SAC (去掉策略侧应急感知):
    - Actor始终按normal模式输出动作
    - 环境层红/橙/黄硬安全投影仍保留，保证不同算法共享安全底线
    - 该消融衡量“策略是否显式感知安全模式”，而不是移除物理保护
    """

    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.name = 'SAC-NoEmergency'

    def select_action(self, state, eval=False, energy_mode='normal',
                      lambda_max=1.0, speed_max=1.0, **kwargs):
        # 忽略策略侧约束；环境仍会根据能量裕度执行不可绕过的安全投影。
        return super().select_action(state, eval, 'normal', 1.0, 1.0)


class SACNoConstraint(SAC):
    """Remove learned CMDP optimization while retaining the safety shield."""

    def __init__(self, state_dim, action_dim, config):
        unconstrained_config = dict(config)
        unconstrained_config['cmdp_enabled'] = False
        super().__init__(state_dim, action_dim, unconstrained_config)
        self.name = 'SAC-NoConstraint'
