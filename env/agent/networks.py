"""神经网络定义: Actor (策略网络), Critic (双Q网络)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Actor(nn.Module):
    """SAC策略网络: 输出动作分布 N(μ, σ)"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        # 输出层用小权重初始化, 防止初始动作极端
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.xavier_uniform_(self.net[2].weight)
        nn.init.uniform_(self.mu_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.net[0].bias)
        nn.init.zeros_(self.net[2].bias)
        nn.init.zeros_(self.mu_head.bias)
        # log_std初始偏置=0 (std=1.0), tanh(N(0,1))在[-0.76,0.76]区间, 覆盖均匀探索
        nn.init.constant_(self.log_std_head.bias, 0.0)

    def forward(self, state):
        x = self.net(state)
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, -20, 2)  # 限制标准差范围
        return mu, log_std.exp()

    def sample(self, state):
        """采样动作 + 计算log概率"""
        mu, std = self.forward(state)
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()  # 重参数化采样
        action = torch.tanh(z)  # 压缩到 [-1, 1]
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def get_deterministic_action(self, state):
        """评估模式下使用确定性动作"""
        mu, _ = self.forward(state)
        return torch.tanh(mu)

    def evaluate_log_prob(self, state, action):
        """计算给定动作在策略下的log概率 (用于BC预训练)"""
        mu, std = self.forward(state)
        dist = torch.distributions.Normal(mu, std)
        z = torch.arctanh(action.clamp(-0.999, 0.999))
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return log_prob.sum(dim=-1, keepdim=True)


class Critic(nn.Module):
    """SAC评价网络: Q(s, a)"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.xavier_uniform_(self.net[2].weight)
        nn.init.uniform_(self.net[4].weight, -3e-3, 3e-3)
        nn.init.zeros_(self.net[0].bias)
        nn.init.zeros_(self.net[2].bias)
        nn.init.zeros_(self.net[4].bias)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
