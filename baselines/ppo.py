"""
PPO基线 (Proximal Policy Optimization)
连续动作空间, 随机策略, 裁剪代理目标
参考: Schulman et al. 2017, PPO-Clip
与SAC对比: on-policy, 无熵正则化, 更保守
"""
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
LOG_RATIO_CLIP = 20.0


class RolloutBuffer:
    """PPO轨迹缓冲区 (on-policy, 每轮后清空)"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.next_values = []

    def push(self, state, action, reward, done, log_prob, value, next_value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.next_values.append(next_value)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.next_values.clear()

    def __len__(self):
        return len(self.states)


class PPOActor(nn.Module):
    """PPO策略网络: 输出动作均值 + 可学习log_std"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        # 初始化
        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.constant_(self.mu.bias, 0)

    def forward(self, state):
        """确定性动作 (评估用): tanh 压缩到 [-1, 1]"""
        x = self.net(state)
        return torch.tanh(self.mu(x))

    def get_dist(self, state):
        """tanh-squashed 高斯策略: 底层分布 N(mu, std), 采样时再 tanh 压缩
        注意: mu 不 tanh —— 若在采样前 tanh, 采样值会无上界 (>1) 导致越界"""
        x = self.net(state)
        mu = self.mu(x)
        # Bound the Gaussian scale so an optimizer step cannot overflow it.
        std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp().expand_as(mu)
        return Normal(mu, std)

    def sample(self, state):
        """tanh-squashed 重参数化采样: z~N(mu,std), a=tanh(z), 带Jacobian修正log_prob"""
        dist = self.get_dist(state)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1)

    def evaluate_log_prob(self, state, action):
        """已存储动作在 tanh-squashed 策略下的 log_prob (反tanh + Jacobian修正)"""
        dist = self.get_dist(state)
        z = torch.arctanh(action.clamp(-0.999, 0.999))
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return log_prob.sum(dim=-1)


class PPOCritic(nn.Module):
    """PPO价值网络: V(s)"""
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.net(state)


class PPO:
    def __init__(self, state_dim, action_dim, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.lr = config.get('lr', 3e-4)
        self.update_epochs = config.get('ppo_epochs', 10)
        self.mini_batch_size = config.get('ppo_mini_batch', 64)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.num_users = config.get('num_users', 5)
        self.name = 'PPO'

        # 网络
        self.actor = PPOActor(state_dim, action_dim).to(self.device)
        self.critic = PPOCritic(state_dim).to(self.device)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.lr
        )

        # 轨迹缓冲区
        self.rollout = RolloutBuffer()

        # 用于与train_baseline.py兼容的接口
        # select_action时暂存log_prob和value, push时配对
        self._pending_log_prob = None
        self._pending_value = None
        self._pending_action = None
        self.last_requested_action = None

    def select_action(self, state, eval=False, energy_mode='normal',
                      lambda_max=1.0, speed_max=1.0, **kwargs):
        """
        选择动作, 遵循PPO on-policy范式
        存储log_prob和value供后续push配对
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if eval:
                action = self.actor(state_t)
                log_prob = torch.zeros(1, device=self.device)
                value = self.critic(state_t).squeeze(-1)
            else:
                action, log_prob = self.actor.sample(state_t)
                value = self.critic(state_t).squeeze(-1)

        action_np = action.cpu().numpy().squeeze(0)
        self._pending_log_prob = log_prob.cpu()
        self._pending_value = value.cpu()
        self._pending_action = action.cpu()
        self.last_requested_action = np.asarray(
            action_np, dtype=np.float32
        ).copy()

        return np.clip(action_np, -1.0, 1.0).astype(np.float32)

    def store_transition(self, state, action, reward, next_state, done):
        """存储环境实际执行动作，并据此重新计算旧策略对数概率。"""
        if any(item is None for item in (
                self._pending_action,
                self._pending_log_prob,
                self._pending_value)):
            raise RuntimeError(
                'PPO transition must immediately follow select_action'
            )
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        executed_action = np.asarray(action, dtype=np.float32).reshape(-1)
        expected_shape = tuple(self._pending_action.shape[1:])
        if executed_action.shape != expected_shape:
            raise ValueError(
                'executed PPO action has shape '
                f'{executed_action.shape}, expected {expected_shape}'
            )
        executed_action_tensor = torch.as_tensor(
            executed_action, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        next_state_tensor = torch.as_tensor(
            next_state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            executed_log_prob = self.actor.evaluate_log_prob(
                state_tensor, executed_action_tensor
            ).cpu()
            next_value = self.critic(next_state_tensor).squeeze(-1).cpu()
        self.rollout.push(
            state,
            executed_action.copy(),
            reward,
            done,
            float(executed_log_prob.item()),
            float(self._pending_value.item()),
            float(next_value.item()),
        )
        self._pending_action = None
        self._pending_log_prob = None
        self._pending_value = None

    def update(self):
        """PPO策略更新: 使用GAE + clipped surrogate (on-policy, 全轨迹后执行)"""
        if len(self.rollout) < 128:
            return {}

        # 提取数据
        states = torch.FloatTensor(np.array(self.rollout.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.rollout.actions)).to(self.device)
        # One scalar joint-action log probability is stored per transition.
        # A (N, 1) tensor would broadcast against (N,) to an invalid (N, N)
        # probability-ratio matrix and quickly destabilize PPO.
        old_log_probs = torch.as_tensor(
            np.asarray(self.rollout.log_probs, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1)
        rewards = np.array(self.rollout.rewards)
        dones = np.array(self.rollout.dones)
        old_values = np.array(self.rollout.values)
        next_values = np.asarray(self.rollout.next_values, dtype=np.float32)

        # === GAE优势估计 ===
        values = old_values.astype(np.float32).reshape(-1)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            next_value = 0.0 if dones[t] else next_values[t]
            delta = rewards[t] + self.gamma * next_value - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * last_gae * (1 - dones[t])
            advantages[t] = last_gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        if not all(torch.isfinite(tensor).all() for tensor in (
                states, actions, old_log_probs, advantages, returns)):
            raise FloatingPointError('PPO rollout contains non-finite values')

        # === PPO迭代更新 ===
        n_samples = len(states)
        indices = np.arange(n_samples)
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0

        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_samples, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_idx = indices[start:end]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                # 新策略的log概率 (tanh-squashed, 含Jacobian修正)
                new_log_probs = self.actor.evaluate_log_prob(mb_states, mb_actions)
                if new_log_probs.shape != mb_old_log_probs.shape:
                    raise RuntimeError(
                        'PPO log-probability shape mismatch: '
                        f'new={tuple(new_log_probs.shape)}, '
                        f'old={tuple(mb_old_log_probs.shape)}'
                    )
                if not torch.isfinite(new_log_probs).all():
                    raise FloatingPointError(
                        'PPO actor produced non-finite log probabilities'
                    )
                # 熵奖励: 用底层高斯熵近似 (tanh-squashed策略熵的标准做法)
                entropy = self.actor.get_dist(mb_states).entropy().sum(dim=-1).mean()

                # 重要性采样比率
                log_ratio = torch.clamp(
                    new_log_probs - mb_old_log_probs,
                    -LOG_RATIO_CLIP,
                    LOG_RATIO_CLIP,
                )
                ratio = log_ratio.exp()

                # Clipped surrogate目标
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # 熵奖励 (鼓励探索)
                actor_loss -= self.entropy_coef * entropy

                # Critic损失 (value function)
                values_pred = self.critic(mb_states).squeeze(-1)
                critic_loss = nn.MSELoss()(values_pred, mb_returns)

                # 联合优化
                loss = actor_loss + 0.5 * critic_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()

        n_updates = self.update_epochs * math.ceil(
            n_samples / self.mini_batch_size
        )

        # 清空rollout
        self.rollout.clear()

        return {
            'actor_loss': total_actor_loss / max(1, n_updates),
            'critic_loss': total_critic_loss / max(1, n_updates),
            'entropy': total_entropy / max(1, n_updates),
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
        print(f'PPO saved to {path}')

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        print(f'PPO loaded from {path}')
