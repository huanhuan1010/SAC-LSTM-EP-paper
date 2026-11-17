"""
DDPG基线 (Deep Deterministic Policy Gradient)
连续动作空间, 确定性策略, 经验回放
与SAC对比: 无熵正则化, 确定性策略
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy


class DDPGActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh(),
        )
    def forward(self, state):
        return self.net(state)


class DDPGCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        indices = np.random.choice(len(self.buffer), batch_size)
        batch = [self.buffer[i] for i in indices]
        s, a, r, ns, d = zip(*batch)
        return (np.array(s, dtype=np.float32),
                np.array(a, dtype=np.float32),
                np.array(r, dtype=np.float32).reshape(-1, 1),
                np.array(ns, dtype=np.float32),
                np.array(d, dtype=np.float32).reshape(-1, 1))

    def __len__(self):
        return len(self.buffer)


class DDPG:
    def __init__(self, state_dim, action_dim, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = int(action_dim)
        if self.action_dim <= 0:
            raise ValueError('action_dim must be positive')
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.lr = config.get('lr', 3e-4)
        self.batch_size = config.get('batch_size', 256)
        self.num_users = config.get('num_users', 5)
        # #4: warmup 与 SAC 统一 (config.warmup_steps), 保证对比公平
        self.warmup_steps = config.get('warmup_steps', 0)
        self.name = 'DDPG'

        # 网络
        self.actor = DDPGActor(state_dim, action_dim).to(self.device)
        self.critic = DDPGCritic(state_dim, action_dim).to(self.device)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)

        # 优化器
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr)

        # OU噪声 (训练时探索)
        self.noise_scale = 0.1

        # 经验回放
        capacity = config.get('buffer_capacity', 100000)
        self.replay_buffer = ReplayBuffer(capacity)

    def select_action(self, state, eval=False, energy_mode='normal',
                      lambda_max=1.0, speed_max=1.0, **kwargs):
        if not eval and len(self.replay_buffer) < self.warmup_steps:
            action = np.random.uniform(
                -1.0, 1.0, size=self.action_dim
            ).astype(np.float32)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.actor(state).cpu().numpy().squeeze(0)
        # 预热结束后，训练时在确定性策略上添加探索噪声。
        if not eval and len(self.replay_buffer) >= self.warmup_steps:
            noise = np.random.randn(len(action)) * self.noise_scale
            action = np.clip(action + noise, -1, 1)
            self.noise_scale *= 0.99995  # 缓慢退火
            self.noise_scale = max(self.noise_scale, 0.01)

        self.last_requested_action = np.asarray(
            action, dtype=np.float32
        ).copy()
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return {}
        # #4: warmup 与 SAC 对齐 (buffer<warmup_steps 时纯随机探索不更新)
        if len(self.replay_buffer) < self.warmup_steps:
            return {}

        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return {}

        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 更新Critic
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # 更新Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # 软更新目标网络
        for tp, p in zip(self.target_actor.parameters(), self.actor.parameters()):
            tp.data.copy_(self.tau * p + (1 - self.tau) * tp)
        for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * p + (1 - self.tau) * tp)

        return {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item()}

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.target_actor.load_state_dict(ckpt['target_actor'])
        self.target_critic.load_state_dict(ckpt['target_critic'])
