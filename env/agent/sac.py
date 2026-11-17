"""
SAC (Soft Actor-Critic) 算法实现
参考: Haarnoja et al. 2018, Soft Actor-Critic
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy
from .networks import Actor, Critic
from .replay_buffer import PrioritizedReplayBuffer


class SAC:
    def __init__(self, state_dim, action_dim, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = int(action_dim)
        # Safety projection is owned exclusively by UAVEnv.step(). Policies
        # return raw normalized actions so requested/executed provenance is
        # preserved for replay and logging.
        self.action_projection = str(
            config.get('action_projection', 'identity')
        )
        if self.action_projection != 'identity':
            raise ValueError(
                "action_projection must be 'identity'; safety projection "
                "is applied exclusively by UAVEnv.step()"
            )
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)           # 软更新系数
        self.lr = config.get('lr', 3e-4)
        self.batch_size = config.get('batch_size', 256)
        self.alpha_lr = config.get('alpha_lr', 3e-4)
        configured_target_entropy = config.get('target_entropy')
        self.target_entropy = float(
            -action_dim
            if configured_target_entropy is None
            else configured_target_entropy
        )
        self.num_users = config.get('num_users', 5)    # 用户数，用于动作索引解包
        self.constraint_enabled = bool(config.get('cmdp_enabled', True))
        self.cmdp_cost_limit = float(config.get(
            'cmdp_cost_limit', config.get('constraint_cost_limit', 5.0)
        ))
        # Compatibility for old checkpoints/callers. New experiment code uses
        # the explicit CMDP name so it cannot be confused with physical
        # evaluation violations.
        self.constraint_cost_limit = self.cmdp_cost_limit
        self.lagrange_lr = float(config.get('lagrange_lr', 1e-4))
        self.lagrange_max = float(config.get('lagrange_max', 100.0))
        self.lagrange_cost_ema_alpha = float(
            config.get('lagrange_cost_ema_alpha', 0.10)
        )
        if self.cmdp_cost_limit < 0.0:
            raise ValueError('cmdp_cost_limit must be nonnegative')
        if self.lagrange_lr <= 0.0 or self.lagrange_max < np.exp(-10.0):
            raise ValueError('Lagrange learning rate and maximum must be positive')
        if not 0.0 < self.lagrange_cost_ema_alpha <= 1.0:
            raise ValueError('lagrange_cost_ema_alpha must lie in (0, 1]')

        # 网络
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic1 = Critic(state_dim, action_dim).to(self.device)
        self.critic2 = Critic(state_dim, action_dim).to(self.device)
        self.target_critic1 = deepcopy(self.critic1)
        self.target_critic2 = deepcopy(self.critic2)
        if self.constraint_enabled:
            # Twin cost critics use the conservative maximum.  This mirrors
            # clipped double-Q for reward while avoiding optimistic
            # under-estimation of unsafe discounted cost.
            self.cost_critic1 = Critic(state_dim, action_dim).to(self.device)
            self.cost_critic2 = Critic(state_dim, action_dim).to(self.device)
            self.target_cost_critic1 = deepcopy(self.cost_critic1)
            self.target_cost_critic2 = deepcopy(self.cost_critic2)
        else:
            self.cost_critic1 = None
            self.cost_critic2 = None
            self.target_cost_critic1 = None
            self.target_cost_critic2 = None

        # 优化器
        # Actor用更小学习率 + 延迟更新, 防止Q值高估导致策略坍缩
        self.actor_lr = config.get('actor_lr', self.lr / 3)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.actor_update_delay = config.get('actor_update_delay', 2)
        # #4: warmup 统一在算法内部生效 (纯随机探索填buffer), 各训练入口行为一致
        self.warmup_steps = config.get('warmup_steps', 0)
        self.critic_optim = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=self.lr
        )
        self.cost_critic_optim = (
            optim.Adam(
                list(self.cost_critic1.parameters())
                + list(self.cost_critic2.parameters()),
                lr=self.lr,
            )
            if self.constraint_enabled else None
        )

        # Nonnegative dual variable for J_C(pi) <= d.  Updating it from the
        # episode-level discounted cost keeps the dual target consistent with
        # the CMDP definition, while the cost critic supplies action gradients.
        initial_lagrange = float(config.get('lagrange_init', 0.1))
        if initial_lagrange <= 0.0:
            raise ValueError('lagrange_init must be positive')
        self.log_lagrange = nn.Parameter(torch.tensor(
            [np.log(initial_lagrange)], dtype=torch.float32,
            device=self.device,
        ))
        self.lagrange_optim = (
            optim.Adam([self.log_lagrange], lr=self.lagrange_lr)
            if self.constraint_enabled else None
        )
        self.cmdp_cost_ema = None
        self.last_lagrange_loss = 0.0

        # 自动温度α
        self.log_alpha = nn.Parameter(torch.zeros(1, device=self.device))
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.alpha_lr)
        # 默认采用均匀经验回放。缓冲区实现保留可配置alpha，但论文主实验
        # 使用alpha=0，不再把PER作为算法创新或消融项。
        capacity = config.get('buffer_capacity', 100000)
        replay_alpha = float(config.get('replay_alpha', 0.0))
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity, state_dim, action_dim, alpha=replay_alpha
        )

        self.update_count = 0
        self.last_requested_action = None

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def lagrange_multiplier(self):
        if not self.constraint_enabled:
            return torch.zeros(1, device=self.device)
        return self.log_lagrange.exp()

    @property
    def constraint_cost_ema(self):
        """Deprecated compatibility alias for ``cmdp_cost_ema``."""
        return self.cmdp_cost_ema

    @constraint_cost_ema.setter
    def constraint_cost_ema(self, value):
        self.cmdp_cost_ema = value

    def update_lagrange(self, episode_discounted_cost):
        """Perform stochastic dual ascent from one completed CMDP episode."""
        if not self.constraint_enabled:
            return {
                'lagrange_multiplier': 0.0,
                'cmdp_cost_ema': float(episode_discounted_cost),
                'cmdp_cost_gap': 0.0,
                'lagrange_loss': 0.0,
            }
        observed = max(float(episode_discounted_cost), 0.0)
        if self.cmdp_cost_ema is None:
            self.cmdp_cost_ema = observed
        else:
            eta = self.lagrange_cost_ema_alpha
            self.cmdp_cost_ema = (
                (1.0 - eta) * self.cmdp_cost_ema + eta * observed
            )
        gap = self.cmdp_cost_ema - self.cmdp_cost_limit
        lagrange_loss = -(self.lagrange_multiplier * gap)
        self.lagrange_optim.zero_grad()
        lagrange_loss.backward()
        self.lagrange_optim.step()
        self.log_lagrange.data.clamp_(
            min=-10.0, max=float(np.log(self.lagrange_max))
        )
        self.last_lagrange_loss = float(lagrange_loss.item())
        return {
            'lagrange_multiplier': float(self.lagrange_multiplier.item()),
            'cmdp_cost_ema': float(self.cmdp_cost_ema),
            'cmdp_cost_gap': float(gap),
            'lagrange_loss': self.last_lagrange_loss,
        }

    def select_action(self, state, eval=False, energy_mode='normal',
                      lambda_max=1.0, speed_max=1.0, speed_min=0.0,
                      cpu_max=1.0, p_tx_max=1.0, **kwargs):
        """
        选择动作
        state: 状态向量
        energy_mode: normal/yellow/orange/red
        """
        # 预热阶段使用真正的均匀随机动作，而不是尚未训练的Actor分布。
        # 这与“纯随机探索填充经验池”的配置语义保持一致。
        if not eval and len(self.replay_buffer) < self.warmup_steps:
            action = np.random.uniform(
                -1.0, 1.0, size=self.action_dim
            ).astype(np.float32)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                if eval:
                    action = self.actor.get_deterministic_action(state)
                else:
                    action, _ = self.actor.sample(state)
            action = action.cpu().numpy().squeeze(0)

        self.last_requested_action = np.asarray(
            action, dtype=np.float32
        ).copy()
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def update(self):
        """单步策略更新"""
        if len(self.replay_buffer) < self.batch_size:
            return {}
        # #4: warmup (config.warmup_steps) 在算法内部统一生效,
        # 保证 train.py / train_baseline.py / ablation.py 三个入口行为一致
        if len(self.replay_buffer) < self.warmup_steps:
            return {}

        # 采样
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return {}

        (states, actions, rewards, costs, next_states, dones,
         indices, weights) = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        costs = torch.FloatTensor(costs).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # === 更新Critic ===
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            # 标准SAC直接使用重参数化采样动作。不能再叠加TD3目标策略
            # 平滑噪声，否则next_actions与next_log_probs不再对应同一动作。
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        # alpha=0时重要性权重恒为1；非零alpha仅作为可选实验配置。
        td1 = (current_q1 - target_q).abs().detach().cpu().numpy().flatten()
        td2 = (current_q2 - target_q).abs().detach().cpu().numpy().flatten()
        td_errors = (td1 + td2) / 2

        critic_loss = (weights * (current_q1 - target_q).pow(2)).mean() + \
                      (weights * (current_q2 - target_q).pow(2)).mean()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            1.0
        )
        self.critic_optim.step()

        # === Update the discounted constraint-cost critic ===
        if self.constraint_enabled:
            with torch.no_grad():
                target_cost_q1 = self.target_cost_critic1(
                    next_states, next_actions)
                target_cost_q2 = self.target_cost_critic2(
                    next_states, next_actions)
                target_cost_q = torch.clamp(
                    torch.max(target_cost_q1, target_cost_q2), min=0.0
                )
                target_cost_q = (
                    costs + (1 - dones) * self.gamma * target_cost_q
                )
            current_cost_q1 = self.cost_critic1(states, actions)
            current_cost_q2 = self.cost_critic2(states, actions)
            cost_critic_loss = (
                weights * (current_cost_q1 - target_cost_q).pow(2)
            ).mean() + (
                weights * (current_cost_q2 - target_cost_q).pow(2)
            ).mean()
            self.cost_critic_optim.zero_grad()
            cost_critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.cost_critic1.parameters())
                + list(self.cost_critic2.parameters()),
                1.0,
            )
            self.cost_critic_optim.step()
        else:
            cost_critic_loss = torch.tensor(0.0, device=self.device)

        # === 更新Actor (延迟更新: 每actor_update_delay步更新一次) ===
        new_actions, log_probs = self.actor.sample(states)
        if self.update_count % self.actor_update_delay == 0:
            q1 = self.critic1(states, new_actions)
            q2 = self.critic2(states, new_actions)
            q = torch.min(q1, q2)

            if self.constraint_enabled:
                policy_cost_q1 = self.cost_critic1(states, new_actions)
                policy_cost_q2 = self.cost_critic2(states, new_actions)
                policy_cost_q = torch.clamp(
                    torch.max(policy_cost_q1, policy_cost_q2), min=0.0
                )
                constraint_term = (
                    self.lagrange_multiplier.detach() * policy_cost_q
                )
            else:
                policy_cost_q = torch.zeros_like(q)
                constraint_term = torch.zeros_like(q)

            actor_loss = (
                self.alpha * log_probs - q + constraint_term
            ).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optim.step()
        else:
            actor_loss = torch.tensor(0.0, device=self.device)

        # === 更新自动温度α (每步更新, 保持探索压力) ===
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.log_alpha.data = self.log_alpha.data.clamp(-5, 5)

        # === 软更新目标网络 ===
        for target_param, param in zip(self.target_critic1.parameters(),
                                       self.critic1.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
        for target_param, param in zip(self.target_critic2.parameters(),
                                       self.critic2.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
        if self.constraint_enabled:
            for target_network, source_network in (
                (self.target_cost_critic1, self.cost_critic1),
                (self.target_cost_critic2, self.cost_critic2),
            ):
                for target_param, param in zip(
                        target_network.parameters(), source_network.parameters()):
                    target_param.data.copy_(
                        self.tau * param + (1 - self.tau) * target_param
                    )

        # alpha=0时优先级不会影响采样分布。
        self.replay_buffer.update_priorities(indices, td_errors)

        self.update_count += 1

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha.item(),
            'cost_critic_loss': cost_critic_loss.item(),
            'lagrange_multiplier': self.lagrange_multiplier.item(),
            'cmdp_cost_ema': (
                0.0 if self.cmdp_cost_ema is None
                else float(self.cmdp_cost_ema)
            ),
        }

    def pretrain_actor_with_bc(self, replay_buffer, steps=10000, batch_size=256):
        """Behavior Cloning上预训练Actor: 最大化buffer中动作的log概率"""
        print(f'Pre-training actor with BC for {steps} steps...')
        bc_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        for i in range(steps):
            batch = replay_buffer.sample(batch_size)
            if batch is None:
                break
            states, actions, *_ = batch
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            log_prob = self.actor.evaluate_log_prob(states, actions)
            loss = -log_prob.mean()  # 最大化log_prob
            bc_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            bc_optim.step()
            if (i + 1) % 2000 == 0:
                print(f'  BC step {i+1}/{steps}, loss={loss.item():.4f}')
        print('BC pre-training done.')

    def save(self, path):
        model_parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(model_parent, exist_ok=True)
        checkpoint = {
            'constraint_enabled': self.constraint_enabled,
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
            'actor_optim': self.actor_optim.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
            'alpha_optim': self.alpha_optim.state_dict(),
            'update_count': self.update_count,
        }
        if self.constraint_enabled:
            checkpoint.update({
                'cost_critic1': self.cost_critic1.state_dict(),
                'cost_critic2': self.cost_critic2.state_dict(),
                'target_cost_critic1': self.target_cost_critic1.state_dict(),
                'target_cost_critic2': self.target_cost_critic2.state_dict(),
                'cost_critic_optim': self.cost_critic_optim.state_dict(),
                'log_lagrange': self.log_lagrange.detach().cpu(),
                'lagrange_optim': self.lagrange_optim.state_dict(),
                'cmdp_cost_ema': self.cmdp_cost_ema,
            })
        torch.save(checkpoint, path)
        print(f'Model saved to {path}')

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        saved_constraint_enabled = bool(
            checkpoint.get('constraint_enabled', False)
        )
        if saved_constraint_enabled != self.constraint_enabled:
            raise ValueError(
                'checkpoint CMDP setting does not match the current agent; '
                'use a checkpoint trained with the same cmdp_enabled value'
            )
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2'])
        self.log_alpha.data.copy_(checkpoint['log_alpha'].to(self.device))
        self.actor_optim.load_state_dict(checkpoint['actor_optim'])
        self.critic_optim.load_state_dict(checkpoint['critic_optim'])
        if 'alpha_optim' in checkpoint:
            self.alpha_optim.load_state_dict(checkpoint['alpha_optim'])
        self.update_count = int(checkpoint.get('update_count', 0))
        if self.constraint_enabled:
            self.cost_critic1.load_state_dict(checkpoint['cost_critic1'])
            self.cost_critic2.load_state_dict(checkpoint['cost_critic2'])
            self.target_cost_critic1.load_state_dict(
                checkpoint['target_cost_critic1'])
            self.target_cost_critic2.load_state_dict(
                checkpoint['target_cost_critic2'])
            self.cost_critic_optim.load_state_dict(
                checkpoint['cost_critic_optim'])
            self.log_lagrange.data.copy_(
                checkpoint['log_lagrange'].to(self.device))
            self.lagrange_optim.load_state_dict(checkpoint['lagrange_optim'])
            self.cmdp_cost_ema = checkpoint.get(
                'cmdp_cost_ema', checkpoint.get('constraint_cost_ema')
            )
        print(f'Model loaded from {path}')
