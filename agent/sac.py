"""
SAC (Soft Actor-Critic) 算法实现
参考: Haarnoja et al. 2018, Soft Actor-Critic
"""
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy
from .networks import Actor, Critic
from .replay_buffer import PrioritizedReplayBuffer
from utils.seeding import derive_seed, isolated_torch_seed


class SAC:
    def __init__(self, state_dim, action_dim, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = int(action_dim)
        component_seed = config.get('component_seed', config.get('seed', 42))

        def component(name):
            return int(config.get(
                f'{name}_seed', derive_seed(component_seed, name)
            ))

        self.component_seeds = {
            name: component(name) for name in (
                'actor_init', 'reward_critic1_init',
                'reward_critic2_init', 'cost_critic1_init',
                'cost_critic2_init', 'policy_exploration',
                'policy_update', 'replay_sampling',
            )
        }
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
        self.gamma = float(config.get('gamma', 0.99))
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError('gamma must lie in [0, 1)')
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
        # Preserve the historical log-parameter Adam update by default so old
        # protocols and checkpoints remain reproducible.  New protocols may
        # explicitly select a budget-normalized projected dual-ascent step.
        self.lagrange_update_mode = str(config.get(
            'lagrange_update_mode', 'log_adam'
        ))
        self.lagrange_normalization_epsilon = float(config.get(
            'lagrange_normalization_epsilon', 1e-8
        ))
        self.cmdp_gradient_diagnostic_interval = int(config.get(
            'cmdp_gradient_diagnostic_interval', 0
        ))
        self.cmdp_actor_penalty_warmup_updates = int(config.get(
            'cmdp_actor_penalty_warmup_updates', 1000
        ))
        self.cmdp_actor_penalty_min_positive_samples = int(config.get(
            'cmdp_actor_penalty_min_positive_samples', self.batch_size
        ))
        self.cmdp_actor_penalty_ramp_updates = int(config.get(
            'cmdp_actor_penalty_ramp_updates', 0
        ))
        self.cmdp_positive_cost_epsilon = float(config.get(
            'cmdp_positive_cost_epsilon', 1e-8
        ))
        self.cmdp_stage_cost_max = float(config.get(
            'cmdp_stage_cost_max',
            max(
                1.0,
                float(config.get('cmdp_terminal_cost', 1.0)),
                float(config.get('cmdp_energy_cost_max', 1.0)),
            ),
        ))
        configured_q_horizon = config.get(
            'cmdp_q_horizon', config.get('max_steps')
        )
        self.cmdp_q_horizon = (
            None if configured_q_horizon is None
            else int(configured_q_horizon)
        )
        self.cmdp_cost_huber_delta = float(config.get(
            'cmdp_cost_huber_delta', 1.0
        ))
        self.cmdp_cost_grad_clip = float(config.get(
            'cmdp_cost_grad_clip', 1.0
        ))
        self.cmdp_finite_guard = bool(config.get(
            'cmdp_finite_guard', True
        ))
        if self.cmdp_stage_cost_max <= 0.0:
            raise ValueError('cmdp_stage_cost_max must be positive')
        if self.cmdp_q_horizon is not None and self.cmdp_q_horizon <= 0:
            raise ValueError('cmdp_q_horizon must be positive')
        if self.cmdp_cost_huber_delta <= 0.0:
            raise ValueError('cmdp_cost_huber_delta must be positive')
        if self.cmdp_cost_grad_clip <= 0.0:
            raise ValueError('cmdp_cost_grad_clip must be positive')
        if config.get('cmdp_q_max') is None:
            if self.cmdp_q_horizon is None:
                geometric_sum = 1.0 / (1.0 - self.gamma)
            else:
                geometric_sum = (
                    (1.0 - self.gamma ** self.cmdp_q_horizon)
                    / (1.0 - self.gamma)
                )
            self.cmdp_q_max = self.cmdp_stage_cost_max * geometric_sum
        else:
            self.cmdp_q_max = float(config['cmdp_q_max'])
        if self.cmdp_q_max <= 0.0 or not np.isfinite(self.cmdp_q_max):
            raise ValueError('cmdp_q_max must be finite and positive')
        if self.cmdp_cost_limit < 0.0:
            raise ValueError('cmdp_cost_limit must be nonnegative')
        if self.lagrange_lr <= 0.0 or self.lagrange_max < np.exp(-10.0):
            raise ValueError('Lagrange learning rate and maximum must be positive')
        if not 0.0 < self.lagrange_cost_ema_alpha <= 1.0:
            raise ValueError('lagrange_cost_ema_alpha must lie in (0, 1]')
        if self.lagrange_update_mode not in ('log_adam', 'projected_normalized'):
            raise ValueError(
                'lagrange_update_mode must be log_adam or projected_normalized'
            )
        if self.lagrange_normalization_epsilon <= 0.0:
            raise ValueError('lagrange_normalization_epsilon must be positive')
        if self.cmdp_gradient_diagnostic_interval < 0:
            raise ValueError(
                'cmdp_gradient_diagnostic_interval must be nonnegative'
            )
        if self.cmdp_actor_penalty_warmup_updates < 0:
            raise ValueError(
                'cmdp_actor_penalty_warmup_updates must be nonnegative'
            )
        if self.cmdp_actor_penalty_min_positive_samples <= 0:
            raise ValueError(
                'cmdp_actor_penalty_min_positive_samples must be positive'
            )
        if self.cmdp_actor_penalty_ramp_updates < 0:
            raise ValueError(
                'cmdp_actor_penalty_ramp_updates must be nonnegative'
            )
        if self.cmdp_positive_cost_epsilon < 0.0:
            raise ValueError('cmdp_positive_cost_epsilon must be nonnegative')

        # 网络
        with isolated_torch_seed(self.component_seeds['actor_init']):
            self.actor = Actor(state_dim, action_dim).to(self.device)
        with isolated_torch_seed(
                self.component_seeds['reward_critic1_init']):
            self.critic1 = Critic(state_dim, action_dim).to(self.device)
        with isolated_torch_seed(
                self.component_seeds['reward_critic2_init']):
            self.critic2 = Critic(state_dim, action_dim).to(self.device)
        self.target_critic1 = deepcopy(self.critic1)
        self.target_critic2 = deepcopy(self.critic2)
        if self.constraint_enabled:
            # Twin cost critics use the conservative maximum.  This mirrors
            # clipped double-Q for reward while avoiding optimistic
            # under-estimation of unsafe discounted cost.
            with isolated_torch_seed(
                    self.component_seeds['cost_critic1_init']):
                self.cost_critic1 = Critic(
                    state_dim, action_dim
                ).to(self.device)
            with isolated_torch_seed(
                    self.component_seeds['cost_critic2_init']):
                self.cost_critic2 = Critic(
                    state_dim, action_dim
                ).to(self.device)
            if bool(config.get('cmdp_zero_initialize_cost_output', True)):
                for critic in (self.cost_critic1, self.cost_critic2):
                    nn.init.zeros_(critic.net[-1].weight)
                    nn.init.zeros_(critic.net[-1].bias)
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
            capacity, state_dim, action_dim, alpha=replay_alpha,
            seed=self.component_seeds['replay_sampling'],
            positive_cost_epsilon=self.cmdp_positive_cost_epsilon,
        )

        self.exploration_rng = np.random.default_rng(
            self.component_seeds['policy_exploration']
        )
        generator_device = (
            'cuda' if self.device.type == 'cuda' else 'cpu'
        )
        self.policy_exploration_generator = torch.Generator(
            device=generator_device
        )
        self.policy_exploration_generator.manual_seed(
            self.component_seeds['policy_exploration']
        )
        self.policy_update_generator = torch.Generator(
            device=generator_device
        )
        self.policy_update_generator.manual_seed(
            self.component_seeds['policy_update']
        )

        self.update_count = 0
        self.cost_critic_update_count = 0
        self.cmdp_actor_penalty_activation_update = None
        self.last_requested_action = None

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def lagrange_multiplier(self):
        if not self.constraint_enabled:
            return torch.zeros(1, device=self.device)
        return self.log_lagrange.exp()

    def _require_finite(self, tensor, name):
        """Fail fast before a non-finite CMDP value contaminates training."""
        if self.cmdp_finite_guard and not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError(f'non-finite CMDP tensor: {name}')
        return tensor

    def _bounded_cost_q(self, tensor, name):
        """Clamp cost-to-go to its finite-horizon theoretical range."""
        self._require_finite(tensor, name)
        return torch.clamp(tensor, min=0.0, max=self.cmdp_q_max)

    @property
    def constraint_cost_ema(self):
        """Deprecated compatibility alias for ``cmdp_cost_ema``."""
        return self.cmdp_cost_ema

    @constraint_cost_ema.setter
    def constraint_cost_ema(self, value):
        self.cmdp_cost_ema = value

    @property
    def cmdp_actor_penalty_active(self):
        """Whether the cost critic has enough real signal to shape the actor."""
        return bool(
            self.constraint_enabled
            and self.cost_critic_update_count
            >= self.cmdp_actor_penalty_warmup_updates
            and self.replay_buffer.positive_cost_count
            >= self.cmdp_actor_penalty_min_positive_samples
        )

    @property
    def cmdp_actor_penalty_scale(self):
        """Return the gradual actor-cost weight after the readiness gates."""
        if not self.cmdp_actor_penalty_active:
            return 0.0
        if self.cmdp_actor_penalty_ramp_updates == 0:
            return 1.0
        activation = self.cmdp_actor_penalty_activation_update
        if activation is None:
            return 0.0
        elapsed = max(
            self.cost_critic_update_count - int(activation) + 1, 1
        )
        return float(min(
            1.0, elapsed / self.cmdp_actor_penalty_ramp_updates
        ))

    def update_lagrange(self, episode_discounted_cost):
        """Perform stochastic dual ascent from one completed CMDP episode."""
        if not self.constraint_enabled:
            return {
                'lagrange_multiplier': 0.0,
                'cmdp_cost_ema': float(episode_discounted_cost),
                'cmdp_cost_gap': 0.0,
                'lagrange_normalized_gap': 0.0,
                'lagrange_update_delta': 0.0,
                'lagrange_projected_update': 0.0,
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
        normalization = max(
            abs(self.cmdp_cost_limit), self.lagrange_normalization_epsilon
        )
        normalized_gap = gap / normalization
        old_lagrange = float(self.lagrange_multiplier.detach().item())
        lagrange_loss = -(self.lagrange_multiplier * gap)
        if self.lagrange_update_mode == 'projected_normalized':
            new_lagrange = float(np.clip(
                old_lagrange + self.lagrange_lr * normalized_gap,
                np.exp(-10.0),
                self.lagrange_max,
            ))
            with torch.no_grad():
                self.log_lagrange.copy_(torch.tensor(
                    [np.log(new_lagrange)],
                    dtype=self.log_lagrange.dtype,
                    device=self.device,
                ))
        else:
            self.lagrange_optim.zero_grad()
            lagrange_loss.backward()
            self.lagrange_optim.step()
            self.log_lagrange.data.clamp_(
                min=-10.0, max=float(np.log(self.lagrange_max))
            )
        new_lagrange = float(self.lagrange_multiplier.detach().item())
        self.last_lagrange_loss = float(lagrange_loss.item())
        return {
            'lagrange_multiplier': new_lagrange,
            'cmdp_cost_ema': float(self.cmdp_cost_ema),
            'cmdp_cost_gap': float(gap),
            'lagrange_normalized_gap': float(normalized_gap),
            'lagrange_update_delta': float(new_lagrange - old_lagrange),
            'lagrange_projected_update': float(
                self.lagrange_update_mode == 'projected_normalized'
            ),
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
            action = self.exploration_rng.uniform(
                -1.0, 1.0, size=self.action_dim
            ).astype(np.float32)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                if eval:
                    action = self.actor.get_deterministic_action(state)
                else:
                    action, _ = self.actor.sample(
                        state, generator=self.policy_exploration_generator
                    )
            action = action.cpu().numpy().squeeze(0)

        self.last_requested_action = np.asarray(
            action, dtype=np.float32
        ).copy()
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def estimate_cmdp_risk(self, state, action):
        """Estimate discounted CMDP risk for one exact state-action pair.

        The environment owns action projection.  Calling this method for the
        requested action and the action actually executed therefore gives a
        paired, same-state pre/post-shield diagnostic.  No-CMDP agents return
        NaN because they intentionally have no learned cost critic.
        """
        if not self.constraint_enabled:
            return float('nan')
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        action_tensor = torch.as_tensor(
            action, dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        with torch.no_grad():
            risk = self._bounded_cost_q(torch.max(
                self.cost_critic1(state_tensor, action_tensor),
                self.cost_critic2(state_tensor, action_tensor),
            ), 'estimated_cmdp_risk')
        return float(risk.item())

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
        self._require_finite(costs, 'replay_stage_cost')
        costs = torch.clamp(
            costs, min=0.0, max=self.cmdp_stage_cost_max
        )
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # === 更新Critic ===
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(
                next_states, generator=self.policy_update_generator
            )
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

        target_cost_q_max = 0.0
        current_cost_q_abs_max = 0.0
        # === Update the discounted constraint-cost critic ===
        if self.constraint_enabled:
            with torch.no_grad():
                target_cost_q1 = self.target_cost_critic1(
                    next_states, next_actions)
                target_cost_q2 = self.target_cost_critic2(
                    next_states, next_actions)
                target_cost_q = self._bounded_cost_q(
                    torch.max(target_cost_q1, target_cost_q2),
                    'bootstrapped_target_cost_q',
                )
                target_cost_q = self._bounded_cost_q(
                    costs + (1 - dones) * self.gamma * target_cost_q,
                    'bellman_target_cost_q',
                )
            current_cost_q1 = self.cost_critic1(states, actions)
            current_cost_q2 = self.cost_critic2(states, actions)
            self._require_finite(current_cost_q1, 'current_cost_q1')
            self._require_finite(current_cost_q2, 'current_cost_q2')
            cost_critic_loss = (
                weights * F.smooth_l1_loss(
                    current_cost_q1, target_cost_q,
                    beta=self.cmdp_cost_huber_delta,
                    reduction='none',
                )
            ).mean() + (
                weights * F.smooth_l1_loss(
                    current_cost_q2, target_cost_q,
                    beta=self.cmdp_cost_huber_delta,
                    reduction='none',
                )
            ).mean()
            self._require_finite(cost_critic_loss, 'cost_critic_loss')
            self.cost_critic_optim.zero_grad()
            cost_critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.cost_critic1.parameters())
                + list(self.cost_critic2.parameters()),
                self.cmdp_cost_grad_clip,
            )
            self.cost_critic_optim.step()
            self.cost_critic_update_count += 1
            target_cost_q_max = float(target_cost_q.max().item())
            current_cost_q_abs_max = float(torch.max(
                current_cost_q1.detach().abs().max(),
                current_cost_q2.detach().abs().max(),
            ).item())
        else:
            cost_critic_loss = torch.tensor(0.0, device=self.device)

        # === 更新Actor (延迟更新: 每actor_update_delay步更新一次) ===
        new_actions, log_probs = self.actor.sample(
            states, generator=self.policy_update_generator
        )
        policy_cost_q = torch.zeros_like(rewards)
        replay_cost_q = torch.zeros_like(rewards)
        constraint_term = torch.zeros_like(rewards)
        actor_reward_gradient_norm = math.nan
        actor_cost_gradient_norm = math.nan
        actor_cost_to_reward_gradient_ratio = math.nan
        penalty_active = self.cmdp_actor_penalty_active
        if (
            penalty_active
            and self.cmdp_actor_penalty_activation_update is None
        ):
            self.cmdp_actor_penalty_activation_update = (
                self.cost_critic_update_count
            )
        penalty_scale = self.cmdp_actor_penalty_scale
        if self.update_count % self.actor_update_delay == 0:
            q1 = self.critic1(states, new_actions)
            q2 = self.critic2(states, new_actions)
            q = torch.min(q1, q2)

            if self.constraint_enabled:
                policy_cost_q1 = self.cost_critic1(states, new_actions)
                policy_cost_q2 = self.cost_critic2(states, new_actions)
                policy_cost_q = self._bounded_cost_q(
                    torch.max(policy_cost_q1, policy_cost_q2),
                    'policy_cost_q',
                )
                replay_cost_q = self._bounded_cost_q(
                    torch.max(
                        self.cost_critic1(states, actions),
                        self.cost_critic2(states, actions),
                    ),
                    'replay_cost_q',
                )
                if penalty_active:
                    constraint_term = (
                        penalty_scale
                        * self.lagrange_multiplier.detach()
                        * policy_cost_q
                    )

            actor_loss = (
                self.alpha * log_probs - q + constraint_term
            ).mean()

            if (
                penalty_active
                and self.cmdp_gradient_diagnostic_interval > 0
                and self.update_count
                % self.cmdp_gradient_diagnostic_interval == 0
            ):
                actor_parameters = tuple(self.actor.parameters())
                reward_objective = (self.alpha * log_probs - q).mean()
                cost_objective = constraint_term.mean()
                reward_gradients = torch.autograd.grad(
                    reward_objective,
                    actor_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                cost_gradients = torch.autograd.grad(
                    cost_objective,
                    actor_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )

                def gradient_norm(gradients):
                    squared = sum(
                        float(gradient.detach().pow(2).sum().item())
                        for gradient in gradients if gradient is not None
                    )
                    return float(np.sqrt(squared))

                actor_reward_gradient_norm = gradient_norm(reward_gradients)
                actor_cost_gradient_norm = gradient_norm(cost_gradients)
                actor_cost_to_reward_gradient_ratio = (
                    actor_cost_gradient_norm
                    / max(actor_reward_gradient_norm, 1e-12)
                )

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
            'cmdp_stage_cost_max': self.cmdp_stage_cost_max,
            'cmdp_q_max': self.cmdp_q_max,
            'target_cost_q_max': target_cost_q_max,
            'current_cost_q_abs_max': current_cost_q_abs_max,
            'cost_critic_update_count': self.cost_critic_update_count,
            'cmdp_positive_cost_samples': (
                self.replay_buffer.positive_cost_count
            ),
            'cmdp_actor_penalty_active': float(penalty_active),
            'cmdp_actor_penalty_scale': float(penalty_scale),
            'policy_cost_q_mean': float(policy_cost_q.detach().mean().item()),
            'policy_cost_q_std': float(
                policy_cost_q.detach().std(unbiased=False).item()
            ),
            'replay_cost_q_mean': float(replay_cost_q.detach().mean().item()),
            # Batch-level learning diagnostics.  These compare a newly sampled
            # policy proposal with replayed applied actions, but are not the
            # exact paired action from one environment step.  Exact same-step
            # raw/applied risk is logged by annotate_cmdp_projection_risk().
            'cmdp_pre_projection_q_mean': float(
                policy_cost_q.detach().mean().item()
            ),
            'cmdp_post_projection_q_mean': float(
                replay_cost_q.detach().mean().item()
            ),
            'constraint_term_mean': float(
                constraint_term.detach().mean().item()
            ),
            'actor_reward_gradient_norm': actor_reward_gradient_norm,
            'actor_cost_gradient_norm': actor_cost_gradient_norm,
            'actor_cost_to_reward_gradient_ratio': (
                actor_cost_to_reward_gradient_ratio
            ),
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
            'cost_critic_update_count': self.cost_critic_update_count,
            'cmdp_actor_penalty_activation_update': (
                self.cmdp_actor_penalty_activation_update
            ),
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
        self.cost_critic_update_count = int(checkpoint.get(
            'cost_critic_update_count', self.update_count
        ))
        self.cmdp_actor_penalty_activation_update = checkpoint.get(
            'cmdp_actor_penalty_activation_update'
        )
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
