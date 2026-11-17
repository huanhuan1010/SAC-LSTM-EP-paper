"""任务生成模型: 核心/常规任务"""
import numpy as np


class TaskGenerator:
    def __init__(self, config):
        # 任务参数
        self.num_users = config.get('num_users', 5)
        self.area_size = config.get('area_size', 500.0)     # 500m × 500m
        self.task_interval = config.get('task_interval', 5)  # 每?步生成一次任务
        self.D_mean = config.get('D_mean', 1e6)              # 平均数据量 1Mb
        self.D_std = config.get('D_std', 0.5e6)
        self.C_mean = config.get('C_mean', 1e9)              # 平均CPU周期 1Gcycles
        self.C_std = config.get('C_std', 0.3e9)
        self.tau_mean = config.get('tau_mean', 2.0)          # 平均最大容忍时延 2s
        self.tau_std = config.get('tau_std', 0.5)            # 容忍时延标准差
        self.core_task_ratio = config.get('core_task_ratio', 0.3)  # 核心任务占比

        self.users = []  # 用户列表
        self.rng = np.random.default_rng()

    def reset(self, rng=None):
        """随机生成用户位置"""
        if rng is not None:
            self.rng = rng
        self.users = []
        for i in range(self.num_users):
            user = {
                'id': i,
                'pos': np.array([
                    self.rng.uniform(0, self.area_size),
                    self.rng.uniform(0, self.area_size),
                    0  # 地面
                ]),
                'queue': [],
            }
            self.users.append(user)
        return self.users

    def generate_task(self, t):
        """在时隙t生成任务"""
        tasks = []
        for user in self.users:
            if t % self.task_interval == 0:
                D = abs(self.rng.normal(self.D_mean, self.D_std))
                C = abs(self.rng.normal(self.C_mean, self.C_std))
                tau = abs(self.rng.normal(self.tau_mean, self.tau_std))
                is_core = self.rng.random() < self.core_task_ratio
                task = {
                    'user_id': user['id'],
                    'D': D,                    # 数据量 (bits)
                    'C': C,                    # CPU周期数 (cycles)
                    'tau': max(0.5, tau),      # 容忍时延 (s)
                    'is_core': is_core,        # 是否核心任务
                    'arrival_time': t,
                    # 卸载比例由任务第一次获得服务时确定，随后保持不变。
                    # 剩余工作量字段由 UAVEnv 在首次调度时初始化。
                    'decision_made': False,
                }
                user['queue'].append(task)
                tasks.append(task)
        return tasks
