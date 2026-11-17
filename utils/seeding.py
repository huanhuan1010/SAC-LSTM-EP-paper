"""实验级随机种子管理。"""
from contextlib import contextmanager
import hashlib
import random

import numpy as np
import torch


def normalize_seed(seed):
    """把种子规范为NumPy/Gymnasium接受的32位非负整数。"""
    if seed is None:
        return None
    return int(seed) % (2 ** 32)


def seed_everything(seed):
    """统一设置Python、NumPy和PyTorch的全局随机种子。"""
    normalized = normalize_seed(seed)
    if normalized is None:
        return None
    random.seed(normalized)
    np.random.seed(normalized)
    torch.manual_seed(normalized)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(normalized)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return normalized


def derive_seed(base_seed, namespace):
    """Derive a stable component seed without consuming a shared RNG stream."""
    normalized = normalize_seed(base_seed)
    if normalized is None:
        return None
    payload = f'{normalized}:{str(namespace)}'.encode('utf-8')
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], 'big') % (2 ** 32)


@contextmanager
def isolated_torch_seed(seed):
    """Temporarily seed PyTorch while preserving the caller's RNG states.

    Network construction consumes random numbers.  Full CMDP creates extra
    critics while No-CMDP does not; without isolation that structural change
    silently shifts actor, LSTM, and exploration randomness even when both
    algorithms declare the same experiment seed.
    """
    normalized = normalize_seed(seed)
    if normalized is None:
        yield
        return
    cpu_state = torch.random.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    try:
        torch.manual_seed(normalized)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(normalized)
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def component_seed_map(base_seed):
    """Return independent streams shared by matched Full/No-CMDP runs."""
    namespaces = (
        'actor_init', 'reward_critic1_init', 'reward_critic2_init',
        'cost_critic1_init', 'cost_critic2_init', 'policy_exploration',
        'policy_update', 'replay_sampling', 'lstm_init', 'lstm_update',
    )
    return {
        f'{name}_seed': derive_seed(base_seed, name)
        for name in namespaces
    }


def episode_seed(base_seed, episode_index):
    """为每个回合生成稳定且互不相同的环境种子。"""
    normalized = normalize_seed(base_seed)
    if normalized is None:
        return None
    return (normalized + int(episode_index)) % (2 ** 32)
