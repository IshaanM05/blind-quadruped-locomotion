"""Shared building blocks for the from-scratch PPO progression.

Kept deliberately small and dependency-light (torch + numpy) so it runs on CPU in CI
without Isaac Sim. The GAE implementation here is the backbone of every later algorithm;
its derivation lives in ``docs/00_foundations/03_variance_and_gae.md``.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch for reproducible single-runs (see docs §10.2)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mlp(
    sizes: list[int],
    activation: type[nn.Module] = nn.Tanh,
    output_activation: type[nn.Module] | None = None,
) -> nn.Sequential:
    """Build a simple MLP. ``sizes`` includes input and output dims."""
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        is_last = i == len(sizes) - 2
        act = output_activation if is_last else activation
        if act is not None:
            layers.append(act())
    return nn.Sequential(*layers)


def compute_gae(
    rewards: torch.Tensor,  # (T,) per-step rewards
    values: torch.Tensor,  # (T,) V(s_t) from the critic
    dones: torch.Tensor,  # (T,) 1.0 if s_{t+1} is a TRUE terminal (not a timeout)
    last_value: torch.Tensor,  # scalar bootstrap value V(s_T)
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation, single backward pass.

    Returns ``(advantages, returns)`` where ``returns = advantages + values`` are the
    critic's regression targets. ``dones`` must encode *terminations only*: timeouts are
    truncations and should bootstrap (see docs/00_foundations/03_variance_and_gae.md).
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros((), dtype=rewards.dtype)
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        last_adv = delta + gamma * lam * non_terminal * last_adv
        advantages[t] = last_adv
    returns = advantages + values
    return advantages, returns
