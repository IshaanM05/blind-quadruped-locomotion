"""GAE checked against hand-computed trajectories.

If this fails, no training curve in the repo is trustworthy — GAE is the backbone of every
algorithm. Derivation: docs/00_foundations/03_variance_and_gae.md.
"""

import torch
from quadruped_distill.algorithms.ppo.common import compute_gae


def test_gae_reduces_to_reward_to_go_when_gamma_lam_one():
    # gamma=lam=1, zero baseline -> advantage = reward-to-go.
    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.zeros(3)
    dones = torch.zeros(3)
    adv, ret = compute_gae(rewards, values, dones, last_value=torch.tensor(0.0), gamma=1.0, lam=1.0)
    assert torch.allclose(adv, torch.tensor([3.0, 2.0, 1.0]))
    assert torch.allclose(ret, torch.tensor([3.0, 2.0, 1.0]))


def test_gae_with_termination_masking():
    # gamma=0.99, lam=0.95, a true terminal after step 1 -> no bootstrap across it.
    rewards = torch.tensor([1.0, 2.0, 3.0])
    values = torch.tensor([0.5, 0.5, 0.5])
    dones = torch.tensor([0.0, 1.0, 0.0])  # s_2 is terminal
    adv, ret = compute_gae(
        rewards, values, dones, last_value=torch.tensor(0.5), gamma=0.99, lam=0.95
    )
    expected_adv = torch.tensor([2.40575, 1.5, 2.995])
    assert torch.allclose(adv, expected_adv, atol=1e-5)
    assert torch.allclose(ret, expected_adv + values, atol=1e-5)
