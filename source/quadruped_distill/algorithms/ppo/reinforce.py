"""Stage 1 — REINFORCE: Monte-Carlo policy gradient, no baseline.

The plainest possible policy gradient (docs/00_foundations/02_policy_gradients.md):
collect a full episode, weight each action's log-prob by the episode's reward-to-go, ascend.
There is *no* variance reduction here — that is the point. Run several seeds and watch the
final-return spread; ``reinforce_baseline.py`` then shows the variance collapse a baseline buys.

Usage:
    python -m quadruped_distill.algorithms.ppo.reinforce --seeds 3
"""

from __future__ import annotations

import argparse
import statistics

import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical

from quadruped_distill.algorithms.ppo.common import mlp, set_seed


def reward_to_go(rewards: list[float], gamma: float) -> torch.Tensor:
    """Discounted return from each step onward: G_t = sum_{k>=t} gamma^{k-t} r_k."""
    out = [0.0] * len(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        out[t] = running
    return torch.tensor(out, dtype=torch.float32)


def train(seed: int, total_steps: int, gamma: float, lr: float, device: str) -> float:
    env = gym.make("CartPole-v1")
    set_seed(seed)
    obs_dim = env.observation_space.shape[0]
    n_act = env.action_space.n
    policy = mlp([obs_dim, 128, n_act], activation=nn.Tanh).to(device)  # outputs logits
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    steps = 0
    recent: list[float] = []
    while steps < total_steps:
        obs, _ = env.reset(seed=seed + steps)
        logps, rewards = [], []
        done = False
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            dist = Categorical(logits=policy(obs_t))
            action = dist.sample()
            logps.append(dist.log_prob(action))
            obs, r, term, trunc, _ = env.step(action.item())
            rewards.append(float(r))
            done = term or trunc
        steps += len(rewards)

        returns = reward_to_go(rewards, gamma).to(device)
        # No baseline: raw returns weight the gradient -> high variance.
        loss = -(torch.stack(logps) * returns).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

        recent.append(sum(rewards))
        recent = recent[-50:]
    env.close()
    return statistics.mean(recent)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--total-steps", type=int, default=150_000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    finals = []
    for s in range(args.seeds):
        score = train(s, args.total_steps, args.gamma, args.lr, args.device)
        finals.append(score)
        print(f"seed {s}: final avg return (last 50 eps) = {score:.1f}")
    spread = statistics.pstdev(finals) if len(finals) > 1 else 0.0
    print(
        f"\nREINFORCE | mean {statistics.mean(finals):.1f} +/- {spread:.1f} across {args.seeds} seeds"
    )
    print("(High variance is expected — compare with reinforce_baseline.py.)")


if __name__ == "__main__":
    main()
