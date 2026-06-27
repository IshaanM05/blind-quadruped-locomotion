"""Stage 2 — REINFORCE + a learned value baseline.

Identical to ``reinforce.py`` except a critic V(s) is trained to predict the return, and the
policy gradient is weighted by ``return - V(s)`` instead of the raw return. The baseline is
action-independent, so it does NOT bias the gradient (proof in
docs/00_foundations/02_policy_gradients.md) but it sharply cuts variance: run the same seeds as
Stage 1 and compare the final-return spread.

Usage:
    python -m quadruped_distill.algorithms.ppo.reinforce_baseline --seeds 3
"""

from __future__ import annotations

import argparse
import statistics

import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical

from quadruped_distill.algorithms.ppo.common import mlp, set_seed
from quadruped_distill.algorithms.ppo.reinforce import reward_to_go


def train(seed: int, total_steps: int, gamma: float, lr: float, device: str) -> float:
    env = gym.make("CartPole-v1")
    set_seed(seed)
    obs_dim = env.observation_space.shape[0]
    n_act = env.action_space.n
    policy = mlp([obs_dim, 128, n_act], activation=nn.Tanh).to(device)
    value = mlp([obs_dim, 128, 1], activation=nn.Tanh).to(device)
    opt = torch.optim.Adam([*policy.parameters(), *value.parameters()], lr=lr)

    steps = 0
    recent: list[float] = []
    while steps < total_steps:
        obs, _ = env.reset(seed=seed + steps)
        logps, rewards, obs_buf = [], [], []
        done = False
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            obs_buf.append(obs_t)
            dist = Categorical(logits=policy(obs_t))
            action = dist.sample()
            logps.append(dist.log_prob(action))
            obs, r, term, trunc, _ = env.step(action.item())
            rewards.append(float(r))
            done = term or trunc
        steps += len(rewards)

        returns = reward_to_go(rewards, gamma).to(device)
        values = value(torch.stack(obs_buf)).squeeze(-1)
        advantages = returns - values.detach()  # detach: critic error must not flow via the actor
        policy_loss = -(torch.stack(logps) * advantages).sum()
        value_loss = nn.functional.mse_loss(values, returns)
        loss = policy_loss + 0.5 * value_loss
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
        f"\nREINFORCE+baseline | mean {statistics.mean(finals):.1f} +/- {spread:.1f} "
        f"across {args.seeds} seeds"
    )
    print("(Compare the +/- spread against plain reinforce.py — that drop is the baseline.)")


if __name__ == "__main__":
    main()
