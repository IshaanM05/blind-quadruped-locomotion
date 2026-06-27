"""Stage 3 — A2C: bootstrapped (TD) advantages instead of full Monte-Carlo returns.

Stages 1-2 used the *whole* episode return (Monte-Carlo: unbiased, high variance). A2C instead
bootstraps the critic — it uses GAE over short rollouts, traversing the MC<->TD bias/variance
dial that ``lam`` controls (docs/00_foundations/03_variance_and_gae.md). This is the conceptual
bridge to PPO: same actor-critic + GAE machinery, but a single gradient step per rollout (no
clipping, no data reuse yet).

Usage:
    python -m quadruped_distill.algorithms.ppo.a2c --seeds 3
"""

from __future__ import annotations

import argparse
import statistics

import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical

from quadruped_distill.algorithms.ppo.common import compute_gae, mlp, set_seed


def train(
    seed: int,
    total_steps: int,
    rollout: int,
    gamma: float,
    lam: float,
    lr: float,
    ent_coef: float,
    device: str,
) -> float:
    env = gym.make("CartPole-v1")
    set_seed(seed)
    obs_dim = env.observation_space.shape[0]
    n_act = env.action_space.n
    actor = mlp([obs_dim, 128, n_act], activation=nn.Tanh).to(device)
    critic = mlp([obs_dim, 128, 1], activation=nn.Tanh).to(device)
    opt = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=lr)

    obs, _ = env.reset(seed=seed)
    ep_ret, recent = 0.0, []
    steps = 0
    while steps < total_steps:
        obs_b, act_b, logp_b, rew_b, val_b, done_b = [], [], [], [], [], []
        for _ in range(rollout):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            dist = Categorical(logits=actor(obs_t))
            action = dist.sample()
            obs_b.append(obs_t)
            act_b.append(action)
            logp_b.append(dist.log_prob(action))
            val_b.append(critic(obs_t).squeeze(-1))
            obs, r, term, trunc, _ = env.step(action.item())
            rew_b.append(float(r))
            done_b.append(1.0 if term else 0.0)  # timeout (trunc) is NOT a terminal -> bootstrap
            ep_ret += r
            steps += 1
            if term or trunc:
                recent.append(ep_ret)
                recent = recent[-50:]
                ep_ret = 0.0
                obs, _ = env.reset()

        with torch.no_grad():
            last_v = critic(torch.as_tensor(obs, dtype=torch.float32, device=device)).squeeze(-1)
        rewards = torch.tensor(rew_b, device=device)
        values = torch.stack(val_b)
        dones = torch.tensor(done_b, device=device)
        adv, returns = compute_gae(rewards, values.detach(), dones, last_v, gamma, lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        logps = torch.stack(logp_b)
        # entropy of a fresh forward pass over the batch keeps exploration alive
        dist = Categorical(logits=actor(torch.stack(obs_b)))
        entropy = dist.entropy().mean()
        policy_loss = -(logps * adv).mean()
        value_loss = nn.functional.mse_loss(values, returns)
        loss = policy_loss + 0.5 * value_loss - ent_coef * entropy
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_([*actor.parameters(), *critic.parameters()], 0.5)
        opt.step()

    env.close()
    return statistics.mean(recent) if recent else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--total-steps", type=int, default=150_000)
    p.add_argument("--rollout", type=int, default=32)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    finals = []
    for s in range(args.seeds):
        score = train(
            s,
            args.total_steps,
            args.rollout,
            args.gamma,
            args.lam,
            args.lr,
            args.ent_coef,
            args.device,
        )
        finals.append(score)
        print(f"seed {s}: final avg return (last 50 eps) = {score:.1f}")
    spread = statistics.pstdev(finals) if len(finals) > 1 else 0.0
    print(f"\nA2C | mean {statistics.mean(finals):.1f} +/- {spread:.1f} across {args.seeds} seeds")


if __name__ == "__main__":
    main()
