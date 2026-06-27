"""Stage 5 — PPO (continuous actions). The bridge to robot control.

Same PPO machinery as ``ppo_discrete.py`` but with a **Gaussian policy**: the network outputs an
action mean, a separate learned (state-independent) log-std sets the spread, and actions are
**tanh-squashed** into the env's bounds — with the change-of-variables Jacobian correction
applied to the log-prob (the classic continuous-control trap; see
docs/00_foundations/05_continuous_control.md).

Pendulum-v1 only ever *truncates* (200-step timeout, never terminates), so the GAE bootstrap
mask is always 1 — every episode end is a truncation and must bootstrap.

Gate: >= -250 mean return on Pendulum-v1 across >= 3 seeds.

Usage:
    python -m quadruped_distill.algorithms.ppo.ppo_continuous --seeds 3
"""

from __future__ import annotations

import argparse
import statistics

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from quadruped_distill.algorithms.ppo.common import set_seed


def layer_init(layer: nn.Linear, std: float = np.sqrt(2)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class GaussianActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, act_scale: float):
        super().__init__()
        self.act_scale = act_scale
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )
        # state-independent learned log-std (docs/00_foundations/05_continuous_control.md)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x).squeeze(-1)

    def act(self, x: torch.Tensor, raw_action: torch.Tensor | None = None):
        """Returns (env_action, raw_action, logprob, entropy, value).

        ``raw_action`` is the pre-tanh Gaussian sample we store and re-evaluate; ``env_action``
        is what we send to the environment (tanh-squashed and scaled).
        """
        mean = self.mean(x)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        if raw_action is None:
            raw_action = dist.rsample()
        # log-prob with tanh change-of-variables correction, summed over action dims
        logprob = dist.log_prob(raw_action).sum(-1)
        logprob -= torch.log(1.0 - torch.tanh(raw_action).pow(2) + 1e-6).sum(-1)
        entropy = dist.entropy().sum(-1)
        env_action = torch.tanh(raw_action) * self.act_scale
        return env_action, raw_action, logprob, entropy, self.value(x)


def train(
    seed: int,
    total_steps: int,
    num_envs: int,
    num_steps: int,
    gamma: float,
    lam: float,
    clip: float,
    epochs: int,
    minibatches: int,
    ent_coef: float,
    vf_coef: float,
    max_grad: float,
    lr: float,
    device: str,
) -> float:
    set_seed(seed)
    envs = [gym.make("Pendulum-v1") for _ in range(num_envs)]
    obs_dim = envs[0].observation_space.shape[0]
    act_dim = envs[0].action_space.shape[0]
    act_scale = float(envs[0].action_space.high[0])  # Pendulum: +-2.0
    agent = GaussianActorCritic(obs_dim, act_dim, act_scale).to(device)
    opt = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    batch_size = num_envs * num_steps
    mb_size = batch_size // minibatches
    num_updates = total_steps // batch_size

    obs = torch.zeros((num_steps, num_envs, obs_dim), device=device)
    raw_actions = torch.zeros((num_steps, num_envs, act_dim), device=device)
    logprobs = torch.zeros((num_steps, num_envs), device=device)
    rewards = torch.zeros((num_steps, num_envs), device=device)
    values = torch.zeros((num_steps, num_envs), device=device)
    next_values = torch.zeros((num_steps, num_envs), device=device)
    terminated = torch.zeros((num_steps, num_envs), device=device)
    done_mask = torch.zeros((num_steps, num_envs), device=device)

    cur = np.stack([e.reset(seed=seed + i)[0] for i, e in enumerate(envs)])
    ep_ret = np.zeros(num_envs)
    ep_returns: list[float] = []

    for update in range(num_updates):
        opt.param_groups[0]["lr"] = lr * (1.0 - update / num_updates)

        for t in range(num_steps):
            cur_t = torch.as_tensor(cur, dtype=torch.float32, device=device)
            with torch.no_grad():
                env_a, raw_a, logp, _, value = agent.act(cur_t)
            obs[t], raw_actions[t], logprobs[t], values[t] = cur_t, raw_a, logp, value

            a_np = env_a.cpu().numpy()
            real_next = np.zeros_like(cur)
            for i, e in enumerate(envs):
                o, r, term, trunc, _ = e.step(a_np[i])
                rewards[t, i] = r
                terminated[t, i] = float(term)
                done_mask[t, i] = float(term or trunc)
                ep_ret[i] += r
                real_next[i] = o
                if term or trunc:
                    ep_returns.append(float(ep_ret[i]))
                    ep_ret[i] = 0.0
                    o, _ = e.reset()
                cur[i] = o
            with torch.no_grad():
                next_values[t] = agent.value(
                    torch.as_tensor(real_next, dtype=torch.float32, device=device)
                )

        advantages = torch.zeros_like(rewards)
        last_adv = torch.zeros(num_envs, device=device)
        for t in reversed(range(num_steps)):
            boot = 1.0 - terminated[t]
            chain = 1.0 - done_mask[t]
            delta = rewards[t] + gamma * next_values[t] * boot - values[t]
            last_adv = delta + gamma * lam * chain * last_adv
            advantages[t] = last_adv
        returns = advantages + values

        b_obs = obs.reshape(-1, obs_dim)
        b_raw = raw_actions.reshape(-1, act_dim)
        b_logp = logprobs.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)
        b_val = values.reshape(-1)

        idx = np.arange(batch_size)
        for _ in range(epochs):
            np.random.shuffle(idx)
            for start in range(0, batch_size, mb_size):
                mb = idx[start : start + mb_size]
                _, _, newlogp, entropy, newval = agent.act(b_obs[mb], b_raw[mb])
                ratio = (newlogp - b_logp[mb]).exp()
                adv = b_adv[mb]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                pg1 = -adv * ratio
                pg2 = -adv * torch.clamp(ratio, 1 - clip, 1 + clip)
                policy_loss = torch.max(pg1, pg2).mean()

                v_clip = b_val[mb] + torch.clamp(newval - b_val[mb], -clip, clip)
                v_loss = (
                    0.5 * torch.max((newval - b_ret[mb]) ** 2, (v_clip - b_ret[mb]) ** 2).mean()
                )
                loss = policy_loss - ent_coef * entropy.mean() + vf_coef * v_loss

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad)
                opt.step()

    for e in envs:
        e.close()
    return statistics.mean(ep_returns[-50:]) if ep_returns else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--total-steps", type=int, default=400_000)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--num-steps", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.9)  # short-horizon Pendulum
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--minibatches", type=int, default=32)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    finals = []
    for s in range(args.seeds):
        score = train(
            s,
            args.total_steps,
            args.num_envs,
            args.num_steps,
            args.gamma,
            args.lam,
            args.clip,
            args.epochs,
            args.minibatches,
            args.ent_coef,
            args.vf_coef,
            args.max_grad,
            args.lr,
            args.device,
        )
        finals.append(score)
        print(f"seed {s}: final avg return (last 50 eps) = {score:.1f}")
    mean = statistics.mean(finals)
    spread = statistics.pstdev(finals) if len(finals) > 1 else 0.0
    gate = "PASS" if mean >= -250 else "FAIL"
    print(
        f"\nPPO (continuous) | mean {mean:.1f} +/- {spread:.1f} across {args.seeds} seeds "
        f"| gate >=-250: {gate}"
    )


if __name__ == "__main__":
    main()
