"""Stage 4 — PPO (discrete actions). The gate of Phase 0's first half.

Full PPO over parallel envs: rollout buffer, GAE, the clipped surrogate, several epochs of
minibatch updates, an entropy bonus, advantage normalization, orthogonal init, value-loss
clipping, gradient clipping, and LR annealing. Read alongside docs/00_foundations/04_ppo.md.

Env handling: we run a list of *single* CartPole envs and reset them manually. Single envs do
not autoreset, so ``step`` returns the true terminal/truncation observation — letting us bootstrap
timeouts honestly (docs/00_foundations/03_variance_and_gae.md). The GAE bootstrap mask is
``1 - terminated`` (timeouts still bootstrap); the advantage *chain* is cut on
``terminated OR truncated``.

Gate: >= 475 mean return on CartPole-v1 across >= 3 seeds. Reaches ~470 by 100k steps and a
stable, perfect 500 by ~150k (the default) — the last-50-episode metric is noisy until the
policy fully saturates, hence the slightly-longer-than-100k default.

Usage:
    python -m quadruped_distill.algorithms.ppo.ppo_discrete --seeds 3
"""

from __future__ import annotations

import argparse
import statistics

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from quadruped_distill.algorithms.ppo.common import set_seed


def layer_init(layer: nn.Linear, std: float = np.sqrt(2)) -> nn.Linear:
    """Orthogonal weight init — one of the '37 details' (docs/00_foundations/04_ppo.md)."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_act: int):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, n_act), std=0.01),  # small gain -> near-uniform start
        )

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x).squeeze(-1)

    def act(self, x: torch.Tensor, action: torch.Tensor | None = None):
        dist = Categorical(logits=self.actor(x))
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.value(x)


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
    envs = [gym.make("CartPole-v1") for _ in range(num_envs)]
    obs_dim = envs[0].observation_space.shape[0]
    n_act = envs[0].action_space.n
    agent = ActorCritic(obs_dim, n_act).to(device)
    opt = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    batch_size = num_envs * num_steps
    mb_size = batch_size // minibatches
    num_updates = total_steps // batch_size

    obs = torch.zeros((num_steps, num_envs, obs_dim), device=device)
    actions = torch.zeros((num_steps, num_envs), device=device)
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
        opt.param_groups[0]["lr"] = lr * (1.0 - update / num_updates)  # LR anneal

        for t in range(num_steps):
            cur_t = torch.as_tensor(cur, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, logp, _, value = agent.act(cur_t)
            obs[t], actions[t], logprobs[t], values[t] = cur_t, action, logp, value

            a_np = action.cpu().numpy()
            real_next = np.zeros_like(cur)
            for i, e in enumerate(envs):
                o, r, term, trunc, _ = e.step(int(a_np[i]))
                rewards[t, i] = r
                terminated[t, i] = float(term)
                done_mask[t, i] = float(term or trunc)
                ep_ret[i] += r
                real_next[i] = o  # true next/terminal obs (single env -> no autoreset)
                if term or trunc:
                    ep_returns.append(float(ep_ret[i]))
                    ep_ret[i] = 0.0
                    o, _ = e.reset()
                cur[i] = o
            with torch.no_grad():
                next_values[t] = agent.value(
                    torch.as_tensor(real_next, dtype=torch.float32, device=device)
                )

        # GAE: bootstrap mask (1-terminated) keeps timeouts bootstrapping; chain mask (1-done)
        # stops advantage propagation across any episode boundary.
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
        b_act = actions.reshape(-1)
        b_logp = logprobs.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)
        b_val = values.reshape(-1)

        idx = np.arange(batch_size)
        for _ in range(epochs):
            np.random.shuffle(idx)
            for start in range(0, batch_size, mb_size):
                mb = idx[start : start + mb_size]
                _, newlogp, entropy, newval = agent.act(b_obs[mb], b_act[mb].long())
                ratio = (newlogp - b_logp[mb]).exp()
                adv = b_adv[mb]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # per-minibatch adv norm

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
    p.add_argument("--total-steps", type=int, default=150_000)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=2.5e-3)
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
    gate = "PASS" if mean >= 475 else "FAIL"
    print(
        f"\nPPO (discrete) | mean {mean:.1f} +/- {spread:.1f} across {args.seeds} seeds "
        f"| gate >=475: {gate}"
    )


if __name__ == "__main__":
    main()
