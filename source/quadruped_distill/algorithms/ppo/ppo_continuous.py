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

Ablation flags (see docs/00_foundations/04_ppo.md and Huang et al. "37 implementation details
of PPO"): each ``--no-*`` flag turns OFF a detail that is ON by default, so the default run is
an exact regression check against the recorded gate result.
    python -m quadruped_distill.algorithms.ppo.ppo_continuous --seeds 3 --no-adv-norm
    python -m quadruped_distill.algorithms.ppo.ppo_continuous --seeds 3 --no-lr-anneal
"""

from __future__ import annotations

import argparse
import statistics

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from quadruped_distill.algorithms.ppo.common import set_seed


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), orthogonal: bool = True) -> nn.Linear:
    if orthogonal:
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, 0.0)
    return layer


class GaussianActorCritic(nn.Module):
    def __init__(
        self, obs_dim: int, act_dim: int, act_scale: float, orthogonal_init: bool = True
    ):
        super().__init__()
        self.act_scale = act_scale
        oi = orthogonal_init
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64), orthogonal=oi),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64), orthogonal=oi),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0, orthogonal=oi),
        )
        self.mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64), orthogonal=oi),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64), orthogonal=oi),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01, orthogonal=oi),
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
    orthogonal_init: bool = True,
    adv_norm: bool = True,
    vloss_clip: bool = True,
    lr_anneal: bool = True,
    grad_clip: bool = True,
    log_dir: str | None = None,
) -> float:
    set_seed(seed)
    # Tiny 64x64 MLP: PyTorch's default (one thread per core) causes thread-launch/sync
    # overhead to dominate the actual compute on a laptop with many cores. Cap it.
    torch.set_num_threads(4)
    writer = SummaryWriter(log_dir) if log_dir else None
    envs = [gym.make("Pendulum-v1") for _ in range(num_envs)]
    obs_dim = envs[0].observation_space.shape[0]
    act_dim = envs[0].action_space.shape[0]
    act_scale = float(envs[0].action_space.high[0])  # Pendulum: +-2.0
    agent = GaussianActorCritic(obs_dim, act_dim, act_scale, orthogonal_init).to(device)
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
        if lr_anneal:
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
                if adv_norm:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                pg1 = -adv * ratio
                pg2 = -adv * torch.clamp(ratio, 1 - clip, 1 + clip)
                policy_loss = torch.max(pg1, pg2).mean()

                if vloss_clip:
                    v_clip = b_val[mb] + torch.clamp(newval - b_val[mb], -clip, clip)
                    v_loss = (
                        0.5
                        * torch.max((newval - b_ret[mb]) ** 2, (v_clip - b_ret[mb]) ** 2).mean()
                    )
                else:
                    v_loss = 0.5 * ((newval - b_ret[mb]) ** 2).mean()
                loss = policy_loss - ent_coef * entropy.mean() + vf_coef * v_loss

                opt.zero_grad()
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(agent.parameters(), max_grad)
                opt.step()

        if writer is not None:
            step = (update + 1) * batch_size
            if ep_returns:
                writer.add_scalar("return/last_50_mean", statistics.mean(ep_returns[-50:]), step)
            writer.add_scalar("loss/policy", policy_loss.item(), step)
            writer.add_scalar("loss/value", v_loss.item(), step)
            writer.add_scalar("entropy", entropy.mean().item(), step)
            writer.add_scalar("lr", opt.param_groups[0]["lr"], step)

    for e in envs:
        e.close()
    if writer is not None:
        writer.close()
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
    p.add_argument("--no-orthogonal-init", action="store_true", help="ablate orthogonal init")
    p.add_argument("--no-adv-norm", action="store_true", help="ablate advantage normalization")
    p.add_argument("--no-vloss-clip", action="store_true", help="ablate value-loss clipping")
    p.add_argument("--no-lr-anneal", action="store_true", help="ablate linear LR annealing")
    p.add_argument("--no-grad-clip", action="store_true", help="ablate global grad-norm clipping")
    p.add_argument("--log-dir", default=None, help="tensorboard log root (per-seed subdirs)")
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
            orthogonal_init=not args.no_orthogonal_init,
            adv_norm=not args.no_adv_norm,
            vloss_clip=not args.no_vloss_clip,
            lr_anneal=not args.no_lr_anneal,
            grad_clip=not args.no_grad_clip,
            log_dir=f"{args.log_dir}/seed{s}" if args.log_dir else None,
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
