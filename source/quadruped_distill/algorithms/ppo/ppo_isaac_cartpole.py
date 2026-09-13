"""Isaac Lab bridge — train Isaac-Cartpole-v0 with our own from-scratch PPO.

This is Phase 0's last deferred item: wrap Isaac Lab's Gym interface (a single vectorized,
GPU-batched-tensor env, NOT a list of separate ``gym.Env``s) and train it with the
``GaussianActorCritic`` from ``ppo_continuous.py``, unmodified except for policy hidden sizes
matching rsl_rl's baseline config for a fair comparison.

Gate: mean episodic reward comparable to rsl_rl's baseline on the same task
(see ``docs/10_isaaclab/README.md`` and ``notes/experiment_log.md`` for the recorded number).

Key differences from ``ppo_continuous.py``'s Gymnasium rollout loop:

- ONE ``gym.make("Isaac-Cartpole-v0", ...)`` call, not N separate envs — actions/observations
  are already batched GPU tensors, no numpy round-trip, no manual per-env reset loop.
- Isaac Lab's ``step()`` auto-resets terminated/timed-out envs internally *before* returning,
  so the observation after a reset is already the new episode's first observation, not the true
  terminal one. We can't recover the true terminal obs (the framework doesn't expose it), so —
  matching rsl_rl's own convention — we bootstrap timeouts using the critic's value on the
  (already-reset) next observation. This is a known, accepted approximation, not a bug: with
  thousands of parallel envs the bias averages out over training.
- Observations arrive as ``{"policy": tensor}`` dicts (Isaac Lab's manager-based obs group
  convention), not raw arrays.

MUST run inside the Isaac Lab venv, not this package's own ``.venv``:
    source /home/ishaan/Desktop/IsaacLab-Proj/activate_isaaclab.sh
    python -m quadruped_distill.algorithms.ppo.ppo_isaac_cartpole --num-envs 4096 --headless
"""

from __future__ import annotations

import argparse
import statistics

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from quadruped_distill.algorithms.ppo.common import set_seed
from quadruped_distill.algorithms.ppo.ppo_continuous import GaussianActorCritic


def train(
    seed: int,
    num_envs: int,
    num_steps: int,
    num_updates: int,
    gamma: float,
    lam: float,
    clip: float,
    epochs: int,
    minibatches: int,
    ent_coef: float,
    vf_coef: float,
    max_grad: float,
    lr: float,
    headless: bool,
    log_dir: str | None = None,
) -> float:
    # Imports that need the Isaac Sim app running go inside train(), after AppLauncher starts
    # (see main()) — importing isaaclab/gymnasium-registered tasks before the sim app exists
    # fails, so this module can still be imported for --help without booting the simulator.
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401  (registers Isaac-Cartpole-v0 with gym)
    from isaaclab_tasks.utils import parse_env_cfg

    set_seed(seed)
    device = "cuda"

    env_cfg = parse_env_cfg("Isaac-Cartpole-v0", device=device, num_envs=num_envs)
    env = gym.make("Isaac-Cartpole-v0", cfg=env_cfg)

    obs_dim = env.observation_space["policy"].shape[-1]
    act_dim = env.action_space.shape[-1]
    # Isaac Lab's action space here is deliberately unbounded (Box(-inf, inf)): the env's own
    # JointEffortActionCfg(scale=100.0) converts a normalized [-1, 1]-ish policy action into real
    # torque internally. So act_scale=1.0 (not env.action_space.high, which is inf) — the policy's
    # tanh output IS the action Isaac Lab expects, unlike Pendulum-v1 which wants raw torque.
    act_scale = 1.0
    agent = GaussianActorCritic(obs_dim, act_dim, act_scale).to(device)
    opt = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    writer = SummaryWriter(log_dir) if log_dir else None

    batch_size = num_envs * num_steps
    mb_size = batch_size // minibatches

    obs_buf = torch.zeros((num_steps, num_envs, obs_dim), device=device)
    raw_actions = torch.zeros((num_steps, num_envs, act_dim), device=device)
    logprobs = torch.zeros((num_steps, num_envs), device=device)
    rewards = torch.zeros((num_steps, num_envs), device=device)
    values = torch.zeros((num_steps, num_envs), device=device)
    next_values = torch.zeros((num_steps, num_envs), device=device)
    terminated_buf = torch.zeros((num_steps, num_envs), device=device)
    done_buf = torch.zeros((num_steps, num_envs), device=device)

    obs_dict, _ = env.reset()
    cur = obs_dict["policy"]
    ep_ret = torch.zeros(num_envs, device=device)
    ep_returns: list[float] = []

    for update in range(num_updates):
        opt.param_groups[0]["lr"] = lr * (1.0 - update / num_updates)

        for t in range(num_steps):
            with torch.no_grad():
                env_a, raw_a, logp, _, value = agent.act(cur)
            obs_buf[t], raw_actions[t], logprobs[t], values[t] = cur, raw_a, logp, value

            next_obs_dict, r, terminated, timed_out, _ = env.step(env_a)
            next_obs = next_obs_dict["policy"]
            rewards[t] = r
            terminated_buf[t] = terminated.float()
            done_buf[t] = (terminated | timed_out).float()
            ep_ret += r
            done_envs = (terminated | timed_out).nonzero(as_tuple=False).squeeze(-1)
            if len(done_envs) > 0:
                ep_returns.extend(ep_ret[done_envs].tolist())
                ep_ret[done_envs] = 0.0
            with torch.no_grad():
                next_values[t] = agent.value(next_obs)
            cur = next_obs

        advantages = torch.zeros_like(rewards)
        last_adv = torch.zeros(num_envs, device=device)
        for t in reversed(range(num_steps)):
            boot = 1.0 - terminated_buf[t]
            chain = 1.0 - done_buf[t]
            delta = rewards[t] + gamma * next_values[t] * boot - values[t]
            last_adv = delta + gamma * lam * chain * last_adv
            advantages[t] = last_adv
        returns = advantages + values

        b_obs = obs_buf.reshape(-1, obs_dim)
        b_raw = raw_actions.reshape(-1, act_dim)
        b_logp = logprobs.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)
        b_val = values.reshape(-1)

        idx = torch.randperm(batch_size, device=device)
        for _ in range(epochs):
            idx = idx[torch.randperm(batch_size, device=device)]
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

        if writer is not None:
            step = (update + 1) * batch_size
            if ep_returns:
                writer.add_scalar("return/last_100_mean", statistics.mean(ep_returns[-100:]), step)
            writer.add_scalar("loss/policy", policy_loss.item(), step)
            writer.add_scalar("loss/value", v_loss.item(), step)
            print(
                f"update {update + 1}/{num_updates} | steps {step} | "
                f"return {statistics.mean(ep_returns[-100:]) if ep_returns else float('nan'):.2f}"
            )

    env.close()
    if writer is not None:
        writer.close()
    return statistics.mean(ep_returns[-100:]) if ep_returns else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--num-steps", type=int, default=16)
    p.add_argument("--num-updates", type=int, default=150)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--ent-coef", type=float, default=0.005)
    p.add_argument("--vf-coef", type=float, default=1.0)
    p.add_argument("--max-grad", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--log-dir", default=None)
    args, _ = p.parse_known_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    score = train(
        args.seed,
        args.num_envs,
        args.num_steps,
        args.num_updates,
        args.gamma,
        args.lam,
        args.clip,
        args.epochs,
        args.minibatches,
        args.ent_coef,
        args.vf_coef,
        args.max_grad,
        args.lr,
        args.headless,
        log_dir=args.log_dir,
    )
    print(f"\nown-PPO on Isaac-Cartpole-v0 | final mean return (last 100 eps) = {score:.2f}")

    simulation_app.close()


if __name__ == "__main__":
    main()
