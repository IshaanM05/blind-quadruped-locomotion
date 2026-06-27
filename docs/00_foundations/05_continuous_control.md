# Continuous control: Gaussian policies

**What.** How a policy outputs *continuous* actions (joint targets, not "left/right") and how we
compute the log-probabilities PPO needs.

**Why it exists.** A quadruped commands 12 continuous joint positions. The discrete categorical
policy from CartPole doesn't apply. We need a continuous distribution we can sample from and
differentiate — and the log-prob bookkeeping is the single most common place to introduce a
silent bug.

**The core idea in one sentence.** Output the *mean* of a Gaussian over actions (with a learned,
state-independent log-std), sample from it during training, and act on the mean at eval.

---

## The Gaussian policy

The network maps state → action mean $\mu_\theta(s)$. The spread is a separate learned
parameter vector $\log\sigma$ (one per action dim), **not** a function of the state — this is the
standard, stable choice (state-dependent std is harder to train and rarely worth it). The action
is sampled $a \sim \mathcal{N}(\mu_\theta(s), \sigma^2)$.

**Log-prob** of an action under a diagonal Gaussian (sum over action dims $i$):

$$\log\pi_\theta(a\mid s) = \sum_i\Big[-\tfrac{1}{2}\Big(\tfrac{a_i-\mu_i}{\sigma_i}\Big)^2 - \log\sigma_i - \tfrac{1}{2}\log 2\pi\Big]$$

Getting this exactly right — summing over dims, using $\log\sigma$ not $\sigma$, matching the
sampled action — is the classic trap. `torch.distributions.Normal(mu, sigma).log_prob(a).sum(-1)`
does it correctly; writing it by hand once (and unit-testing against the closed form) is worth it.

## Exploration: the std *is* the entropy

A diagonal Gaussian's entropy is $\sum_i(\tfrac12\log 2\pi e + \log\sigma_i)$ — it grows with
$\log\sigma$. So the **init noise std** (1.0 here) sets initial exploration, and the entropy
bonus in [04_ppo](04_ppo.md) resists premature collapse of $\sigma$. Watch $\sigma$ during
training: if it drops below ~0.2 within the first 100 iters, you collapsed too early (entropy
coef too low or LR too high) — this is item 3 in the debugging playbook.

## Squashing: keeping actions in bounds, honestly

Joint targets are bounded. Two approaches:

1. **Action scale + clip** (what the locomotion task uses): the Gaussian is unbounded; multiply
   by a small scale (~0.25 rad) and let the env clip. Simple; the log-prob is the plain Gaussian.
2. **tanh-squash** (`ppo_continuous.py` on Pendulum): pass the sample through `tanh`. This bounds
   actions to $(-1,1)$ but **changes the density** — you must add the
   change-of-variables Jacobian correction $-\sum_i \log(1-\tanh(a_i)^2)$ to the log-prob.
   Forgetting this term is *the* continuous-control bug; we do it honestly and document it.

## Why position targets, not torques (the locomotion choice)

The 12 actions are interpreted as **joint position offsets from a default stance**, fed to a PD
controller (stiffness ~25–40, damping ~0.5–1). The PD loop is a built-in stabilizer: small
action noise produces gentle, safe motion instead of violent torque spikes, so exploration is
far safer and learning dramatically faster. Raw-torque control is possible but much harder — a
clean ablation for later. (Foreshadowed here; implemented in `docs/20_locomotion/`.)

## How it appears in this repo

- `algorithms/ppo/ppo_continuous.py` — Gaussian head, honest tanh log-prob, the 37 details.
- **Gate:** Pendulum-v1 ≥ −250 avg return across seeds.
- The Isaac Lab bridge (`scripts/isaac_bridge_cartpole.py`) reuses this core on batched envs.

## Explain-it-back

1. Why a state-independent learned $\log\sigma$ rather than a network output?
2. Write the diagonal-Gaussian log-prob and explain each term.
3. What is the tanh Jacobian correction and what goes wrong without it?
4. Relate init noise std, entropy, and premature determinism.
5. Why do PD position targets make learning easier than torque control?
