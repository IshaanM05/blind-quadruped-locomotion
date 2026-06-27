# Variance, bias, and GAE

**What.** Generalized Advantage Estimation — a single knob $\lambda$ that interpolates between
two ways of estimating the advantage, trading bias against variance.

**Why it exists.** The advantage $A_t$ in the policy gradient must be *estimated* from samples.
The two natural estimators sit at opposite extremes: Monte-Carlo returns (unbiased, very noisy)
and one-step TD (low noise, biased if the critic is wrong). Neither is ideal. GAE gives you the
whole dial in between, and in practice $\lambda\approx0.95$ is near-optimal.

**The core idea in one sentence.** An exponentially-weighted average of multi-step TD errors
lets you keep most of TD's low variance while bleeding off most of its bias.

---

## The bias–variance dial

The **TD error** (a one-step advantage estimate): $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$.
Under the *true* $V$, $\mathbb{E}[\delta_t]=A_t$ exactly (this is interview question #7 — it
follows from the Bellman expectation equation).

- **Monte-Carlo** ($A_t = \hat G_t - V(s_t)$): no bootstrapping, **unbiased**, but $\hat G_t$
  sums many random rewards → **high variance**. This is the REINFORCE regime; the seed plots
  are noisy.
- **One-step TD** ($A_t = \delta_t$): bootstraps on $V(s_{t+1})$, so **low variance**, but
  **biased** whenever $V$ is imperfect (it always is early in training).

MC ←→ TD is exactly the dial you traverse by hand in `a2c.py` (n-step) before GAE formalizes it.

## GAE — the derivation in three lines

Define the $n$-step advantage $A_t^{(n)} = \sum_{l=0}^{n-1}\gamma^l \delta_{t+l}$ (telescopes
to $\hat G_t^{(n)} - V(s_t)$). GAE is the exponentially-weighted average of all of them:

$$A_t^{\text{GAE}(\gamma,\lambda)} = (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1} A_t^{(n)}
= \sum_{l=0}^{\infty}(\gamma\lambda)^l\,\delta_{t+l}.$$

The right-hand form is what you implement — a **single backward pass** over the rollout:

```
adv[T] = delta[T]
for t in reversed(range(T)):
    adv[t] = delta[t] + gamma * lam * (1 - done[t]) * adv[t+1]
```

- $\lambda = 0$ → $A_t = \delta_t$ (pure one-step TD; low variance, biased).
- $\lambda = 1$ → $A_t = \hat G_t - V(s_t)$ (Monte-Carlo; unbiased, high variance).
- $\lambda \approx 0.95$ → the sweet spot used everywhere in this repo.

## The `done` vs `timeout` subtlety (a real bug source)

The `(1 - done[t])` mask stops bootstrapping across episode boundaries — **but only for true
terminations.** If an episode ends because of a **time limit** (truncation), the value *should*
still bootstrap (the world didn't actually end). Treating a timeout as a termination zeros a
real future and biases every advantage in that episode. Isaac Lab / rsl_rl expose `time_outs`
separately for exactly this reason. This is both a classic silent bug and a great interview
talking point — see [01_mdp_and_returns](01_mdp_and_returns.md) and the Phase 1 termination notes.

## How it appears in this repo

- `compute_gae(...)` in `algorithms/ppo/common.py` — the backward recursion above.
- `tests/test_gae.py` — checks it against a **hand-computed** 3-step toy trajectory (do this
  before trusting any training curve).

## Explain-it-back

1. What exactly does $\lambda$ trade off? What do $\lambda=0$ and $\lambda=1$ reduce to?
2. Show $\mathbb{E}[\delta_t]=A_t$ under the true value function.
3. Why must timeouts bootstrap but terminations not? What breaks if you confuse them?
4. Write the GAE backward recursion from memory and explain each term.
