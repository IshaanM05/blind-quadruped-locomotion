# PPO: from trust regions to the clip

**What.** Proximal Policy Optimization — the algorithm that trains every policy in this project
(your from-scratch version in Phase 0, then rsl_rl's for locomotion).

**Why it exists.** Plain policy gradient takes one noisy step per batch of data and throws the
data away — sample-inefficient. We'd like to take *several* optimization steps on the same
batch. But reusing data means the policy moves away from the one that collected it, and a large
move can destroy performance (the gradient is only trustworthy locally). PPO lets you reuse data
for several epochs while *softly* preventing destructive updates — without TRPO's heavy math.

**The core idea in one sentence.** Maximize an importance-weighted advantage objective, but
**clip** the importance ratio so the policy gains nothing from moving too far in one update.

---

## The problem PPO solves (TRPO's question, PPO's answer)

Reusing off-batch data needs importance sampling: $r_t(\theta) = \dfrac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_\text{old}}(a_t\mid s_t)}$.
The naive surrogate $\mathbb{E}[r_t(\theta) A_t]$ can be pushed arbitrarily high by making
$r_t$ huge on a single high-advantage action — i.e. by taking a reckless step. TRPO fixes this
with a hard KL trust-region constraint (a constrained optimization). PPO approximates the same
intent with a cheap, unconstrained, **clipped** objective.

## The clipped surrogate objective

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(r_t(\theta)\,A_t,\;\;
\text{clip}(r_t(\theta),\,1-\epsilon,\,1+\epsilon)\,A_t\big)\Big], \quad \epsilon=0.2$$

Read it by cases:

- **$A_t > 0$** (good action): objective rises with $r_t$ up to $1+\epsilon$, then the clip
  **flattens** it — no reward for pushing the probability higher than +20%.
- **$A_t < 0$** (bad action): objective rises as $r_t$ drops to $1-\epsilon$, then flattens.
- The outer **`min`** makes the bound *pessimistic*: it only clips when clipping makes the
  objective *worse*, so the policy can always *undo* a too-large previous step but never
  *exploit* one. This asymmetry is the subtle, essential bit.

We clip the **objective**, not the gradient, because the goal is to remove the *incentive* to
move far — a clipped gradient would still chase the same bad optimum, just slower.

## The full loss

$$L = \underbrace{L^{\text{CLIP}}}_{\text{policy}} - c_1\,\underbrace{(V_\theta(s_t)-\hat G_t)^2}_{\text{value, optionally clipped}} + c_2\,\underbrace{\mathcal{H}[\pi_\theta(\cdot\mid s_t)]}_{\text{entropy bonus}}$$

The **entropy bonus** keeps the action distribution wide enough to keep exploring; set it to 0
and the policy can collapse to a deterministic, unexplored gait early (you'll verify this
empirically in Phase 1 — gait diversity dies). $c_2 \approx 0.005$–$0.01$.

## The "37 implementation details" that actually matter

PPO's paper is simple; making it *work* is in the details. The ones implemented in
`ppo_continuous.py` and ablated on Pendulum:

- **Advantage normalization** (per-minibatch, zero mean/unit std) — stabilizes the scale.
- **Orthogonal weight init** + small policy-head gain — well-conditioned start.
- **Value-loss clipping** — mirrors the policy clip on the critic.
- **Global gradient-norm clipping** (max 0.5–1.0) — caps occasional huge updates.
- **Learning-rate annealing** — or, in rsl_rl, **adaptive LR from a target KL** (≈0.01): if the
  measured policy KL overshoots, shrink the LR; if it's tiny, grow it. This "hidden gem" largely
  prevents the approx-KL spikes / clip-fraction blowups that wreck naive PPO. See
  [glossary](../glossary.md) and the rsl_rl diff note in `notes/`.

## Why PPO here (and not DQN/SAC)?

On-policy PPO handles **continuous, high-dimensional** action spaces (12 joints) natively via a
Gaussian policy ([05_continuous_control](05_continuous_control.md)), is stable and simple to
tune, and shines with **massively parallel** sims (4096 envs) where sample-efficiency matters
less than wall-clock throughput. DQN is value-based and discrete-action; SAC is off-policy and
more sample-efficient but fiddlier — a fine answer to "why not SAC?" is "PPO + 4096 envs trains
a gait in minutes, so sample-efficiency isn't the bottleneck; stability and simplicity are."

## How it appears in this repo

- `algorithms/ppo/ppo_discrete.py`, `ppo_continuous.py` — the loss above, end to end.
- `tests/` — clip-branch logic on crafted tensors.
- Locomotion uses rsl_rl's PPO (same math, adaptive-KL LR) — `docs/10_isaaclab/`.

## Explain-it-back

1. Why clip the *ratio/objective* rather than the gradient? What TRPO problem does this address?
2. Walk through $L^{\text{CLIP}}$ for $A_t>0$ and $A_t<0$; why the outer `min`?
3. Why is PPO on-policy, and what does that imply about sample reuse?
4. What does the entropy bonus do; what happens at $c_2=0$?
5. How does adaptive-KL LR prevent destructive updates, and what would you log to see it working?
