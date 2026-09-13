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
  [glossary](../glossary.md) and [10_isaaclab/01_rsl_rl_diff](../10_isaaclab/01_rsl_rl_diff.md).

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
- Locomotion uses rsl_rl's PPO (same math, adaptive-KL LR) — see [10_isaaclab/01_rsl_rl_diff](../10_isaaclab/01_rsl_rl_diff.md).

### Ablation: which details actually matter (Pendulum-v1, 3 seeds each)

`ppo_continuous.py` exposes `--no-orthogonal-init`, `--no-adv-norm`, `--no-vloss-clip`,
`--no-lr-anneal`, `--no-grad-clip` (run via `scripts/ablate_ppo_details.py`):

| Config | Mean ± std | Gate (>=−250) |
|---|---|---|
| baseline (all details on) | −198.9 ± 16.0 | PASS |
| no advantage normalization | −251.5 ± 43.7 | **FAIL** |
| no LR annealing | −217.0 ± 23.0 | PASS (worse) |

Dropping **advantage normalization** doesn't just hurt the mean — it nearly **triples the
seed-to-seed variance** (16.0 → 43.7). Without it, the policy-gradient scale isn't normalized
per-minibatch, so a batch with unusually large-magnitude advantages produces a disproportionately
large update; some seeds get a lucky, well-scaled batch early and converge fine, others don't —
that's the mechanism behind the variance blowup, not just a worse average. **LR annealing** has a
smaller but still real effect (mean drops slightly, variance up ~1.4x): without decay, the
optimizer keeps taking full-sized steps late in training when the policy is already close to
converged, occasionally kicking it back out of a good region.

## Explain-it-back

1. Why clip the *ratio/objective* rather than the gradient? What TRPO problem does this address?
2. Walk through $L^{\text{CLIP}}$ for $A_t>0$ and $A_t<0$; why the outer `min`?
3. Why is PPO on-policy, and what does that imply about sample reuse?
4. What does the entropy bonus do; what happens at $c_2=0$?
5. How does adaptive-KL LR prevent destructive updates, and what would you log to see it working?

## §4.4 concept-check answers (guide questions 4, 5)

**Q4. Why does PPO clip the probability ratio instead of the gradient? What failure mode of
vanilla PG / what idea from TRPO does this address?** Vanilla policy gradient (and the naive
importance-weighted surrogate $\mathbb{E}[r_t(\theta)A_t]$ that lets you reuse data) can be pushed
arbitrarily high by making $r_t$ huge on a single high-advantage sample — nothing stops the
optimizer from taking a reckless step that destroys the policy. TRPO fixes this with a hard KL
trust-region constraint; PPO approximates the same intent far more cheaply by clipping the
*objective* itself, not the gradient. Clipping the objective removes the *incentive* to move the
ratio far from 1 (the "min" makes the bound pessimistic — see "The clipped surrogate objective"
above); clipping the *gradient* instead would still chase the same unbounded optimum, just more
slowly, so it doesn't fix the underlying problem — only the clipped objective changes what the
optimizer is even trying to do.

**Q5. Why is PPO on-policy, and what does that imply about sample reuse?** PPO's clipped
surrogate is only a valid (importance-weighted) approximation of the true policy gradient when
the *current* policy $\pi_\theta$ hasn't drifted too far from the policy $\pi_{\theta_\text{old}}$
that collected the data — the clip explicitly bounds how far that drift is allowed to go before
the objective stops rewarding it. This means data can be reused for several epochs (unlike a
single-sample REINFORCE step) but *not indefinitely*: once the policy has moved on, the collected
rollout is stale and must be discarded and re-collected. This is the direct trade PPO makes
against fully off-policy methods (e.g. SAC), which can reuse a replay buffer indefinitely at the
cost of a fiddlier, less stable algorithm — see "Why PPO here" above.
