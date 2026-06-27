# Documentation — the learning core

These docs exist so you can **explain every component of this project from first principles**,
not just run it. Each page follows the same shape:

1. **What** — the thing in one sentence.
2. **Why it exists** — the problem it solves; what breaks without it.
3. **The core idea** — derivation or mechanism, with the minimum math that actually matters.
4. **How it appears in this repo** — links to the exact code/config.
5. **Explain-it-back** — questions you should be able to answer out loud. If you can't, reread.

Read in order. The empirical companion is [`../notes/experiment_log.md`](../notes/experiment_log.md)
— what each training run actually taught us.

## Reading order

### 00 — Foundations (the "why" behind PPO)
1. [MDPs, returns, and Bellman equations](00_foundations/01_mdp_and_returns.md)
2. [Policy gradients, derived properly](00_foundations/02_policy_gradients.md)
3. [Variance, bias, and GAE](00_foundations/03_variance_and_gae.md)
4. [PPO: from trust regions to the clip](00_foundations/04_ppo.md)
5. [Continuous control: Gaussian policies](00_foundations/05_continuous_control.md)

### 10 — Isaac Lab architecture
- [The manager-based environment model](10_isaaclab/README.md)

### 20 — Flat locomotion (Phase 1)
- [MDP design for velocity tracking, term by term](20_locomotion/README.md)

### 30 — The teacher (Phase 2)
- [Terrain curriculum, domain randomization, asymmetric critic](30_teacher/README.md)

### 40 — Distillation (Phase 3, capstone)
- [Privileged → proprioceptive, DAgger, recurrence](40_distillation/README.md)

### 50 — Vision (Phase 4, stretch)
- [Perceptive locomotion](50_vision/README.md)

### Reference
- [Glossary](glossary.md)

> Pages for phases not yet built contain the outline + the questions they will answer, so the
> structure is visible and the gaps are explicit. They get filled as each phase lands.
