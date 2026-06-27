# Experiment log

One line per run. The empirical companion to `docs/`. Columns:
**date · phase · hypothesis · config · key metric · conclusion · commit**.

Discipline (docs §10.2): one experiment = one config = one logged commit hash. Headline
results run with >=3 seeds (report mean +- std); single-seed RL numbers are noise.

| Date | Phase | Hypothesis / change | Config | Metric | Conclusion | Commit |
|---|---|---|---|---|---|---|
| 2026-06-27 | 0 | Repo + docs scaffold; verified GAE against hand-computed trajectories | — | tests pass | foundation laid | _init_ |
| 2026-06-27 | 0 | REINFORCE (no baseline), CartPole | 40k steps, 2 seeds | 104.7 ± 34.1 | learns but noisy/low — the MC-variance baseline case | _phase0_ |
| 2026-06-27 | 0 | REINFORCE + value baseline, CartPole | 40k steps, 2 seeds | 355.4 ± 41.8 | big jump over plain REINFORCE — baseline cuts variance | _phase0_ |
| 2026-06-27 | 0 | A2C (GAE bootstrap), CartPole | 60k steps, 2 seeds | 253.5 ± 64.3 | learns; still climbing at 60k — bridge to PPO | _phase0_ |
| 2026-06-27 | 0 | **PPO discrete — GATE** | 150k steps, 3 seeds | **500.0 ± 0.0** | **PASS (>=475)**; perfect, all seeds | _phase0_ |
| 2026-06-27 | 0 | **PPO continuous — GATE** | 400k steps, 3 seeds | **−198.9 ± 16.0** | **PASS (>=−250)**; all seeds under the bar | _phase0_ |

<!-- Append new rows below. Keep newest at the bottom. -->
