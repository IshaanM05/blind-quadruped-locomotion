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
| 2026-09-13 | 0 | 37-details ablation: baseline regression check | 400k steps, 3 seeds, all details on | −198.9 ± 16.0 | exact match to recorded gate — new ablation flags are no-ops by default | `65b2034` |
| 2026-09-13 | 0 | 37-details ablation: no advantage normalization | 400k steps, 3 seeds, `--no-adv-norm` | −251.5 ± 43.7 | **FAIL**; mean drops below gate AND variance ~2.7x baseline — adv-norm both centers and stabilizes the update | _pending_ |
| 2026-09-13 | 0 | 37-details ablation: no LR annealing | 400k steps, 3 seeds, `--no-lr-anneal` | −217.0 ± 23.0 | PASS but worse mean + ~1.4x variance vs baseline — smaller, real effect | _pending_ |
| 2026-09-13 | 0 | rsl_rl baseline on Isaac-Cartpole-v0 (ground truth for bridge gate) | 4096 envs, 16 steps/env, 150 iters (default `CartpolePPORunnerCfg`) | mean reward 4.95 | reference number for own-PPO bridge comparison | _n/a, framework baseline_ |
| 2026-09-13 | 0 | **Isaac Lab bridge — GATE**: own PPO on Isaac-Cartpole-v0 via Isaac Lab's batched Gym API | same config as rsl_rl run above, 9.83M total env steps | mean return 4.78 (last 100 eps) | **PASS**; within ~3.4% of rsl_rl. Found+fixed: `action_space.high` is `inf` for effort-controlled joints (Isaac Lab scales internally via `JointEffortActionCfg`); act_scale must be 1.0, not read from the env | _pending_ |
| 2026-09-13 | 0.5 | **Custom reacher env — GATE**: from-scratch `ManagerBasedRLEnvCfg` (double-pendulum, custom `JointTargetCommand`, custom reward), own PPO | 512 envs, 300 updates, 2.46M total env steps | mean return −87.16 → **−1.19** (last 100 eps) | **PASS ("custom env trains")**; ~98.6% reduction in summed squared joint-angle error. `ppo_isaac_cartpole.py` generalized with `--task` to train it, no PPO code duplicated | _pending_ |
| 2026-09-13 | 1 | Baseline reference: stock `Isaac-Velocity-Flat-Unitree-Go2-v0` + rsl_rl (§6.1) | 4096 envs, `UnitreeGo2FlatPPORunnerCfg` default (300 iters) | mean reward 34.6; tracking error xy 0.187 m/s, yaw 0.355 rad/s (last iter) | reference number/gait our rebuilt reward stack is compared against | _n/a, framework baseline_ |

<!-- Append new rows below. Keep newest at the bottom. -->
