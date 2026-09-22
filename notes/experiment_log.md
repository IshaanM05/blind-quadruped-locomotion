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
| 2026-09-13 | 1 | **Stage A** (task-tracking rewards only): our from-scratch reward stack, `QuadrupedDistill-Flat-Go2-StageA-v0` | 4096 envs, 500 iters, `FlatGo2PPORunnerCfg` | mean reward 41.0; tracking error xy 0.156 m/s, yaw 0.415 rad/s; episode length 1000/1000 (no falls) | Tracking numbers already close to target despite no smoothness terms. **Confirmed visually (GUI playback, 16 envs)**: gait looks "off and a bit weird" (user's own words) — exactly the twitchy/broken gait the guide predicts for task-only rewards. Good demonstration that tracking-error alone doesn't capture gait quality; this is the "before" video for the report | _pending_ |
| 2026-09-13 | 1 | **Stage B** (+ regularization: `lin_vel_z_l2`, `ang_vel_xy_l2`, `joint_torques_l2`, `joint_acc_l2`, `action_rate_l2`, `flat_orientation_l2`) | 4096 envs, 500 iters, `QuadrupedDistill-Flat-Go2-StageB-v0` | mean reward 39.2; tracking error xy 0.131 m/s, yaw 0.237 rad/s; episode length 1000/1000 | Tracking error improved on BOTH axes vs Stage A (0.156→0.131 xy, 0.415→0.237 yaw) despite mean reward dropping slightly (penalties now subtract from it) — regularization didn't just smooth motion, it made tracking itself more efficient. Not yet visually confirmed | _pending_ |
| 2026-09-14 | 1 | **Stage C — GATE** (+ gait shaping: `feet_air_time`, `undesired_contacts`) | 4096 envs, 500 iters, `QuadrupedDistill-Flat-Go2-StageC-v0` | training-time: mean reward 39.5, tracking error xy 0.118 m/s, yaw 0.225 rad/s; `play_policy.py` eval (32 envs, 32 episodes, deterministic policy): xy **0.054 m/s**, yaw **0.051 rad/s** | **PASS** — well under the guide's <0.2 m/s target on both axes. Eval numbers are notably tighter than training-time metrics (deterministic policy, no exploration noise). Checkpoint: `logs/rsl_rl/quadruped_distill_flat_go2/2026-09-14_00-08-34/model_499.pt` | _pending_ |
| 2026-09-14 | 1 | **Reward ablation suite** — 8 short runs (4096 envs, 200 iters each, `scripts/ablate_flat_rewards.py`), each evaluated via `play_policy.py` (32 envs, 32 episodes) | Hydra CLI overrides on `QuadrupedDistill-Flat-Go2-StageC-v0` (`env.rewards.<term>.weight=0.0` or `agent.algorithm.entropy_coef=0.0`) | see table below | short runs (200 vs Stage C's 500 iters) — read as directional evidence, not final numbers | _pending_ |

**Ablation results** (tracking error, mean over 32 episodes; baseline = all Stage C terms at 200 iters):

| Config | xy err (m/s) | yaw err (rad/s) | vs baseline |
|---|---|---|---|
| baseline (all terms) | 0.0843 | 0.0619 | — |
| no `action_rate_l2` | 0.0955 | **0.1130** | yaw nearly **2x worse** — biggest single effect, confirms the guide's claim this term is the main smoothness driver |
| no `flat_orientation_l2` | **0.1010** | 0.0963 | worst xy of all, yaw +56% — orientation regularization matters for stable tracking, not just cosmetics |
| no `feet_air_time` | 0.0802 | 0.0675 | ~unchanged — expected: this term shapes gait *quality* (stepping vs shuffling), not tracking accuracy, so tracking error doesn't capture its effect |
| no `undesired_contacts` | 0.0749 | 0.0571 | ~unchanged/slightly better — same caveat: this guards against illegal-contact gaits (e.g. knee-walking), which tracking error alone can't detect; needs a contact-count metric or video, not measured here |
| no `joint_torques_l2` | 0.0756 | 0.0692 | ~unchanged — energy/actuator-stress proxy, mild effect within 200 iters |
| no `ang_vel_xy_l2` | 0.0742 | 0.0674 | ~unchanged/slightly better — wobble guard; may need a longer horizon to show its value |
| zero `entropy_coef` | 0.0885 | 0.0644 | mild degradation (xy +5%) — see Q6 discussion below |
| 2026-09-20 | 2 | Baseline reference: stock `Isaac-Velocity-Rough-Unitree-Go2-v0` + rsl_rl | 4096 envs, `UnitreeGo2RoughPPORunnerCfg` default (1500 iters) | mean reward 22.77; tracking error xy 0.375 m/s, yaw 0.416 rad/s; mean terrain level 5.69/10 | reference number our rebuilt rough teacher is compared against. Run took ~2h18m wall-clock (4096 envs, terrain+raycaster overhead) — much longer than Phase 1's flat runs, budget accordingly for the full teacher + ablation runs | _n/a, framework baseline_ |
| 2026-09-22 | 2 | **Full teacher — GATE**: `QuadrupedDistill-Rough-Go2-v0` (terrain curriculum + DR + asymmetric critic) | 4096 envs, 1900/2000 iters (stopped manually — reward plateaued 22-24 over the last ~300 iters, well past the 1500-iter stock reference already) | training-time: mean reward ~23, tracking error xy 0.28-0.33 m/s; `play_policy.py` eval (32 envs, 20 episodes, deterministic): xy **0.123 m/s**, yaw **0.139 rad/s** | **PASS** — well under the guide's <0.2 m/s target on both axes, beats the stock baseline (0.375/0.416) by a wide margin. One real hang hit mid-run (~iter 539, likely a laptop suspend/resume CUDA-context issue, not a code bug) — resumed cleanly from checkpoint. Checkpoint: `logs/rsl_rl/quadruped_distill_rough_go2/2026-09-22_10-54-27/model_1900.pt` | _pending_ |

<!-- Append new rows below. Keep newest at the bottom. -->
