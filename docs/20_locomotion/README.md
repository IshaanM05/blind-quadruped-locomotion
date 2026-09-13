# MDP design for velocity tracking, term by term (Phase 1)

> **Status: in progress.** Stages A and B trained (below); Stage C and the reward ablation table
> (`scripts/ablate_flat_rewards.py`) still running — the "Will answer" section fills in once
> those land. Implementation: `quadruped_distill/tasks/flat/`, `quadruped_distill/mdp/locomotion.py`.

**What.** The full MDP for a Go2 that tracks commanded $(v_x, v_y, \omega_z)$ on flat ground with
a natural trot.

**Why it exists.** This is where RL theory meets a real robot. The reward function *is* the
design: a tiny weight change is the difference between a smooth trot and a twitching robot that
games the reward.

## Observations (policy)
| Term | Dim | Why |
|---|---|---|
| base lin vel (body frame) | 3 | what we control — note: hard to measure on a real robot (Phase 3 drops it) |
| base ang vel | 3 | IMU gyro — realistic |
| projected gravity | 3 | orientation w/o yaw — "which way is down" |
| velocity commands | 3 | the target |
| joint pos (rel. default) | 12 | proprioception |
| joint vel | 12 | proprioception |
| previous action | 12 | smoothness / self-dynamics |

## Actions
12-d → joint **position offsets** from default stance → PD controller. Action scale ~0.25 rad.
(Why position not torque: [05_continuous_control](../00_foundations/05_continuous_control.md).)

## Rewards — built in three stages, retraining each time
- **A (task):** `track_lin_vel_xy_exp`, `track_ang_vel_z_exp` → expect a twitchy gait (save the video).
- **B (regularization, negative):** `lin_vel_z_l2`, `ang_vel_xy_l2`, `joint_torques_l2`, `joint_acc_l2`, `action_rate_l2`, `flat_orientation_l2`.
- **C (gait shaping):** `feet_air_time` (the trot-maker), `undesired_contacts`, optional `feet_slide`.

### Results so far (4096 envs, 500 iterations each; `notes/experiment_log.md` has the full rows)

| Stage | Mean reward | Tracking error xy (m/s) | Tracking error yaw (rad/s) | Gait |
|---|---|---|---|---|
| stock baseline (reference) | 34.6 | 0.187 | 0.355 | — |
| A (task only) | 41.0 | 0.156 | 0.415 | **Confirmed visually: "off and a bit weird"** — twitchy, exactly as predicted |
| B (+ regularization) | 39.2 | 0.131 | 0.237 | not yet watched, but *both* tracking axes improved despite lower raw reward |
| C (+ gait shaping) | 39.5 | 0.118 (train) / **0.054** (eval) | 0.225 (train) / **0.051** (eval) | **PASS** target <0.2 m/s |

Stage B's result is the more interesting one methodologically: adding penalty terms *dropped*
the raw mean reward (penalties are subtracted), which could look like "worse" at a glance — but
the actual tracking accuracy *improved* on both axes. The lesson: don't read total reward as a
proxy for task performance once the reward function has multiple competing terms; read the
task-specific metric (`Metrics/base_velocity/error_vel_*`) directly.

## Terminations
Trunk contact = fell; timeout = **truncation, not termination**
([03_variance_and_gae](../00_foundations/03_variance_and_gae.md)).

## Per-term results (from `scripts/ablate_flat_rewards.py`, 200-iter short runs — see
`notes/experiment_log.md` for the full table; read these as directional, not final, since Stage
C itself trains 500 iterations)

**`track_lin_vel_xy_exp` / `track_ang_vel_z_exp`.** The only two positive/task terms — everything
else is either a penalty or a shaping bonus layered on top. Exponential kernel (not a hard
threshold) so the gradient stays informative even far from the target.

**`action_rate_l2`** — penalizes `||a_t - a_{t-1}||²`. **Confirmed the guide's claim**: removing
it nearly doubled yaw tracking error (0.062→0.113 rad/s), the single biggest degradation in the
whole ablation table. Mechanism: without it, PPO has no reason to prefer a smooth action
trajectory over a noisy one with the same expected task reward — the policy exploits any slack in
the tracking reward with jittery, high-frequency joint commands that happen to average out
correctly but destabilize yaw control specifically (yaw depends on differential leg timing,
which jitter disrupts more than straight-line xy motion does).

**`flat_orientation_l2`** — penalizes nonzero xy-components of projected gravity (base tilt).
Removing it gave the **worst xy tracking of any ablation** (0.101 m/s) — a tilted base doesn't
just look wrong, it actively corrupts the body-frame velocity the tracking reward is computed
against, so orientation regularization turns out to be load-bearing for the task reward itself,
not merely cosmetic.

**`lin_vel_z_l2` / `ang_vel_xy_l2` / `joint_torques_l2` / `joint_acc_l2`** — all showed only mild
effects within 200 iterations (roughly ±10% on tracking error, within the ablation's noise band).
This is expected: these are secondary regularizers (bounce, roll/pitch wobble, energy,
jerk) whose value compounds over longer training and shows up more in gait *naturalness* — a
qualitative property tracking error doesn't measure — than in raw tracking accuracy.

**`feet_air_time` / `undesired_contacts`** — both showed **no measurable tracking-error
penalty when removed** (if anything, marginally better numbers). This is not evidence they're
unimportant — it's a limitation of the metric: these terms exist to prevent specific *gait
pathologies* (shuffling instead of stepping; knee/thigh-walking) that a velocity-tracking-error
number cannot see at all. Properly evaluating them needs either video or a contact-count/air-time
statistic, neither of which this ablation pass collected. Honest gap, not a finding that they
don't matter.

**Why a fraction of envs are commanded to stand still** (`rel_standing_envs=0.1` in
`CommandsCfg`): without some zero-command envs, the policy never practices the "stay balanced
with no velocity target" regime, which is a genuinely different control problem (active
balancing without directional momentum) from tracking a nonzero command.

## Entropy coefficient — answering Phase 0's deferred Q6

Phase 0's concept-check Q6 ("what happens to gait diversity if you set the entropy coefficient to
0?") was explicitly deferred because it needed real locomotion data. The zero-entropy ablation
here gives a partial, honest answer: killing entropy caused a **mild tracking degradation** (xy
error +5%, 0.084→0.089 m/s) rather than a dramatic collapse — consistent with the policy
converging faster to a narrower, less-explored behavior that's slightly worse at handling the
full range of commanded velocities. **What this does *not* show**: literal gait diversity (foot
placement variety, trot-pattern robustness) isn't something tracking-error alone measures — that
would need either video comparison across seeds or a foot-contact-pattern statistic, neither of
which was collected here. So Q6 is now backed by real data on *tracking robustness*, but the
*visual gait diversity* half of the question would need a dedicated video-based follow-up to
answer as fully as the "gait shaping" terms above.
