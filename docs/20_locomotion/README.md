# MDP design for velocity tracking, term by term (Phase 1)

> **Status: outline.** Filled as Phase 1 is built. Each reward term gets one paragraph here:
> what it does, and what breaks without it (verified by the ablation table).

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

## Terminations
Trunk contact = fell; timeout = **truncation, not termination**
([03_variance_and_gae](../00_foundations/03_variance_and_gae.md)).

## Will answer (one paragraph each, post-ablation)
For every reward term: its mathematical form, its weight, the failure mode it prevents, and the
qualitative gait change when it's removed. Plus: why `action_rate_l2` is the single biggest
contributor to non-jittery motion; why a fraction of envs are commanded to stand still.
