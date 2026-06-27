# Distillation: privileged → proprioceptive, with memory (Phase 3, capstone)

> **Status: outline.** The portfolio centrepiece (Lee et al. 2020). Filled during Phase 3.

**What.** Compress the privileged teacher into a **blind student**: a recurrent policy that sees
only noisy onboard proprioception — no height-scan, no measured base velocity — and must *infer*
the terrain and its own velocity from the history of motion and contacts.

**Why it exists.** A real robot has no terrain map and a poor velocity estimate. The teacher was
*allowed* to cheat so it could discover good gaits cheaply; the student makes those gaits
*deployable*. Memory is essential because instantaneous blind proprioception is **not Markov**
w.r.t. terrain — the recurrent hidden state becomes a learned *belief* over the hidden world
state. (This closes the loop opened in [01_mdp_and_returns](../00_foundations/01_mdp_and_returns.md).)

## Why not just train the blind student with RL directly?
Exploration from scratch under partial observability + no terrain info is brutally slow and
unstable. Supervising on a competent teacher's actions (DAgger) sidesteps the exploration
problem entirely — you already know what good looks like. (This contrast *is* an ablation:
distilled student vs blind-RL-from-scratch.)

## Components (expanded with code + the money table)
- **Student architecture:** GRU/LSTM over the proprioceptive history → action; student obs =
  Phase 1 set **minus** base lin vel, **plus** realistic noise, **without** the height scan.
- **DAgger loop:** roll out the *student*, query the *teacher* for the action it would take,
  supervise the student toward it; aggregate data. Rolling out the student (not the teacher) is
  what fixes covariate shift — the key DAgger insight.
- **Recurrent buffer:** sequences + **hidden-state masking** at episode boundaries (unit-tested —
  a subtle, bug-prone detail).

## The "money table" (held-out terrain)
teacher (privileged) vs distilled student vs blind-RL-from-scratch — tracking error, success
rate, time-to-fall. The student approaching the teacher *without* privileged inputs is the result.

## Will answer
1. Why distill instead of RL-from-scratch for the blind policy?
2. In what sense is the recurrent hidden state a belief state over hidden terrain?
3. Why roll out the student but supervise with the teacher (covariate shift / DAgger)?
4. Why must hidden state be masked at resets, and what breaks otherwise?
