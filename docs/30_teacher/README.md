# The teacher: terrain, curriculum, randomization, asymmetric critic (Phase 2)

> **Status: outline.** Filled during Phase 2. This trains the **privileged teacher** that Phase 3 distills.

**What.** The same Go2 crossing stairs, slopes, and rough ground — trained with an automatic
terrain curriculum and randomized physics — using *privileged* information a real robot lacks.

**Why it exists.** Hard terrain is unlearnable from scratch (every episode dies in 2 s before
exploration finds stair-climbing). A curriculum grows difficulty with competence. Domain
randomization makes the policy robust. The height-scanner gives the teacher terrain knowledge it
will later have to *infer* without — this gap is the whole point of the teacher/student split.

## Components (each expanded with code + a learning-curve plot)
- **Procedural terrain:** `TerrainGenerator` grid — rows = difficulty, cols = type (pyramid
  stairs ↑/↓, slopes 5–25°, random rough 2–10 cm, discrete obstacles).
- **Height-scanner `RayCaster`:** ~187 downward rays around the base → terrain height. *Privileged.*
- **Terrain curriculum (self-written):** walked far enough → promote a level; barely moved →
  demote; beat the top → re-randomize. Log **mean terrain level** (the satisfying rising curve).
- **Domain randomization (events):** friction, base mass, motor strength, external pushes,
  per-step observation noise — modes `startup` / `reset` / `interval`.
- **Asymmetric actor-critic:** a `"critic"` obs group with privileged state (true base velocity,
  friction, contacts) → better value estimates → lower-variance advantages → faster learning.

## Ablations (keep the curves)
curriculum vs none · symmetric vs asymmetric critic · randomization on/off.

## Will answer
1. Why does training directly on hard terrain fail, and how does the curriculum fix exploration?
2. Domain randomization as "training on a distribution of MDPs" — and the robustness↔performance trade-off.
3. Why can the critic use information the actor can't? (PPO discards the critic at deploy time.)
4. What makes the height-scan "privileged," and why deliberately build a policy that depends on it?
