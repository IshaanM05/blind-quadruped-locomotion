# Perceptive locomotion (Phase 4, stretch)

> **Status: outline / stretch goal.** Built only if Phases 0–3 land well. Miki et al. 2022.

**What.** Extend the blind student with **vision** (depth cameras) so it can anticipate terrain
instead of only reacting to it underfoot.

**Why it exists.** Proprioception-only locomotion is reactive — the robot discovers a gap with its
foot. Exteroception lets it plan footholds ahead. Miki 2022 fuses proprioception and vision so the
robot degrades gracefully to the blind policy when perception is unreliable (fog, motion blur).

## Components (sketch)
- Depth-camera **tiled rendering** in Isaac Lab (the real VRAM cost → lower env counts, 256–1024).
- A perception encoder feeding the recurrent policy; proprioception remains the safety fallback.
- Robustness when vision drops out → recover the Phase 3 blind behavior.

## Will answer
1. What does exteroception buy over the blind policy, concretely?
2. How is graceful degradation to proprioception achieved and tested?
3. Why are cameras the VRAM bottleneck, and how does tiled rendering help?
