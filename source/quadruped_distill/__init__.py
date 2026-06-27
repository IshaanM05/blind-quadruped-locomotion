"""quadruped_distill — learning blind quadruped locomotion.

From-scratch PPO (Phase 0) → flat locomotion (Phase 1) → rough-terrain teacher (Phase 2)
→ teacher-student distillation into a recurrent, proprioception-only student (Phase 3).

See ``docs/`` for the conceptual derivations behind every component.
"""

__version__ = "0.1.0"
