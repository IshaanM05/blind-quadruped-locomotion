"""Custom MDP terms: reward / observation / curriculum / event / command functions.

Phase 0.5: `reacher.py` — a from-scratch manager-based env's custom command generator,
observation term, and reward term (proves understanding of the manager framework
independent of anything copied from Isaac Lab's stock tasks).

Phase 1 will add the Go2 flat-locomotion reward stack here, rebuilt term-by-term rather than
imported from `isaaclab_tasks.manager_based.locomotion.velocity.mdp`.
"""

from .reacher import JointTargetCommand, JointTargetCommandCfg, joint_target_distance

__all__ = [
    "JointTargetCommand",
    "JointTargetCommandCfg",
    "joint_target_distance",
]
