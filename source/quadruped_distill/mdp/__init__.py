"""Custom MDP terms: reward / observation / curriculum / event / command functions.

Phase 0.5: `reacher.py` — a from-scratch manager-based env's custom command generator,
observation term, and reward term (proves understanding of the manager framework
independent of anything copied from Isaac Lab's stock tasks).

Phase 1: `locomotion.py` — the Go2 flat-locomotion reward stack, rebuilt term-by-term rather
than imported from `isaaclab_tasks.manager_based.locomotion.velocity.mdp`.
"""

from .locomotion import (
    action_rate_l2,
    ang_vel_xy_l2,
    feet_air_time,
    flat_orientation_l2,
    joint_acc_l2,
    joint_torques_l2,
    lin_vel_z_l2,
    track_ang_vel_z_exp,
    track_lin_vel_xy_exp,
    undesired_contacts,
)
from .reacher import JointTargetCommand, JointTargetCommandCfg, joint_target_distance

__all__ = [
    "JointTargetCommand",
    "JointTargetCommandCfg",
    "joint_target_distance",
    "track_lin_vel_xy_exp",
    "track_ang_vel_z_exp",
    "lin_vel_z_l2",
    "ang_vel_xy_l2",
    "joint_torques_l2",
    "joint_acc_l2",
    "action_rate_l2",
    "flat_orientation_l2",
    "feet_air_time",
    "undesired_contacts",
]
