"""Custom MDP terms: reward / observation / curriculum / event / command functions.

Phase 0.5: `reacher.py` — a from-scratch manager-based env's custom command generator,
observation term, and reward term (proves understanding of the manager framework
independent of anything copied from Isaac Lab's stock tasks).

Phase 1: `locomotion.py` — the Go2 flat-locomotion reward stack, rebuilt term-by-term rather
than imported from `isaaclab_tasks.manager_based.locomotion.velocity.mdp`.

Phase 2: `privileged.py` — terrain curriculum (own reimplementation of `terrain_levels_vel`) and
the asymmetric-critic's privileged observation functions.
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
from .privileged import (
    feet_contact_bool,
    privileged_friction,
    privileged_push_velocity,
    push_and_record,
    randomize_friction_and_record,
    terrain_levels,
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
    "terrain_levels",
    "randomize_friction_and_record",
    "push_and_record",
    "privileged_friction",
    "privileged_push_velocity",
    "feet_contact_bool",
]
