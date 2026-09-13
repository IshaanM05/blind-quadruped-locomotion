"""Unitree Go2 robot config — re-exported from Isaac Lab's asset zoo.

Phase 1 doesn't need to modify the physical asset (USD, actuator gains) beyond what Isaac Lab
already ships; re-exporting keeps a single, explicit import point for `tasks/flat/` and leaves
room to override actuator gains here later without touching the task cfg, should that become
necessary.
"""

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

__all__ = ["UNITREE_GO2_CFG"]
