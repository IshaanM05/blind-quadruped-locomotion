"""Thin wrapper around Isaac Lab's own rsl_rl play.py that registers this project's own gym
tasks first — same reasoning as `train_rsl_rl.py` (see its docstring).

Usage (from the Isaac Lab venv, drop --headless to get a GUI window):
    python scripts/play_rsl_rl.py --task QuadrupedDistill-Flat-Go2-StageA-v0 --num_envs 16
"""

import os
import runpy
import sys

import quadruped_distill.tasks  # noqa: F401  (registers QuadrupedDistill-* gym IDs)

_ISAACLAB_PLAY_PY = "/home/ishaan/Desktop/IsaacLab-Proj/IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py"

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(_ISAACLAB_PLAY_PY))
    runpy.run_path(_ISAACLAB_PLAY_PY, run_name="__main__")
