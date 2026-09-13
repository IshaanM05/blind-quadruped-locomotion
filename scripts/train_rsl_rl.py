"""Thin wrapper around Isaac Lab's own rsl_rl train.py that registers this project's own gym
tasks first.

Isaac Lab's stock `scripts/reinforcement_learning/rsl_rl/train.py` only does
`import isaaclab_tasks` (its own tasks), so our `QuadrupedDistill-*` gym IDs never get
registered when running through it directly. Rather than fork/copy that ~200-line script (with
all its AppLauncher/hydra/checkpoint-resume plumbing) and let it drift from upstream, this file
imports our tasks package for its registration side effect, then runs Isaac Lab's script
unmodified via `runpy` so `sys.argv` still reaches its own argparse untouched.

Usage (from the Isaac Lab venv):
    python scripts/train_rsl_rl.py --task QuadrupedDistill-Flat-Go2-StageA-v0 --headless \\
        --num_envs 4096
"""

import os
import runpy
import sys

import quadruped_distill.tasks  # noqa: F401  (registers QuadrupedDistill-* gym IDs)

_ISAACLAB_TRAIN_PY = "/home/ishaan/Desktop/IsaacLab-Proj/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py"

if __name__ == "__main__":
    # train.py does `import cli_args` (a sibling module), which only resolves if that directory
    # is on sys.path — true when running `python train.py` directly (script dir is sys.path[0]),
    # not when running this wrapper from a different cwd via runpy.
    sys.path.insert(0, os.path.dirname(_ISAACLAB_TRAIN_PY))
    runpy.run_path(_ISAACLAB_TRAIN_PY, run_name="__main__")
