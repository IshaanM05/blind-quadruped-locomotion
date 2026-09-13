#!/usr/bin/env bash
# Isaac Lab bridge: train Isaac-Cartpole-v0 with our own PPO.
#
# MUST run under the Isaac Lab venv (Isaac Sim + Isaac Lab), NOT this package's own .venv.
# See source/quadruped_distill/algorithms/ppo/ppo_isaac_cartpole.py for the implementation.
set -euo pipefail

source /home/ishaan/Desktop/IsaacLab-Proj/activate_isaaclab.sh
cd "$(dirname "$0")/.."

python -m quadruped_distill.algorithms.ppo.ppo_isaac_cartpole --headless --log-dir runs/isaac_cartpole "$@"
