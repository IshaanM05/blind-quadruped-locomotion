"""Reward ablation suite for the Stage C flat Go2 task (guide §6.4's "6-8 runs" deliverable).

For each major reward term, retrains from scratch with that term's weight zeroed out (via
Hydra's CLI override on the registered env cfg — `env.rewards.<term>.weight=0.0` — rather than
registering a separate gym ID per ablation) and reports the final tracking-error stats via
`play_policy.py`. Also includes the entropy-coefficient ablation the guide ties back to Phase
0's deferred concept-check Q6.

Usage (from the Isaac Lab venv):
    python scripts/ablate_flat_rewards.py --iterations 250
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TASK = "QuadrupedDistill-Flat-Go2-StageC-v0"

# (label, hydra override) — None override = the unablated baseline for comparison
ABLATIONS = [
    ("baseline (all terms)", None),
    ("no action_rate_l2", "env.rewards.action_rate_l2.weight=0.0"),
    ("no flat_orientation_l2", "env.rewards.flat_orientation_l2.weight=0.0"),
    ("no feet_air_time", "env.rewards.feet_air_time.weight=0.0"),
    ("no undesired_contacts", "env.rewards.undesired_contacts.weight=0.0"),
    ("no joint_torques_l2", "env.rewards.joint_torques_l2.weight=0.0"),
    ("no ang_vel_xy_l2", "env.rewards.ang_vel_xy_l2.weight=0.0"),
    ("zero entropy coef", "agent.algorithm.entropy_coef=0.0"),
]


def run_one(label: str, override: str | None, iterations: int, num_envs: int) -> str:
    run_name = label.replace(" ", "_").replace("(", "").replace(")", "")
    cmd = [
        sys.executable,
        "scripts/train_rsl_rl.py",
        "--task",
        TASK,
        "--headless",
        "--num_envs",
        str(num_envs),
        "--max_iterations",
        str(iterations),
        "--run_name",
        run_name,
    ]
    if override:
        cmd.append(override)
    print(f"\n=== {label} ===\n{' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return run_name


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iterations", type=int, default=250, help="shorter than the full 500 — directional comparison")
    p.add_argument("--num-envs", type=int, default=4096)
    args = p.parse_args()

    for label, override in ABLATIONS:
        run_one(label, override, args.iterations, args.num_envs)

    print("\nAll ablation runs done. Evaluate each with scripts/play_policy.py against its")
    print("checkpoint under logs/rsl_rl/quadruped_distill_flat_go2/<run>_<timestamp>/, then")
    print("record results in notes/experiment_log.md and docs/20_locomotion/README.md.")


if __name__ == "__main__":
    main()
