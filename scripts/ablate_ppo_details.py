"""Ablate 2 of the "37 implementation details" in ppo_continuous.py on Pendulum-v1.

Runs the baseline (all details on) plus one run per ablated detail, each across N seeds, and
prints a comparison table. See docs/00_foundations/04_ppo.md and notes/experiment_log.md for
the recorded results.

Usage:
    python scripts/ablate_ppo_details.py --seeds 3
"""

from __future__ import annotations

import argparse
import statistics

from quadruped_distill.algorithms.ppo.ppo_continuous import train

CONFIGS = {
    "baseline": {},
    "no-adv-norm": {"adv_norm": False},
    "no-lr-anneal": {"lr_anneal": False},
}

DEFAULTS = dict(
    total_steps=400_000,
    num_envs=4,
    num_steps=256,
    gamma=0.9,
    lam=0.95,
    clip=0.2,
    epochs=10,
    minibatches=32,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad=0.5,
    lr=3e-4,
    device="cpu",
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument(
        "--log-dir",
        default="runs/ablation",
        help="tensorboard log root; run `tensorboard --logdir <this>` to watch live",
    )
    args = p.parse_args()

    print(f"{'config':<28} {'mean':>8} {'std':>8}  gate(>=-250)")
    print("-" * 60)
    for name, overrides in CONFIGS.items():
        cfg = {**DEFAULTS, **overrides}
        finals = [
            train(seed=s, log_dir=f"{args.log_dir}/{name}/seed{s}", **cfg)
            for s in range(args.seeds)
        ]
        mean = statistics.mean(finals)
        spread = statistics.pstdev(finals) if len(finals) > 1 else 0.0
        gate = "PASS" if mean >= -250 else "FAIL"
        print(f"{name:<28} {mean:>8.1f} {spread:>8.1f}  {gate}")


if __name__ == "__main__":
    main()
