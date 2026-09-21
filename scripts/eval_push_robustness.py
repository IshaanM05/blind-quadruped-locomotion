"""Push-robustness evaluation: sweep push-impulse magnitude, measure survival rate and
time-to-fall at each — the guide's explicit Phase 2 robustness metric ("max impulse survived").

Implemented as a magnitude sweep across many parallel envs (not a literal per-env binary search)
— with 4096 envs, each magnitude already gets thousands of samples in one pass, which is a more
efficient way to find the same "survival drops off around magnitude X" answer. Report which
magnitude survival rate first drops below 50% as the practical "max impulse survived."

Usage (from the Isaac Lab venv, note the required -u — see play_policy.py's docstring for why):
    python -u scripts/eval_push_robustness.py --task QuadrupedDistill-Rough-Go2-v0 \\
        --checkpoint /path/to/model_1499.pt --num-envs 512
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--settle-steps", type=int, default=100, help="steps to walk before applying the push")
parser.add_argument("--post-push-steps", type=int, default=150, help="steps to watch for a fall after the push")
parser.add_argument(
    "--magnitudes", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], help="push speed (m/s) to sweep"
)
parser.add_argument("--headless", action="store_true", default=True)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import quadruped_distill.tasks  # noqa: E402,F401  (registers QuadrupedDistill-* gym IDs)
import torch  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402


def run_at_magnitude(env, policy, magnitude: float, settle_steps: int, post_push_steps: int) -> tuple[float, float]:
    """Reset, walk for `settle_steps`, push every env at `magnitude` m/s in a random xy direction,
    then watch `post_push_steps` for a base-contact termination. Returns (survival_rate,
    mean_time_to_fall_among_fallen)."""
    obs, _ = env.reset()
    with torch.inference_mode():
        for _ in range(settle_steps):
            obs, _, _, _ = env.step(policy(obs))

        num_envs = env.unwrapped.num_envs
        device = env.unwrapped.device
        robot = env.unwrapped.scene["robot"]
        angle = torch.rand(num_envs, device=device) * 2 * torch.pi
        push = torch.zeros(num_envs, 6, device=device)
        push[:, 0] = magnitude * torch.cos(angle)
        push[:, 1] = magnitude * torch.sin(angle)
        vel = robot.data.root_vel_w + push
        robot.write_root_velocity_to_sim(vel)

        fell = torch.zeros(num_envs, dtype=torch.bool, device=device)
        fall_step = torch.full((num_envs,), post_push_steps, device=device)
        for t in range(post_push_steps):
            obs, _, dones, extras = env.step(policy(obs))
            newly_fallen = dones.bool() & ~fell
            fall_step[newly_fallen] = t
            fell |= dones.bool()
            if fell.all():
                break

    survival_rate = 1.0 - fell.float().mean().item()
    fallen_steps = fall_step[fell]
    mean_time_to_fall = fallen_steps.float().mean().item() if len(fallen_steps) > 0 else float("nan")
    return survival_rate, mean_time_to_fall


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device="cuda", num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    resume_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO] loading checkpoint: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    print(f"\n{args_cli.task} | checkpoint {resume_path}")
    print(f"{'push (m/s)':>12} {'survival':>10} {'mean steps-to-fall':>20}")
    max_survived = 0.0
    for magnitude in args_cli.magnitudes:
        survival, ttf = run_at_magnitude(env, policy, magnitude, args_cli.settle_steps, args_cli.post_push_steps)
        print(f"{magnitude:>12.2f} {survival:>10.2%} {ttf:>20.1f}")
        if survival >= 0.5:
            max_survived = magnitude

    print(f"\nmax push magnitude with >=50% survival: {max_survived:.2f} m/s")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
