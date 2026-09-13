"""Load a trained checkpoint, run N episodes headless, and print velocity-tracking error stats.

Reuses rsl_rl's `OnPolicyRunner` purely for inference (same as Isaac Lab's own `play.py`, whose
loading logic this mirrors), but instead of opening a GUI window it just accumulates the
`base_velocity` command term's own tracking-error metrics and reports mean/std at the end —
the reusable eval tool the guide's Phase 1 checklist asks for, meant to outlive Phase 1
(Phase 2/3 policies can be evaluated the same way).

Usage (from the Isaac Lab venv):
    python scripts/play_policy.py --task QuadrupedDistill-Flat-Go2-StageC-v0 \\
        --checkpoint /path/to/model_499.pt --num-envs 64 --episodes 20
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=20, help="stop once this many episodes have completed")
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

    obs, _ = env.get_observations()
    error_xy_samples: list[float] = []
    error_yaw_samples: list[float] = []
    completed_episodes = 0
    max_steps = env.unwrapped.max_episode_length * (args_cli.episodes // args_cli.num_envs + 2)

    step = 0
    with torch.inference_mode():
        while completed_episodes < args_cli.episodes and step < max_steps:
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

            cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
            error_xy_samples.extend(cmd_term.metrics["error_vel_xy"].tolist())
            error_yaw_samples.extend(cmd_term.metrics["error_vel_yaw"].tolist())
            completed_episodes += int(dones.sum().item())
            step += 1

    xy = torch.tensor(error_xy_samples)
    yaw = torch.tensor(error_yaw_samples)
    print(f"\n{args_cli.task} | checkpoint {resume_path}")
    print(f"steps sampled: {len(error_xy_samples)} (~{completed_episodes} episodes completed)")
    print(f"tracking error xy : mean {xy.mean():.4f} m/s, std {xy.std():.4f}")
    print(f"tracking error yaw: mean {yaw.mean():.4f} rad/s, std {yaw.std():.4f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
