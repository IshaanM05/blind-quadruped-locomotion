"""Phase 0.5 deliverable: a from-scratch manager-based env (double-pendulum joint-target reacher)."""

import gymnasium as gym

gym.register(
    id="QuadrupedDistill-Reacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reacher_env_cfg:ReacherEnvCfg",
    },
)
