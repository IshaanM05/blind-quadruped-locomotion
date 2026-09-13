"""Phase 1: Go2 flat-ground velocity tracking, reward stack rebuilt term-by-term.

Registers three gym IDs, one per reward-build stage (see `flat_env_cfg.py`):
    QuadrupedDistill-Flat-Go2-StageA-v0 — task-tracking only
    QuadrupedDistill-Flat-Go2-StageB-v0 — + regularization
    QuadrupedDistill-Flat-Go2-StageC-v0 — + gait shaping (the "current best" cfg)
"""

import gymnasium as gym

from . import agents

gym.register(
    id="QuadrupedDistill-Flat-Go2-StageA-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:FlatGo2StageAEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlatGo2PPORunnerCfg",
    },
)

gym.register(
    id="QuadrupedDistill-Flat-Go2-StageB-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:FlatGo2StageBEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlatGo2PPORunnerCfg",
    },
)

gym.register(
    id="QuadrupedDistill-Flat-Go2-StageC-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:FlatGo2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlatGo2PPORunnerCfg",
    },
)
