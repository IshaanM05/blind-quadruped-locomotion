"""Phase 2: rough-terrain teacher — curriculum, domain randomization, asymmetric critic.

Registers three gym IDs (see `rough_env_cfg.py`):
    QuadrupedDistill-Rough-Go2-v0                — full teacher
    QuadrupedDistill-Rough-Go2-NoCurriculum-v0    — ablation: no terrain curriculum
    QuadrupedDistill-Rough-Go2-SymmetricCritic-v0 — ablation: no privileged critic
"""

import gymnasium as gym

from . import agents

gym.register(
    id="QuadrupedDistill-Rough-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:RoughGo2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughGo2PPORunnerCfg",
    },
)

gym.register(
    id="QuadrupedDistill-Rough-Go2-NoCurriculum-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:RoughGo2NoCurriculumEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughGo2PPORunnerCfg",
    },
)

gym.register(
    id="QuadrupedDistill-Rough-Go2-SymmetricCritic-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:RoughGo2SymmetricCriticEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughGo2PPORunnerCfg",
    },
)
