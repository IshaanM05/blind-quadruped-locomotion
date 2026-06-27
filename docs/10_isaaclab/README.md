# The manager-based environment model (Isaac Lab)

> **Status: outline.** Filled during Phase 0.5. The questions below are what this page will answer.

**What.** How Isaac Lab decomposes an RL environment into declarative, composable config managers
operating on GPU-batched tensors across thousands of parallel envs.

**Why it exists.** To build *custom* environments (Phase 1+) instead of only editing configs, you
must understand the manager model and the "everything is a `(num_envs, …)` tensor, no Python
per-env loops" execution model.

## The mental model (to be expanded with annotated code links)

```
ManagerBasedRLEnvCfg
├── scene          InteractiveSceneCfg   robot, terrain, lights, sensors
├── observations   ObservationGroupCfg   groups: "policy", "critic" (asymmetric — Phase 2)
├── actions        ActionTermCfg         12-d → PD joint targets
├── rewards        RewardTermCfg[]        weighted sum (Phase 1 rebuilds this)
├── terminations   TerminationTermCfg[]   fell over / timeout (truncation!)
├── events         EventTermCfg[]         startup / reset / interval randomization (Phase 2)
├── curriculum     CurriculumTermCfg[]    terrain levels (Phase 2)
└── commands        CommandTermCfg[]       sampled (vx, vy, ωz)
```

Each term is a Python function receiving the **whole `(num_envs,)` batch** and returning a
`(num_envs,)` tensor — vectorized on GPU.

## Will answer
1. What does each of the 8 managers own, and how do they compose at each sim step?
2. Why are there no per-env Python loops, and what does that imply for how you write a reward fn?
3. How does the `"policy"` vs `"critic"` observation-group split enable asymmetric actor-critic?
4. How does Isaac Lab surface `time_outs` so truncation bootstraps correctly? (links to [03_variance_and_gae](../00_foundations/03_variance_and_gae.md))
5. Deliverable: a from-scratch 2-DOF reacher env proving the model is understood.

## Source to read & annotate
`source/isaaclab_tasks/.../locomotion/velocity/velocity_env_cfg.py`, its `mdp/rewards.py`, and
the Go2 configs — then `rsl_rl` `ppo.py` + `on_policy_runner.py` (diff note in `notes/`).
