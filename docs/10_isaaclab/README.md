# The manager-based environment model (Isaac Lab)

**What.** How Isaac Lab decomposes an RL environment into declarative, composable config managers
operating on GPU-batched tensors across thousands of parallel envs.

**Why it exists.** To build *custom* environments (Phase 1+) instead of only editing configs, you
must understand the manager model and the "everything is a `(num_envs, …)` tensor, no Python
per-env loops" execution model.

## The mental model

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

## Answered, from building the reacher (`quadruped_distill/tasks/reacher/`, `mdp/reacher.py`)

1. **What does each manager own, and how do they compose per step?** `step()` runs, in order:
   actions (write control targets) → physics substeps (`decimation` of them) → terminations
   (compute done/timeout) → reset terminated envs → events (`interval`-mode ones) → rewards
   (computed *after* the reset, against the *new* state for reset envs — a subtlety worth
   remembering) → observations. Commands resample on their own clock (`resampling_time_range`),
   independent of the step loop. Building `JointTargetCommand` made this concrete: it's the only
   manager whose state must *persist* across steps (a target that holds until resampled) — reward
   and observation functions are stateless, reading whatever the environment/other managers
   currently hold.
2. **Why no per-env Python loops, and what does that imply for reward functions?** Every manager
   term is a plain function `(env, ...) -> Tensor[num_envs, ...]`, called once per step for *all*
   envs simultaneously — GPU-batched from the start. It implies reward/observation/termination
   functions must be written as batched tensor ops (indexing, `torch.sum(..., dim=1)`, no
   Python-level `for env in envs`), exactly the same discipline the Phase 0 Isaac Lab bridge
   already forced on the *rollout loop*; the manager model pushes that same discipline down into
   every individual term.
3. **`"policy"` vs `"critic"` observation groups → asymmetric actor-critic:** not exercised by the
   reacher (single `"policy"` group, symmetric) — deferred to Phase 2 as planned, where the critic
   will see privileged terrain/contact state the actor can't (see
   [01_rsl_rl_diff](01_rsl_rl_diff.md)'s asymmetric actor-critic section for how rsl_rl already
   supports this).
4. **How does Isaac Lab surface `time_outs` for correct bootstrapping?** `step()` returns
   `(obs, reward, terminated, time_outs, extras)` — two separate boolean tensors, not one
   combined `done`. `ppo_isaac_cartpole.py` (reused unmodified for the reacher) already threads
   this correctly: `boot = 1 - terminated` (timeouts still bootstrap), `chain = 1 - (terminated |
   time_outs)` (both break the GAE recursion). See
   [03_variance_and_gae](../00_foundations/03_variance_and_gae.md).
5. **Deliverable — done:** `quadruped_distill/tasks/reacher/` — a double-pendulum (existing Isaac
   Lab asset, `CART_DOUBLE_PENDULUM_CFG`) reconfigured from scratch: a custom `CommandTerm`
   (`JointTargetCommand`, the manager none of Phase 0's bridge work exercised), a custom reward
   (`joint_target_distance`), and generic observation/action/event/termination terms wired
   ourselves — registered as `QuadrupedDistill-Reacher-v0`. Trained with the same own-PPO from
   Phase 0 (generalized to take `--task`, no code duplicated): mean return −87.16 → **−1.19**
   over 300 updates (2.46M env steps, 512 envs) — gate **PASS**. See
   `notes/experiment_log.md`.

## Source to read & annotate
`source/isaaclab_tasks/.../locomotion/velocity/velocity_env_cfg.py`, its `mdp/rewards.py`, and
the Go2 configs — then `rsl_rl` `ppo.py` + `on_policy_runner.py` (diff note: [01_rsl_rl_diff](01_rsl_rl_diff.md)).

## First bridge result (Phase 0 deferred item, done ahead of the formal 0.5 study)

Before writing custom envs, Phase 0 required proving the from-scratch PPO (`ppo_continuous.py`'s
`GaussianActorCritic`, unmodified) works through Isaac Lab's actual batched-tensor Gym interface,
not just Gymnasium's. `algorithms/ppo/ppo_isaac_cartpole.py` wraps `Isaac-Cartpole-v0` (4096
parallel envs, one GPU tensor, no per-env Python loop) and trains with our own rollout/PPO-update
loop instead of rsl_rl's.

**Gate: comparable to rsl_rl's baseline reward on the same task, same config** (4096 envs, 16
steps/env, 150 iterations — rsl_rl's own default `CartpolePPORunnerCfg`):

| | Mean return (last 100 episodes) |
|---|---|
| rsl_rl baseline | 4.95 |
| our PPO | **4.78** (~3.4% below) |

**PASS.** One real Isaac Lab gotcha this surfaced: `Isaac-Cartpole-v0`'s action space is
`Box(-inf, inf)` — Isaac Lab's `JointEffortActionCfg(scale=100.0)` already converts a
normalized policy action into real torque internally, so the policy should output actions in
roughly `[-1, 1]` directly. Naively reading `env.action_space.high` for our own external scaling
(the Gymnasium-style pattern `ppo_continuous.py` uses for Pendulum) gives `inf`, and multiplying
a tanh-squashed action by `inf` corrupts the physics on the very first step. Fix: `act_scale=1.0`
for Isaac Lab manager-/direct-based tasks — the env's own action manager does the unit conversion.
