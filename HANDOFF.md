# Handoff — start here in a new session

This file exists so a **fresh chat thread** can pick up this project without re-deriving
context. Read this first, then only dig into the plan/code files it points to as needed —
don't try to read the whole git history or prior conversation.

## What this project is

Blind-to-vision quadruped locomotion (Unitree Go2) via teacher-student distillation in Isaac
Lab, following a self-authored 16-week guide (`/home/ishaan/Desktop/IsaacLab-Proj/isaaclab_quadruped_rl_guide.md`).
Repo: `/home/ishaan/Desktop/IsaacLab-Proj/quadruped-distill/` (git, public on GitHub as
[blind-quadruped-locomotion](https://github.com/IshaanM05/blind-quadruped-locomotion)).

**Environment**: always `source /home/ishaan/Desktop/IsaacLab-Proj/activate_isaaclab.sh` before
running anything Isaac-Lab-related (strips ROS env pollution, sets EULA/cmake vars). Don't use
plain venv activate.

## Where things stand (as of this commit)

- **Phase 0** (from-scratch PPO), **Phase 0.5** (manager-architecture reacher), **Phase 1** (Go2
  flat locomotion, reward stack rebuilt from scratch) — all fully complete, gates passed, pushed.
- **Phase 2** (rough-terrain teacher: curriculum, domain randomization, asymmetric actor-critic)
  — **in progress**. See `/home/ishaan/.claude/plans/get-context-on-this-greedy-platypus.md` for
  the full detailed plan (context, design decisions, milestones 2-A through 2-H).

### Phase 2 status specifically
- Code scaffold DONE and committed: `source/quadruped_distill/tasks/rough/` (own
  `TerrainGeneratorCfg`, height-scanner, full DR `EventCfg`, asymmetric `"critic"` observation
  group), `source/quadruped_distill/mdp/privileged.py` (own terrain curriculum + DR events that
  cache privileged state), `scripts/eval_push_robustness.py`. Three gym IDs registered:
  `QuadrupedDistill-Rough-Go2-v0` (full teacher), `-NoCurriculum-v0`, `-SymmetricCritic-v0`.
- Stock baseline reference run DONE: `Isaac-Velocity-Rough-Unitree-Go2-v0` + rsl_rl, 4096 envs,
  1500 iterations — mean reward 22.77, tracking error 0.375 m/s xy / 0.416 rad/s yaw, mean
  terrain level 5.69/10. Recorded in `notes/experiment_log.md`. **Took ~2h18m wall-clock** —
  terrain+raycaster overhead is real; budget accordingly for remaining runs. User explicitly
  chose to keep full 1500 iterations for all runs (not reduced) despite this cost.
- **One real bug found and fixed**: our `RoughGo2EnvCfg` was missing
  `self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15` (Isaac Lab's stock rough cfg sets this;
  I missed it). Without it, PhysX silently drops contacts past its default buffer size at 4096
  envs on rough terrain ("Patch buffer overflow" errors — a physics-correctness bug, not just a
  perf warning). Fixed in `rough_env_cfg.py::RoughGo2EnvCfg.__post_init__`.
- **Full teacher training** (`QuadrupedDistill-Rough-Go2-v0`, 1500 iters, 4096 envs) — **paused
  mid-run, not finished**. Latest checkpoint: `model_500.pt` in run folder
  `2026-09-21_23-25-57` (i.e. `logs/rsl_rl/quadruped_distill_rough_go2/2026-09-21_23-25-57/model_500.pt`).
  One thing worth knowing: the first long-run attempt hung (process alive, GPU idle, no log
  progress for 30+ min) partway through — likely a CUDA-context issue from a laptop
  suspend/resume during the multi-hour run, not a code bug (the exact same command resumed and
  ran cleanly from the checkpoint afterward). If a future run hangs the same way (log stops
  advancing, `nvidia-smi` shows no real compute activity), it's the same class of issue: kill and
  resume from the latest `model_N.pt`, don't assume the code regressed.
- **Not yet done**: the two ablation runs (no-curriculum, symmetric-critic — both required per
  the guide, expected results: no-curriculum should learn much slower/fail, symmetric-critic
  should show worse sample efficiency), then `eval_push_robustness.py` needs to actually be run
  against the finished teacher, GUI videos across terrain types, and `docs/30_teacher/README.md`'s
  "Will answer" section filled in (same treatment `docs/20_locomotion/README.md` got for Phase 1).

## Resuming: check training state first

Any background process from a prior session is dead once that session/terminal closed — check
before assuming anything is still running:
```bash
ps aux | grep train_rsl_rl
find /home/ishaan/Desktop/IsaacLab-Proj/quadruped-distill/logs/rsl_rl/quadruped_distill_rough_go2 -maxdepth 1
```
If a checkpoint exists but training didn't reach 1500 iterations, either resume from the latest
`model_N.pt` or just restart — checkpoints save every 50 iterations (`save_interval=50`) so
little is lost either way. As of this handoff, the latest checkpoint is
`logs/rsl_rl/quadruped_distill_rough_go2/2026-09-21_23-25-57/model_500.pt` (500/1500 iterations
done). To resume from it:
```bash
source /home/ishaan/Desktop/IsaacLab-Proj/activate_isaaclab.sh
cd /home/ishaan/Desktop/IsaacLab-Proj/quadruped-distill
python scripts/train_rsl_rl.py --task QuadrupedDistill-Rough-Go2-v0 --headless --num_envs 4096 \
    --resume --load_run 2026-09-21_23-25-57 --checkpoint model_500.pt
```
(Or check for a later checkpoint first — `find logs/rsl_rl/quadruped_distill_rough_go2 -name "model_*.pt" | sort`
— in case a later session advanced it further before this handoff was last updated. To start
fully fresh instead, drop `--resume --load_run ... --checkpoint ...`.)
(`scripts/train_rsl_rl.py` is a thin wrapper that registers our `QuadrupedDistill-*` gym IDs
before running Isaac Lab's own rsl_rl `train.py` — required because Isaac Lab's script only
registers its own tasks by default. `scripts/play_rsl_rl.py` is the same pattern for GUI
playback, `scripts/play_policy.py` for headless tracking-error eval.)

**Expect ~2-2.5+ hours wall-clock** for a full 1500-iteration run at 4096 envs on rough terrain
(confirmed from the baseline run) — don't expect quick turnaround, plan the session accordingly.

## After the full teacher finishes

1. Record results in `notes/experiment_log.md` (mean reward, tracking error, mean terrain level,
   compare against the stock baseline numbers above).
2. Run the two ablations (`QuadrupedDistill-Rough-Go2-NoCurriculum-v0`,
   `-SymmetricCritic-v0`) — can use shorter iteration counts for directional evidence, matching
   how Phase 1's reward-ablation suite was done (200 iters vs the full 500).
3. Run `scripts/eval_push_robustness.py` against the teacher's final checkpoint.
4. GUI playback videos (`scripts/play_rsl_rl.py`, no `--headless`) across terrain types.
5. Fill in `docs/30_teacher/README.md`'s "Will answer" section with real findings.
6. Commit each piece separately (matching the existing granular commit history), push.

## Conventions this project follows (don't relitigate these)

- **No AI/Claude attribution** anywhere pushed — commit messages, docs, code comments. (User's
  standing instruction.)
- Guide's core philosophy, repeated at every phase: use Isaac Lab's stock tasks **only as a
  baseline reference run**, never subclass/import them wholesale — rebuild the interesting parts
  (rewards, curriculum, DR composition) yourself. Generic infrastructure (action terms,
  observation getters, event functions) is fine to reuse directly.
- Every trained checkpoint gets smoke-tested (small `num_envs`, few iterations) before a long
  run — this project has hit real bugs this way (Hydra override syntax quirks, `act_scale=inf`,
  stdout buffering with `simulation_app.close()`, the PhysX patch-buffer bug above) cheaply
  rather than after burning an hour of training.
- `docs/WORKLOG.md` (at `/home/ishaan/Desktop/IsaacLab-Proj/docs/WORKLOG.md`, one level up from
  this repo — NOT git-tracked, it's outside this repo) gets a dated entry after substantive
  session work. `notes/experiment_log.md` (inside this repo) gets one row per training run.
- `study/` (inside this repo) is gitignored and must never be pushed — personal study notes only.
- User's memory system (persisted across Claude Code sessions, auto-loaded) already has entries
  for env setup, project goal, worklog convention, disk/artifacts, and remote GPU access via
  molab/marimo pairing (useful for a heavier run than this laptop's RTX 4080 can handle, if
  Phase 3's distillation training ends up needing it).

## After Phase 2: what's next

Per the plan file's "Looking ahead" section, Phase 3 will NOT be a straight ETH-lineage
reproduction — three deliberate differentiators, in dependency order: (1) standard DAgger
teacher→GRU-student baseline first (needed as the comparison point), (2) an RMA-style
(Kumar et al. 2021) online adaptation module compared against it, (3) an alternative student
architecture (state-space/attention vs GRU), (4) sim-to-sim MuJoCo cross-validation across all
trained policies, done last. This needs its own detailed plan once Phase 2's teacher exists.
