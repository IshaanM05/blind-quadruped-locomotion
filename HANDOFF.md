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
- **Full teacher training — DONE, GATE PASS.** `QuadrupedDistill-Rough-Go2-v0`, trained to
  iteration 1900 (stopped manually once mean reward plateaued 22-24 with no trend over ~300
  iterations — well past the 1500-iter stock reference already). `play_policy.py` eval: tracking
  error **0.123 m/s xy / 0.139 rad/s yaw** — comfortably under the guide's <0.2 m/s target, well
  ahead of the stock baseline (0.375/0.416). Checkpoint:
  `logs/rsl_rl/quadruped_distill_rough_go2/2026-09-22_10-54-27/model_1900.pt`.
  - **Gotcha hit**: the run hung once mid-training (process alive, GPU idle, log stalled 30+ min)
    — likely a CUDA-context issue from the laptop suspending/resuming during the multi-hour run,
    not a code bug. Killed and resumed cleanly from the last checkpoint with `--resume --load_run
    <folder> --checkpoint model_N.pt`, no code changes needed. If a future run hangs the same way
    (log stops advancing, `nvidia-smi` shows no real compute), it's the same class of issue.
  - **Gotcha**: rsl_rl's `--resume` treats `max_iterations` as "iterations to run from here," not
    "total target" — resuming from iteration 500 with `max_iterations=1500` targets 2000 total,
    not 1500. Also, resuming writes checkpoints to a **new** timestamped run folder, not the
    original one — check `find logs/rsl_rl/quadruped_distill_rough_go2 -maxdepth 1 -type d` for
    the latest folder, don't assume it's the one you passed to `--load_run`.
  - Since rsl_rl has no built-in early stopping/patience, an external watcher script was used to
    monitor the log and auto-kill on plateau (20-iteration rolling-average reward, 300-iteration
    patience) — see `/tmp/.../scratchpad/early_stop_watch.sh` if reusing this pattern (that path
    is session-local and won't exist in a new session — recreate if needed, it's ~30 lines).
- **Not yet done**: the two ablation runs (no-curriculum, symmetric-critic — both required per
  the guide, expected results: no-curriculum should learn much slower/fail, symmetric-critic
  should show worse sample efficiency), then `eval_push_robustness.py` needs to actually be run
  against the finished teacher, GUI videos across terrain types, and `docs/30_teacher/README.md`'s
  "Will answer" section filled in (same treatment `docs/20_locomotion/README.md` got for Phase 1).

## Resuming: next steps (teacher is DONE, start here)

The full teacher is trained and gated — nothing to resume for it. Any background process from a
prior session is dead once that session/terminal closed, so no need to check for a live
training process. What's left, in order:

1. **Two ablation runs** (both required per the guide):
   ```bash
   source /home/ishaan/Desktop/IsaacLab-Proj/activate_isaaclab.sh
   cd /home/ishaan/Desktop/IsaacLab-Proj/quadruped-distill
   python scripts/train_rsl_rl.py --task QuadrupedDistill-Rough-Go2-NoCurriculum-v0 --headless --num_envs 4096
   python scripts/train_rsl_rl.py --task QuadrupedDistill-Rough-Go2-SymmetricCritic-v0 --headless --num_envs 4096
   ```
   Can use shorter iteration counts (`--max_iterations 400` or similar) for directional evidence
   rather than the full 1500-2000 — matches how Phase 1's reward-ablation suite used 200 iters
   vs the full run's 500. Expected results: no-curriculum should learn much slower or fail
   outright (guide's prediction — confirm the actual curve, don't assume); symmetric-critic
   should show worse sample efficiency than the full teacher's curve.
   - **Watch for the same hang class** noted above on any run over ~30-40 min — if the log stalls
     with the process still alive and GPU idle, kill and resume from the latest checkpoint rather
     than assuming something broke. Consider re-arming an early-stopping watcher (see the pattern
     above) for these too, especially since ablations don't need full convergence — just enough
     to see the trend versus the full teacher.
2. Record both ablation results in `notes/experiment_log.md`, comparing against the teacher's
   0.123/0.139 numbers above.
3. Run `scripts/eval_push_robustness.py` against the teacher's checkpoint:
   ```bash
   python -u scripts/eval_push_robustness.py --task QuadrupedDistill-Rough-Go2-v0 \
       --checkpoint "$(pwd)/logs/rsl_rl/quadruped_distill_rough_go2/2026-09-22_10-54-27/model_1900.pt" \
       --num-envs 512
   ```
   (Note the `-u` — same stdout-buffering gotcha as `play_policy.py`.)
4. GUI playback videos (`scripts/play_rsl_rl.py`, no `--headless`) across terrain types, using
   the same checkpoint.
5. Fill in `docs/30_teacher/README.md`'s "Will answer" section with real findings from the above.
6. Commit each piece separately, push, then Phase 2 is fully closed out — move to Phase 3.

(`scripts/train_rsl_rl.py` is a thin wrapper that registers our `QuadrupedDistill-*` gym IDs
before running Isaac Lab's own rsl_rl `train.py` — required because Isaac Lab's script only
registers its own tasks by default. `scripts/play_rsl_rl.py` is the same pattern for GUI
playback, `scripts/play_policy.py` for headless tracking-error eval.)
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
