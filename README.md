# blind-quadruped-locomotion

**From RL first principles to a proprioception-only policy that walks over rough terrain it
cannot see** — on the Unitree Go2 in [Isaac Lab](https://isaac-sim.github.io/IsaacLab/).

This repo reproduces the modern ETH legged-robotics recipe end to end and **documents every
component well enough to explain it from scratch**. The `docs/` tree is a first-class part of
the project: each concept is derived, then linked to the exact code that implements it.

> Status: **Phase 0–1 in progress.** See the [roadmap](#roadmap) and
> [`notes/experiment_log.md`](notes/experiment_log.md).

---

## The idea in one paragraph

A real legged robot can't measure its own body velocity well and has no map of the ground.
So we train in two stages. First a **teacher** policy that *cheats* — it gets a terrain
height-scanner and clean state — learns to walk over stairs, slopes, and rubble using PPO
with a terrain curriculum and domain randomization. Then we **distill** that teacher into a
**blind student**: a recurrent network that sees only noisy onboard proprioception and must
*infer* the terrain and its own velocity from the history of motion and contacts. This is the
Lee et al. 2020 paradigm, the conceptual core of how quadrupeds like ANYmal walk in the wild.

## Roadmap

| Phase | What | Key artifact |
|---|---|---|
| **0** | RL from first principles — PPO written from a blank file (REINFORCE → A2C → PPO → continuous → Isaac Lab bridge) | `source/.../algorithms/ppo/`, `docs/00_foundations/` |
| **0.5** | Isaac Lab's manager-based env model | `docs/10_isaaclab/` |
| **1** | Flat-ground velocity tracking (Go2 trot), reward stack rebuilt term-by-term | `tasks/flat/`, `docs/20_locomotion/` |
| **2** | Rough terrain + curriculum + domain randomization + asymmetric critic → the **teacher** | `tasks/rough/`, `docs/30_teacher/` |
| **3** | Teacher→student **DAgger distillation** into a recurrent, proprioception-only student | `algorithms/distill/`, `docs/40_distillation/` |
| **4** | *(stretch)* Vision / perceptive locomotion | `docs/50_vision/` |

## Technical highlights

**Phase 0 — PPO from first principles (complete)**

Wrote the full policy-gradient progression from a blank file — no RL library, no `stable-baselines`:

| Stage | Key idea | CartPole mean (3 seeds) |
|---|---|---|
| REINFORCE (MC, no baseline) | log-derivative trick, reward-to-go | 104.7 ± 34.1 |
| REINFORCE + value baseline | E[∇log π · b(s)] = 0 unbiasedness | 355.4 ± 41.8 (+240) |
| A2C (n-step / GAE bootstrap) | bias-variance dial, TD target | 253.5 ± 64.3 |
| **PPO discrete** (CartPole-v1) | clipped surrogate, K-epoch minibatches, adv-norm | **500.0 ± 0.0** ✓ |
| **PPO continuous** (Pendulum-v1) | diagonal Gaussian, learned log-std, tanh-squash Jacobian | **−198.9 ± 16.0** ✓ |

Implementation follows Huang et al. "37 implementation details of PPO" (ICLR 2022): orthogonal
weight init, global grad-norm clipping, value-loss clipping, linear LR annealing, per-batch
advantage normalization. Correct **truncation vs termination** bootstrapping under gymnasium 1.x
`NEXT_STEP` autoreset semantics (separate boot mask and episode-chain mask in GAE).

GAE unit-tested against hand-computed trajectories; CI: ruff + pytest on every push (GitHub
Actions, CPU-only — Phase 0 runs without Isaac Sim).

**Phases 1–3 — in progress (see [roadmap](#roadmap))**

Go2 flat locomotion training in Isaac Lab, followed by rough-terrain teacher with curriculum and
domain randomization, followed by DAgger distillation into a recurrent blind student.

## Results

| Metric | Value | Status |
|---|---|---|
| PPO @ CartPole-v1 (3 seeds, 150k steps) | **500.0 ± 0.0** | ✓ Phase 0 complete |
| PPO @ Pendulum-v1 (3 seeds, 400k steps) | **−198.9 ± 16.0** | ✓ Phase 0 complete |
| Go2 flat velocity tracking error | _target < 0.2 m/s_ | Phase 1 in progress |
| Teacher: max negotiable stair height | _Phase 2_ | — |
| Student vs teacher on held-out terrain | _Phase 3_ | — |

## Quickstart

```bash
# environment (Isaac Lab is installed separately — see docs/10_isaaclab/)
source ../activate_isaaclab.sh
pip install -e .

# Phase 0: from-scratch PPO (no Isaac Sim needed, runs on CPU/GPU)
python -m quadruped_distill.algorithms.ppo.ppo_discrete --seeds 3      # CartPole, target ≥475
python -m quadruped_distill.algorithms.ppo.ppo_continuous              # Pendulum, target ≥-250

# tests & lint
pytest && ruff check .
```

## How this repo is organized

```
source/quadruped_distill/   installable package: algorithms (ppo, distill), tasks, mdp, assets
docs/                       the learning core — derive a concept, then link to its code
notes/                      experiment log (one line per run) + working notes
scripts/                    train / evaluate / record_video entry points
configs/                    one config per experiment (reproducibility)
tests/                      GAE, reward fns, buffer masking — the cheap high-value checks
```

Start reading at [`docs/README.md`](docs/README.md).

## References (the lineage this reproduces)

- Rudin et al. 2021, *Learning to Walk in Minutes Using Massively Parallel Deep RL* (CoRL).
- Lee et al. 2020, *Learning quadrupedal locomotion over challenging terrain* (Science Robotics).
- Schulman et al. 2017, *Proximal Policy Optimization Algorithms* (PPO).
- Schulman et al. 2016, *High-Dimensional Continuous Control Using GAE*.
- Miki et al. 2022, *Learning robust perceptive locomotion* (Science Robotics) — Phase 4.

## License

MIT — see [LICENSE](LICENSE).
