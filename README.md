# quadruped-distill

**Learning blind quadruped locomotion** — from RL first principles to a proprioception-only
policy that walks over rough terrain it cannot see, on the Unitree Go2 in
[Isaac Lab](https://isaac-sim.github.io/IsaacLab/).

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

## Results

_Filled in as phases land._

| Metric | Result |
|---|---|
| PPO @ CartPole-v1 (3 seeds) | _≥ 475 target_ |
| PPO @ Pendulum-v1 (3 seeds) | _≥ −250 target_ |
| Go2 flat tracking error | _< 0.2 m/s target_ |
| Teacher: max stair height | _Phase 2_ |
| Student vs teacher on held-out terrain | _Phase 3_ |

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
