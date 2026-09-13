# rsl_rl vs our PPO: a diff-style note

**What.** Every place rsl_rl's `PPO` (`algorithms/ppo.py`) and `OnPolicyRunner`
(`runners/on_policy_runner.py`) — the library Isaac Lab's own examples use, and the one this
project's Isaac Lab bridge (`ppo_isaac_cartpole.py`) had to interoperate with — differs from our
from-scratch PPO in `ppo_continuous.py`, and why.

**Why it exists.** Phase 0's own PPO passed both gates without ever reading a production RL
library. Before locomotion (Phase 1) hands training over to rsl_rl, it's worth knowing exactly
which of its choices are "the same math, different code" versus genuinely different design
decisions — so nothing rsl_rl does later looks unfamiliar or magical.

**The core idea in one sentence.** rsl_rl implements the identical clipped-surrogate PPO loss we
derived from scratch, but replaces our fixed linear LR anneal with an **adaptive, KL-driven**
schedule, and is architected from the ground up for **asymmetric actor-critic** and
**teacher-student distillation** — two things our Phase 0 code never needed.

---

## The headline diff: adaptive KL-based learning rate

Our `ppo_continuous.py` anneals LR linearly and unconditionally:
```python
opt.param_groups[0]["lr"] = lr * (1.0 - update / num_updates)
```
This is schedule-blind: it doesn't look at how the policy is actually behaving, just how far
through training we are.

rsl_rl's default (`schedule="adaptive"`, `desired_kl=0.01`) instead computes the actual KL
divergence between the old and new policy after each minibatch update (`ppo.py:269-303`), using
the closed-form diagonal-Gaussian KL:

```python
kl = torch.sum(
    torch.log(sigma_new / sigma_old + 1e-5)
    + (sigma_old**2 + (mu_old - mu_new)**2) / (2 * sigma_new**2)
    - 0.5,
    axis=-1,
)
```

then adjusts the learning rate multiplicatively based on how far that KL is from a target:

```python
if kl_mean > desired_kl * 2.0:
    learning_rate = max(1e-5, learning_rate / 1.5)   # moved too far — shrink LR
elif kl_mean < desired_kl / 2.0:
    learning_rate = min(1e-2, learning_rate * 1.5)   # barely moved — grow LR
```

**Why this matters more than it looks:** a fixed anneal schedule doesn't know the difference
between "training is going smoothly" and "the policy is oscillating wildly" — it just decays on
a clock. The adaptive scheme directly targets the quantity PPO's clip is *approximating* control
over (how far the policy moved), and self-corrects in both directions: it shrinks LR after a
destructive-looking update *and* grows it back when updates go stale. This is the actual
trust-region idea TRPO used a hard constraint for, applied here as a soft, cheap feedback loop
on top of PPO's clip — not a replacement for it.

## Second diff: asymmetric actor-critic (privileged critic observations)

Our `GaussianActorCritic.act()` takes one observation and feeds it to both the actor and the
critic. rsl_rl's `PPO.act()` takes two: `act(self, obs, critic_obs)` (`ppo.py:136`), and stores
both in the rollout (`transition.privileged_observations = critic_obs`, `ppo.py:147`). The critic
can see privileged simulator state the actor (and the real robot) never will — exact terrain
height, contact forces, friction coefficients — while the actor only sees what's actually
observable. This is exactly the asymmetric-actor-critic pattern the locomotion phases will use:
better value estimates (hence better advantages) without cheating on what the deployed policy
can act on. Our Phase 0 tasks had no privileged information to exploit, so this distinction never
came up.

## Third diff: distillation is a first-class algorithm, not bolted on

rsl_rl ships a separate `algorithms/distillation.py` implementing `Distillation`, operating on a
`StudentTeacher` or `StudentTeacherRecurrent` policy pair (`rsl_rl.modules`). This is *exactly*
Phase 3's DAgger-style teacher→recurrent-student pipeline — rsl_rl already has the plumbing
(rollout storage, gradient truncation via `gradient_length`, MSE/other student-teacher losses)
that this project will otherwise have had to write from scratch. Worth reading closely before
Phase 3 starts rather than reimplementing.

## What's the same

The clipped surrogate objective, value-loss clipping, orthogonal init, global grad-norm
clipping, entropy bonus, and GAE itself are the same math we already implemented and tested in
Phase 0 — rsl_rl's version is just batched across thousands of GPU-resident envs instead of a
handful of CPU Gymnasium envs. The Isaac Lab bridge (`ppo_isaac_cartpole.py`) proved this
directly: our unmodified `GaussianActorCritic` reached a comparable reward (4.78 vs rsl_rl's
4.95) on the identical task and training budget — the loss function was never the hard part.

## How it appears in this repo

- `algorithms/ppo/ppo_continuous.py` — our fixed-anneal, single-obs-stream version.
- `algorithms/ppo/ppo_isaac_cartpole.py` — proof our PPO interoperates with Isaac Lab's batched
  env API, the same API rsl_rl trains against.
- `notes/experiment_log.md` — the head-to-head reward comparison.
- Phase 2 (asymmetric critic) and Phase 3 (distillation) will lean on the two diffs above.

## Explain-it-back

1. Why does targeting KL divergence directly generalize better than a fixed LR anneal schedule?
2. Derive the diagonal-Gaussian KL formula rsl_rl uses — where does each term come from?
3. Why is it safe for the critic to see privileged state the actor can't, but not vice versa?
4. What would go wrong if Phase 3's student policy were trained with the teacher's privileged
   observations instead of only what the real robot can sense?
