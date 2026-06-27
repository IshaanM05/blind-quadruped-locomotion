# MDPs, returns, and Bellman equations

**What.** The mathematical frame for every decision-making problem in this project: an agent
acting in an environment to maximize cumulative reward.

**Why it exists.** Before we can *optimize* a policy we need to define precisely what we are
optimizing. The Markov Decision Process (MDP) is that definition. Every later concept — value
functions, advantages, PPO's objective — is built on these few symbols.

**The core idea in one sentence.** Maximize the expected discounted sum of future rewards,
exploiting the Markov property so that the *current state* is a sufficient summary of history.

---

## The MDP

An MDP is the tuple $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$:

- $\mathcal{S}$ — states. For Go2: base velocity, orientation, joint angles, … (the obs vector).
- $\mathcal{A}$ — actions. For Go2: 12 joint position targets.
- $P(s' \mid s, a)$ — transition dynamics (the simulator). **We never differentiate through this.**
- $r(s, a)$ — reward. The thing we *design* (Phase 1 is mostly reward design).
- $\gamma \in [0,1)$ — discount factor (0.99 here).

**Markov property:** $P(s_{t+1} \mid s_t, a_t) = P(s_{t+1} \mid s_0,a_0,\dots,s_t,a_t)$. The
future depends on the past *only through the present state*. This is an assumption we *engineer*
into the observation — and a key tension in this project: a blind robot's instantaneous
proprioception is **not** Markov w.r.t. the terrain, which is exactly why the Phase 3 student
needs **memory** (a recurrent net) to reconstruct a sufficient state. Hold onto that thread.

## Return — what we actually maximize

The **discounted return** from time $t$:

$$G_t = \sum_{k=0}^{\infty} \gamma^k\, r_{t+k+1}$$

Why discount? (1) Convergence — bounds the infinite sum. (2) It encodes "sooner is better" and
hedges model uncertainty about the far future. $\gamma=0.99$ gives an effective horizon of
$\sim 1/(1-\gamma)=100$ steps.

## Value functions

- **State value** $V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]$ — how good is it to be in $s$ under policy $\pi$.
- **Action value** $Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid s_t=s, a_t=a]$ — how good is taking $a$ in $s$, then following $\pi$.
- **Advantage** $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$ — how much better than average is $a$. **This is the quantity PPO pushes on.**

## Bellman equations

**Expectation** (value of *this* policy — recursive consistency):

$$V^\pi(s) = \mathbb{E}_{a\sim\pi,\, s'\sim P}\big[r(s,a) + \gamma V^\pi(s')\big]$$

**Optimality** (value of the *best* policy):

$$V^*(s) = \max_a \mathbb{E}_{s'}\big[r(s,a) + \gamma V^*(s')\big]$$

The expectation equation is what our critic learns to satisfy (TD learning bootstraps on it).
The optimality equation underlies value-based methods (DQN) — which we *don't* use, and
[04_ppo](04_ppo.md) explains why policy-gradient methods suit continuous control better.

## How it appears in this repo

- $\gamma$, the value/advantage estimates: `source/quadruped_distill/algorithms/ppo/common.py`.
- The reward $r(s,a)$ is *designed*, not given — Phase 1, `docs/20_locomotion/`.
- The Markov-violation-by-blindness motivates Phase 3 recurrence — `docs/40_distillation/`.

## Explain-it-back

1. State the Markov property and give one way the Go2 *blind* observation violates it.
2. Why is the advantage, not the raw return, the natural learning signal? (Preview of variance — [03](03_variance_and_gae.md).)
3. Derive the Bellman expectation equation for $V^\pi$ from the definition of $G_t$.
4. What does $\gamma \to 1$ do to the effective horizon and to gradient variance?
