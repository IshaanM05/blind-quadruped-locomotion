# Policy gradients, derived properly

**What.** A way to improve a *parameterized* policy $\pi_\theta(a\mid s)$ by gradient ascent on
expected return — without ever differentiating the environment.

**Why it exists.** The policy is a neural net; we want $\nabla_\theta J(\theta)$. But the
return depends on the environment dynamics $P$, which we can't differentiate (it's a simulator,
or reality). The policy gradient theorem rescues us: it expresses the gradient as an
expectation we can estimate from sampled trajectories.

**The core idea in one sentence.** Push up the log-probability of actions that led to
better-than-expected outcomes, weighted by how much better — the "log-derivative trick" turns a
gradient-of-an-expectation into an expectation-of-a-gradient.

---

## The objective

$$J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\big[R(\tau)\big], \qquad R(\tau)=\sum_t \gamma^t r_t$$

where $\tau=(s_0,a_0,s_1,\dots)$ is a trajectory with probability
$p_\theta(\tau)=p(s_0)\prod_t \pi_\theta(a_t\mid s_t)\,P(s_{t+1}\mid s_t,a_t)$.

## The log-derivative trick (the engine)

For any distribution: $\nabla_\theta p_\theta(\tau) = p_\theta(\tau)\,\nabla_\theta \log p_\theta(\tau)$,
because $\nabla \log f = \nabla f / f$. Therefore

$$\nabla_\theta J = \nabla_\theta \int p_\theta(\tau) R(\tau)\,d\tau
= \int p_\theta(\tau)\,\nabla_\theta \log p_\theta(\tau)\,R(\tau)\,d\tau
= \mathbb{E}_\tau\big[\nabla_\theta \log p_\theta(\tau)\, R(\tau)\big].$$

Now the **key cancellation**: $\log p_\theta(\tau) = \log p(s_0) + \sum_t \log \pi_\theta(a_t\mid s_t) + \sum_t \log P(\cdots)$.
The dynamics terms $p(s_0)$ and $P$ **don't depend on $\theta$**, so their gradient is zero:

$$\boxed{\nabla_\theta J = \mathbb{E}_\tau\Big[\sum_t \nabla_\theta \log \pi_\theta(a_t\mid s_t)\; R(\tau)\Big]}$$

**This is why we never differentiate the simulator.** That single fact is the whole reason
model-free RL is possible. (Interview question #1.)

## Three refinements (each a valid, lower-variance estimator)

1. **Reward-to-go.** An action at time $t$ can't affect rewards before $t$. Replace $R(\tau)$
   with $\hat G_t = \sum_{k\ge t}\gamma^{k-t} r_k$. Same expectation, less variance.
2. **Baseline.** Subtract any function $b(s_t)$ that doesn't depend on the action:
   $\nabla_\theta J = \mathbb{E}\big[\sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)\,(\hat G_t - b(s_t))\big]$.
   **It's unbiased** because $\mathbb{E}_{a\sim\pi}[\nabla_\theta\log\pi_\theta(a\mid s)\,b(s)] = b(s)\,\nabla_\theta\!\sum_a \pi_\theta(a\mid s) = b(s)\,\nabla_\theta 1 = 0$.
   The variance-minimizing choice is $b(s)=V^\pi(s)$.
3. **Advantage.** With $b=V^\pi$, the weight becomes $\hat G_t - V(s_t) \approx A^\pi(s_t,a_t)$.
   So the practical gradient is $\mathbb{E}[\sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)\,A_t]$.

## The algorithm ladder this repo walks

| Algorithm | Weight on $\nabla\log\pi$ | File |
|---|---|---|
| REINFORCE | full-trajectory return $R(\tau)$ | `algorithms/ppo/reinforce.py` |
| + baseline | $\hat G_t - V(s_t)$ | `algorithms/ppo/reinforce_baseline.py` |
| A2C | bootstrapped advantage (n-step/TD) | `algorithms/ppo/a2c.py` |
| PPO | clipped surrogate on $A_t$ (GAE) | `algorithms/ppo/ppo_discrete.py` |

You will *feel* the variance drop from REINFORCE → +baseline in the seed-spread plots. That
visceral experience is the point of writing all four.

## Explain-it-back

1. Why does the policy gradient avoid differentiating the dynamics? (Show the cancellation.)
2. Prove $\mathbb{E}[\nabla\log\pi \cdot b(s)] = 0$ for an action-independent baseline.
3. Why is reward-to-go a valid (unbiased) estimator?
4. Mechanistically, why does the value baseline reduce variance without adding bias?

## §4.4 concept-check answers (guide questions 1, 2, 6, 8)

**Q1. Why does the policy gradient theorem let us avoid differentiating through the environment
dynamics?** Because $\log p_\theta(\tau) = \log p(s_0) + \sum_t\log\pi_\theta(a_t\mid s_t) +
\sum_t\log P(s_{t+1}\mid s_t,a_t)$, and the dynamics terms $p(s_0)$, $P$ don't depend on $\theta$
— their gradient is exactly zero. See the boxed result above; this cancellation is the entire
reason model-free RL doesn't need a differentiable simulator.

**Q2. What goes wrong with plain REINFORCE that the value baseline fixes? Why doesn't the
baseline bias the gradient?** Plain REINFORCE weights $\nabla\log\pi$ by the full Monte-Carlo
return $\hat G_t$, which sums many random future rewards — high variance, most visible as huge
seed-to-seed spread in the learning curves (`reinforce.py`, no baseline). Subtracting $b(s_t) =
V^\pi(s_t)$ centers that weight around zero without changing its expectation, because for any
$b$ that doesn't depend on the sampled action, $\mathbb{E}_{a\sim\pi}[\nabla_\theta\log\pi_\theta(a\mid s)\,b(s)]
= b(s)\,\nabla_\theta\sum_a\pi_\theta(a\mid s) = b(s)\,\nabla_\theta 1 = 0$ — subtracting zero in
expectation, so no bias, only variance reduction (see "Baseline" above).

**Q6. Why do we add an entropy bonus, and what happens to gait diversity if you set it to 0?**
Deferred to Phase 1. The entropy bonus keeps the action distribution from collapsing to a
deterministic point estimate before the policy has explored enough to find a good gait; setting
$c_2=0$ risks premature convergence to a single, possibly suboptimal behavior. But "what happens
to gait diversity" is an empirical claim about *locomotion*, which doesn't exist yet in Phase
0 — CartPole and Pendulum don't have a notion of "gait." Answering this honestly requires running
the Go2 flat-locomotion task (Phase 1) with entropy coefficient on vs. off and observing the
actual policies, not asserting a result we haven't measured. See `study/prep/progress.md` for the
tracked deferral.

**Q8. Your Week 3 variance plots: explain mechanistically why the baseline reduced variance in
your own runs.** Mechanistically: REINFORCE's per-step gradient weight is the *full* trajectory
return, which is identical for every timestep in an episode regardless of how good that specific
action was — a global, high-magnitude, high-variance signal applied uniformly. The value baseline
replaces it with $\hat G_t - V(s_t)$, a *local*, typically much smaller-magnitude signal (how much
better than expected this state's continuation was), so the same gradient estimator variance
formula ($\mathrm{Var}[X - c]$ minimized at $c=\mathbb{E}[X]$ for the optimal constant baseline)
directly predicts a variance drop when $b\approx V^\pi$.

Honesty check against our own recorded numbers (`notes/experiment_log.md`): REINFORCE was
104.7 ± 34.1 and REINFORCE+baseline was 355.4 ± 41.8 (both 2 seeds). The **mean** jumped exactly
as theory predicts, but the raw **std** did not visibly drop (34.1 → 41.8) — with only 2 seeds,
final-return standard deviation is far too noisy an estimator to demonstrate the effect cleanly
(2 samples barely constrain a std estimate at all). The theoretical variance reduction is about
the *policy-gradient estimator itself* within a run, not necessarily the *across-seed spread of
final returns* — those are related but distinct quantities, and this repo's Phase 0 runs measured
the latter with too few seeds to isolate the former. A cleaner demonstration would log the
gradient norm's variance across minibatches directly, or use ≥10 seeds for the final-return
comparison — neither was done here, so this note says so rather than overclaiming.
