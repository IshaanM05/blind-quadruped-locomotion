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
