"""From-scratch PPO progression (Phase 0).

Staged, readable reference implementations — read alongside ``docs/00_foundations/``:

* ``common``            — shared building blocks (GAE, MLP, seeding).  [implemented]
* ``reinforce``         — Monte-Carlo policy gradient, no baseline.    [next]
* ``reinforce_baseline``— + learned value baseline (variance drops).   [next]
* ``a2c``               — bootstrapped advantages (the MC↔TD dial).     [next]
* ``ppo_discrete``      — clipped PPO; gate: CartPole-v1 >= 475.        [next]
* ``ppo_continuous``    — Gaussian policy; gate: Pendulum-v1 >= -250.   [next]
"""
