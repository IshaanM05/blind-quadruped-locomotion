"""From-scratch PPO progression (Phase 0).

Staged, readable reference implementations — read alongside ``docs/00_foundations/``:

* ``common``            — shared building blocks (GAE, MLP, seeding).
* ``reinforce``         — Monte-Carlo policy gradient, no baseline.
* ``reinforce_baseline``— + learned value baseline (variance drops).
* ``a2c``               — bootstrapped advantages (the MC↔TD dial).
* ``ppo_discrete``      — clipped PPO; gate CartPole-v1 >= 475: PASS (500.0 ± 0.0, 3 seeds).
* ``ppo_continuous``    — Gaussian policy; gate Pendulum-v1 >= -250: PASS (-198.9 ± 16.0, 3 seeds).
"""
