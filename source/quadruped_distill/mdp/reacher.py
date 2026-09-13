"""Phase 0.5 deliverable: a from-scratch manager-based env proving the manager model.

Task: the double-pendulum's two actuated joints (`cart_to_pole`, `pole_to_pendulum`) must reach
and hold a randomly resampled target joint-angle pair — a 2-DOF "reacher" built on an asset
Isaac Lab already ships (`CART_DOUBLE_PENDULUM_CFG`), reconfigured with entirely our own
manager terms rather than copying an existing task cfg (see `docs/10_isaaclab/README.md`).

Exercises the one manager Phase 0's Isaac Lab bridge never needed: **commands** — a term whose
state must persist across steps and resample periodically, unlike a reward/observation/event
function which is stateless. `JointTargetCommand` is a minimal custom `CommandTerm`, modeled on
`isaaclab.envs.mdp.commands.velocity_command.UniformVelocityCommand`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_ROBOT_ASSET_CFG = SceneEntityCfg("robot")


class JointTargetCommand(CommandTerm):
    """Samples a random target angle (rad) for each of `cfg.joint_names`, resampled periodically."""

    cfg: JointTargetCommandCfg

    def __init__(self, cfg: JointTargetCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.joint_ids, _ = self.robot.find_joints(cfg.joint_names)
        self.joint_target = torch.zeros(self.num_envs, len(self.joint_ids), device=self.device)
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return (
            f"JointTargetCommand: joints={self.cfg.joint_names}, "
            f"resample every {self.cfg.resampling_time_range} s"
        )

    @property
    def command(self) -> torch.Tensor:
        return self.joint_target

    def _update_metrics(self):
        current = self.robot.data.joint_pos[:, self.joint_ids]
        self.metrics["error_pos"] = torch.norm(current - self.joint_target, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), len(self.joint_ids), device=self.device)
        for i, (lo, hi) in enumerate(self.cfg.ranges):
            r[:, i].uniform_(lo, hi)
        self.joint_target[env_ids] = r

    def _update_command(self):
        pass  # target holds steady between resamples — nothing to do per-step


@configclass
class JointTargetCommandCfg(CommandTermCfg):
    class_type: type = JointTargetCommand

    asset_name: str = MISSING
    joint_names: list[str] = MISSING
    """Joints to sample a target angle for, in order."""
    ranges: list[tuple[float, float]] = MISSING
    """Per-joint (min, max) target angle range (rad), same order as `joint_names`."""


def joint_target_distance(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = _ROBOT_ASSET_CFG,
) -> torch.Tensor:
    """Negative squared distance between the commanded and actual joint angles — the "point at
    a target" reward. Written from scratch (no equivalent generic term exists in
    `isaaclab.envs.mdp.rewards`, which has no notion of a joint-space target)."""
    command_term: JointTargetCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    current = asset.data.joint_pos[:, command_term.joint_ids]
    return -torch.sum(torch.square(current - command_term.joint_target), dim=-1)
