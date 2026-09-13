"""Phase 1: the Go2 flat-locomotion reward stack, rebuilt term-by-term from a blank slate.

Same reward *math* as Isaac Lab's own locomotion tasks (there's no reason to invent different
physics), but written ourselves rather than imported from `isaaclab_tasks.manager_based.
locomotion.velocity.mdp` — per the guide's explicit instruction (`docs/00_foundations`'s
"tutorial-clone trap" warning): using the stock task as a baseline reference is fine, importing
its reward functions wholesale defeats the point of rebuilding them.

See `docs/20_locomotion/README.md` for the three-stage build (A: task-only, B: +regularization,
C: +gait shaping) these terms are assembled into.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_ROBOT_ASSET_CFG = SceneEntityCfg("robot")


# --- stage A: task ---------------------------------------------------------------------------


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = _ROBOT_ASSET_CFG
) -> torch.Tensor:
    """Exponential reward for tracking commanded (vx, vy) in the robot's body frame.

    exp(-||v_cmd - v_actual||^2 / std^2): 1.0 at perfect tracking, decaying smoothly (not
    a hard threshold) as error grows — a smooth reward is easier for PPO's gradient to climb
    than a sparse/thresholded one.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    error = torch.sum(torch.square(cmd - asset.data.root_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-error / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = _ROBOT_ASSET_CFG
) -> torch.Tensor:
    """Exponential reward for tracking commanded yaw rate ωz. Same shape as the linear-vel term."""
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, 2]
    error = torch.square(cmd - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-error / std**2)


# --- stage B: regularization (all penalties, negative weights) -------------------------------


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = _ROBOT_ASSET_CFG) -> torch.Tensor:
    """Penalize vertical (bounce) velocity — the task only commands planar motion."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = _ROBOT_ASSET_CFG) -> torch.Tensor:
    """Penalize roll/pitch angular velocity — discourages rocking/wobbling."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = _ROBOT_ASSET_CFG) -> torch.Tensor:
    """Penalize squared joint torque — energy/actuator-stress proxy."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = _ROBOT_ASSET_CFG) -> torch.Tensor:
    """Penalize squared joint acceleration — discourages jerky, high-frequency motion."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the change in action between consecutive steps.

    The guide calls this "the single biggest contributor to non-jittery motion": without it,
    PPO has no reason to prefer a smooth action trajectory over a noisy one that happens to
    average out to the same reward.
    """
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = _ROBOT_ASSET_CFG) -> torch.Tensor:
    """Penalize non-flat base orientation via the xy-components of projected gravity.

    ``projected_gravity`` is the world gravity vector expressed in the base frame — (0, 0, -1)
    when level, growing nonzero xy components as the base tilts. Penalizing those components
    keeps the trunk upright without needing an explicit angle computation.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


# --- stage C: gait shaping ---------------------------------------------------------------------


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward feet staying airborne past `threshold` seconds on touchdown — the trot-maker.

    Without this, the cheapest way to satisfy the velocity-tracking reward is often a shuffle
    (feet barely leaving the ground) rather than a proper stepping gait. Rewarding air time past
    a threshold at the moment of *first contact* pushes the policy toward genuine steps. Zeroed
    for near-zero commands so standing-still envs aren't penalized for not stepping.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def undesired_contacts(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Penalize contact force above `threshold` on bodies that shouldn't touch the ground
    (thighs/knees) — without this the policy can find gaits that "kneel-walk" on non-foot links.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact.float(), dim=1)
