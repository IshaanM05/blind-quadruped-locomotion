"""Phase 2: terrain curriculum + privileged (asymmetric-critic) observations.

Three things live here that Isaac Lab's stock building blocks don't provide as-is:

1. `terrain_levels` — our own reimplementation of `terrain_levels_vel`'s promote/demote
   algorithm (the guide's explicit instruction: read it, then write your own version).
2. `randomize_friction_and_record` / `push_and_record` — DR events that ALSO cache the sampled
   value into a lazily-created per-env buffer on the env, because the stock
   `randomize_rigid_body_material` (bucketed, per-shape) and `push_by_setting_velocity` (fire-
   and-forget) don't expose what they sampled — needed so the asymmetric critic can observe
   "current friction" / "last push" as privileged state. Same cached-state pattern as Phase 0.5's
   `JointTargetCommand`.
3. `privileged_friction` / `privileged_push_velocity` / `feet_contact_bool` — the paired
   observation functions the critic group reads.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_ROBOT_ASSET_CFG = SceneEntityCfg("robot")


# --- curriculum ---------------------------------------------------------------------------


def terrain_levels(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = _ROBOT_ASSET_CFG
) -> torch.Tensor:
    """Promote a terrain level if the robot walked past half a terrain tile; demote if it walked
    less than half its commanded-achievable distance. Own reimplementation of `terrain_levels_vel`
    (isaaclab_tasks/.../velocity/mdp/curriculums.py) — same algorithm, written independently.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down &= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


# --- DR events that cache what they sampled, for the critic to read back -------------------


def _priv_buffer(env: ManagerBasedRLEnv, name: str, dim: int) -> torch.Tensor:
    """Lazily create/return a (num_envs, dim) buffer stored on the env instance."""
    if not hasattr(env, name):
        setattr(env, name, torch.zeros(env.num_envs, dim, device=env.device))
    return getattr(env, name)


def randomize_friction_and_record(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    static_friction_range: tuple[float, float],
    dynamic_friction_range: tuple[float, float],
    restitution_range: tuple[float, float],
) -> None:
    """Assign ONE (static, dynamic, restitution) triple per env (not Isaac Lab's stock
    per-shape/bucketed scheme — simpler, and lets us cache exactly one friction value per env
    for the privileged observation) and cache the static-friction value for `privileged_friction`.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    ranges = torch.tensor([static_friction_range, dynamic_friction_range, restitution_range], device="cpu")
    samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu")

    total_num_shapes = asset.root_physx_view.max_shapes
    materials = asset.root_physx_view.get_material_properties()
    materials[env_ids] = samples.unsqueeze(1).expand(-1, total_num_shapes, -1)
    asset.root_physx_view.set_material_properties(materials, env_ids)

    friction_buf = _priv_buffer(env, "_priv_friction", 1)
    friction_buf[env_ids.to(env.device)] = samples[:, 0:1].to(env.device)


def push_and_record(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = _ROBOT_ASSET_CFG,
) -> None:
    """Same effect as `isaaclab.envs.mdp.push_by_setting_velocity`, but also caches the sampled
    push velocity for `privileged_push_velocity` (the stock function discards it)."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    vel_w = asset.data.root_vel_w[env_ids]
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    delta = sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
    asset.write_root_velocity_to_sim(vel_w + delta, env_ids=env_ids)

    push_buf = _priv_buffer(env, "_priv_push_vel", 6)
    push_buf[env_ids] = delta


# --- privileged observation functions -------------------------------------------------------


def privileged_friction(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _priv_buffer(env, "_priv_friction", 1)


def privileged_push_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _priv_buffer(env, "_priv_push_vel", 6)


def feet_contact_bool(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Binary contact state per foot — privileged (a real robot may lack reliable contact
    sensing; the teacher's critic gets ground truth from the simulator's contact sensor)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    return (torch.max(torch.norm(net_forces, dim=-1), dim=1)[0] > 1.0).float()
