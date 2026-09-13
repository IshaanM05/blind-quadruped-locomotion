"""Phase 0.5: a from-scratch `ManagerBasedRLEnvCfg` — every manager wired ourselves, not copied
from an existing Isaac Lab task cfg (see docs/10_isaaclab/README.md for the mental model this
proves out).

Task: the double-pendulum's two actuated joints (`cart_to_pole`, `pole_to_pendulum`) must reach
and hold a randomly resampled target angle pair (`quadruped_distill.mdp.JointTargetCommand`).
The cart joint (`slider_to_cart`) is left uncontrolled (only its shipped passive damping=10
resists drift) — a known simplification, noted rather than hidden.
"""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.cart_double_pendulum import CART_DOUBLE_PENDULUM_CFG

from quadruped_distill import mdp as reacher_mdp

ACTUATED_JOINTS = ["cart_to_pole", "pole_to_pendulum"]


@configclass
class ReacherSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -2.0)),
    )
    robot: ArticulationCfg = CART_DOUBLE_PENDULUM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class ActionsCfg:
    joint_effort = base_mdp.JointEffortActionCfg(asset_name="robot", joint_names=ACTUATED_JOINTS, scale=20.0)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=base_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=base_mdp.joint_vel_rel)
        target = ObsTerm(func=base_mdp.generated_commands, params={"command_name": "target_joint_pos"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class CommandsCfg:
    target_joint_pos = reacher_mdp.JointTargetCommandCfg(
        asset_name="robot",
        joint_names=ACTUATED_JOINTS,
        ranges=[(-1.0, 1.0), (-1.0, 1.0)],  # rad, per joint
        resampling_time_range=(3.0, 5.0),
        debug_vis=False,
    )


@configclass
class RewardsCfg:
    reach_target = RewTerm(func=reacher_mdp.joint_target_distance, weight=1.0, params={"command_name": "target_joint_pos"})
    joint_effort_penalty = RewTerm(
        func=base_mdp.joint_torques_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS)}
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class EventCfg:
    reset_joints = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.1, 0.1),
        },
    )


@configclass
class ReacherEnvCfg(ManagerBasedRLEnvCfg):
    scene: ReacherSceneCfg = ReacherSceneCfg(num_envs=512, env_spacing=3.0, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.viewer.eye = (3.0, 0.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 1.5)
