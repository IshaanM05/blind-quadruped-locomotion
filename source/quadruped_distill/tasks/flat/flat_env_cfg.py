"""Phase 1: Go2 flat-ground velocity tracking — a from-scratch `ManagerBasedRLEnvCfg`.

Built directly from `ManagerBasedRLEnvCfg`, NOT by subclassing Isaac Lab's
`LocomotionVelocityRoughEnvCfg` (that would just be tweaking the stock task — the guide's
explicit "tutorial-clone trap" warning). Isaac Lab's stock `Isaac-Velocity-Flat-Unitree-Go2-v0`
is used only as a baseline reference run (`notes/experiment_log.md`), not as a base class.

Generic managers (commands, actions, observations, terminations, the minimal reset events) reuse
Isaac Lab's `isaaclab.envs.mdp` building blocks — that's infrastructure, not a design choice worth
re-deriving. The REWARDS are our own (`quadruped_distill.mdp.locomotion`), per the guide's
instruction to rebuild that section term-by-term.

No curriculum, no domain-randomization events (`physics_material`, `add_base_mass`,
`base_external_force_torque`, `push_robot`) — that's Phase 2's job (`tasks/rough/`). Only a
minimal reset-time pose/joint randomization is kept, which is basic episode-restart hygiene, not
systematic domain randomization.

Three stages (`docs/20_locomotion/README.md`), each a full env cfg registered as its own gym ID
so each can be trained and compared independently:
    Stage A (`FlatGo2StageAEnvCfg`): task-tracking rewards only — expect a twitchy/broken gait.
    Stage B (`FlatGo2StageBEnvCfg`): + regularization penalties.
    Stage C (`FlatGo2StageCEnvCfg`): + gait shaping (feet_air_time, undesired_contacts) — target.
"""

from __future__ import annotations

import math

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
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from quadruped_distill import mdp as own_mdp
from quadruped_distill.assets import UNITREE_GO2_CFG


@configclass
class FlatSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class CommandsCfg:
    base_velocity = base_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=base_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0)
        ),
    )


@configclass
class ActionsCfg:
    joint_pos = base_mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=base_mdp.projected_gravity)
        velocity_commands = ObsTerm(func=base_mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=base_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=base_mdp.joint_vel_rel)
        last_action = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Basic episode-restart hygiene only — no systematic domain randomization (Phase 2)."""

    reset_base = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
        },
    )
    reset_robot_joints = EventTerm(
        func=base_mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.9, 1.1), "velocity_range": (0.0, 0.0)},
    )


@configclass
class RewardsCfg:
    # stage A -- task
    track_lin_vel_xy_exp = RewTerm(
        func=own_mdp.track_lin_vel_xy_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=own_mdp.track_ang_vel_z_exp, weight=0.75, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # stage B -- regularization
    lin_vel_z_l2 = RewTerm(func=own_mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=own_mdp.ang_vel_xy_l2, weight=-0.05)
    joint_torques_l2 = RewTerm(func=own_mdp.joint_torques_l2, weight=-0.0002)
    joint_acc_l2 = RewTerm(func=own_mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=own_mdp.action_rate_l2, weight=-0.01)
    flat_orientation_l2 = RewTerm(func=own_mdp.flat_orientation_l2, weight=-2.5)
    # stage C -- gait shaping
    feet_air_time = RewTerm(
        func=own_mdp.feet_air_time,
        weight=0.25,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "command_name": "base_velocity", "threshold": 0.5},
    )
    undesired_contacts = RewTerm(
        func=own_mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"), "threshold": 1.0},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=base_mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class FlatGo2EnvCfg(ManagerBasedRLEnvCfg):
    """Stage C (full reward stack) — the "current best" env; also the base the A/B stage cfgs
    subclass and prune, mirroring how Isaac Lab's own `_PLAY` variants subclass-and-prune."""

    scene: FlatSceneCfg = FlatSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class FlatGo2StageAEnvCfg(FlatGo2EnvCfg):
    """Task-tracking rewards only — no regularization, no gait shaping."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.lin_vel_z_l2 = None
        self.rewards.ang_vel_xy_l2 = None
        self.rewards.joint_torques_l2 = None
        self.rewards.joint_acc_l2 = None
        self.rewards.action_rate_l2 = None
        self.rewards.flat_orientation_l2 = None
        self.rewards.feet_air_time = None
        self.rewards.undesired_contacts = None


@configclass
class FlatGo2StageBEnvCfg(FlatGo2EnvCfg):
    """Task + regularization — no gait shaping yet."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_air_time = None
        self.rewards.undesired_contacts = None
