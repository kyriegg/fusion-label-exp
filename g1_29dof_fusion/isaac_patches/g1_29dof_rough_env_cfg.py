# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# 项目专用扩展 (D:\robot paper): G1RoughEnvCfg 的 29-DOF 版本。
#
# 动机: 三策略融合训练 (D:\robot paper\fusion_label_exp\isaac\readme 9.1) 需要把
# rough-terrain 策略、SMP 自然行走策略、Macarena 舞蹈参考轨迹三者的动作按 alpha 加权相加。
# 官方 G1RoughEnvCfg 用 G1_MINIMAL_CFG (37 个执行关节, 含手指), 而 smp 策略和舞蹈参考轨迹
# 都是 29 维 (G1_29DOF_CFG 对应的关节集, 无手指)。直接相加会因为形状不一致而报错，
# 所以这里复制一份 rough_env_cfg.py, 只把机器人换成 G1_29DOF_CFG, 让新策略原生输出 29 维动作。
#
# 与 rough_env_cfg.py 的差异:
#   1. self.scene.robot 用 G1_29DOF_CFG 而不是 G1_MINIMAL_CFG
#   2. joint_deviation_arms 的关节名改成 G1_29DOF_CFG 实际存在的
#      ".*_elbow_joint" / ".*_wrist_.*_joint" (G1_MINIMAL_CFG 用的是
#      ".*_elbow_pitch_joint"/".*_elbow_roll_joint", 29dof 版没有这两个关节)
#   3. joint_deviation_fingers 整项删除 (G1_29DOF_CFG 用的 g1.usd 不含手指关节)
#   4. 加了 smp_style 奖励项 (D:\robot paper\fusion_label_exp\isaac\smp_bridge.py, 把
#      smp 的扩散去噪打分网络接成奖励), 用户 2026-09-10 要求这次重训就解决旧 rough_policy
#      "动作呆板" 的问题, 不等到 readme 9.1 的融合阶段再加
#   5. 2026-09-13: 参考 project-instinct/InstinctLab (Hiking in the Wild 论文,
#      https://project-instinct.github.io/hiking-in-the-wild/) 的 flat-patch-sampling 思路 -
#      base_velocity 指令改用 D:\robot paper\fusion_label_exp\isaac\flatpatch_command.py 的
#      TargetVelocityCommand, 只朝地形上真实存在的可达平坦点走, 而不是随机瞎给一个可能压根走不到
#      的速度指令。同时加了 is_alive 奖励 (跟其他项解耦的"活着就加分"), 两者都是为了压制融合训练
#      里反复出现的"策略靠摔倒也能拿到不错分数"的 reward hacking 症状 (见 fusion 部分的记忆记录)。
#   6. 2026-09-13: 加了简化版 foot volume points 穿透惩罚 (D:\robot paper\fusion_label_exp\isaac\
#      foot_penetration.py) - InstinctLab 原版把体积点挂在脚部刚体上, 检测跟"地形边缘虚拟障碍物"
#      的穿透 (依赖第 5 项之外还没搬的 terrain-edge-detection 机制); 这里简化成脚下 raycaster 网格
#      直接跟地形表面比高度, 不依赖边缘检测, 经用户确认走这个简化路线。

import sys

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import FlatPatchSamplingCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

##
# Pre-defined configs
##
from isaaclab_assets import G1_29DOF_CFG  # isort: skip

_FUSION_ISAAC_DIR = "D:/robot paper/fusion_label_exp/isaac"
if _FUSION_ISAAC_DIR not in sys.path:
    sys.path.insert(0, _FUSION_ISAAC_DIR)
import smp_bridge  # isort: skip
import flatpatch_command  # isort: skip
import foot_penetration  # isort: skip

SMP_CKPT_PATH = "D:/robot paper/smp/datasets/pretrain_ckpt/pretrained_loco.pt"

# stock ROUGH_TERRAINS_CFG doesn't sample any flat patches - add a "target" flat-patch pool to every
# sub-terrain so TargetVelocityCommand has reachable waypoints to aim at on every terrain type.
_TARGET_FLAT_PATCH_CFG = FlatPatchSamplingCfg(num_patches=50, patch_radius=[0.3, 0.5], max_height_diff=0.05)
_ROUGH_TERRAINS_WITH_TARGETS_CFG = ROUGH_TERRAINS_CFG.replace(
    sub_terrains={
        name: sub_cfg.replace(flat_patch_sampling={"target": _TARGET_FLAT_PATCH_CFG})
        for name, sub_cfg in ROUGH_TERRAINS_CFG.sub_terrains.items()
    }
)


@configclass
class G1_29DOF_Rewards(RewardsCfg):
    """Reward terms for the MDP (29-DOF G1, 无手指)."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    # decoupled from every other term (and from track_lin_vel_xy_exp etc. which need real motion to
    # score anything): a flat "staying alive" bonus, so the policy always has a clean incentive not to
    # fall regardless of whatever else it's doing. See InstinctLab parkour reward's own is_alive=3.0.
    is_alive = RewTerm(func=mdp.is_alive, weight=3.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                ],
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*_joint")},
    )
    # smp 扩散去噪打分: exp(-err*ws) in (0,1], 量级跟 track_lin_vel_xy_exp/track_ang_vel_z_exp
    # 差不多, 权重先给 0.5 (给风格明确信号但不压过速度跟踪这个地形策略的本职任务), 需要跑起来看
    # 日志里 rew/smp_style 的分布再调。fixed_timesteps/ws 是 smp 原始默认值 (未调过)。
    smp_style = RewTerm(
        func=smp_bridge.smp_guidance_reward,
        weight=0.5,
        params={"fixed_timesteps": (8, 15, 22), "ws": 4.0, "normalize": True},
    )
    # simplified foot volume points penetration (see foot_penetration.py docstring for the scope cut vs.
    # InstinctLab's original virtual-obstacle-based version). Weight is a first guess, not tuned - check
    # rew/foot_volume_penetration's scale in the first training run's logs same as smp_style was.
    foot_volume_penetration = RewTerm(
        func=foot_penetration.foot_volume_penetration,
        weight=-1.0,
        params={
            "left_scanner_cfg": SceneEntityCfg("left_foot_scanner"),
            "right_scanner_cfg": SceneEntityCfg("right_foot_scanner"),
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
            "sole_offset": 0.03,
        },
    )


@configclass
class G1_29DOF_RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: G1_29DOF_Rewards = G1_29DOF_Rewards()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_generator = _ROUGH_TERRAINS_WITH_TARGETS_CFG
        # G1_29DOF_CFG defaults activate_contact_sensors=False (tuned for locomanipulation,
        # no contact-based rewards/terminations there); rough-terrain feet/illegal-contact
        # rewards and the contact_forces sensor below need it on.
        self.scene.robot.spawn.activate_contact_sensors = True
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # Simplified foot volume points (see foot_penetration.py): a small footprint-sized raycaster grid
        # under each ankle, yaw-aligned so it follows the foot rather than the world frame. offset.pos
        # x=0.04 shifts the grid forward to roughly center it under the sole (ankle joint isn't foot-centered);
        # z=20.0 starts the rays high above so they reliably hit the ground mesh on their way down.
        _foot_scanner_kwargs = dict(
            offset=RayCasterCfg.OffsetCfg(pos=(0.04, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.02, size=(0.16, 0.06)),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
            update_period=0.02,
        )
        self.scene.left_foot_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link", **_foot_scanner_kwargs
        )
        self.scene.right_foot_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link", **_foot_scanner_kwargs
        )

        # This g1.usd asset actually ships 43 actuated joints (29 body + 14 Inspire-hand
        # finger joints), not 29 - restrict the action term to the 29 body joints only, so
        # the action space matches smp / the dance reference trajectories (both 29-dim) for
        # the alpha-weighted fusion in readme 9.1. Hand joints stay at their PD default pose.
        self.actions.joint_pos.joint_names = [
            ".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint",
            ".*_ankle_pitch_joint", ".*_ankle_roll_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint",
            ".*_elbow_joint", ".*_wrist_.*_joint",
        ]

        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # smp guidance reward wiring (see smp_bridge.py docstring for scope/caveats)
        self.events.init_smp_state = EventTerm(
            func=smp_bridge.init_smp_state, mode="startup", params={"ckpt_path": SMP_CKPT_PATH},
        )
        self.events.reset_smp_buffer = EventTerm(func=smp_bridge.reset_smp_buffer, mode="reset")

        # Rewards
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )

        # Commands
        # Flat-patch-targeted velocity command (see flatpatch_command.py docstring) instead of the stock
        # UniformVelocityCommand: samples a real reachable waypoint from the terrain and P-controls a
        # velocity command toward it, rather than a possibly-unreachable random velocity.
        # debug_vis stays False regardless: this isaacsim build's CreateShaderPrimFromSdrCommand dropped
        # the "name" kwarg that isaaclab's debug-vis marker spawning still passes, breaking marker
        # creation on env creation - not needed for headless training anyway.
        self.commands.base_velocity = flatpatch_command.TargetVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(8.0, 12.0),
            debug_vis=False,
            rel_standing_envs=0.02,
            target_dist_threshold=0.5,
            velocity_control_stiffness=0.5,
            heading_control_stiffness=0.5,
            ranges=flatpatch_command.TargetVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 1.0), lin_vel_y=(-0.0, 0.0), ang_vel_z=(-1.0, 1.0),
            ),
        )

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"


@configclass
class G1_29DOF_RoughEnvCfg_PLAY(G1_29DOF_RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # (heading is now derived from the sampled flat-patch target itself, not a separate range)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
