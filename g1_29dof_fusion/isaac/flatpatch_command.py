"""Flat-patch-targeted velocity command generator.

Ported/simplified from project-instinct/InstinctLab's PoseVelocityCommand
(source/instinctlab/instinctlab/tasks/parkour/mdp/commands/pose_velocity_command.py, referenced by the
"Hiking in the Wild" paper: https://project-instinct.github.io/hiking-in-the-wild/), which is itself built
on IsaacLab's stock TerrainBasedPose2dCommand (isaaclab/envs/mdp/commands/pose_2d_command.py).

Motivation: UniformVelocityCommand samples an arbitrary velocity command with no regard for whether the
robot can actually reach anywhere with it (e.g. it may point straight at a gap or a wall). TargetVelocityCommand
instead samples a real, precomputed flat-patch waypoint from the terrain (terrain.flat_patches["target"],
populated by FlatPatchSamplingCfg on each sub-terrain) and P-controls a velocity command toward it - this is
the "eliminate reward hacking via feasible targets" idea from the paper. Output shape/semantics match
UniformVelocityCommand's `command` property ([lin_vel_x, lin_vel_y, ang_vel_z] in the robot base frame), so
existing reward/observation terms (track_lin_vel_xy_yaw_frame_exp, generated_commands, etc.) work unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.terrains import TerrainImporter
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class TargetVelocityCommand(CommandTerm):
    """Velocity command generator that tracks a randomly sampled flat-patch waypoint."""

    cfg: "TargetVelocityCommandCfg"

    def __init__(self, cfg: "TargetVelocityCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.terrain: TerrainImporter = env.scene["terrain"]
        if "target" not in self.terrain.flat_patches:
            raise RuntimeError(
                "TargetVelocityCommand requires the terrain generator's sub-terrains to define a 'target'"
                f" flat_patch_sampling entry. Found: {list(self.terrain.flat_patches.keys())}"
            )
        # valid targets: (terrain_level, terrain_type, num_patches, 3)
        self.valid_targets: torch.Tensor = self.terrain.flat_patches["target"]

        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.max_command_b = torch.zeros(self.num_envs, 2, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["target_dist"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return (
            "TargetVelocityCommand (flat-patch waypoint tracking):\n"
            f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
            f"\tResampling time range: {self.cfg.resampling_time_range}"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )
        self.metrics["target_dist"] = torch.norm((self.pos_command_w - self.robot.data.root_pos_w)[:, :2], dim=1)

    def _resample_command(self, env_ids: Sequence[int]):
        # sample a reachable flat-patch waypoint from the terrain
        ids = torch.randint(0, self.valid_targets.shape[2], size=(len(env_ids),), device=self.device)
        self.pos_command_w[env_ids] = self.valid_targets[
            self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids], ids
        ]
        # sample this episode's max approach speed (per-axis) toward that waypoint
        r = torch.empty(len(env_ids), device=self.device)
        self.max_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        self.max_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        """Re-target the velocity command to point at the current waypoint from the current root state."""
        target_vec_w = self.pos_command_w - self.robot.data.root_pos_w
        target_dist = torch.norm(target_vec_w[:, :2], dim=1)
        target_vec_b = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w), target_vec_w)

        desired_xy = target_vec_b[:, :2] * self.cfg.velocity_control_stiffness
        max_xy = torch.abs(self.max_command_b)
        self.vel_command_b[:, :2] = torch.clamp(desired_xy, min=-max_xy, max=max_xy)

        target_direction = torch.atan2(target_vec_w[:, 1], target_vec_w[:, 0])
        heading_error = wrap_to_pi(target_direction - self.robot.data.heading_w)
        self.vel_command_b[:, 2] = torch.clamp(
            heading_error * self.cfg.heading_control_stiffness, *self.cfg.ranges.ang_vel_z
        )

        # stop once close enough to the waypoint (a new one is sampled at the next resample tick)
        reached_env_ids = (target_dist < self.cfg.target_dist_threshold).nonzero(as_tuple=False).flatten()
        self.vel_command_b[reached_env_ids, :] = 0.0

        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
        return arrow_scale, arrow_quat


@configclass
class TargetVelocityCommandCfg(CommandTermCfg):
    """Configuration for TargetVelocityCommand."""

    class_type: type = TargetVelocityCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    heading_control_stiffness: float = 0.5
    """Scale factor converting heading-to-target error (rad) into an angular velocity command."""

    velocity_control_stiffness: float = 0.5
    """Scale factor converting position error (m, base frame) into a linear velocity command."""

    target_dist_threshold: float = 0.5
    """Distance (m) within which the waypoint is considered reached and the command is zeroed."""

    rel_standing_envs: float = 0.0
    """The sampled probability of environments that should be standing still. Defaults to 0.0."""

    @configclass
    class Ranges:
        """Bounds for the per-episode max approach speed and for the angular velocity command."""

        lin_vel_x: tuple[float, float] = MISSING
        lin_vel_y: tuple[float, float] = MISSING
        ang_vel_z: tuple[float, float] = MISSING

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
