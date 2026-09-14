"""Simplified "foot volume points" penetration reward.

Loosely inspired by project-instinct/InstinctLab's VolumePoints sensor
(source/instinctlab/instinctlab/sensors/volume_points/, referenced by the "Hiking in the Wild" paper:
https://project-instinct.github.io/hiking-in-the-wild/). The original mechanism attaches a dense 3D grid
of points to each foot's rigid body and checks them against registered "virtual obstacle" colliders (their
terrain-edge cylinders) for penetration - porting that faithfully requires also porting their edge-detection
/ virtual-obstacle machinery (PhysX prim queries, trimesh edge extraction), which we're deliberately skipping
here (see project memory reference_hiking_in_the_wild.md and project_robotpaper_fusion.md, 2026-09-13).

Simplified version: instead of a rigid-body-attached 3D point volume checked against virtual obstacles, we
raycast a small footprint grid straight down from each foot (RayCasterCfg with a GridPatternCfg sized to
roughly cover the sole) and compare each sampled ground height to the foot's own height. Whenever the ground
sampled under some part of the footprint sits above where the sole physically is, that part of the foot is
clipping into the terrain (e.g. the toe poking into a step riser) - this is checked continuously, not gated
on the contact sensor, so it catches near-miss clipping before/without a registered contact event.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def compute_foot_volume_penetration(
    left_scanner: RayCaster,
    right_scanner: RayCaster,
    left_sole_z: torch.Tensor,
    right_sole_z: torch.Tensor,
) -> torch.Tensor:
    """Core computation, callable directly (e.g. from G1FusionEnv, which isn't a ManagerBasedRLEnv reward
    term) as well as from the ``foot_volume_penetration`` reward-manager wrapper below.

    Args:
        left_scanner/right_scanner: RayCaster sensors attached to the left/right ankle link, each casting a
            small footprint-sized grid straight down.
        left_sole_z/right_sole_z: (N,) world-frame height of each foot's sole surface.
    """
    left_ground_z = left_scanner.data.ray_hits_w[..., 2]  # (N, P)
    right_ground_z = right_scanner.data.ray_hits_w[..., 2]  # (N, P)
    left_ground_z = torch.where(torch.isinf(left_ground_z), left_sole_z.unsqueeze(-1), left_ground_z)
    right_ground_z = torch.where(torch.isinf(right_ground_z), right_sole_z.unsqueeze(-1), right_ground_z)

    left_penetration = torch.clamp(left_ground_z - left_sole_z.unsqueeze(-1), min=0.0)  # (N, P)
    right_penetration = torch.clamp(right_ground_z - right_sole_z.unsqueeze(-1), min=0.0)  # (N, P)

    return torch.sum(left_penetration, dim=-1) + torch.sum(right_penetration, dim=-1)


def foot_volume_penetration(
    env: ManagerBasedRLEnv,
    left_scanner_cfg: SceneEntityCfg,
    right_scanner_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sole_offset: float = 0.03,
) -> torch.Tensor:
    """Reward-manager wrapper: penalize the feet's footprint sampling points sitting below the terrain
    surface under them.

    Args:
        left_scanner_cfg/right_scanner_cfg: RayCaster sensors attached to the left/right ankle link, each
            casting a small footprint-sized grid straight down.
        asset_cfg: the robot, with ``body_ids`` resolved to ``[left_ankle_link, right_ankle_link]`` (same
            order as the two scanners).
        sole_offset: approximate vertical distance (m) from the ankle link origin down to the sole surface.
    """
    asset = env.scene[asset_cfg.name]
    left_sole_z = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2] - sole_offset  # (N,)
    right_sole_z = asset.data.body_pos_w[:, asset_cfg.body_ids[1], 2] - sole_offset  # (N,)

    left_scanner: RayCaster = env.scene.sensors[left_scanner_cfg.name]
    right_scanner: RayCaster = env.scene.sensors[right_scanner_cfg.name]
    return compute_foot_volume_penetration(left_scanner, right_scanner, left_sole_z, right_sole_z)
