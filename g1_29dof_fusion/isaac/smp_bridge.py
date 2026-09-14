"""Isaac Lab port of smp's guidance reward (D:\\robot paper\\smp\\src\\smp\\rl\\{rewards,events,utils}.py).

smp's own code is written against mjlab (MuJoCo), not Isaac Lab. mjlab's
``mjlab.utils.lab_api.math`` module is a deliberate Isaac Lab-API mirror (same
function names/semantics as ``isaaclab.utils.math``, confirmed by inspecting
both), and smp's own ``ArticulationData`` field names (``root_link_pos_w`` etc)
also match Isaac Lab's real ``ArticulationData`` 1:1 - so this port is a
straight translation, not a redesign.

Ground-truth joint/body order comes from smp's own
``D:\\robot paper\\smp\\scripts\\csv_to_npz.py`` (``JOINT_NAMES``,
``EE_BODY_NAMES``) - this is NOT the same as Isaac Lab's native
``robot.data.joint_names`` order (which is USD-tree traversal order); indices
are resolved once at startup via ``find_joints``/``find_bodies`` with
``preserve_order=True`` and reused every step.

Scope cut vs the original mjlab code: no GSI (diffusion-sampled reset state
init) - resets just freeze the feature-buffer window to the post-reset pose
(see ``reset_smp_buffer``) instead of sampling a "natural" init window from
the denoiser. GSI is an initialization-quality optimization, not required for
the guidance reward itself to produce a valid learning signal; can be added
later if reward-only integration proves too weak a signal early in training.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import torch

import isaaclab.utils.math as math_utils

_SMP_SRC = "D:/robot paper/smp/src"
if _SMP_SRC not in sys.path:
    sys.path.insert(0, _SMP_SRC)

from smp.pretrain.model import DiffusionDenoiser  # noqa: E402
from smp.pretrain.scheduler import DDPMScheduler  # noqa: E402

# ---- ground truth from D:\robot paper\smp\scripts\csv_to_npz.py ----
JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
NUM_JOINTS = len(JOINT_NAMES)

EE_BODY_NAMES: tuple[str, ...] = (
    "left_ankle_roll_link", "right_ankle_roll_link", "torso_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
)
NUM_EE = len(EE_BODY_NAMES)


# ---- verbatim from smp/rl/utils.py (pure torch, no framework dependency) ----

def load_denoiser(
    ckpt_path: str,
    device: torch.device | str,
) -> tuple[DiffusionDenoiser, DDPMScheduler, torch.Tensor, torch.Tensor, int, int]:
    """Load a frozen pretrained denoiser checkpoint -> (model, scheduler, q_low, q_high, feature_dim, window_size)."""
    device = torch.device(device)
    ckpt: dict[str, Any] = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    feature_dim = int(cfg["feature_dim"])
    window_size = int(cfg["window_size"])

    model = DiffusionDenoiser(
        feature_dim=feature_dim,
        window_size=window_size,
        d_model=int(cfg.get("d_model", 256)),
        nhead=int(cfg.get("nhead", 8)),
        num_layers=int(cfg.get("num_layers", 2)),
        dropout=float(cfg.get("dropout", 0.0)),
    ).to(device)
    state = ckpt.get("model_ema") or ckpt["model"]
    model.load_state_dict(state)
    model.eval()
    model.requires_grad_(False)

    scheduler = DDPMScheduler(num_timesteps=int(cfg.get("num_timesteps", 50))).to(device)
    q_low = torch.from_numpy(np.asarray(ckpt["q_low"], dtype=np.float32)).to(device)
    q_high = torch.from_numpy(np.asarray(ckpt["q_high"], dtype=np.float32)).to(device)
    return model, scheduler, q_low, q_high, feature_dim, window_size


class DiffNormalizer:
    """Count-based running mean per diffusion timestep (verbatim from smp/rl/utils.py)."""

    def __init__(self, num_timesteps: int, device: torch.device, min_value: float = 1e-4, max_count: int = 50_000_000) -> None:
        self.min_value = min_value
        self.max_count = max_count
        self.mean = torch.ones(num_timesteps, device=device)
        self.count = torch.zeros(num_timesteps, device=device, dtype=torch.long)

    def update_and_normalize(self, t: int, mse_per_env: torch.Tensor) -> torch.Tensor:
        if self.count[t] > self.max_count:
            return mse_per_env / self.mean[t].clamp(min=self.min_value)
        n = mse_per_env.numel()
        batch_mean = mse_per_env.mean()
        old_count = self.count[t].item()
        new_count = old_count + n
        if old_count == 0:
            self.mean[t] = batch_mean
        else:
            w_old = old_count / new_count
            w_new = n / new_count
            self.mean[t] = w_old * self.mean[t] + w_new * batch_mean
        self.count[t] = new_count
        return mse_per_env / self.mean[t].clamp(min=self.min_value)


class MotionFeatureBuffer:
    """Rolling per-env buffer of the last window_size kinematic samples (verbatim from smp/rl/utils.py,
    isaaclab.utils.math in place of mjlab.utils.lab_api.math)."""

    def __init__(self, num_envs: int, window_size: int, num_joints: int, num_ee: int, device: torch.device | str) -> None:
        self.num_envs = num_envs
        self.window_size = window_size
        self.num_joints = num_joints
        self.num_ee = num_ee
        self.device = torch.device(device)

        self.root_pos_w = torch.zeros(num_envs, window_size, 3, device=self.device)
        self.root_quat_w = torch.zeros(num_envs, window_size, 4, device=self.device)
        self.root_quat_w[..., 0] = 1.0
        self.root_lin_vel_w = torch.zeros(num_envs, window_size, 3, device=self.device)
        self.root_ang_vel_w = torch.zeros(num_envs, window_size, 3, device=self.device)
        self.ee_pos_w = torch.zeros(num_envs, window_size, num_ee, 3, device=self.device)
        self.joint_pos = torch.zeros(num_envs, window_size, num_joints, device=self.device)
        self.joint_vel = torch.zeros(num_envs, window_size, num_joints, device=self.device)

    def reset(self, env_ids, root_pos_w, root_quat_w, root_lin_vel_w, root_ang_vel_w, ee_pos_w, joint_pos, joint_vel) -> None:
        if env_ids.numel() == 0:
            return
        self.root_pos_w[env_ids] = root_pos_w
        self.root_quat_w[env_ids] = root_quat_w
        self.root_lin_vel_w[env_ids] = root_lin_vel_w
        self.root_ang_vel_w[env_ids] = root_ang_vel_w
        self.ee_pos_w[env_ids] = ee_pos_w
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

    def update(self, root_pos_w, root_quat_w, root_lin_vel_w, root_ang_vel_w, ee_pos_w, joint_pos, joint_vel) -> None:
        self.root_pos_w = torch.roll(self.root_pos_w, shifts=-1, dims=1)
        self.root_quat_w = torch.roll(self.root_quat_w, shifts=-1, dims=1)
        self.root_lin_vel_w = torch.roll(self.root_lin_vel_w, shifts=-1, dims=1)
        self.root_ang_vel_w = torch.roll(self.root_ang_vel_w, shifts=-1, dims=1)
        self.ee_pos_w = torch.roll(self.ee_pos_w, shifts=-1, dims=1)
        self.joint_pos = torch.roll(self.joint_pos, shifts=-1, dims=1)
        self.joint_vel = torch.roll(self.joint_vel, shifts=-1, dims=1)
        self.root_pos_w[:, -1] = root_pos_w
        self.root_quat_w[:, -1] = root_quat_w
        self.root_lin_vel_w[:, -1] = root_lin_vel_w
        self.root_ang_vel_w[:, -1] = root_ang_vel_w
        self.ee_pos_w[:, -1] = ee_pos_w
        self.joint_pos[:, -1] = joint_pos
        self.joint_vel[:, -1] = joint_vel

    def compute_features(self) -> torch.Tensor:
        """(num_envs, W, 3+6+J+E*3+3+3), anchored to the LAST frame's yaw-only local frame."""
        N = self.num_envs
        W = self.window_size
        E = self.num_ee

        anchor_pos_T = self.root_pos_w[:, -1]
        anchor_quat_T = self.root_quat_w[:, -1]
        yaw_T = math_utils.yaw_quat(anchor_quat_T)
        heading_inv_T = math_utils.quat_conjugate(yaw_T)
        heading_inv_T_W = heading_inv_T[:, None, :].expand(N, W, 4)
        yaw_T_W = yaw_T[:, None, :].expand(N, W, 4).reshape(-1, 4)

        root_offset = self.root_pos_w - anchor_pos_T[:, None, :]
        root_pos_local = math_utils.quat_apply_inverse(yaw_T_W, root_offset.reshape(-1, 3)).reshape(N, W, 3)
        root_pos_local = root_pos_local.clone()
        root_pos_local[..., 2] = self.root_pos_w[..., 2]

        root_rot_local_quat = math_utils.quat_mul(
            heading_inv_T_W.reshape(-1, 4), self.root_quat_w.reshape(-1, 4)
        ).reshape(N, W, 4)
        root_rot_mat = math_utils.matrix_from_quat(root_rot_local_quat.reshape(-1, 4)).reshape(N, W, 3, 3)
        root_rot_6d = torch.cat([root_rot_mat[..., :, 0], root_rot_mat[..., :, 2]], dim=-1)

        ee_offset_w = self.ee_pos_w - self.root_pos_w[:, :, None, :]
        yaw_T_E = yaw_T[:, None, None, :].expand(N, W, E, 4).reshape(-1, 4)
        ee_pos_local = math_utils.quat_apply_inverse(yaw_T_E, ee_offset_w.reshape(-1, 3)).reshape(N, W, E * 3)

        lin_vel_local = math_utils.quat_apply_inverse(yaw_T_W, self.root_lin_vel_w.reshape(-1, 3)).reshape(N, W, 3)
        ang_vel_local = math_utils.quat_apply_inverse(yaw_T_W, self.root_ang_vel_w.reshape(-1, 3)).reshape(N, W, 3)

        return torch.cat(
            [root_pos_local, root_rot_6d, self.joint_pos, ee_pos_local, lin_vel_local, ang_vel_local], dim=-1
        )


# ---- isaaclab-side glue (replaces smp/rl/events.py's mjlab glue) ----

def _update_buffer_from_sim(env) -> None:
    robot = env.scene["robot"]
    joint_ids = env._smp_joint_ids  # type: ignore[attr-defined]
    ee_ids = env._smp_ee_ids  # type: ignore[attr-defined]
    buffer: MotionFeatureBuffer = env._smp_buffer  # type: ignore[attr-defined]
    origins = env.scene.env_origins
    buffer.update(
        robot.data.root_link_pos_w - origins,
        robot.data.root_link_quat_w,
        robot.data.root_link_lin_vel_w,
        robot.data.root_link_ang_vel_w,
        robot.data.body_link_pos_w[:, ee_ids] - origins[:, None, :],
        robot.data.joint_pos[:, joint_ids],
        robot.data.joint_vel[:, joint_ids],
    )


def init_smp_state(env, env_ids, ckpt_path: str = "") -> None:
    """Startup event: load the frozen denoiser, resolve smp's canonical joint/EE order
    against this asset's actual indices, and allocate the feature buffer + normalizer."""
    del env_ids
    if not ckpt_path:
        raise RuntimeError(
            "init_smp_state called without ckpt_path. Set it on the EventTermCfg: "
            "EventTermCfg(func=init_smp_state, mode='startup', params={'ckpt_path': '/path/to/pretrained.pt'})."
        )
    model, scheduler, q_low, q_high, feature_dim, window_size = load_denoiser(ckpt_path, env.device)
    env._smp_bundle = (model, scheduler, q_low, q_high, feature_dim, window_size)  # type: ignore[attr-defined]

    robot = env.scene["robot"]
    joint_ids, found_joints = robot.find_joints(list(JOINT_NAMES), preserve_order=True)
    if found_joints != list(JOINT_NAMES):
        raise RuntimeError(
            f"smp_bridge: could not resolve all {NUM_JOINTS} JOINT_NAMES on this robot asset "
            f"(found {found_joints}). Asset joint naming must match smp/scripts/csv_to_npz.py's JOINT_NAMES."
        )
    ee_ids, found_ee = robot.find_bodies(list(EE_BODY_NAMES), preserve_order=True)
    if found_ee != list(EE_BODY_NAMES):
        raise RuntimeError(f"smp_bridge: could not resolve all {NUM_EE} EE_BODY_NAMES (found {found_ee}).")
    env._smp_joint_ids = torch.tensor(joint_ids, dtype=torch.long, device=env.device)  # type: ignore[attr-defined]
    env._smp_ee_ids = torch.tensor(ee_ids, dtype=torch.long, device=env.device)  # type: ignore[attr-defined]

    env._smp_buffer = MotionFeatureBuffer(  # type: ignore[attr-defined]
        num_envs=env.num_envs, window_size=window_size, num_joints=NUM_JOINTS, num_ee=NUM_EE, device=env.device,
    )
    env._smp_normalizer = DiffNormalizer(scheduler.num_timesteps, env.device)  # type: ignore[attr-defined]

    # No GSI here (see module docstring) - just prime every window slot with the
    # post-startup-reset pose so compute_features() is well-defined from step 0.
    all_ids = torch.arange(env.num_envs, device=env.device)
    reset_smp_buffer(env, all_ids)


def reset_smp_buffer(env, env_ids: torch.Tensor) -> None:
    """Reset event: freeze the feature-buffer window to the just-reset pose for env_ids.
    (No GSI - see module docstring.)"""
    if env_ids.numel() == 0 or "_smp_buffer" not in env.__dict__:
        return
    robot = env.scene["robot"]
    joint_ids = env._smp_joint_ids  # type: ignore[attr-defined]
    ee_ids = env._smp_ee_ids  # type: ignore[attr-defined]
    buffer: MotionFeatureBuffer = env._smp_buffer  # type: ignore[attr-defined]
    origins = env.scene.env_origins[env_ids]
    n = env_ids.numel()
    W = buffer.window_size

    def rep(x: torch.Tensor) -> torch.Tensor:
        return x[env_ids].unsqueeze(1).expand(n, W, *x.shape[1:]).clone()

    buffer.reset(
        env_ids,
        rep(robot.data.root_link_pos_w) - origins[:, None, :],
        rep(robot.data.root_link_quat_w),
        rep(robot.data.root_link_lin_vel_w),
        rep(robot.data.root_link_ang_vel_w),
        rep(robot.data.body_link_pos_w[:, ee_ids]) - origins[:, None, None, :],
        rep(robot.data.joint_pos[:, joint_ids]),
        rep(robot.data.joint_vel[:, joint_ids]),
    )


def build_smp_obs(env, last_action: torch.Tensor) -> torch.Tensor:
    """Reconstruct smp's own 101-dim policy observation (see D:\\robot paper\\smp\\src\\smp\\rl\\env_cfg.py's
    ``g1_smp_env_cfg`` actor_terms + \\smp\\src\\smp\\rl\\tasks\\steering\\forward_env_cfg.py's steering command):
    base_lin_vel_b(3) + base_ang_vel_b(3) + projected_gravity_b(3) + joint_pos_rel(29, JOINT_NAMES order) +
    joint_vel_rel(29, JOINT_NAMES order) + last_action(29) + steering_command(5).

    Caveat: the fusion env doesn't run smp's own mjlab SteeringCommand task, so the 5-dim steering
    command ([tar_dir_x, tar_dir_y, tar_speed, face_dir_x, face_dir_y], heading-frame) is approximated
    from this env's own `base_velocity` command instead of smp's exact original semantics: tar_dir/
    face_dir set to the commanded xy-velocity's direction (treating "facing" == "heading of travel"),
    tar_speed to its magnitude. Reasonable in-distribution stand-in, not a faithful reproduction -
    revisit if smp_policy's behavior looks off in the fused rollout.
    """
    robot = env.scene["robot"]
    joint_ids = env._smp_joint_ids  # type: ignore[attr-defined]

    joint_pos_rel = (robot.data.joint_pos - robot.data.default_joint_pos)[:, joint_ids]
    joint_vel_rel = (robot.data.joint_vel - robot.data.default_joint_vel)[:, joint_ids]

    cmd = env.command_manager.get_command("base_velocity")  # (N, 3): lin_vel_x_b, lin_vel_y_b, ang_vel_z_b
    xy = cmd[:, :2]
    speed = xy.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    dir_b = xy / speed
    steering_command = torch.cat([dir_b, speed, dir_b], dim=-1)  # tar_dir(2) + tar_speed(1) + face_dir(2)

    return torch.cat(
        [
            robot.data.root_lin_vel_b,
            robot.data.root_ang_vel_b,
            robot.data.projected_gravity_b,
            joint_pos_rel,
            joint_vel_rel,
            last_action,
            steering_command,
        ],
        dim=-1,
    )


def smp_guidance_reward(
    env, fixed_timesteps: tuple[int, ...] = (8, 15, 22), ws: float = 4.0, normalize: bool = True,
) -> torch.Tensor:
    """SDS-style guidance reward (verbatim algorithm from smp/rl/rewards.py):
    exp(-w_s/|K| * sum_{i in K} ||eps_hat_i - eps_i||^2)."""
    device = torch.device(env.device)
    model, scheduler, q_low, q_high, _, _ = env._smp_bundle  # type: ignore[attr-defined]
    normalizer: DiffNormalizer = env._smp_normalizer  # type: ignore[attr-defined]
    _update_buffer_from_sim(env)

    buffer: MotionFeatureBuffer = env._smp_buffer  # type: ignore[attr-defined]
    features = buffer.compute_features()
    x_0 = 2.0 * (features - q_low) / (q_high - q_low + 1e-8) - 1.0
    num_envs = x_0.shape[0]

    total_err = torch.zeros(num_envs, device=device)
    with torch.no_grad():
        for t_scalar in fixed_timesteps:
            if not 0 <= t_scalar < scheduler.num_timesteps:
                raise ValueError(f"fixed_timestep {t_scalar} out of range [0, {scheduler.num_timesteps})")
            t = torch.full((num_envs,), t_scalar, dtype=torch.long, device=device)
            noise = torch.randn_like(x_0)
            x_t = scheduler.add_noise(x_0, noise, t)
            eps_hat = model(x_t, t)
            mse_per_env = ((eps_hat - noise) ** 2).mean(dim=(-1, -2))
            if normalize:
                total_err += normalizer.update_and_normalize(t_scalar, mse_per_env)
            else:
                total_err += mse_per_env

    err = total_err / len(fixed_timesteps)
    return torch.exp(-err * ws)
