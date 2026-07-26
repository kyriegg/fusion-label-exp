"""
Isaac Lab 环境: 第一阶段 Function Label 训练 (DirectRLEnv 风格)。

一步 (step) 内的数据流:
  1. 从 sim 取 motion 相位/骨骼观测 -> 冻结模仿 policy -> label1
  2. 从 sim 取 速度指令 + height scanner 地形 -> 冻结感知 policy -> label2
  3. RL 的 action = FusionNet 的 α (由 rsl_rl 的 actor 输出, 见 fusion_actor.py)
     hybrid = α1 ⊙ label1 + α2 ⊙ label2
  4. hybrid 作为 PD 目标下发给机器人关节
  5. 从物理状态计算四项约束 reward

注意: 这里给的是骨架 —— Isaac Lab 的 import 路径与你本地版本相关
(isaaclab / omni.isaac.lab), 场景资产 (robot cfg / terrain cfg) 按你们
已有的 locomotion 任务复用即可, 需要改的位置都标了 TODO。
"""
from __future__ import annotations
import torch

# TODO: 按你本地 Isaac Lab 版本调整 import (>=2.0 是 isaaclab.*)
# from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
# from isaaclab.sensors import ContactSensor, RayCaster
# from isaaclab.utils import configclass

from isaac.rewards import (total_reward, compute_contact_mask_from_label)


class FusionLabelEnv:  # (DirectRLEnv):
    """
    观测 (给 rsl_rl actor):  concat[label1, label2, proprio(可选)]
    动作 (rsl_rl 输出):      α 的 raw logits, 在 actor 内归一化
    """

    def __init__(self, cfg, imitation_policy, perception_policy, **kwargs):
        # super().__init__(cfg, **kwargs)
        self.cfg = cfg
        self.imi = imitation_policy.eval()   # 冻结
        self.per = perception_policy.eval()  # 冻结
        self.prev_alpha = None

    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        with torch.no_grad():
            motion_obs = self._get_motion_obs()          # TODO: 相位/参考motion帧
            vel_cmd = self._get_velocity_command()       # (N, 3)
            terrain = self._get_height_scan()            # (N, S) RayCaster 采样

            self.label1 = self.imi(motion_obs)           # (N, D)
            self.label2 = self.per(vel_cmd, terrain)     # (N, D)

        obs = torch.cat([self.label1, self.label2], dim=-1)
        # 可选: 拼上本体感知 (关节位置/速度/姿态), 通常能让 α 学得更稳
        # obs = torch.cat([obs, self._get_proprio()], dim=-1)
        return {"policy": obs}

    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        """actions: (N, 2) 或 (N, 2*D) 的 α raw logits"""
        alpha = self._normalize_alpha(actions)           # softmax -> α1+α2=1
        self.alpha1, self.alpha2 = alpha[..., 0], alpha[..., 1]
        if self.cfg.fusion_mode == "scalar":
            a1 = self.alpha1.unsqueeze(-1)
            a2 = self.alpha2.unsqueeze(-1)
        else:
            a1, a2 = self.alpha1, self.alpha2            # (N, D)

        self.hybrid = a1 * self.label1 + a2 * self.label2

        # hybrid label 作为关节 PD 目标下发
        # TODO: 若 label 含 root 信息, 只取关节部分做 target
        joint_targets = self._label_to_joint_targets(self.hybrid)
        self._apply_pd_targets(joint_targets)

    def _normalize_alpha(self, raw: torch.Tensor) -> torch.Tensor:
        if self.cfg.fusion_mode == "scalar":
            return torch.softmax(raw, dim=-1)                       # (N, 2)
        N = raw.shape[0]
        a = raw.view(N, 2, -1)
        return torch.softmax(a, dim=1).permute(1, 0, 2)             # (2, N, D) -> 取用时转

    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        alpha_flat = torch.stack([self.alpha1.flatten(1) if self.alpha1.dim() > 1
                                  else self.alpha1.unsqueeze(-1),
                                  self.alpha2.flatten(1) if self.alpha2.dim() > 1
                                  else self.alpha2.unsqueeze(-1)], dim=-1)
        if self.prev_alpha is None:
            self.prev_alpha = alpha_flat.clone()

        with torch.no_grad():
            l1_foot_vel = self._label1_foot_velocity_xy()   # (N, F, 2) 参考侧
            cmask = compute_contact_mask_from_label(
                l1_foot_vel, self.cfg.rewards["contact_vel_thresh"])

        state = {
            "hybrid": self.hybrid, "label1": self.label1,
            "alpha": alpha_flat, "prev_alpha": self.prev_alpha,
            "contact_forces": self._get_contact_forces(),   # (N, B, 3) ContactSensor
            "illegal_body_ids": self.cfg.illegal_body_ids,  # 除脚以外的 body
            "foot_pos_z": self._get_foot_pos_z(),           # (N, F)
            "terrain_h_at_foot": self._get_terrain_h_at_foot(),  # (N, F)
            "contact_mask": cmask,
            "joint_acc": self._get_joint_acc(),             # (N, J)
        }
        reward, logs = total_reward(state, self.cfg.rewards)
        self.extras["log"] = logs
        self.prev_alpha = alpha_flat.clone()
        return reward

    # ------------------------------------------------------------------
    def _get_dones(self):
        # 沿用你们 locomotion 任务的终止条件: 躯干接触地面 / 姿态超限 / 超时
        raise NotImplementedError

    # ====== 以下为需要接 Isaac Lab API 的取状态函数 (TODO) ======
    def _get_motion_obs(self): ...
    def _get_velocity_command(self): ...
    def _get_height_scan(self): ...          # RayCaster (height scanner)
    def _get_contact_forces(self): ...       # ContactSensor.data.net_forces_w
    def _get_foot_pos_z(self): ...           # robot.data.body_pos_w[:, foot_ids, 2]
    def _get_terrain_h_at_foot(self): ...    # 脚部 (x,y) 处 raycast 高度
    def _get_joint_acc(self): ...            # robot.data.joint_acc
    def _label1_foot_velocity_xy(self): ...  # 参考侧脚速度 (判定支撑相)
    def _label_to_joint_targets(self, label): ...
    def _apply_pd_targets(self, targets): ...
