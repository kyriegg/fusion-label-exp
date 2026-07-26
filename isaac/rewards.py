"""
四项约束的 Isaac 版本: 从【仿真物理状态】直接计算 reward (不再依赖可微 FK)。

与离线 loss 版 (losses/constraints.py) 一一对应, 但语义升级到物理层面:
  1. 相似性   r_sim       : hybrid label 与 label1 接近 -> 保模仿
  2. 不穿模   r_penetrate : 用 ContactSensor 的接触力惩罚非法接触
                            (非足部 link 出现接触力 = 穿模/摔倒/擦地)
  3. 不悬空   r_float     : 支撑相脚部离地高度惩罚 (地形高度从 height scanner 查)
  4. 不抖动   r_jitter    : 关节加速度 + α 帧间变化率惩罚

全部为 batched tensor 运算, 形状约定 num_envs = N。
Isaac Lab 与 legged_gym 均可直接调用 (只是取状态的 API 不同, 在 env 里取好传进来)。
"""
from __future__ import annotations
import torch


# ---------- 1. 相似性 (保模仿能力) ----------
def reward_similarity(hybrid: torch.Tensor, label1: torch.Tensor,
                      sigma: float = 0.5) -> torch.Tensor:
    """
    hybrid, label1: (N, D)   当前帧的融合标签与模仿标签
    指数核奖励, 值域 (0,1], 完全一致时=1
    """
    err = ((hybrid - label1) ** 2).mean(dim=-1)
    return torch.exp(-err / (sigma ** 2))


# ---------- 2. 不穿模 (物理接触惩罚) ----------
def penalty_penetration(contact_forces: torch.Tensor,
                        illegal_body_ids: list[int],
                        force_thresh: float = 1.0) -> torch.Tensor:
    """
    contact_forces:   (N, B, 3)  各刚体接触力 (ContactSensor.net_forces_w)
    illegal_body_ids: 非法接触的 body 索引 (躯干/大腿/小腿/膝, 即除脚以外)
    返回 (N,) 惩罚值 (>=0), 有非法接触即惩罚。
    穿模在物理仿真里表现为: 不该碰地的 link 出现接触力 / 穿透深度。
    """
    f = contact_forces[:, illegal_body_ids].norm(dim=-1)        # (N, K)
    return ((f > force_thresh).float() * (1.0 + f.clamp(max=200.0) / 200.0)).sum(dim=-1)


# ---------- 3. 不悬空 (支撑相贴地) ----------
def penalty_float(foot_pos_z: torch.Tensor,
                  terrain_h_at_foot: torch.Tensor,
                  contact_mask: torch.Tensor,
                  max_height: float = 0.03) -> torch.Tensor:
    """
    foot_pos_z:        (N, F)  脚部世界高度
    terrain_h_at_foot: (N, F)  脚下地形表面高度 (height scanner 采样)
    contact_mask:      (N, F)  1 = 参考侧(label1)判定该脚处于支撑相
    """
    clearance = foot_pos_z - terrain_h_at_foot
    viol = (clearance - max_height).clamp(min=0.0) * contact_mask
    return (viol ** 2).sum(dim=-1)


def compute_contact_mask_from_label(label1_foot_vel_xy: torch.Tensor,
                                    vel_thresh: float = 0.15) -> torch.Tensor:
    """
    (N, F, 2) 参考侧脚部水平速度 -> (N, F) 支撑相 mask。
    仍然坚持: 用 label1 侧判定, 不让 hybrid 自己判定自己。
    """
    return (label1_foot_vel_xy.norm(dim=-1) < vel_thresh).float()


# ---------- 4. 不抖动 ----------
def penalty_jitter(joint_acc: torch.Tensor,
                   alpha: torch.Tensor, prev_alpha: torch.Tensor,
                   w_acc: float = 1.0, w_alpha_rate: float = 1.0) -> torch.Tensor:
    """
    joint_acc:  (N, J)  关节加速度 (Isaac 直接给)
    alpha:      (N, ...) 当前 α, prev_alpha: 上一帧 α
    双重抑制: 物理层面的关节加速度 + 决策层面的 α 突变
    """
    j_acc = (joint_acc ** 2).mean(dim=-1)
    a_rate = ((alpha - prev_alpha) ** 2).flatten(1).mean(dim=-1)
    return w_acc * j_acc + w_alpha_rate * a_rate


# ---------- 汇总 ----------
def total_reward(state: dict, cfg: dict) -> tuple[torch.Tensor, dict]:
    """
    state 由 env 每步组装 (见 fusion_env.py), cfg 为 reward 权重配置。
    返回 (reward (N,), 分项日志)
    """
    r_sim = reward_similarity(state["hybrid"], state["label1"], cfg["sim_sigma"])
    p_pen = penalty_penetration(state["contact_forces"],
                                state["illegal_body_ids"], cfg["force_thresh"])
    p_flt = penalty_float(state["foot_pos_z"], state["terrain_h_at_foot"],
                          state["contact_mask"], cfg["float_max_height"])
    p_jit = penalty_jitter(state["joint_acc"], state["alpha"], state["prev_alpha"],
                           cfg["w_joint_acc"], cfg["w_alpha_rate"])

    reward = (cfg["w_sim"] * r_sim
              - cfg["w_penetrate"] * p_pen
              - cfg["w_float"] * p_flt
              - cfg["w_jitter"] * p_jit)

    logs = {"rew/sim": r_sim.mean().item(),
            "pen/penetrate": p_pen.mean().item(),
            "pen/float": p_flt.mean().item(),
            "pen/jitter": p_jit.mean().item(),
            "rew/total": reward.mean().item()}
    return reward, logs
