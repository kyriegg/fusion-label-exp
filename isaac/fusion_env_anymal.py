"""
Anymal-C 替身实验环境 (适配新版 rsl_rl: TensorDict 观测接口)

数据流:
  底层 rough env 观测(235) ─┬─ 前48维 ──> [冻结] 平地policy ──> label1 (12)
                            └─ 全235维 ─> [冻结] 崎岖policy ──> label2 (12)
  RL action = α logits (2) -> softmax -> hybrid = α1*label1 + α2*label2
  hybrid 作为底层 env 动作下发; reward = 四项约束
"""
from __future__ import annotations
import torch
from tensordict import TensorDict

from isaac.rewards import reward_similarity, penalty_penetration, penalty_jitter

FLAT_OBS_DIM = 48
NUM_ACTIONS_ROBOT = 12


class AnymalFusionEnv:
    """对外暴露新版 rsl_rl 的 VecEnv 接口 (get_observations 返回 TensorDict)"""

    def __init__(self, base_env, flat_policy_jit: str, rough_policy_jit: str,
                 reward_cfg: dict, device: str = "cuda:0"):
        self.env = base_env
        self.uenv = base_env.unwrapped
        self.device = device
        self.rcfg = reward_cfg

        self.flat_policy = torch.jit.load(flat_policy_jit, map_location=device).eval()
        self.rough_policy = torch.jit.load(rough_policy_jit, map_location=device).eval()

        scene = self.uenv.scene
        self.robot = scene["robot"]
        self.contact = scene["contact_forces"]

        self.foot_ids, _ = self.contact.find_bodies(".*FOOT")
        self.illegal_ids, _ = self.contact.find_bodies(
            ["base", ".*THIGH", ".*SHANK", ".*HIP"])

        # ---- rsl_rl VecEnv 接口属性 ----
        self.num_envs = self.uenv.num_envs
        self.num_actions = 2                       # α logits
        self.max_episode_length = self.uenv.max_episode_length

        self.label1 = self.label2 = None
        self.prev_alpha = torch.full((self.num_envs, 2), 0.5, device=device)

        # rsl_rl runner 不调用 reset, 在这里先 reset 一次
        obs_dict, _ = self.env.reset()
        self._raw_obs = obs_dict["policy"]
        self._compute_labels()
        self.num_obs = self._fusion_obs().shape[-1]

    # ---- 属性代理 ----
    @property
    def cfg(self):
        return self.uenv.cfg

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.uenv.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        self.uenv.episode_length_buf = value

    # ------------------------------------------------------------------
    def _compute_labels(self):
        with torch.no_grad():
            self.label1 = self.flat_policy(self._raw_obs[:, :FLAT_OBS_DIM])
            self.label2 = self.rough_policy(self._raw_obs)

    def _fusion_obs(self) -> torch.Tensor:
        return torch.cat([self.label1, self.label2,
                          self._raw_obs[:, :FLAT_OBS_DIM]], dim=-1)

    def get_observations(self) -> TensorDict:
        return TensorDict({"policy": self._fusion_obs()},
                          batch_size=[self.num_envs])

    def reset(self):
        obs_dict, extras = self.env.reset()
        self._raw_obs = obs_dict["policy"]
        self._compute_labels()
        return TensorDict({"policy": self._fusion_obs()},
                          batch_size=[self.num_envs]), extras

    # ------------------------------------------------------------------
    def step(self, alpha_logits: torch.Tensor):
        alpha = torch.softmax(alpha_logits, dim=-1)          # (N,2)
        a1, a2 = alpha[:, 0:1], alpha[:, 1:2]
        hybrid = a1 * self.label1 + a2 * self.label2         # (N,12)

        obs_dict, _base_rew, terminated, truncated, extras = self.env.step(hybrid)
        self._raw_obs = obs_dict["policy"]

        reward, logs = self._compute_reward(hybrid, alpha)
        self.prev_alpha = alpha.detach()

        dones = (terminated | truncated).to(torch.long)
        extras = dict(extras or {})
        extras.setdefault("log", {}).update(logs)
        extras["time_outs"] = truncated

        self._compute_labels()
        return (TensorDict({"policy": self._fusion_obs()},
                           batch_size=[self.num_envs]),
                reward, dones, extras)

    # ------------------------------------------------------------------
    def _compute_reward(self, hybrid, alpha):
        cfg = self.rcfg
        r_sim = reward_similarity(hybrid, self.label1, cfg["sim_sigma"])

        forces = self.contact.data.net_forces_w              # (N,B,3)
        p_pen = penalty_penetration(forces, self.illegal_ids, cfg["force_thresh"])

        foot_f = forces[:, self.foot_ids].norm(dim=-1)       # (N,4)
        in_contact = (foot_f > cfg["force_thresh"]).float()
        p_float = (in_contact.sum(dim=-1) < 0.5).float()

        p_jit = penalty_jitter(self.robot.data.joint_acc, alpha, self.prev_alpha,
                               cfg["w_joint_acc"], cfg["w_alpha_rate"])

        reward = (cfg["w_sim"] * r_sim - cfg["w_penetrate"] * p_pen
                  - cfg["w_float"] * p_float - cfg["w_jitter"] * p_jit)

        logs = {"rew/sim": r_sim.mean().item(),
                "pen/illegal_contact": p_pen.mean().item(),
                "pen/all_feet_air": p_float.mean().item(),
                "pen/jitter": p_jit.mean().item(),
                "alpha/a2_mean": alpha[:, 1].mean().item()}
        return reward, logs

    def close(self):
        self.env.close()
