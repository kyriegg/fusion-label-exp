r"""
G1 人形机器人 融合环境 (与 Anymal 版同构, 但去掉了所有硬编码)

相比 fusion_env_anymal.py 的三处关键改动:
  1. FLAT_OBS_DIM 不再写死 48 —— 从 observation_manager 里按 term 名自动算出
     (排除 height_scan 后的维度和), 换任何机器人都不用改。
  2. body 名字模式从 ROBOT_CFG 传入 (G1 的脚是 ankle_roll_link, 不是 FOOT)。
  3. 支持 alpha 覆盖 (alpha_override), 用于评估基线:
     [1,0]=纯模仿  [0,1]=纯感知  [0.5,0.5]=朴素平均  None=用训练好的网络
"""
from __future__ import annotations
import torch
from tensordict import TensorDict

from isaac.rewards import reward_similarity, penalty_penetration, penalty_jitter

# ---------------- 机器人配置 ----------------
ROBOT_CFG = {
    "g1": {
        "foot_pattern": [".*ankle_roll_link"],
        # 非法接触: 除脚以外任何该着地的部位 (躯干/骨盆/膝/髋)
        "illegal_pattern": ["torso_link", "pelvis", ".*knee_link", ".*hip_.*_link"],
    },
    "anymal_c": {
        "foot_pattern": [".*FOOT"],
        "illegal_pattern": ["base", ".*THIGH", ".*SHANK", ".*HIP"],
    },
}


class G1FusionEnv:
    """对外暴露 rsl_rl VecEnv 接口; robot 参数决定 body 名字模式"""

    def __init__(self, base_env, flat_policy_jit: str, rough_policy_jit: str,
                 reward_cfg: dict, robot: str = "g1", device: str = "cuda:0",
                 alpha_override: torch.Tensor | None = None):
        self.env = base_env
        self.uenv = base_env.unwrapped
        self.device = device
        self.rcfg = reward_cfg
        self.alpha_override = alpha_override   # (2,) 或 None
        self.dagger_net = None   # 若设置, 每帧用它从观测算 alpha2 (监督训的网络)

        self.flat_policy = torch.jit.load(flat_policy_jit, map_location=device).eval()
        self.rough_policy = torch.jit.load(rough_policy_jit, map_location=device).eval()

        scene = self.uenv.scene
        self.robot = scene["robot"]
        self.contact = scene["contact_forces"]

        rcfg_body = ROBOT_CFG[robot]
        self.foot_ids, foot_names = self.contact.find_bodies(rcfg_body["foot_pattern"])
        self.illegal_ids, illegal_names = self.contact.find_bodies(
            rcfg_body["illegal_pattern"])
        if len(self.foot_ids) == 0:
            raise RuntimeError(
                f"没匹配到脚部 body! 可用 body 列表:\n{self.contact.body_names}\n"
                f"请修改 ROBOT_CFG['{robot}']['foot_pattern']")
        print(f"[fusion] 脚部 body ({len(foot_names)}): {foot_names}")
        print(f"[fusion] 非法接触 body ({len(illegal_names)}): {illegal_names}")

        # ---- 自动推断 flat policy 的观测维度 (排除 height_scan) ----
        self.flat_obs_dim = self._infer_flat_obs_dim()

        self.num_envs = self.uenv.num_envs
        self.num_actions = 2
        self.max_episode_length = self.uenv.max_episode_length

        # ---- 相似性奖励只在"有意义"的关节上计算 ----
        # G1 的手部关节 (zero/one/.../six) 对 locomotion 无贡献, 却在两路策略
        # 之间差异巨大(纯噪声), 会稀释 alpha 的优化信号 -> 排除
        jn = list(self.robot.data.joint_names)
        exclude = reward_cfg.get("sim_exclude_patterns",
                                 ["_zero_", "_one_", "_two_", "_three_",
                                  "_four_", "_five_", "_six_"])
        self.sim_joint_ids = [i for i, n in enumerate(jn)
                              if not any(p in n for p in exclude)]
        print(f"[fusion] 相似性奖励关节: {len(self.sim_joint_ids)}/{len(jn)} "
              f"(排除 {len(jn) - len(self.sim_joint_ids)} 个手部关节)", flush=True)

        # ---- DAgger 路径1: 切换奖励用的下肢关节掩码 ----
        leg_kw = ("_hip_pitch", "_hip_roll", "_hip_yaw", "_knee",
                  "_ankle_pitch", "_ankle_roll")
        self.leg_joint_ids = [i for i, n in enumerate(jn)
                              if any(k in n for k in leg_kw)]
        self.w_switch = float(reward_cfg.get("w_switch", 0.0))
        # 目标 alpha2 的映射参数: 下肢两专家分歧 d 归一化后线性映射到 [a_lo, a_hi]
        self.sw_a_lo = float(reward_cfg.get("switch_a_lo", 0.15))
        self.sw_a_hi = float(reward_cfg.get("switch_a_hi", 0.90))
        self.sw_d_lo = float(reward_cfg.get("switch_d_lo", 0.05))  # 平地分歧下界
        self.sw_d_hi = float(reward_cfg.get("switch_d_hi", 0.40))  # 崎岖分歧上界
        self.sw_ema = float(reward_cfg.get("switch_ema", 0.0))     # 目标时间平滑
        self._alpha_tgt_prev = None
        if self.w_switch > 0:
            print(f"[fusion] 切换奖励开启 w_switch={self.w_switch} | "
                  f"下肢关节 {len(self.leg_joint_ids)} 个 | "
                  f"目标 a2 映射 d[{self.sw_d_lo},{self.sw_d_hi}]"
                  f"->a2[{self.sw_a_lo},{self.sw_a_hi}]", flush=True)

        # v3: 是否把 height_scan 地形观测喂给 alpha 网络
        self.use_terrain_obs = bool(reward_cfg.get("use_terrain_obs", False))
        print(f"[fusion] alpha 网络地形观测: "
              f"{'开启 (v3)' if self.use_terrain_obs else '关闭 (v1/v2)'}", flush=True)

        self.label1 = self.label2 = None
        self.prev_alpha = torch.full((self.num_envs, 2), 0.5, device=device)

        obs_dict, _ = self.env.reset()
        self._raw_obs = obs_dict["policy"]
        self._check_flat_policy()
        self._compute_labels()
        self.num_obs = self._fusion_obs().shape[-1]

    # ------------------------------------------------------------------
    def _infer_flat_obs_dim(self) -> int:
        """flat env 观测 = rough 观测去掉 height_scan, 且 height_scan 排在最后"""
        om = self.uenv.observation_manager
        names = om.active_terms["policy"]
        dims = om.group_obs_term_dim["policy"]
        total, scan = 0, 0
        for n, d in zip(names, dims):
            size = int(torch.tensor(d).prod())
            if "height" in n or "scan" in n:
                scan += size
            else:
                total += size
        print(f"[fusion] 观测项: {list(zip(names, dims))}")
        print(f"[fusion] 本体观测={total}  地形扫描={scan}  合计={total + scan}")
        if scan == 0:
            raise RuntimeError("没找到 height_scan 观测项, 请确认用的是 Rough 任务")
        return total

    def _check_flat_policy(self):
        """确认 flat policy 的输入维度和推断值一致, 不一致立刻报清楚"""
        try:
            with torch.no_grad():
                self.flat_policy(self._raw_obs[:1, :self.flat_obs_dim])
        except Exception as e:
            raise RuntimeError(
                f"flat policy 不接受 {self.flat_obs_dim} 维输入。\n"
                f"通常说明这个 ckpt 是用不同版本/配置训练的, 建议重新训练 flat policy。\n"
                f"原始错误: {e}")

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
            self.label1 = self.flat_policy(self._raw_obs[:, :self.flat_obs_dim])
            self.label2 = self.rough_policy(self._raw_obs)

    def _fusion_obs(self) -> torch.Tensor:
        if self.use_terrain_obs:
            # label1 + label2 + 完整本体观测(含 height_scan 地形扫描)
            return torch.cat([self.label1, self.label2, self._raw_obs], dim=-1)
        # v1/v2: 只给本体, 不含地形
        return torch.cat([self.label1, self.label2,
                          self._raw_obs[:, :self.flat_obs_dim]], dim=-1)

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
        if self.dagger_net is not None:
            # 监督训的 alpha 网络: 输入 [label1, label2, raw_obs] -> a2
            with torch.no_grad():
                x = torch.cat([self.label1, self.label2, self._raw_obs], dim=-1)
                a2 = self.dagger_net(x).view(-1)              # (N,)
            alpha = torch.stack([1.0 - a2, a2], dim=-1)       # (N,2)
        elif self.alpha_override is not None:
            alpha = self.alpha_override.to(self.device).expand(self.num_envs, 2)
        else:
            alpha = torch.softmax(alpha_logits, dim=-1)
        a1, a2 = alpha[:, 0:1], alpha[:, 1:2]
        hybrid = a1 * self.label1 + a2 * self.label2
        self._last_hybrid, self._last_alpha = hybrid, alpha
        self._last_label1, self._last_label2 = self.label1, self.label2

        obs_dict, _r, terminated, truncated, extras = self.env.step(hybrid)
        self._raw_obs = obs_dict["policy"]

        reward, logs = self._compute_reward(hybrid, alpha)
        self.prev_alpha = alpha.detach()

        dones = (terminated | truncated).to(torch.long)
        extras = dict(extras or {})
        extras.setdefault("log", {}).update(logs)
        extras["time_outs"] = truncated
        extras["terminated"] = terminated          # 评估用: 区分摔倒与超时

        self._compute_labels()
        return (TensorDict({"policy": self._fusion_obs()},
                           batch_size=[self.num_envs]),
                reward, dones, extras)

    # ------------------------------------------------------------------
    def _compute_reward(self, hybrid, alpha):
        cfg = self.rcfg
        ids = self.sim_joint_ids
        r_sim = reward_similarity(hybrid[:, ids], self.label1[:, ids],
                                  cfg["sim_sigma"])

        forces = self.contact.data.net_forces_w
        p_pen = penalty_penetration(forces, self.illegal_ids, cfg["force_thresh"])

        foot_f = forces[:, self.foot_ids].norm(dim=-1)
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

        # ---- 切换奖励 (路径1): 让 alpha2 跟踪"下肢两专家分歧"导出的目标 ----
        if self.w_switch > 0:
            lids = self.leg_joint_ids
            # 每个 env 的下肢分歧 (label2 vs label1 的 L2 距离, 按关节数归一)
            d = (self.label2[:, lids] - self.label1[:, lids]).norm(dim=-1) \
                / (len(lids) ** 0.5)                              # (N,)
            # 线性映射到目标 alpha2, clamp 到 [a_lo, a_hi]
            frac = ((d - self.sw_d_lo) / (self.sw_d_hi - self.sw_d_lo)).clamp(0, 1)
            a2_tgt = self.sw_a_lo + frac * (self.sw_a_hi - self.sw_a_lo)   # (N,)
            if self.sw_ema > 0 and self._alpha_tgt_prev is not None:
                a2_tgt = self.sw_ema * self._alpha_tgt_prev + (1 - self.sw_ema) * a2_tgt
            self._alpha_tgt_prev = a2_tgt.detach()

            switch_err = (alpha[:, 1] - a2_tgt) ** 2                       # (N,)
            reward = reward - self.w_switch * switch_err
            logs["switch/target_a2"] = a2_tgt.mean().item()
            logs["switch/leg_divergence"] = d.mean().item()
            logs["switch/err"] = switch_err.mean().item()
        return reward, logs

    # ---- 评估用: 当前速度跟踪误差 ----
    def velocity_error(self) -> torch.Tensor:
        cmd = self.uenv.command_manager.get_command("base_velocity")   # (N,3)
        actual = self.robot.data.root_lin_vel_b                        # (N,3)
        return (cmd[:, :2] - actual[:, :2]).norm(dim=-1)

    def joint_snapshot(self, n_envs: int) -> dict:
        """取前 n_envs 个环境的逐帧关节/标签数据 (numpy), 供录制用"""
        d = self.robot.data
        torque = getattr(d, "applied_torque", None)
        if torque is None:
            torque = getattr(d, "computed_torque", d.joint_effort_target)
        f = lambda t: t[:n_envs].detach().cpu().numpy().copy()
        return {
            "joint_pos": f(d.joint_pos),
            "joint_vel": f(d.joint_vel),
            "joint_acc": f(d.joint_acc),
            "joint_torque": f(torque),
            "label1": f(self._last_label1),
            "label2": f(self._last_label2),
            "hybrid": f(self._last_hybrid),
            "alpha": f(self._last_alpha),
            "foot_force": f(self.contact.data.net_forces_w[:, self.foot_ids]),
            "root_lin_vel": f(d.root_lin_vel_b),
            "root_ang_vel": f(d.root_ang_vel_b),
            "projected_gravity": f(d.projected_gravity_b),
        }

    def close(self):
        self.env.close()
