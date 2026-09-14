"""
G1 人形机器人 三策略融合环境 (SMP + 地形 + 舞蹈参考轨迹)

三策略融合: SMP (自然行走) + 地形 (崎岖稳定) + 舞蹈 (Macarena 参考轨迹)

用法:
  env = G1FusionEnv(
      base_env=base_env,
      smp_policy_jit="path/to/smp_model.pt",
      rough_policy_jit="path/to/rough_policy.pt", 
      dance_ref_csv="path/to/macarena_q.csv",  # 舞蹈参考轨迹 CSV
      reward_cfg=reward_cfg,
      robot="g1",
      device="cuda:0",
  )
"""
from __future__ import annotations
import os
import sys
import torch
import pandas as pd
import numpy as np
from tensordict import TensorDict

from rewards import reward_similarity, penalty_penetration, penalty_jitter
from policy_loader import load_rsl_rl_policy
import smp_bridge
import foot_penetration

# ---------------- 机器人配置 ----------------
ROBOT_CFG = {
    "g1": {
        "foot_pattern": [".*ankle_roll_link"],
        "illegal_pattern": ["torso_link", "pelvis", ".*knee_link", ".*hip_.*_link"],
    },
    "anymal_c": {
        "foot_pattern": [".*FOOT"],
        "illegal_pattern": ["base", ".*THIGH", ".*SHANK", ".*HIP"],
    },
}


class G1FusionEnv:
    """
    三策略融合环境: SMP(自然行走) + 地形(崎岖稳定) + 舞蹈(Macarena参考轨迹)
    对外暴露 rsl_rl VecEnv 接口
    """

    def __init__(
        self,
        base_env,
        smp_policy_jit: str,               # SMP 自然行走策略 checkpoint (原始 rsl_rl 格式, 非 jit)
        rough_policy_jit: str,             # 地形感知策略 (崎岖) checkpoint (原始 rsl_rl 格式, 非 jit)
        dance_ref_csv: str | None,         # 舞蹈参考轨迹 CSV 路径 (可选)
        reward_cfg: dict,
        robot: str = "g1",
        device: str = "cuda:0",
        alpha_override: torch.Tensor | None = None,
    ):
        self.env = base_env
        self.uenv = base_env.unwrapped
        self.device = device
        self.rcfg = reward_cfg
        self.alpha_override = alpha_override
        self.dagger_net = None

        # 加载 SMP 和地形策略 (两者都是原始 rsl_rl 训练 checkpoint, 不是 TorchScript,
        # 之前这里用 torch.jit.load 会直接报错 - 见 policy_loader.py)
        self.smp_policy = load_rsl_rl_policy(smp_policy_jit, device)
        self.rough_policy = load_rsl_rl_policy(rough_policy_jit, device)

        # ---- 加载舞蹈参考轨迹 ----
        self.use_dance = dance_ref_csv is not None and os.path.exists(dance_ref_csv)
        if self.use_dance:
            print(f"[fusion] 加载舞蹈参考轨迹: {dance_ref_csv}")
            df = pd.read_csv(dance_ref_csv)
            # 提取 q_0 到 q_28 (29个关节)
            q_cols = [f'q_{i}' for i in range(29)]
            dance_traj = torch.tensor(df[q_cols].values, dtype=torch.float32, device=device)
            self.dance_trajectory = dance_traj  # (T, 29)
            print(f"[fusion] 舞蹈轨迹帧数: {dance_traj.shape[0]}")
            print(f"[fusion] 三策略模式: SMP + 地形 + 舞蹈参考轨迹")
        else:
            self.dance_trajectory = None
            if dance_ref_csv is not None:
                print(f"[fusion] 警告: 舞蹈参考文件不存在: {dance_ref_csv}")
            print("[fusion] 双策略模式: SMP + 地形 (舞蹈未启用)")

        # 场景与接触
        scene = self.uenv.scene
        self.robot = scene["robot"]
        self.contact = scene["contact_forces"]

        rcfg_body = ROBOT_CFG[robot]
        self.foot_ids, foot_names = self.contact.find_bodies(rcfg_body["foot_pattern"])
        self.illegal_ids, illegal_names = self.contact.find_bodies(rcfg_body["illegal_pattern"])
        
        if len(self.foot_ids) == 0:
            raise RuntimeError(f"没匹配到脚部 body! 可用 body 列表:\n{self.contact.body_names}")
        print(f"[fusion] 脚部 body ({len(foot_names)}): {foot_names}")
        print(f"[fusion] 非法接触 body ({len(illegal_names)}): {illegal_names}")

        # 简化版 foot volume points (见 foot_penetration.py) - 跟接触力惩罚(p_pen)不同,
        # 这个不依赖接触传感器触发, 在脚还没被判定"接触"之前就能发现"卡/踩进地形"的迹象。
        # G1_29DOF_RoughEnvCfg.__post_init__ 已经把 left_foot_scanner/right_foot_scanner
        # 挂到 self.uenv.scene 上了 (跟 base env 自己的 foot_volume_penetration 奖励项共用
        # 同一对传感器, 但那个奖励项算出来的值会跟 base env 其他奖励一起被 G1FusionEnv.step()
        # 丢弃, 所以这里单独再算一遍接到融合奖励里, 用法跟 r_smp_style 直接调
        # smp_bridge.smp_guidance_reward(self.uenv) 是同一个模式)。
        self._ankle_ids, _ankle_names = self.robot.find_bodies(
            ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
        )
        self._left_foot_scanner = scene["left_foot_scanner"]
        self._right_foot_scanner = scene["right_foot_scanner"]

        # 推断观测维度
        self.flat_obs_dim = self._infer_flat_obs_dim()
        self.num_envs = self.uenv.num_envs
        # alpha 网络的动作空间: 2维(smp+rough) 或 3维(smp+rough+dance)。原代码硬编码成2，
        # 舞蹈启用时(use_dance=True)会跟下面 step()/prev_alpha 用的 3维 alpha 对不上。
        self.num_actions = 3 if self.use_dance else 2
        self.max_episode_length = self.uenv.max_episode_length

        # hybrid/label_smp/dance_action 现在统一是 29 维, 顺序 = smp_bridge.JOINT_NAMES
        # (smp/scripts/csv_to_npz.py 的权威定义, 见 smp_bridge.py 顶部说明) - 手指关节已经
        # 不在动作空间里了 (G1_29DOF_RoughEnvCfg 训练时就排除了), 相似性奖励直接用全部 29 维,
        # 不用再像旧代码那样按名字排除手指。
        self.sim_joint_ids = list(range(smp_bridge.NUM_JOINTS))
        print(f"[fusion] 相似性奖励关节: {len(self.sim_joint_ids)}/{smp_bridge.NUM_JOINTS} (29维动作空间已不含手指)", flush=True)

        # rough_policy 是在 G1_29DOF_RoughEnvCfg 里训的, 它的 29 维输出顺序是 Isaac Lab
        # 原生资产关节树顺序 (robot.data.joint_names 前29个, preserve_order=False 的默认行为),
        # 不是 smp_bridge.JOINT_NAMES 那种按肢体分组的顺序。两者用同一批 find_joints 索引即可
        # 互相转换: canon_ids[i] 是 robot 原生关节表里 JOINT_NAMES[i] 的下标, 而
        # rough_policy 原始输出的下标本来就是 robot 原生关节表的下标(0..28, 因为手指排在后面),
        # 所以 label_rough_native[:, canon_ids] 就是按 JOINT_NAMES 顺序重排后的结果。
        canon_ids, found_names = self.robot.find_joints(list(smp_bridge.JOINT_NAMES), preserve_order=True)
        if found_names != list(smp_bridge.JOINT_NAMES):
            raise RuntimeError(
                f"fusion_env: 没能在这个资产上找全 smp_bridge.JOINT_NAMES 的 29 个关节 (找到: {found_names})"
            )
        self._canon_ids = torch.tensor(canon_ids, dtype=torch.long, device=device)

        # 切换奖励用下肢关节 (按 JOINT_NAMES 里的下标, 不是 robot 原生关节表下标, 因为
        # hybrid/label_rough(重排后)/label_smp/dance_action 现在都是 JOINT_NAMES 顺序)
        leg_kw = ("_hip_pitch", "_hip_roll", "_hip_yaw", "_knee", "_ankle_pitch", "_ankle_roll")
        self.leg_joint_ids = [i for i, n in enumerate(smp_bridge.JOINT_NAMES) if any(k in n for k in leg_kw)]
        self.w_switch = float(reward_cfg.get("w_switch", 0.0))
        self.sw_a_lo = float(reward_cfg.get("switch_a_lo", 0.15))
        self.sw_a_hi = float(reward_cfg.get("switch_a_hi", 0.90))
        self.sw_d_lo = float(reward_cfg.get("switch_d_lo", 0.05))
        self.sw_d_hi = float(reward_cfg.get("switch_d_hi", 0.40))
        self.sw_ema = float(reward_cfg.get("switch_ema", 0.0))
        self._alpha_tgt_prev = None
        if self.w_switch > 0:
            print(f"[fusion] 切换奖励开启 w_switch={self.w_switch}", flush=True)

        # 地形观测开关
        self.use_terrain_obs = bool(reward_cfg.get("use_terrain_obs", False))
        print(f"[fusion] alpha 网络地形观测: {'开启' if self.use_terrain_obs else '关闭'}", flush=True)

        # 舞蹈风格奖励权重
        self.w_dance_style = float(reward_cfg.get("w_dance_style", 0.0))
        if self.use_dance and self.w_dance_style > 0:
            print(f"[fusion] 舞蹈奖励开启, 权重={self.w_dance_style}", flush=True)
        else:
            print("[fusion] 舞蹈奖励关闭", flush=True)

        # 初始化状态
        num_alpha = 3 if self.use_dance else 2
        self.prev_alpha = torch.full(
            (self.num_envs, num_alpha), 
            1.0 / num_alpha, 
            device=device
        )

        # smp_policy 的 last_action 观测项在第一次调用时还没有"上一步"可用, 用0初始化
        # (对应 smp 自己训练时 mdp.last_action 在 reset 后的默认值)
        self._last_hybrid = torch.zeros(self.num_envs, smp_bridge.NUM_JOINTS, device=device)

        obs_dict, _ = self.env.reset()
        self._raw_obs = obs_dict["policy"]
        self._check_smp_policy()
        self._compute_labels()
        self.num_obs = self._fusion_obs().shape[-1]

    def _infer_flat_obs_dim(self) -> int:
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
        print(f"[fusion] 本体观测={total}  地形扫描={scan}")
        if scan == 0:
            raise RuntimeError("没找到 height_scan 观测项")
        return total

    def _check_smp_policy(self):
        try:
            with torch.no_grad():
                self.smp_policy(smp_bridge.build_smp_obs(self.uenv, self._last_hybrid)[:1])
        except Exception as e:
            raise RuntimeError(f"SMP policy 观测构造有问题 (smp_bridge.build_smp_obs): {e}")

    @property
    def cfg(self):
        return self.uenv.cfg

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.uenv.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        self.uenv.episode_length_buf = value

    def _compute_labels(self):
        with torch.no_grad():
            self.label_smp = self.smp_policy(smp_bridge.build_smp_obs(self.uenv, self._last_hybrid))
            # rough_policy 原始输出是 robot 原生关节表顺序 (0..28); 重排到 JOINT_NAMES 顺序
            # 才能跟 label_smp/dance_action 按同一套下标相加 (见 __init__ 里 self._canon_ids 的注释)
            self.label_rough = self.rough_policy(self._raw_obs)[:, self._canon_ids]
            # 舞蹈策略通过参考轨迹实现，不需要独立的策略网络

    def _fusion_obs(self) -> torch.Tensor:
        parts = [self.label_smp, self.label_rough]
        if self.use_dance:
            # 舞蹈参考轨迹作为观测的一部分
            # 获取当前步数对应的舞蹈帧索引
            step_idx = self.uenv.episode_length_buf.long() % self.dance_trajectory.shape[0]
            dance_ref = self.dance_trajectory[step_idx]  # (N, 29)
            parts.append(dance_ref)
        if self.use_terrain_obs:
            parts.append(self._raw_obs)
        else:
            parts.append(self._raw_obs[:, :self.flat_obs_dim])
        return torch.cat(parts, dim=-1)

    def get_observations(self) -> TensorDict:
        return TensorDict({"policy": self._fusion_obs()}, batch_size=[self.num_envs])

    def reset(self):
        obs_dict, extras = self.env.reset()
        self._raw_obs = obs_dict["policy"]
        self._compute_labels()
        return TensorDict({"policy": self._fusion_obs()}, batch_size=[self.num_envs]), extras

    def step(self, alpha_logits: torch.Tensor):
        # 计算 alpha
        if self.alpha_override is not None:
            alpha = self.alpha_override.to(self.device).expand(self.num_envs, -1)
        else:
            alpha = torch.softmax(alpha_logits, dim=-1)

        # 三策略融合
        if self.use_dance:
            # 获取舞蹈参考动作（当前帧）
            step_idx = self.uenv.episode_length_buf.long() % self.dance_trajectory.shape[0]
            dance_action = self.dance_trajectory[step_idx]  # (N, 29)
            hybrid = (alpha[:, 0:1] * self.label_smp + 
                      alpha[:, 1:2] * self.label_rough +
                      alpha[:, 2:3] * dance_action)
        else:
            hybrid = (alpha[:, 0:1] * self.label_smp + 
                      alpha[:, 1:2] * self.label_rough)

        self._last_hybrid, self._last_alpha = hybrid, alpha
        self._last_label_smp, self._last_label_rough = self.label_smp, self.label_rough

        obs_dict, _r, terminated, truncated, extras = self.env.step(hybrid)
        self._raw_obs = obs_dict["policy"]
        reward, logs = self._compute_reward(hybrid, alpha, terminated)
        self.prev_alpha = alpha.detach()

        dones = (terminated | truncated).to(torch.long)
        extras = dict(extras or {})
        extras.setdefault("log", {}).update(logs)
        extras["time_outs"] = truncated
        extras["terminated"] = terminated

        self._compute_labels()
        return (
            TensorDict({"policy": self._fusion_obs()}, batch_size=[self.num_envs]),
            reward,
            dones,
            extras,
        )

    def _compute_reward(self, hybrid, alpha, terminated):
        cfg = self.rcfg
        ids = self.sim_joint_ids
        r_sim = reward_similarity(hybrid[:, ids], self.label_smp[:, ids], cfg["sim_sigma"])
        # smp 扩散去噪打分: 给实际执行完(self.env.step(hybrid)之后)的真实机器人动作打"像不像人"的分,
        # 不管当前 alpha 更偏向地形策略还是舞蹈, 融合出来的动作都要持续被这个分数约束 - 用户要求
        # "一直按smp的风格走", 不能只是训 rough_policy 时加一次就完事, 融合阶段自己的奖励里也要有。
        # G1_29DOF_RoughEnvCfg 的 init_smp_state startup 事件已经在 self.uenv 上装好了
        # _smp_bundle/_smp_buffer/_smp_normalizer/_smp_joint_ids/_smp_ee_ids, 这里直接复用。
        # 注意: 不要再让 base env 自己的 rewards.smp_style 也算一遍 - 那个也会触发
        # _update_buffer_from_sim() 里的滑动窗口更新, 一步里更新两次会把窗口错位
        # (见 train_fusion_v2.py create_isaac_env() 里 env_cfg.rewards.smp_style = None)。
        r_smp_style = smp_bridge.smp_guidance_reward(self.uenv)
        # exposed so external callers (e.g. dagger data collection) can read the per-env raw score
        # without calling smp_guidance_reward a second time - it has the same double-buffer-update
        # hazard as base env's own rewards.smp_style (see the comment above): calling it twice per
        # real timestep rolls MotionFeatureBuffer's window twice and corrupts it.
        self._last_smp_score = r_smp_style.detach()
        forces = self.contact.data.net_forces_w
        p_pen = penalty_penetration(forces, self.illegal_ids, cfg["force_thresh"])
        foot_f = forces[:, self.foot_ids].norm(dim=-1)
        in_contact = (foot_f > cfg["force_thresh"]).float()
        p_float = (in_contact.sum(dim=-1) < 0.5).float()
        p_jit = penalty_jitter(
            self.robot.data.joint_acc, alpha, self.prev_alpha,
            cfg["w_joint_acc"], cfg["w_alpha_rate"]
        )
        left_sole_z = self.robot.data.body_pos_w[:, self._ankle_ids[0], 2] - 0.03
        right_sole_z = self.robot.data.body_pos_w[:, self._ankle_ids[1], 2] - 0.03
        p_foot_pen = foot_penetration.compute_foot_volume_penetration(
            self._left_foot_scanner, self._right_foot_scanner, left_sole_z, right_sole_z
        )

        # ---- 舞蹈参考轨迹奖励 ----
        if self.use_dance and self.w_dance_style > 0:
            step_idx = self.uenv.episode_length_buf.long() % self.dance_trajectory.shape[0]
            dance_ref = self.dance_trajectory[step_idx]  # (N, 29)
            r_dance = reward_similarity(hybrid[:, ids], dance_ref[:, ids], cfg["sim_sigma"])
            dance_reward = self.w_dance_style * r_dance
        else:
            r_dance = torch.zeros_like(r_sim)
            dance_reward = torch.zeros_like(r_sim)

        # 摔倒(提前 terminate, 不是超时 truncate)的惩罚 - 之前融合奖励里完全没有这一项,
        # 旧的 G1_29DOF_RoughEnvCfg 里虽然有 termination_penalty=-200, 但那是 base env
        # 自己的奖励, G1FusionEnv.step() 里 self.env.step(hybrid) 返回的 _r 整个被丢弃了,
        # alpha 网络对"摔倒"这件事几乎没有直接负反馈, 只能靠 illegal_contact/all_feet_air
        # 这些间接信号 - 权重明显不够, 训出来的 alpha 直接学会全权重压舞蹈骗高分(舞蹈参考轨迹
        # 是固定的、容易拟合, rew/dance 能稳定到 0.98, 比认真走地形好赚), 然后一直摔。
        p_terminate = terminated.float()
        # 跟 p_terminate 不是同一件事: p_terminate 只在摔倒那一步给一次性惩罚, r_alive 是
        # "每多活一步就有一份持续的正反馈", 覆盖 p_terminate 之前那些"还没摔但也没拿到其他
        # 奖励"的步数 (跟 InstinctLab parkour reward 的 is_alive 同一个思路, 见
        # g1_29dof_rough_env_cfg.py 的 is_alive 奖励项)。
        r_alive = (~terminated).float()

        reward = (
            cfg["w_sim"] * r_sim
            + cfg.get("w_smp_style", 0.0) * r_smp_style
            + cfg.get("w_alive", 0.0) * r_alive
            - cfg["w_penetrate"] * p_pen
            - cfg["w_float"] * p_float
            - cfg["w_jitter"] * p_jit
            - cfg.get("w_terminate", 0.0) * p_terminate
            - cfg.get("w_foot_penetrate", 0.0) * p_foot_pen
            + dance_reward
        )

        logs = {
            "rew/sim": r_sim.mean().item(),
            "rew/smp_style": r_smp_style.mean().item(),
            "rew/alive": r_alive.mean().item(),
            "pen/illegal_contact": p_pen.mean().item(),
            "pen/all_feet_air": p_float.mean().item(),
            "pen/jitter": p_jit.mean().item(),
            "pen/terminate": p_terminate.mean().item(),
            "pen/foot_volume": p_foot_pen.mean().item(),
            "alpha/a2_mean": alpha[:, 1].mean().item(),
        }
        if self.use_dance:
            logs["rew/dance"] = r_dance.mean().item()
            if alpha.shape[-1] >= 3:
                logs["alpha/a3_mean"] = alpha[:, 2].mean().item()

        # ---- 切换奖励 ----
        if self.w_switch > 0:
            lids = self.leg_joint_ids
            d = (self.label_rough[:, lids] - self.label_smp[:, lids]).norm(dim=-1) / (len(lids) ** 0.5)
            frac = ((d - self.sw_d_lo) / (self.sw_d_hi - self.sw_d_lo)).clamp(0, 1)
            a2_tgt = self.sw_a_lo + frac * (self.sw_a_hi - self.sw_a_lo)
            if self.sw_ema > 0 and self._alpha_tgt_prev is not None:
                a2_tgt = self.sw_ema * self._alpha_tgt_prev + (1 - self.sw_ema) * a2_tgt
            self._alpha_tgt_prev = a2_tgt.detach()
            switch_err = (alpha[:, 1] - a2_tgt) ** 2
            reward = reward - self.w_switch * switch_err
            logs["switch/target_a2"] = a2_tgt.mean().item()
            logs["switch/leg_divergence"] = d.mean().item()
            logs["switch/err"] = switch_err.mean().item()

        return reward, logs

    def velocity_error(self) -> torch.Tensor:
        cmd = self.uenv.command_manager.get_command("base_velocity")
        actual = self.robot.data.root_lin_vel_b
        return (cmd[:, :2] - actual[:, :2]).norm(dim=-1)

    def joint_snapshot(self, n_envs: int) -> dict:
        d = self.robot.data
        torque = getattr(d, "applied_torque", getattr(d, "computed_torque", d.joint_effort_target))
        f = lambda t: t[:n_envs].detach().cpu().numpy().copy()
        result = {
            "joint_pos": f(d.joint_pos),
            "joint_vel": f(d.joint_vel),
            "joint_acc": f(d.joint_acc),
            "joint_torque": f(torque),
            "label_smp": f(self._last_label_smp),
            "label_rough": f(self._last_label_rough),
            "hybrid": f(self._last_hybrid),
            "alpha": f(self._last_alpha),
            "foot_force": f(self.contact.data.net_forces_w[:, self.foot_ids]),
            "root_lin_vel": f(d.root_lin_vel_b),
            "root_ang_vel": f(d.root_ang_vel_b),
            "projected_gravity": f(d.projected_gravity_b),
        }
        return result

    def close(self):
        self.env.close()