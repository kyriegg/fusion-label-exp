r"""
DAgger 第一步: 采集监督训练数据集 (需 Isaac)

参考用户早前的 fusion-label-exp 仓库 (github.com/kyriegg/fusion-label-exp,
isaac/dagger/collect_dagger_data.py) 的思路移植 - 在混合地形(50%平面 + 现有几种崎岖地形)
上用感知策略(rough_policy, 经诊断证明比 smp 称职得多, 见 project 记忆)驱动机器人跑,
保证覆盖两种地形下都稳定的状态, 每帧存:
  - raw_obs    (base env 自己的 314 维观测, 含 127 维本体 + 187 维 height_scan)
  - label_smp  (smp_policy 在当前状态下会输出的动作, 29 维, 典范关节序)
  - label_rough(rough_policy 在当前状态下会输出的动作, 29 维, 已转典范关节序)
  - smp_score  (smp_bridge 扩散去噪打分, 衡量"刚执行完的真实动作像不像人", 直接从
                G1FusionEnv._last_smp_score 读, 不重复调用 smp_guidance_reward 以免
                MotionFeatureBuffer 滑动窗口被重复更新)

跟老仓库的差异: 老仓库离线训练脚本用的目标信号是 height_scan 粗糙度(物理地形属性);
这次改用 smp_score(用户 2026-09-14 的想法) —— 低分(动作不自然, 比如在崎岖地形上勉强
稳住重心)时目标 alpha 应偏向 rough, 这个映射在 train_dagger_offline.py 里做, 这里只负责
把 smp_score 存下来。

用法 (D:\IsaacLab\isaaclab_venv 环境下):
    python collect_dagger_data.py --num-envs 256 --steps 3000 --headless

产出: D:\robot paper\fusion_label_exp\dagger_data\dagger_dataset.npz
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="DAgger 数据采集")
parser.add_argument(
    "--smp-policy", type=str,
    default="D:/robot paper/smp/logs/rsl_rl/smp_forward_g1/2026-08-31_15-01-57_smp_forward_g1/model_4999.pt",
)
parser.add_argument(
    "--rough-policy", type=str,
    default="D:/robot paper/fusion_label_exp/logs/rsl_rl/g1_29dof_rough/2026-09-11_00-24-06/model_3599.pt",
)
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--sample-every", type=int, default=4, help="每N帧存一次, 控制数据体积")
parser.add_argument("--out", type=str, default="D:/robot paper/fusion_label_exp/dagger_data/dagger_dataset.npz")
parser.add_argument(
    "--dagger-ckpt", type=str, default=None,
    help="迭代 DAgger 用: 传入上一轮训出来的 alpha 网络, 改用它(而不是纯 rough)驱动这轮采集, "
         "让新采到的数据覆盖'部署时真实会遇到的状态分布', 修正单轮行为克隆的分布漂移问题 "
         "(2026-09-14 diagnostic 发现纯 rough 驱动采集训出来的网络一部署就塌缩回常数, 见项目记忆)",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- 以下全部要在 simulation_app 起来之后才能 import ----
import os  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.terrains as terrain_gen  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

_FUSION_ISAAC_DIR = "D:/robot paper/fusion_label_exp/isaac"
if _FUSION_ISAAC_DIR not in sys.path:
    sys.path.insert(0, _FUSION_ISAAC_DIR)
import smp_bridge  # noqa: E402
from fusion_env_g1_v2 import G1FusionEnv  # noqa: E402

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.g1_29dof_rough_env_cfg import (  # noqa: E402
    G1_29DOF_RoughEnvCfg,
    ROUGH_TERRAINS_CFG,
    _TARGET_FLAT_PATCH_CFG,
)


def build_mixed_terrain():
    """50% 纯平面 + 50%(原比例内分配)现有崎岖子地形, 每种都带 flat_patch "target"
    (TargetVelocityCommand 需要), 让采集到的数据同时覆盖平地(smp该被信任)和崎岖
    (rough该被信任)两种情形, 给 DAgger 目标信号足够的对比度 - 跟老仓库 v6 发现"地形
    生成器里从来没有真平地, 导致 alpha 无从学习切换"是同一个坑, 这里从一开始就避开。
    """
    sub_terrains = {
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5, flat_patch_sampling={"target": _TARGET_FLAT_PATCH_CFG}),
    }
    for name, cfg in ROUGH_TERRAINS_CFG.sub_terrains.items():
        sub_terrains[name] = cfg.replace(
            proportion=cfg.proportion * 0.5, flat_patch_sampling={"target": _TARGET_FLAT_PATCH_CFG}
        )
    return ROUGH_TERRAINS_CFG.replace(sub_terrains=sub_terrains)


@torch.inference_mode()
def main():
    env_cfg = G1_29DOF_RoughEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.scene.terrain.terrain_generator = build_mixed_terrain()
    env_cfg.actions.joint_pos.joint_names = list(smp_bridge.JOINT_NAMES)
    env_cfg.actions.joint_pos.preserve_order = True
    env_cfg.rewards.smp_style = None  # 避免跟 G1FusionEnv 自己的 smp_guidance_reward 调用重复

    base_env = ManagerBasedRLEnv(cfg=env_cfg)

    # reward_cfg 只是为了让 G1FusionEnv._compute_reward 别 KeyError, 数值本身在数据采集阶段不重要
    # (不训练 PPO), 照抄 train_fusion_v2.py 的 get_reward_cfg() 保证键齐全。
    reward_cfg = {
        "w_sim": 1.0, "sim_sigma": 0.1, "w_smp_style": 1.0, "w_dance_style": 0.3,
        "w_penetrate": 1.0, "force_thresh": 5.0, "w_float": 0.5, "w_jitter": 0.1,
        "w_terminate": 5.0, "w_alive": 0.1, "w_foot_penetrate": 0.5,
        "w_joint_acc": 1.25e-7, "w_alpha_rate": 0.1,
        "w_switch": 0.5, "switch_a_lo": 0.15, "switch_a_hi": 0.90,
        "switch_d_lo": 0.5, "switch_d_hi": 2.5, "switch_ema": 0.3,
        "use_terrain_obs": True,
    }

    dagger_net = None
    if args.dagger_ckpt is not None:
        from dagger_alpha_net import alpha2_to_alpha, alpha_net_input, load_alpha_net
        dagger_net = load_alpha_net(args.dagger_ckpt, args.device)
        print(f"[dagger-collect] 迭代模式: 用上一轮 alpha 网络驱动采集 ({args.dagger_ckpt})")

    fusion_env = G1FusionEnv(
        base_env=base_env,
        smp_policy_jit=args.smp_policy,
        rough_policy_jit=args.rough_policy,
        dance_ref_csv=None,
        reward_cfg=reward_cfg,
        robot="g1",
        device=args.device,
        # 第一轮: 用感知策略驱动采集(诊断已证明它比 smp 称职得多), 保证混合地形上机器人大部分
        # 时候都走得稳、能覆盖两种地形下的真实状态分布。迭代轮(dagger_net 不为 None)改成动态
        # 每步设置, 见下面循环体。
        alpha_override=None if dagger_net is not None else torch.tensor([[0.0, 1.0]]),
    )

    obs_dict = fusion_env.get_observations()
    buf = {"raw_obs": [], "label_smp": [], "label_rough": [], "smp_score": []}
    dummy_alpha_logits = torch.zeros(args.num_envs, 2, device=args.device)

    print(f"[dagger-collect] 开始采集: {args.num_envs} envs x {args.steps} 步, 每 {args.sample_every} 帧存一次")
    for step in range(args.steps):
        if step % 250 == 0:
            print(f"    ... {step}/{args.steps} 步", flush=True)
        raw_obs_before = fusion_env._raw_obs
        label_smp_before = fusion_env.label_smp
        label_rough_before = fusion_env.label_rough

        if dagger_net is not None:
            x = alpha_net_input(label_smp_before, label_rough_before, raw_obs_before)
            alpha2 = dagger_net(x)
            fusion_env.alpha_override = alpha2_to_alpha(alpha2)

        obs_dict, reward, dones, extras = fusion_env.step(dummy_alpha_logits)

        if step % args.sample_every == 0:
            buf["raw_obs"].append(raw_obs_before.detach().cpu().numpy().copy())
            buf["label_smp"].append(label_smp_before.detach().cpu().numpy().copy())
            buf["label_rough"].append(label_rough_before.detach().cpu().numpy().copy())
            buf["smp_score"].append(fusion_env._last_smp_score.cpu().numpy().copy())

    arrs = {}
    for k, v in buf.items():
        a = np.stack(v)  # (T', E, D) or (T', E) for smp_score
        arrs[k] = a.reshape(-1, a.shape[-1]) if a.ndim == 3 else a.reshape(-1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **arrs)
    N = arrs["raw_obs"].shape[0]
    print(f"\n[dagger-collect] 数据集 -> {args.out}")
    print(
        f"       样本数 N={N}  raw_obs={arrs['raw_obs'].shape[1]}维  "
        f"label={arrs['label_smp'].shape[1]}维"
    )
    q = np.percentile(arrs["smp_score"], [5, 25, 50, 75, 95])
    print(f"[dagger-collect] smp_score 分布: 5%={q[0]:.4f} 25%={q[1]:.4f} 中位={q[2]:.4f} 75%={q[3]:.4f} 95%={q[4]:.4f}")
    print(f"       -> {'分布跨度大, 适合训练' if q[4] - q[0] > 0.05 else '跨度偏小, 留意信号是否有区分度'}")

    sys.stdout.flush()
    os._exit(0)  # 避开 Isaac Sim 不可靠的正常关闭流程 (今天已反复卡死数小时, 见项目记忆)


if __name__ == "__main__":
    main()
