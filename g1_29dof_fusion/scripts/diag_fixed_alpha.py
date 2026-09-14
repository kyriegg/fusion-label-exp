#!/usr/bin/env python
"""
两路融合诊断: 固定 alpha (不训练 alpha 网络), 对比几个定值混合比在同一地形分布下的表现。

背景 (2026-09-12): smp_rough_nodance 训练到 iter~1369/3000 时, 摔倒率稳定卡在
95%~100%, Curriculum/terrain_levels 全程钉死 0.0000, alpha/a2_mean(rough权重)
从 0.53 跌到 0.20~0.25 附近不再变化, 而 switch/target_a2 一直认为应该给到 ~0.70~0.75。
奖励量级核算: w_sim*r_sim (~0.75) 量级上跟 w_penetrate*p_pen (~0.6) 相当, 但
w_terminate*p_terminate 因为 terminate 是"每回合只触发一次"的稀疏信号, 按步平均后
只有 ~0.06~0.08, 远小于 r_sim 的持续每步拉力。

这个脚本想回答的是: 到底是"奖励量级失衡让 alpha 学不到该往 rough 靠"(可以靠调权重解决),
还是"两个策略的关节目标角度线性相加本身就是个语义有问题的融合方式"(线性混合两个策略的动作,
在楼梯/箱子这种需要精确抬脚高度的场景下, 折中出来的动作可能比两个单独策略都差, 不管 alpha
怎么调都救不回来)。做法: 完全绕开 alpha 网络 (用 G1FusionEnv 自带的 alpha_override), 跑几个
固定混合比 (纯smp / 纯rough / 训练现在卡住的~0.22 / 0.5 / switch想要的~0.7), 同一地形分布下
统计摔倒率和地形等级。如果纯 rough(alpha=[0,1]) 本身就摔得少, 但混合比一高摔倒率就跳升,
说明是线性混合语义问题, 不是权重问题。

用法:
    D:\\IsaacLab\\isaaclab_venv\\Scripts\\python.exe diag_fixed_alpha.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="固定 alpha 融合诊断 (SMP + 地形感知, 无舞蹈)")
parser.add_argument(
    "--smp-policy", type=str,
    default="D:/robot paper/smp/logs/rsl_rl/smp_forward_g1/2026-08-31_15-01-57_smp_forward_g1/model_4999.pt",
)
parser.add_argument(
    "--rough-policy", type=str,
    default="D:/robot paper/fusion_label_exp/logs/rsl_rl/g1_29dof_rough/2026-09-11_00-24-06/model_3599.pt",
)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=600, help="每个 alpha 定值跑多少步")
parser.add_argument("--out", type=str, default="D:/robot paper/fusion_label_exp/eval_results/fixed_alpha_diag.csv")
parser.add_argument(
    "--mode", type=str, required=True,
    choices=["pure_smp", "pure_rough", "stuck_022", "naive_avg", "switch_want"],
    help="每次进程只跑一个模式再退出 - 实测在同一个 Kit 进程里连续 close()+新建第二个"
         "ManagerBasedRLEnv 会在第二个环境刚建完时被静默杀掉(exit 0, 无 traceback), "
         "不是 OOM 也不是脚本逻辑错误, 是 isaacsim 6.1 这个 Kit 版本对进程内重建仿真"
         "上下文不稳定。所以改成单模式单进程, 外层脚本/命令行分别起 5 次。",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- 以下全部要在 simulation_app 起来之后才能 import ----
import csv  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, "D:/robot paper/fusion_label_exp/isaac")

import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.g1_29dof_rough_env_cfg import (  # noqa: E402
    G1_29DOF_RoughEnvCfg,
)

import smp_bridge  # noqa: E402
from fusion_env_g1_v2 import G1FusionEnv  # noqa: E402


REWARD_CFG = {
    "w_sim": 1.0, "sim_sigma": 0.1,
    "w_smp_style": 1.0,
    "w_dance_style": 0.0,
    "w_penetrate": 1.0, "force_thresh": 5.0, "w_float": 0.5,
    "w_jitter": 0.1, "w_joint_acc": 1.25e-7, "w_alpha_rate": 0.1,
    "w_terminate": 5.0,
    "w_switch": 0.0,  # 诊断时不需要切换奖励参与(不训练), 只是构造函数要用到默认值
    "switch_a_lo": 0.15, "switch_a_hi": 0.90, "switch_d_lo": 0.5, "switch_d_hi": 2.5, "switch_ema": 0.3,
    "use_terrain_obs": True,
}

# alpha = [w_smp, w_rough]。0.22 是当前 smp_rough_nodance 训练卡住不动的均值,
# 0.72 是 switch 奖励认为"该给"的目标值 - 两头都测, 加上纯策略基线和朴素对半。
MODES = {
    "pure_smp":    (1.00, 0.00),
    "pure_rough":  (0.00, 1.00),
    "stuck_022":   (0.78, 0.22),
    "naive_avg":   (0.50, 0.50),
    "switch_want": (0.28, 0.72),
}


def make_env():
    env_cfg = G1_29DOF_RoughEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.actions.joint_pos.joint_names = list(smp_bridge.JOINT_NAMES)
    env_cfg.actions.joint_pos.preserve_order = True
    env_cfg.rewards.smp_style = None
    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    return base_env


@torch.inference_mode()
def run_mode(name, alpha_pair, base_env):
    alpha_t = torch.tensor(list(alpha_pair), device=args.device)
    fusion_env = G1FusionEnv(
        base_env=base_env, smp_policy_jit=args.smp_policy, rough_policy_jit=args.rough_policy,
        dance_ref_csv=None, reward_cfg=REWARD_CFG, robot="g1", device=args.device,
        alpha_override=alpha_t,
    )
    obs = fusion_env.get_observations()
    n_fall = n_done = 0
    illegal_sum = feet_air_sum = terrain_sum = 0.0
    steps = 0
    for _step in range(args.steps):
        act = torch.zeros(fusion_env.num_envs, fusion_env.num_actions, device=args.device)  # 占位, alpha_override 生效时不读这个
        obs, _rew, dones, extras = fusion_env.step(act)
        log = extras.get("log", {})
        illegal_sum += log.get("pen/illegal_contact", 0.0)
        feet_air_sum += log.get("pen/all_feet_air", 0.0)
        terrain_sum += fusion_env.uenv.scene.terrain.terrain_levels.float().mean().item()
        term = extras.get("terminated")
        if term is not None:
            n_fall += int(term.sum().item())
        n_done += int(dones.sum().item())
        steps += 1

    res = {
        "mode": name,
        "alpha_smp_rough": f"{alpha_pair[0]:.2f}/{alpha_pair[1]:.2f}",
        "回合数": n_done,
        "平均存活步数": round(steps * fusion_env.num_envs / max(n_done, 1), 1),
        "摔倒率": round(n_fall / max(n_done, 1), 4),
        "非法接触惩罚(均值)": round(illegal_sum / steps, 4),
        "双脚离地率(均值)": round(feet_air_sum / steps, 4),
        "地形等级(均值)": round(terrain_sum / steps, 4),
    }
    fusion_env.close()
    return res


def main():
    print("=" * 70)
    print(f"固定 alpha 融合诊断 (单模式单进程): {args.mode}")
    print("=" * 70)
    for path in [args.smp_policy, args.rough_policy]:
        if not os.path.exists(path):
            print(f"错误: 文件不存在: {path}")
            return

    alpha_pair = MODES[args.mode]
    print(f"\n{'='*60}\n评估模式: {args.mode} alpha(smp,rough)={alpha_pair}\n{'='*60}", flush=True)
    base_env = make_env()
    row = run_mode(args.mode, alpha_pair, base_env)
    base_env.close()
    print("结果: " + " | ".join(f"{k}={v}" for k, v in row.items()), flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    exists = os.path.isfile(args.out)
    keys = list(row.keys())
    with open(args.out, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if not exists:
            w.writeheader()
        w.writerow(row)
    print(f"\n结果已追加 -> {args.out}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
