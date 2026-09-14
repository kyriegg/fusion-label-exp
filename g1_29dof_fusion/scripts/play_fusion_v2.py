#!/usr/bin/env python
"""
三策略融合训练结果回放 (SMP + 地形 + 舞蹈参考轨迹)

默认非 headless, 会弹出 Isaac Sim 窗口实时显示机器人。

用法:
    D:\\IsaacLab\\isaaclab_venv\\Scripts\\python.exe play_fusion_v2.py ^
        --alpha-checkpoint "D:\\robot paper\\fusion_label_exp\\logs\\rsl_rl\\smp_rough_dance\\model_3499.pt"
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="三策略融合训练结果回放")
parser.add_argument(
    "--alpha-checkpoint", type=str,
    default="D:/robot paper/fusion_label_exp/logs/rsl_rl/smp_rough_dance/model_3499.pt",
    help="训好的 alpha 融合网络 checkpoint",
)
parser.add_argument(
    "--smp-policy", type=str,
    default="D:/robot paper/smp/logs/rsl_rl/smp_forward_g1/2026-08-31_15-01-57_smp_forward_g1/model_4999.pt",
    help="SMP 自然行走策略路径",
)
parser.add_argument(
    "--rough-policy", type=str,
    default="D:/robot paper/fusion_label_exp/logs/rsl_rl/g1_29dof_rough/2026-09-11_00-24-06/model_3599.pt",
    help="地形感知策略路径",
)
parser.add_argument(
    "--dance-ref", type=str,
    default="D:/robot paper/fusion_label_exp/reference/macarena/q.csv",
    help="舞蹈参考轨迹 CSV 路径 (被 --no-dance 覆盖)",
)
parser.add_argument("--no-dance", action="store_true", help="回放两路融合(无舞蹈)的 alpha checkpoint")
parser.add_argument("--num-envs", type=int, default=4, help="并行环境数量 (回放用小一点方便看)")
parser.add_argument(
    "--fixed-alpha", type=str, default=None, choices=["smp", "rough"],
    help="诊断用: 绕过训好的 alpha 网络, 强制锁定成纯 smp 或纯 rough (用 G1FusionEnv 自带的 "
         "alpha_override), 用来单独测每个冻结子策略在当前地形/指令分布下自己站不站得住 - "
         "不需要 --alpha-checkpoint",
)
parser.add_argument(
    "--dagger-ckpt", type=str, default=None,
    help="用 train_dagger_offline.py 训出来的监督式 alpha 网络 (dagger_alpha_net.py 的格式) "
         "代替 --alpha-checkpoint 的 PPO 网络, 每步真正跑一次前向(不是像 --fixed-alpha 那样锁死), "
         "用来验证 alpha 是否真的随情境切换",
)
parser.add_argument(
    "--max-steps", type=int, default=None,
    help="诊断用: 跑够这么多步就自动退出并打印摔倒率统计, 不设置则跟原来一样一直跑到窗口关闭",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# AppLauncher 的 --headless 默认就是 False (开窗口), 这里不用额外处理,
# 想无头运行就自己加 --headless
if args.no_dance:
    args.dance_ref = None

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- 以下全部要在 simulation_app 起来之后才能 import ----
import os  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, "D:/robot paper/fusion_label_exp/isaac")

import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.g1_29dof_rough_env_cfg import (  # noqa: E402
    G1_29DOF_RoughEnvCfg_PLAY,
)

import smp_bridge  # noqa: E402
from fusion_env_g1_v2 import G1FusionEnv  # noqa: E402
from policy_loader import load_rsl_rl_policy  # noqa: E402


def get_reward_cfg():
    # 跟 train_fusion_v2.py 的 get_reward_cfg() 保持同步 (回放脚本这份此前一直没跟上
    # 训练脚本 09-11 诊断后的修复 - w_joint_acc/switch_d_lo/switch_d_hi 数值走的是老草稿值,
    # 只影响回放时打印的 log 数值是否准确, 不影响已训好的 alpha 网络实际怎么走, 但会让
    # 回放日志跟训练日志对不上, 顺手修一致)
    return {
        "w_sim": 1.0, "sim_sigma": 0.1,
        "w_smp_style": 1.0,
        "w_dance_style": 0.3,
        "w_penetrate": 1.0, "force_thresh": 5.0, "w_float": 0.5,
        "w_jitter": 0.1, "w_joint_acc": 1.25e-7, "w_alpha_rate": 0.1,
        "w_terminate": 5.0,
        "w_alive": 0.1,
        "w_foot_penetrate": 0.5,
        "w_switch": 0.5, "switch_a_lo": 0.15, "switch_a_hi": 0.90,
        "switch_d_lo": 0.5, "switch_d_hi": 2.5, "switch_ema": 0.3,
        "use_terrain_obs": True,
    }


def main():
    env_cfg = G1_29DOF_RoughEnvCfg_PLAY()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.actions.joint_pos.joint_names = list(smp_bridge.JOINT_NAMES)
    env_cfg.actions.joint_pos.preserve_order = True
    env_cfg.rewards.smp_style = None  # 避免跟 G1FusionEnv 自己的 smp_guidance_reward 调用重复

    base_env = ManagerBasedRLEnv(cfg=env_cfg)

    alpha_override = None
    if args.fixed_alpha is not None:
        num_actions = 3 if args.dance_ref is not None else 2
        alpha_override = torch.zeros(1, num_actions)
        alpha_override[0, 0 if args.fixed_alpha == "smp" else 1] = 1.0
        print(f"[play] 诊断模式: alpha 强制锁定为纯 {args.fixed_alpha} {alpha_override.tolist()}")

    fusion_env = G1FusionEnv(
        base_env=base_env,
        smp_policy_jit=args.smp_policy,
        rough_policy_jit=args.rough_policy,
        dance_ref_csv=args.dance_ref,
        reward_cfg=get_reward_cfg(),
        robot="g1",
        device=args.device,
        alpha_override=alpha_override,
    )

    dagger_net = None
    if args.dagger_ckpt is not None:
        from dagger_alpha_net import alpha2_to_alpha, alpha_net_input, load_alpha_net
        print(f"[play] 加载 DAgger 监督式 alpha 网络: {args.dagger_ckpt}")
        dagger_net = load_alpha_net(args.dagger_ckpt, args.device)
    elif args.fixed_alpha is None:
        print(f"[play] 加载 alpha 网络: {args.alpha_checkpoint}")
        alpha_policy = load_rsl_rl_policy(args.alpha_checkpoint, args.device)
    else:
        alpha_policy = None

    obs_dict = fusion_env.get_observations()
    print("[play] 开始回放, 关掉窗口或 Ctrl+C 结束" + ("" if args.max_steps is None else f" (最多 {args.max_steps} 步自动退出)"))
    # 诊断统计: 每个 env 各自"活了多久"(距上次 reset 的步数), 每次 terminate (真摔倒, 不是 time_out)
    # 就把这条命记进 episode_lengths_at_fall, 用来算平均存活步数/摔倒率, 不用非要盯着窗口看。
    steps_alive = torch.zeros(args.num_envs)
    episode_lengths_at_fall = []
    num_episodes_ended = 0
    num_falls = 0
    alpha2_history = []
    step_i = 0
    with torch.no_grad():
        while simulation_app.is_running():
            if dagger_net is not None:
                # DAgger 网络每步真正跑前向(不是 --fixed-alpha 那种锁死常量), 通过
                # G1FusionEnv 的 alpha_override 机制直接注入这一步算出来的 alpha, 绕开
                # step() 内部默认的 softmax(alpha_logits) 路径 (dagger_net 自己的输出
                # 已经是 sigmoid 过的合法概率, 不需要也不能再 softmax 一次)。
                x = alpha_net_input(fusion_env.label_smp, fusion_env.label_rough, fusion_env._raw_obs)
                alpha2 = dagger_net(x)
                fusion_env.alpha_override = alpha2_to_alpha(alpha2)
                alpha2_history.append(alpha2.mean().item())
                alpha_logits = torch.zeros(args.num_envs, 2, device=args.device)  # 占位, 会被 alpha_override 覆盖
            elif args.fixed_alpha is not None:
                alpha_logits = torch.zeros(args.num_envs, alpha_override.shape[-1], device=args.device)
            else:
                alpha_logits = alpha_policy(obs_dict["policy"])
            obs_dict, reward, dones, extras = fusion_env.step(alpha_logits)

            terminated = extras.get("terminated")
            steps_alive += 1
            if terminated is not None:
                fell_ids = terminated.nonzero(as_tuple=False).flatten().cpu()
                for i in fell_ids.tolist():
                    episode_lengths_at_fall.append(steps_alive[i].item())
                    num_falls += 1
                num_episodes_ended += int(dones.sum().item())
                steps_alive[dones.cpu().bool()] = 0

            step_i += 1
            if args.max_steps is not None and step_i >= args.max_steps:
                break

    if args.max_steps is not None:
        avg_len = sum(episode_lengths_at_fall) / len(episode_lengths_at_fall) if episode_lengths_at_fall else float("nan")
        fall_frac = num_falls / num_episodes_ended if num_episodes_ended > 0 else float("nan")
        print(
            f"[play] 诊断统计 ({step_i} 步, {args.num_envs} 个并行env): "
            f"结束的episode数={num_episodes_ended}, 其中摔倒结束={num_falls} ({fall_frac:.1%}), "
            f"摔倒前平均存活步数={avg_len:.1f}"
        )
        if alpha2_history:
            a2t = torch.tensor(alpha2_history)
            print(
                f"[play] DAgger alpha2 统计: 均值={a2t.mean():.3f} 标准差={a2t.std():.3f} "
                f"最小={a2t.min():.3f} 最大={a2t.max():.3f} (标准差越大说明alpha真的在随情境变化, 不是收敛到常数)"
            )
        # 诊断模式是一次性跑批, 没有需要干净保留的状态 - simulation_app.close() 这条关闭路径
        # 今天已经反复卡死(有一次卡了 13 小时才被手动杀掉), 打完统计直接硬退避开它, 不再走
        # 正常关闭流程。
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[play] 用户中断")
    simulation_app.close()
