r"""
统一评估: 同一协议下对比四种策略, 输出数据表 (这是"采集数据"的主脚本)

  flat    : α=[1,0]     纯模仿侧 (基线)
  rough   : α=[0,1]     纯感知侧 (基线)
  avg     : α=[0.5,0.5] 朴素平均 (基线, 证明"学习式融合"必要性的关键对照)
  fusion  : 训练好的 α 网络

每种模式跑同样步数、同样地形、同样随机种子, 统计:
  存活步数 / 摔倒率 / 速度跟踪误差 / 非法接触率 / 双脚离地率 / α 统计

用法:
  isaaclab.bat -p ".../isaac/eval_policies.py" ^
     --flat_policy <..> --rough_policy <..> --checkpoint <融合ckpt> ^
     --task Isaac-Velocity-Rough-G1-v0 --robot g1 --num_envs 256 --steps 2000 --headless
"""
from __future__ import annotations
import argparse, csv, glob, os, re, sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--flat_policy", required=True)
parser.add_argument("--rough_policy", required=True)
parser.add_argument("--checkpoint", default=None, help="融合 model_*.pt, 默认自动找最新")
parser.add_argument("--task", default="Isaac-Velocity-Rough-G1-v0")
parser.add_argument("--robot", default="g1", choices=["g1", "anymal_c"])
parser.add_argument("--exp_name", default="g1_fusion_stage1")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--out", default="eval_results")
parser.add_argument("--record", action="store_true",
                    help="录制逐帧关节数据到 npz")
parser.add_argument("--record_envs", type=int, default=4,
                    help="录制前几个环境 (默认4, 控制文件大小)")
parser.add_argument("--record_steps", type=int, default=1000,
                    help="录制前多少步")
parser.add_argument("--modes", default="all",
                    help="逗号分隔: flat,rough,avg,fusion 或 all")
parser.add_argument("--terrain", default="rough", choices=["rough", "flat"],
                    help="评估地形: rough=崎岖(默认), flat=平坦")
parser.add_argument("--dagger_ckpt", default=None,
                    help="监督训的 alpha 网络 (dagger_alpha_net.pt), 设了则跑 dagger 模式")
parser.add_argument("--terrain_obs", action="store_true",
                    help="alpha 网络使用地形观测 (评估 v3 模型时必须加)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym                                   # noqa: E402
import torch                                              # noqa: E402
from rsl_rl.runners import OnPolicyRunner                 # noqa: E402
import isaaclab_tasks                                     # noqa: F401, E402
from isaaclab_tasks.utils.parse_cfg import (              # noqa: E402
    parse_env_cfg, load_cfg_from_registry)
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg  # noqa: E402
from importlib import metadata                            # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from isaac.fusion_env_g1 import G1FusionEnv               # noqa: E402

REWARD_CFG = {
    "w_sim": 1.0, "sim_sigma": 0.5,
    "w_penetrate": 2.0, "force_thresh": 1.0,
    "w_float": 1.0,
    "w_jitter": 1.0, "w_joint_acc": 2.5e-7, "w_alpha_rate": 0.1,
    "use_terrain_obs": False,   # 运行时按 --terrain_obs 覆盖
}

MODES = {
    "dagger": "DAGGER",   # 监督训的 alpha 网络 (需 --dagger_ckpt)
    "flat":   torch.tensor([1.0, 0.0]),    # 纯模仿侧
    "rough":  torch.tensor([0.0, 1.0]),    # 纯感知侧
    "avg":    torch.tensor([0.5, 0.5]),    # 朴素平均
    "fusion": None,                        # 学到的 alpha
}


def find_latest_ckpt(exp: str) -> str:
    cands = glob.glob(os.path.join("logs", "rsl_rl", exp, "*", "model_*.pt"))
    if not cands:
        raise FileNotFoundError(f"logs/rsl_rl/{exp} 下没找到 checkpoint")
    return max(cands, key=lambda p: (os.path.basename(os.path.dirname(p)),
                                     int(re.search(r"model_(\d+)", p).group(1))))


@torch.inference_mode()
def run_mode(name, alpha_override, ckpt, args, device):
    """跑一种模式, 返回指标字典"""
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    env_cfg.seed = 12345                       # 各模式同一种子, 保证可比
    REWARD_CFG["use_terrain_obs"] = args.terrain_obs
    # 评估地形: flat=纯平面, rough=纯崎岖。直接重构地形生成器, 保证 height_scan
    # 维度不变(采样点数由 scanner 决定, 与子地形种类无关)。
    try:
        import isaaclab.terrains as terrain_gen
        from isaaclab.terrains import TerrainGeneratorCfg
        if args.terrain == "flat":
            subs = {"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)}
            note = "纯平面"
        else:
            subs = {
                "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                    proportion=0.4, step_height_range=(0.05, 0.20),
                    step_width=0.3, platform_width=3.0, border_width=1.0, holes=False),
                "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                    proportion=0.3, grid_width=0.45,
                    grid_height_range=(0.05, 0.2), platform_width=2.0),
                "rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.3, noise_range=(0.02, 0.10),
                    noise_step=0.02, border_width=0.25),
            }
            note = "纯崎岖"
        env_cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(8.0, 8.0), border_width=20.0, num_rows=10, num_cols=20,
            horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75,
            use_cache=False, curriculum=False, sub_terrains=subs)
        env_cfg.scene.terrain.max_init_terrain_level = 5
        print(f"[eval] 评估地形: {note} ({args.terrain})", flush=True)
    except Exception as e:
        print(f"[warn] 地形重构失败, 用默认: {e}", flush=True)

    base_env = gym.make(args.task, cfg=env_cfg)
    env = G1FusionEnv(base_env, args.flat_policy, args.rough_policy, REWARD_CFG,
                      robot=args.robot, device=device, alpha_override=alpha_override)

    # ---- dagger 模式: 加载监督训的 alpha 网络, 挂到 env ----
    policy = None
    if alpha_override == "DAGGER":
        import torch.nn as nn
        pack = torch.load(args.dagger_ckpt, map_location=device)
        hid = pack.get("hid", [256, 128]); in_dim = pack["in_dim"]
        layers, dd = [], in_dim
        for h in hid:
            layers += [nn.Linear(dd, h), nn.ELU()]; dd = h
        layers.append(nn.Linear(dd, 1))
        net = nn.Sequential(*layers).to(device)
        # 重建时用 Sequential, 但保存的是带 sigmoid 的 forward -> 包一层
        class _Wrap(nn.Module):
            def __init__(s, seq): super().__init__(); s.seq = seq
            def forward(s, x): return torch.sigmoid(s.seq(x))
        wrap = _Wrap(net).to(device)
        # state_dict 的 key 是 net.0/net.2...; 适配
        sd = pack["alpha_net"]
        new_sd = {k.replace("net.", "seq."): v for k, v in sd.items()}
        wrap.load_state_dict(new_sd)
        wrap.eval()
        env.dagger_net = wrap
        print(f"    已加载 dagger alpha 网络: {args.dagger_ckpt}", flush=True)

    # ---- fusion 模式: 在这个 env 上直接构建 runner 加载权重 ----
    elif alpha_override is None:
        agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        agent_cfg.experiment_name = args.exp_name
        agent_cfg = handle_deprecated_rsl_rl_cfg(
            agent_cfg, metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
        runner.load(ckpt, map_location=device)
        policy = runner.get_inference_policy(device=device)
        print(f"    已加载融合 checkpoint: {ckpt}", flush=True)

    obs = env.get_observations().to(device)

    # ---- 逐帧关节数据录制缓冲 ----
    rec = None
    if args.record:
        ne = min(args.record_envs, env.num_envs)
        rec = {k: [] for k in ["joint_pos", "joint_vel", "joint_acc", "joint_torque",
                               "label1", "label2", "hybrid", "alpha",
                               "foot_force", "root_lin_vel", "root_ang_vel",
                               "projected_gravity", "done"]}
    n_fall = n_done = 0
    vel_err_sum = illegal_sum = feet_air_sum = a2_sum = 0.0
    steps = 0

    for _step in range(args.steps):
        if _step % 250 == 0:
            print(f"    ... {_step}/{args.steps} 步", flush=True)
        if env.dagger_net is not None:
            act = torch.zeros(env.num_envs, 2, device=device)  # 占位, env内部用dagger_net
        elif policy is not None:
            act = policy(obs)
        else:
            act = torch.zeros(env.num_envs, 2, device=device)
        obs, _rew, dones, extras = env.step(act)
        obs = obs.to(device)
        log = extras.get("log", {})

        if rec is not None and _step < args.record_steps:
            snap = env.joint_snapshot(ne)
            for k, v in snap.items():
                rec[k].append(v)
            rec["done"].append(dones[:ne].cpu().numpy().copy())

        vel_err_sum += env.velocity_error().mean().item()
        illegal_sum += log.get("pen/illegal_contact", 0.0)
        feet_air_sum += log.get("pen/all_feet_air", 0.0)
        a2_sum += log.get("alpha/a2_mean", 0.0)
        steps += 1

        term = extras.get("terminated")
        if term is not None:
            n_fall += int(term.sum().item())
        n_done += int(dones.sum().item())

    res = {
        "mode": name,
        "回合数": n_done,
        # 平均回合长度 = 总环境步数 / 结束的回合数 (不受重置时机影响)
        "平均存活步数": round(steps * args.num_envs / max(n_done, 1), 1),
        "摔倒率": round(n_fall / max(n_done, 1), 4),
        "速度跟踪误差(m/s)": round(vel_err_sum / steps, 4),
        "非法接触惩罚": round(illegal_sum / steps, 4),
        "双脚离地率": round(feet_air_sum / steps, 4),
        "α2均值": round(a2_sum / steps, 4),
    }
    if rec is not None:
        import numpy as np
        os.makedirs(args.out, exist_ok=True)
        arrs = {k: np.stack(v) for k, v in rec.items() if len(v)}   # (T, E, D)
        arrs["joint_names"] = np.array(env.robot.data.joint_names)
        path = os.path.join(args.out, f"joints_{name}_{args.terrain}.npz")
        np.savez_compressed(path, **arrs)
        print(f"    关节数据已保存 -> {path}  "
              f"(形状 T={arrs['joint_pos'].shape[0]}, E={arrs['joint_pos'].shape[1]}, "
              f"J={arrs['joint_pos'].shape[2]})", flush=True)

        # 汇总统计追加进结果行
        res["关节力矩均值(Nm)"] = round(float(np.abs(arrs["joint_torque"]).mean()), 3)
        res["关节力矩峰值(Nm)"] = round(float(np.abs(arrs["joint_torque"]).max()), 2)
        res["hybrid与label1偏差"] = round(
            float(np.abs(arrs["hybrid"] - arrs["label1"]).mean()), 4)

    env.close()
    return res


def main():
    device = "cuda:0"
    sel = list(MODES) if args.modes == "all" else [
        m.strip() for m in args.modes.split(",")]
    for m in sel:
        if m not in MODES:
            raise SystemExit(f"未知模式 {m}, 可选: {list(MODES)}")

    ckpt = None
    if "fusion" in sel:
        ckpt = args.checkpoint or find_latest_ckpt(args.exp_name)
        print(f"[eval] 融合 checkpoint: {ckpt}")

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, f"policy_comparison_{args.terrain}.csv")

    for i, name in enumerate(sel, 1):
        print(f"\n{'='*60}\n[{i}/{len(sel)}] 评估模式: {name}  "
              f"(建场景约需1-2分钟, 请耐心等待, 不要 Ctrl+C)\n{'='*60}", flush=True)
        override = MODES[name]
        row = run_mode(name, override, ckpt, args, device)
        torch.cuda.empty_cache()

        # ---- 立即落盘 (追加), 中途停了也不丢已跑完的模式 ----
        keys = list(row.keys())
        exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            if not exists:
                w.writeheader()
            w.writerow(row)
        print("\n结果: " + " | ".join(f"{k}={v}" for k, v in row.items()), flush=True)
        print(f"已追加到 {csv_path}", flush=True)

    # ---- 汇总成 markdown ----
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    keys = list(rows[0].keys())
    md = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in rows:
        md.append("| " + " | ".join(str(r[k]) for k in keys) + " |")
    md_text = "\n".join(md)
    with open(os.path.join(args.out, f"policy_comparison_{args.terrain}.md"), "w",
              encoding="utf-8") as f:
        f.write(f"# 策略对比 ({args.task}, {args.num_envs} envs x {args.steps} steps)\n\n")
        f.write(md_text + "\n")
    print("\n" + "=" * 70 + "\n" + md_text + "\n" + "=" * 70)
    print(f"\n数据已保存 -> {args.out}/policy_comparison.csv 和 .md")
    simulation_app.close()


if __name__ == "__main__":
    main()
