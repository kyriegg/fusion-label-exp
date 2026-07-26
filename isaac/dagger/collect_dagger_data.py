r"""
离线 DAgger 第一步: 采集监督训练数据集 (需 Isaac)

在【混合地形 50%平面+50%崎岖】上, 用感知策略驱动机器人跑, 每帧存下:
  - raw_obs   (含 height_scan, 用于算地形粗糙度 + 作为 alpha 网络输入)
  - label1    (模仿 policy 输出)
  - label2    (感知 policy 输出)
这些是离线监督训练 alpha 网络的原料。alpha 目标标签在训练脚本里用
height_scan 粗糙度实时算 (物理直连, 无需再标定)。

用感知策略驱动是为了保证机器人在混合地形上都走得稳、能覆盖两种地形的状态。

用法 (D:\IsaacLab 下):
  isaaclab.bat -p "D:\robot paper\fusion_label_exp\isaac\dagger\collect_dagger_data.py" ^
      --flat_policy <flat/policy.pt> --rough_policy <rough/policy.pt> ^
      --num_envs 256 --steps 3000 --headless

产出: eval_results/dagger_dataset.npz
      含 raw_obs (N, obs_dim), label1 (N, 37), label2 (N, 37), height_scan (N, 187)
      N = num_envs * steps (抽样后)
"""
from __future__ import annotations
import argparse, os, sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--flat_policy", required=True)
parser.add_argument("--rough_policy", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--sample_every", type=int, default=4, help="每N帧存一次, 控制体积")
parser.add_argument("--task", default="Isaac-Velocity-Rough-G1-v0")
parser.add_argument("--out", default="eval_results/dagger_dataset.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym                                    # noqa: E402
import numpy as np                                         # noqa: E402
import torch                                               # noqa: E402
import isaaclab_tasks                                      # noqa: F401, E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg   # noqa: E402
import isaaclab.terrains as terrain_gen                    # noqa: E402
from isaaclab.terrains import TerrainGeneratorCfg          # noqa: E402


def infer_flat_dim(uenv):
    """rough 观测 = 本体 + height_scan(末尾), 返回本体维度"""
    om = uenv.observation_manager
    terms = om.active_terms["policy"]
    dims = om.group_obs_term_dim["policy"]
    total = 0
    for name, dim in zip(terms, dims):
        d = dim[0] if isinstance(dim, (list, tuple)) else dim
        if "height_scan" in name:
            return total, d       # 本体维度, height_scan 维度
        total += d
    raise RuntimeError("未找到 height_scan 观测项")


@torch.inference_mode()
def main():
    device = "cuda:0"
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)

    # 混合地形: 50%平面 + 50%崎岖, 关缓存
    mixed = TerrainGeneratorCfg(
        size=(8.0, 8.0), border_width=20.0, num_rows=10, num_cols=20,
        horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75,
        use_cache=False, curriculum=True,
        sub_terrains={
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
            "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=0.2, step_height_range=(0.05, 0.20),
                step_width=0.3, platform_width=3.0, border_width=1.0, holes=False),
            "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                proportion=0.15, grid_width=0.45,
                grid_height_range=(0.05, 0.2), platform_width=2.0),
            "rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.15, noise_range=(0.02, 0.10),
                noise_step=0.02, border_width=0.25),
        })
    env_cfg.scene.terrain.terrain_type = "generator"
    env_cfg.scene.terrain.terrain_generator = mixed
    env_cfg.scene.terrain.max_init_terrain_level = 9

    env = gym.make(args.task, cfg=env_cfg)
    uenv = env.unwrapped
    flat_dim, hs_dim = infer_flat_dim(uenv)
    print(f"[data] 本体维度={flat_dim}  height_scan维度={hs_dim}", flush=True)

    flat_policy = torch.jit.load(args.flat_policy, map_location=device).eval()
    rough_policy = torch.jit.load(args.rough_policy, map_location=device).eval()

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    buf = {"raw_obs": [], "label1": [], "label2": [], "height_scan": []}
    for step in range(args.steps):
        if step % 250 == 0:
            print(f"    ... {step}/{args.steps} 步", flush=True)
        l1 = flat_policy(obs[:, :flat_dim])
        l2 = rough_policy(obs)
        # 用感知策略驱动 (保证混合地形上都走得稳)
        obs_dict, _r, _term, _trunc, _ex = env.step(l2)
        nxt = obs_dict["policy"]

        if step % args.sample_every == 0:
            buf["raw_obs"].append(obs.detach().cpu().numpy().copy())
            buf["label1"].append(l1.detach().cpu().numpy().copy())
            buf["label2"].append(l2.detach().cpu().numpy().copy())
            buf["height_scan"].append(
                obs[:, flat_dim:flat_dim + hs_dim].detach().cpu().numpy().copy())
        obs = nxt

    # 合并 (T', E, D) -> (T'*E, D)
    arrs = {}
    for k, v in buf.items():
        a = np.stack(v)                     # (T', E, D)
        arrs[k] = a.reshape(-1, a.shape[-1])
    arrs["flat_dim"] = np.array([flat_dim])
    arrs["hs_dim"] = np.array([hs_dim])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **arrs)
    N = arrs["raw_obs"].shape[0]
    print(f"\n[data] 数据集 -> {args.out}")
    print(f"       样本数 N={N}  raw_obs={arrs['raw_obs'].shape[1]}维  "
          f"label={arrs['label1'].shape[1]}维  height_scan={arrs['height_scan'].shape[1]}维")

    # 快速自检: 地形粗糙度分布 (height_scan 标准差)
    rough_metric = arrs["height_scan"].std(axis=-1)      # (N,)
    q = np.percentile(rough_metric, [5, 25, 50, 75, 95])
    print(f"\n[data] 地形粗糙度 (height_scan std) 分布:")
    print(f"       5%={q[0]:.4f} 25%={q[1]:.4f} 中位={q[2]:.4f} "
          f"75%={q[3]:.4f} 95%={q[4]:.4f}")
    print(f"       -> 平地样本(低粗糙)和崎岖样本(高粗糙)都有, "
          f"{'分布跨度大, 适合训练' if q[4] > 2 * q[1] + 1e-6 else '跨度偏小, 留意'}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
