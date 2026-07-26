r"""
分歧信号标定 (离线 DAgger 第一步)

目的: 用真实采集数据, 测量两专家在【真平地】vs【真崎岖】上的下肢分歧
      ‖label2 - label1‖, 从而把"分歧 -> 目标α2"的映射阈值定准, 而非拍脑袋。

输入: joints_fusion_flat.npz / joints_fusion_rough.npz
      (这两份里同时存了 label1 和 label2, 才能算分歧)

输出: 终端打印两种地形下分歧的分布 (均值/分位数), 并给出建议的映射阈值
      calib_divergence.png  分歧分布直方图对比

用法:
  python calibrate_divergence.py --flat eval_results/joints_fusion_flat.npz ^
                                 --rough eval_results/joints_fusion_rough.npz
"""
from __future__ import annotations
import argparse, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--flat", required=True, help="joints_fusion_flat.npz")
ap.add_argument("--rough", required=True, help="joints_fusion_rough.npz")
ap.add_argument("--out", default="eval_results")
args = ap.parse_args()

LEG_KW = ("_hip_pitch", "_hip_roll", "_hip_yaw", "_knee",
          "_ankle_pitch", "_ankle_roll")


def leg_divergence(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    if "label1" not in d or "label2" not in d:
        raise SystemExit(f"{npz_path} 里没有 label1/label2, 无法算分歧 "
                         f"(需用 joints_fusion_*.npz, 不是 expert_*.npz)")
    names = [str(x) for x in d["joint_names"]]
    lids = [i for i, n in enumerate(names) if any(k in n for k in LEG_KW)]
    l1, l2 = d["label1"], d["label2"]                 # (T,E,J)
    # 每帧每环境的下肢分歧 (L2 距离, 按下肢关节数归一)
    diff = np.linalg.norm((l2 - l1)[..., lids], axis=-1) / (len(lids) ** 0.5)
    return diff.reshape(-1), len(lids)                 # 展平成 (T*E,)


def stats(x, label):
    q = np.percentile(x, [5, 25, 50, 75, 95])
    print(f"\n[{label}] 下肢分歧 ‖label2-label1‖:")
    print(f"  均值={x.mean():.4f}  标准差={x.std():.4f}")
    print(f"  分位数  5%={q[0]:.4f}  25%={q[1]:.4f}  中位={q[2]:.4f}  "
          f"75%={q[3]:.4f}  95%={q[4]:.4f}")
    return q


def main():
    df, nleg = leg_divergence(args.flat)
    dr, _ = leg_divergence(args.rough)
    print(f"下肢关节数: {nleg}")
    qf = stats(df, "平地")
    qr = stats(dr, "崎岖")

    # ---- 建议阈值: 用平地的高分位 与 崎岖的低分位 作为映射两端 ----
    # d_lo: 平地典型分歧上界 (映射到目标 a2 = a_lo, 偏模仿)
    # d_hi: 崎岖典型分歧下界 (映射到目标 a2 = a_hi, 偏感知)
    d_lo = round(float(np.percentile(df, 75)), 3)
    d_hi = round(float(np.percentile(dr, 25)), 3)
    print("\n" + "=" * 56)
    print("建议的 分歧->目标α2 映射阈值 (写进离线训练配置):")
    print(f"  switch_d_lo = {d_lo}   # 平地75分位: 分歧≤此值 -> 目标α2=a_lo")
    print(f"  switch_d_hi = {d_hi}   # 崎岖25分位: 分歧≥此值 -> 目标α2=a_hi")
    if d_hi <= d_lo:
        print("  [警告] 崎岖低分位 <= 平地高分位, 两地形分歧区分度低!")
        print("         说明两专家在平地/崎岖上的下肢输出差异不够, ")
        print("         分歧信号可能不足以区分地形 -- 需换信号(如直接用height_scan)。")
    else:
        sep = (d_hi - d_lo) / (qf[2] + 1e-6)
        print(f"  区分度: 崎岖低位-平地高位 = {d_hi - d_lo:.3f} "
              f"({'良好' if sep > 0.3 else '偏弱'})")
    print("=" * 56)

    # ---- 画分布对比 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(args.out, exist_ok=True)
        plt.figure(figsize=(9, 5))
        bins = np.linspace(0, max(df.max(), dr.max()), 60)
        plt.hist(df, bins=bins, alpha=0.6, label="flat (平地)", color="tab:blue", density=True)
        plt.hist(dr, bins=bins, alpha=0.6, label="rough (崎岖)", color="tab:red", density=True)
        plt.axvline(d_lo, ls="--", c="tab:blue", lw=1.5, label=f"d_lo={d_lo}")
        plt.axvline(d_hi, ls="--", c="tab:red", lw=1.5, label=f"d_hi={d_hi}")
        plt.xlabel("下肢分歧 ‖label2-label1‖".encode("ascii","ignore").decode() or "leg divergence")
        plt.xlabel("leg divergence ||label2 - label1||")
        plt.ylabel("density"); plt.legend()
        plt.title("Expert divergence: flat vs rough (calibration)")
        plt.tight_layout()
        p = os.path.join(args.out, "calib_divergence.png")
        plt.savefig(p, dpi=130)
        print(f"\n分布图 -> {p}")
    except Exception as e:
        print(f"[warn] 画图失败: {e}")


if __name__ == "__main__":
    main()
