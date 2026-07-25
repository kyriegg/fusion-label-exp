r"""
关节运动数据分析 (不依赖 Isaac, 普通 python 环境即可运行)

输入: eval_policies.py --record 产出的 eval_results/joints_<mode>.npz
     数组形状 (T 帧, E 环境, J 关节)

产出:
  joint_stats_<mode>.csv     每个关节一行: 活动范围/速度/力矩/饱和率/标签偏差
  joint_timeseries_<mode>.csv 逐帧长表 (可选, --dump_timeseries)
  joint_motion_<mode>.png    腿部关节的位置/速度/力矩曲线
  joint_labels_<mode>.png    label1 / label2 / hybrid 三条标签轨迹对比

用法:
  python analyze_joints.py --npz eval_results/joints_fusion.npz
  python analyze_joints.py --npz eval_results/joints_fusion.npz --env 0 --dump_timeseries
"""
from __future__ import annotations
import argparse, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True, help="joints_<mode>.npz 路径")
ap.add_argument("--env", type=int, default=0, help="画图用第几个环境 (默认0)")
ap.add_argument("--fps", type=float, default=50.0, help="控制频率, 用于时间轴")
ap.add_argument("--torque_limit", type=float, default=300.0,
                help="力矩饱和判定阈值 (Nm)")
ap.add_argument("--dump_timeseries", action="store_true",
                help="额外导出逐帧长表 CSV (文件较大)")
ap.add_argument("--out", default=None, help="输出目录, 默认与 npz 同目录")
args = ap.parse_args()

d = np.load(args.npz, allow_pickle=True)
mode = os.path.basename(args.npz).replace("joints_", "").replace(".npz", "")
out = args.out or os.path.dirname(args.npz) or "."
os.makedirs(out, exist_ok=True)

names = [str(x) for x in d["joint_names"]]
qpos, qvel = d["joint_pos"], d["joint_vel"]          # (T,E,J)
qacc = d["joint_acc"] if "joint_acc" in d else np.zeros_like(qpos)
tau = d["joint_torque"]
l1, l2, hyb = d["label1"], d["label2"], d["hybrid"]
T, E, J = qpos.shape
print(f"[{mode}] {T} 帧 x {E} 环境 x {J} 关节  ({T/args.fps:.1f} 秒/环境)")

# ---------------- 逐关节统计 ----------------
flat = lambda a: a.reshape(-1, a.shape[-1])          # (T*E, J)
rows = []
for j, nm in enumerate(names):
    p, v, t = flat(qpos)[:, j], flat(qvel)[:, j], flat(tau)[:, j]
    rows.append({
        "joint": nm,
        "位置均值(rad)": round(float(p.mean()), 4),
        "活动范围(rad)": round(float(p.max() - p.min()), 4),
        "位置min": round(float(p.min()), 4),
        "位置max": round(float(p.max()), 4),
        "速度均值|q̇|(rad/s)": round(float(np.abs(v).mean()), 4),
        "速度峰值(rad/s)": round(float(np.abs(v).max()), 3),
        "力矩均值|τ|(Nm)": round(float(np.abs(t).mean()), 3),
        "力矩峰值(Nm)": round(float(np.abs(t).max()), 2),
        "力矩饱和率": round(float((np.abs(t) >= args.torque_limit * 0.99).mean()), 4),
        "hybrid-label1偏差": round(float(np.abs(flat(hyb)[:, j] - flat(l1)[:, j]).mean()), 4),
        "label2-label1差异": round(float(np.abs(flat(l2)[:, j] - flat(l1)[:, j]).mean()), 4),
    })

import csv
stats_path = os.path.join(out, f"joint_stats_{mode}.csv")
with open(stats_path, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"逐关节统计 -> {stats_path}")

# ---------------- 控制台摘要 ----------------
def top(key, n=6, rev=True):
    s = sorted(rows, key=lambda r: r[key], reverse=rev)[:n]
    return "\n".join(f"    {r['joint']:<28s} {r[key]}" for r in s)

print(f"\n活动范围最大的关节:\n{top('活动范围(rad)')}")
print(f"\n力矩峰值最高的关节:\n{top('力矩峰值(Nm)')}")
sat = [r for r in rows if r["力矩饱和率"] > 0.001]
print(f"\n出现力矩饱和(≥{args.torque_limit}Nm)的关节: {len(sat)} 个")
for r in sorted(sat, key=lambda r: -r["力矩饱和率"])[:8]:
    print(f"    {r['joint']:<28s} 饱和率={r['力矩饱和率']:.3f}  峰值={r['力矩峰值(Nm)']}")
print(f"\nhybrid 偏离 label1 最多的关节 (融合真正改动的地方):\n{top('hybrid-label1偏差')}")
print(f"\n两路标签本身差异最大的关节:\n{top('label2-label1差异')}")

# ---------------- 逐帧长表 ----------------
if args.dump_timeseries:
    ts_path = os.path.join(out, f"joint_timeseries_{mode}.csv")
    e = args.env
    with open(ts_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["t", "joint", "q", "qd", "qdd", "tau", "label1", "label2", "hybrid"])
        for i in range(T):
            for j, nm in enumerate(names):
                w.writerow([round(i / args.fps, 4), nm,
                            round(float(qpos[i, e, j]), 5), round(float(qvel[i, e, j]), 5),
                            round(float(qacc[i, e, j]), 4), round(float(tau[i, e, j]), 4),
                            round(float(l1[i, e, j]), 5), round(float(l2[i, e, j]), 5),
                            round(float(hyb[i, e, j]), 5)])
    print(f"\n逐帧长表 (env {e}) -> {ts_path}")

# ---------------- 画图 ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

e = args.env
tt = np.arange(T) / args.fps
# 优先挑腿部关节画图, 找不到就用活动范围最大的 6 个
leg_kw = ("hip_pitch", "knee", "ankle_pitch")
sel = [j for j, nm in enumerate(names) if any(k in nm for k in leg_kw)][:6]
if not sel:
    sel = [names.index(r["joint"]) for r in
           sorted(rows, key=lambda r: -r["活动范围(rad)"])[:6]]

fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
for j in sel:
    ax[0].plot(tt, qpos[:, e, j], lw=1.1, label=names[j])
    ax[1].plot(tt, qvel[:, e, j], lw=1.0)
    ax[2].plot(tt, tau[:, e, j], lw=1.0)
ax[0].set_ylabel("q (rad)"); ax[1].set_ylabel("qdot (rad/s)")
ax[2].set_ylabel("torque (Nm)"); ax[2].set_xlabel("time (s)")
ax[0].legend(fontsize=7, ncol=3); ax[0].set_title(f"joint motion - {mode} (env {e})")
for a in ax:
    a.grid(alpha=0.3)
plt.tight_layout()
p1 = os.path.join(out, f"joint_motion_{mode}.png")
plt.savefig(p1, dpi=130); plt.close()

# 标签对比图: 挑 hybrid 偏离 label1 最大的 4 个关节
sel2 = [names.index(r["joint"]) for r in
        sorted(rows, key=lambda r: -r["hybrid-label1偏差"])[:4]]
fig, axes = plt.subplots(len(sel2), 1, figsize=(12, 2.4 * len(sel2)), sharex=True)
axes = np.atleast_1d(axes)
for a, j in zip(axes, sel2):
    a.plot(tt, l1[:, e, j], "--", lw=1.0, label="label1 (imitation)")
    a.plot(tt, l2[:, e, j], ":", lw=1.0, label="label2 (perception)")
    a.plot(tt, hyb[:, e, j], "-", lw=1.5, label="hybrid")
    a.set_ylabel(names[j], fontsize=7); a.grid(alpha=0.3)
axes[0].legend(fontsize=8); axes[0].set_title(f"label trajectories - {mode} (env {e})")
axes[-1].set_xlabel("time (s)")
plt.tight_layout()
p2 = os.path.join(out, f"joint_labels_{mode}.png")
plt.savefig(p2, dpi=130); plt.close()
print(f"\n图已保存 -> {p1}\n            {p2}")
