r"""
DAgger 第二步: 监督训练 alpha 网络 (纯 PyTorch, 不需要 Isaac)

移植自用户早前的 fusion-label-exp 仓库 (isaac/dagger/train_dagger_offline.py)。

2026-09-14 先试了把目标信号换成 smp_score(动作自然度), 离线评估看着分化不错(自然0.27/
不自然0.76), 但接回真实环境端到端验证后 alpha2 基本塌缩回常数(std仅0.042, 见项目记忆),
没有真的学会切换, 摔倒率也没有改善。跟用户对齐后明确: alpha 该信任地形感知还是该信任风格,
这件事本来就该用老实验里**验证有效**的物理信号(height_scan 地形粗糙度)来判断 - 这是"什么时候
该谨慎"的问题; 而"学人类风格"是另一件事, 已经有独立通道在做(融合奖励里的 smp_style / rew/
smp_style, 权重1.0, 一直都在把动作往人类自然姿态拉), 不需要也不该靠 alpha 切换信号去兼顾。
所以这里改回 --signal height_scan 为默认, smp_score 保留作为 --signal smp_score 可选项
(留着方便以后对比, 不建议作为主路径)。

两种信号都是从 raw_obs 现成算出来的(raw_obs = [本体127维, height_scan 187维]), 不需要重新
采集数据集, 同一份 dagger_dataset.npz 换个目标信号重训就行。

alpha 网络: 输入 = [label_smp, label_rough, raw_obs], 输出 = alpha2 (sigmoid, 0~1)。
损失: MSE(alpha2_pred, alpha2_target)。

用法:
    python train_dagger_offline.py --data "D:\robot paper\fusion_label_exp\dagger_data\dagger_dataset.npz" --epochs 100 --auto_thresh
产出: D:\robot paper\fusion_label_exp\dagger_data\dagger_alpha_net.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "D:/robot paper/fusion_label_exp/isaac")
from dagger_alpha_net import AlphaNet, save_alpha_net  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument(
    "--data", required=True, nargs="+",
    help="一个或多个 .npz 数据集路径(空格分隔), 会拼接聚合 - 迭代 DAgger 时把上一轮的数据集"
         "和新采的数据集一起传进来, 不需要重新采集全部数据",
)
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--batch", type=int, default=4096)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument(
    "--signal", type=str, default="height_scan", choices=["height_scan", "smp_score"],
    help="目标 alpha2 的信号来源。height_scan(默认, 推荐): 地形粗糙度, 老实验验证过有效的物理"
         "信号, 判断'该不该谨慎'。smp_score: 动作自然度, 2026-09-14 试过, 离线看着分化不错但"
         "端到端部署后塌缩回常数(见项目记忆), 保留仅供对比, 不建议作为主路径。",
)
ap.add_argument("--flat_dim", type=int, default=127, help="raw_obs 里本体部分的维度, 之后是 height_scan")
# 阈值(信号 -> 目标 a2 的映射, height_scan 版本: 粗糙度高->a2高; smp_score 版本: 分数低->a2高)
ap.add_argument("--sig_lo", type=float, default=None, help="信号低阈值(不设则必须配合 --auto_thresh)")
ap.add_argument("--sig_hi", type=float, default=None, help="信号高阈值(不设则必须配合 --auto_thresh)")
ap.add_argument("--a2_lo", type=float, default=0.15, help="安全/自然情形的目标 a2 (偏 smp)")
ap.add_argument("--a2_hi", type=float, default=0.90, help="危险/不自然情形的目标 a2 (偏 rough)")
ap.add_argument("--auto_thresh", action="store_true", help="用数据分位数自动定信号阈值(推荐, 绕开信号本身的基线偏移)")
ap.add_argument("--pct_lo", type=float, default=30, help="auto: 低分位%")
ap.add_argument("--pct_hi", type=float, default=70, help="auto: 高分位%")
ap.add_argument("--out", default="D:/robot paper/fusion_label_exp/dagger_data/dagger_alpha_net.pt")
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
raw_obs_parts, label_smp_parts, label_rough_parts, smp_score_parts = [], [], [], []
for data_path in args.data:
    d = np.load(data_path, allow_pickle=True)
    raw_obs_parts.append(torch.tensor(d["raw_obs"], dtype=torch.float32))
    label_smp_parts.append(torch.tensor(d["label_smp"], dtype=torch.float32))
    label_rough_parts.append(torch.tensor(d["label_rough"], dtype=torch.float32))
    smp_score_parts.append(torch.tensor(d["smp_score"], dtype=torch.float32))
    print(f"[dagger] 加载 {data_path}: {d['raw_obs'].shape[0]} 条")
raw_obs = torch.cat(raw_obs_parts, dim=0)
label_smp = torch.cat(label_smp_parts, dim=0)
label_rough = torch.cat(label_rough_parts, dim=0)
smp_score_all = torch.cat(smp_score_parts, dim=0)
N = raw_obs.shape[0]
print(f"[dagger] 样本 N={N}(聚合 {len(args.data)} 个数据集)  obs={raw_obs.shape[1]}  label={label_smp.shape[1]}  信号={args.signal}")

# ---- 目标 alpha2 ----
if args.signal == "height_scan":
    # 地形粗糙度(height_scan 标准差), 跟老实验完全同一套逻辑: 平地(粗糙度低)->a2低(偏smp保风格),
    # 崎岖(粗糙度高)->a2高(偏rough保稳定)。height_scan 是 raw_obs 的后半段(本体127维之后)。
    signal = raw_obs[:, args.flat_dim:].std(dim=-1)
    direction = 1  # 信号越大, a2 越大
else:
    signal = smp_score_all
    direction = -1  # 信号越大(动作越自然), a2 越小

if args.auto_thresh:
    pct_a, pct_b = (args.pct_lo, args.pct_hi) if direction == 1 else (args.pct_hi, args.pct_lo)
    sig_lo = float(torch.quantile(signal, pct_a / 100))
    sig_hi = float(torch.quantile(signal, pct_b / 100))
else:
    sig_lo, sig_hi = args.sig_lo, args.sig_hi
print(f"[dagger] 信号阈值: 低={sig_lo:.4f}({'->a2低' if direction==1 else '->a2高'}) "
      f"高={sig_hi:.4f}({'->a2高' if direction==1 else '->a2低'})")

if direction == 1:
    frac = ((signal - sig_lo) / (sig_hi - sig_lo + 1e-9)).clamp(0, 1)
else:
    frac = ((sig_hi - signal) / (sig_hi - sig_lo + 1e-9)).clamp(0, 1)
a2_target = args.a2_lo + frac * (args.a2_hi - args.a2_lo)  # (N,)
print(f"[dagger] 目标 a2: 安全={args.a2_lo} 危险={args.a2_hi}")
print(f"[dagger] 目标 a2 分布: 均值={a2_target.mean():.3f} "
      f"(接近{args.a2_lo}的比例={((a2_target<0.3).float().mean()):.2f}, "
      f"接近{args.a2_hi}的比例={((a2_target>0.7).float().mean()):.2f})")
if (a2_target < 0.3).float().mean() < 0.05 or (a2_target > 0.7).float().mean() < 0.05:
    print("[警告] 安全或危险样本过少, 训练可能学不到切换 —— "
          "检查数据集地形是否真的混合, 或调整信号阈值")

# ---- alpha 网络输入 = [label_smp, label_rough, raw_obs] ----
X = torch.cat([label_smp, label_rough, raw_obs], dim=-1)  # (N, D)
Y = a2_target.unsqueeze(-1)  # (N, 1)
in_dim = X.shape[1]

net = AlphaNet(in_dim).to(device)
opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

# train/val split
idx = torch.randperm(N)
n_val = N // 10
val_idx, tr_idx = idx[:n_val], idx[n_val:]
Xtr, Ytr = X[tr_idx].to(device), Y[tr_idx].to(device)
Xval, Yval = X[val_idx].to(device), Y[val_idx].to(device)
signal_val = signal[val_idx].to(device)
safe_mask = (signal_val <= sig_lo) if direction == 1 else (signal_val >= sig_hi)
danger_mask = (signal_val >= sig_hi) if direction == 1 else (signal_val <= sig_lo)

print(f"[dagger] 训练 {len(tr_idx)} / 验证 {len(val_idx)} | in_dim={in_dim}")
for ep in range(args.epochs):
    net.train()
    perm = torch.randperm(Xtr.shape[0], device=device)
    tot = 0.0
    for i in range(0, Xtr.shape[0], args.batch):
        b = perm[i:i + args.batch]
        pred = net(Xtr[b])
        loss = nn.functional.mse_loss(pred, Ytr[b])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tot += loss.item() * len(b)
    sched.step()
    if ep % 10 == 0 or ep == args.epochs - 1:
        net.eval()
        with torch.no_grad():
            vpred = net(Xval)
            vmse = nn.functional.mse_loss(vpred, Yval).item()
            safe_m = vpred[safe_mask].mean().item() if safe_mask.any() else float("nan")
            danger_m = vpred[danger_mask].mean().item() if danger_mask.any() else float("nan")
        print(f"  ep {ep:3d} | train_mse={tot/len(tr_idx):.4f} "
              f"val_mse={vmse:.4f} | 安全样本预测a2={safe_m:.3f} "
              f"危险样本预测a2={danger_m:.3f}")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
save_alpha_net(net, in_dim, (256, 128), args.out)
print(f"\n[dagger] alpha 网络 -> {args.out}")
print("下一步: 接回 play_fusion_v2.py --dagger-ckpt 在真实环境里验证 alpha 是否真的随情境切换")
