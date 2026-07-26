r"""
离线 DAgger 第二步: 监督训练 alpha 网络 (纯 PyTorch, 不需要 Isaac)

原料: collect_dagger_data.py 产出的 dagger_dataset.npz
      (raw_obs 含 height_scan, label1, label2)

目标标签: 用 height_scan 的粗糙度 (标准差) 算每帧的目标 alpha2 ——
  平地(粗糙度≈0) -> 目标 a2 低(偏模仿)
  崎岖(粗糙度大) -> 目标 a2 高(偏感知)
这是物理直连信号, 不像"专家分歧"那样被网络差异污染。

alpha 网络: 输入 = [label1, label2, raw_obs](与 v3 一致, 含地形观测),
           输出 = alpha2 (标量, sigmoid 到 0~1), alpha1 = 1 - alpha2。
损失: MSE(alpha2_pred, alpha2_target)。

用法:
  python train_dagger_offline.py --data eval_results/dagger_dataset.npz --epochs 100
产出: outputs/dagger_alpha_net.pt  (可接回 eval 验证切换)
"""
from __future__ import annotations
import argparse, os
import numpy as np
import torch
import torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--batch", type=int, default=4096)
ap.add_argument("--lr", type=float, default=1e-3)
# 粗糙度 -> 目标 a2 的映射 (物理阈值, 单位: m)
ap.add_argument("--rough_lo", type=float, default=0.01, help="粗糙度≤此值->平地")
ap.add_argument("--rough_hi", type=float, default=0.05, help="粗糙度≥此值->崎岖")
ap.add_argument("--a2_lo", type=float, default=0.15, help="平地目标 a2")
ap.add_argument("--a2_hi", type=float, default=0.90, help="崎岖目标 a2")
ap.add_argument("--auto_thresh", action="store_true",
                help="用数据分位数自动定粗糙度阈值(推荐, 绕开 height_scan 基线偏移)")
ap.add_argument("--pct_lo", type=float, default=30, help="auto: 低阈值分位%")
ap.add_argument("--pct_hi", type=float, default=70, help="auto: 高阈值分位%")
ap.add_argument("--out", default="outputs/dagger_alpha_net.pt")
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
d = np.load(args.data, allow_pickle=True)
raw_obs = torch.tensor(d["raw_obs"], dtype=torch.float32)
label1 = torch.tensor(d["label1"], dtype=torch.float32)
label2 = torch.tensor(d["label2"], dtype=torch.float32)
hs = torch.tensor(d["height_scan"], dtype=torch.float32)
N = raw_obs.shape[0]
print(f"[dagger] 样本 N={N}  obs={raw_obs.shape[1]}  label={label1.shape[1]}")

# ---- 目标 alpha2: 由 height_scan 粗糙度映射 ----
roughness = hs.std(dim=-1)                                  # (N,)
if args.auto_thresh:
    # height_scan 有固定基线偏移(纯平地粗糙度也≈0.055), 绝对阈值会把全部样本
    # 判成崎岖。改用这批数据自身的分位数定阈值, 相对区分平地/崎岖。
    r_lo = float(torch.quantile(roughness, args.pct_lo / 100))
    r_hi = float(torch.quantile(roughness, args.pct_hi / 100))
    print(f"[dagger] auto阈值: 粗糙度 {args.pct_lo}%分位={r_lo:.4f} -> 平地, "
          f"{args.pct_hi}%分位={r_hi:.4f} -> 崎岖")
else:
    r_lo, r_hi = args.rough_lo, args.rough_hi
frac = ((roughness - r_lo) / (r_hi - r_lo + 1e-9)).clamp(0, 1)
a2_target = args.a2_lo + frac * (args.a2_hi - args.a2_lo)   # (N,)
print(f"[dagger] 目标 a2: 平地={args.a2_lo} 崎岖={args.a2_hi} | "
      f"粗糙度阈值 [{args.rough_lo}, {args.rough_hi}]")
print(f"[dagger] 目标 a2 分布: 均值={a2_target.mean():.3f} "
      f"(接近{args.a2_lo}的比例={((a2_target<0.3).float().mean()):.2f}, "
      f"接近{args.a2_hi}的比例={((a2_target>0.7).float().mean()):.2f})")
if (a2_target < 0.3).float().mean() < 0.05 or (a2_target > 0.7).float().mean() < 0.05:
    print("[警告] 平地或崎岖样本过少, 训练可能学不到切换 —— "
          "检查数据集地形是否真的混合, 或调整粗糙度阈值")

# ---- alpha 网络输入 = [label1, label2, raw_obs] (与 v3 一致) ----
X = torch.cat([label1, label2, raw_obs], dim=-1)           # (N, D)
Y = a2_target.unsqueeze(-1)                                 # (N, 1)
in_dim = X.shape[1]


class AlphaNet(nn.Module):
    def __init__(self, in_dim, hid=(256, 128)):
        super().__init__()
        layers, dd = [], in_dim
        for h in hid:
            layers += [nn.Linear(dd, h), nn.ELU()]
            dd = h
        layers.append(nn.Linear(dd, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x))                  # a2 in (0,1)


net = AlphaNet(in_dim).to(device)
opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

# train/val split
idx = torch.randperm(N)
n_val = N // 10
val_idx, tr_idx = idx[:n_val], idx[n_val:]
Xtr, Ytr = X[tr_idx].to(device), Y[tr_idx].to(device)
Xval, Yval = X[val_idx].to(device), Y[val_idx].to(device)

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
            # 分地形看拟合: 用验证集粗糙度分平地/崎岖两组
            vr = hs[val_idx].std(dim=-1).to(device)
            flat_m = vpred[vr <= r_lo].mean().item() if (vr <= r_lo).any() else float('nan')
            rough_m = vpred[vr >= r_hi].mean().item() if (vr >= r_hi).any() else float('nan')
        print(f"  ep {ep:3d} | train_mse={tot/len(tr_idx):.4f} "
              f"val_mse={vmse:.4f} | 平地样本预测a2={flat_m:.3f} "
              f"崎岖样本预测a2={rough_m:.3f}")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
torch.save({"alpha_net": net.state_dict(), "in_dim": in_dim,
            "arch": "sigmoid_a2", "hid": [256, 128]}, args.out)
print(f"\n[dagger] alpha 网络 -> {args.out}")
print("下一步: 接回 eval 在真平地/真崎岖分别测, 看 a2 是否分化 (平地低/崎岖高)")
