# DAgger 离线监督训练(路径2)

用监督学习让 α 网络学会随地形切换 —— PPO 软奖励(v1-v6)做不到的事。
标签信号: height_scan 地形粗糙度(物理直连), 平地→低 α2、崎岖→高 α2。

## 三步流程

### 0. (可选)标定信号 —— 验证信号有没有区分力
```bat
python "D:\robot paper\fusion_label_exp\isaac\dagger\calibrate_divergence.py" ^
    --flat eval_results\joints_fusion_flat.npz --rough eval_results\joints_fusion_rough.npz
```
> 注: 该脚本标定的是"下肢分歧"信号, 实测区分度不足(已弃用)。
> 最终采用 height_scan 粗糙度, 无需标定, 直接进第1步。

### 1. 采数据集(需 Isaac, ~10min)
```bat
isaaclab.bat -p "D:\robot paper\fusion_label_exp\isaac\dagger\collect_dagger_data.py" ^
    --flat_policy <flat/policy.pt> --rough_policy <rough/policy.pt> ^
    --num_envs 256 --steps 3000 --headless
```
产出 `eval_results\dagger_dataset.npz`(raw_obs 含 height_scan + label1 + label2)。

### 2. 离线训练(纯 PyTorch, ~1min)
```bat
python "D:\robot paper\fusion_label_exp\isaac\dagger\train_dagger_offline.py" ^
    --data eval_results\dagger_dataset.npz --epochs 100 --auto_thresh
```
`--auto_thresh` 用数据分位数定粗糙度阈值(绕开 height_scan 基线偏移)。
产出 `outputs\dagger_alpha_net.pt`。看每10轮 "平地样本预测a2 / 崎岖样本预测a2" 是否分化。

### 3. 接回 Isaac 端到端验证(需 Isaac)
用 `isaac\eval_policies.py` 的 dagger 模式(--dagger_ckpt), 在真平地/真崎岖分别测:
```bat
isaaclab.bat -p "D:\robot paper\fusion_label_exp\isaac\eval_policies.py" ^
    --flat_policy <..> --rough_policy <..> ^
    --dagger_ckpt "D:\IsaacLab\outputs\dagger_alpha_net.pt" ^
    --num_envs 256 --steps 2000 --headless --terrain_obs --terrain rough --modes dagger --record
```
把 `--terrain rough` 换 `--terrain flat` 再跑一次。看两地形 α2 是否分化。

## 训练态结果(2026-07-25)
平地预测 α2=0.26、崎岖预测 α2=0.93 —— 分化 0.67, 监督训练成功学会切换。
(对比 v1-v6 纯 RL: 两地形 α2 恒为 0.88, 无切换)
