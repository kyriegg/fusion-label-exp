# G1 29-DOF 融合实验（第二阶段）：遇到的问题与结论

> 记录周期: 2026-09-10 ～ 2026-09-14
> 平台: Isaac Lab (isaacsim 6.1) / rsl_rl / Unitree G1 29-DOF
> 承接: 本仓库第一阶段（[../README.md](../README.md)、[../docs/EXPERIMENT_REPORT.md](../docs/EXPERIMENT_REPORT.md)）用 stock 策略替身验证过的 `hybrid = α1·label1 + α2·label2` 融合架构，这一阶段换成真实的 SMP（人类风格）策略 + 29-DOF 地形感知策略，目标是再加入舞蹈参考轨迹作为第三路。

## 目录结构

```
g1_29dof_fusion/
  isaac/
    fusion_env_g1_v2.py     融合环境: 双冻结策略 + α融合 + 奖励 + DAgger接口
    smp_bridge.py           SMP 扩散去噪打分接入 (观测重建 + guidance reward)
    flatpatch_command.py    地形 flat-patch 目标点速度指令 (09-13 新增)
    foot_penetration.py     简化版脚部体积点穿透检测 (09-13 新增)
    dagger_alpha_net.py     DAgger 监督式 alpha 网络定义 + 存取 (09-14 新增)
    policy_loader.py        从 rsl_rl 原始 checkpoint 重建 MLP 策略网络
    rewards.py               相似度/穿透/抖动等奖励函数 (沿用第一阶段)
  isaac_patches/
    g1_29dof_rough_env_cfg.py  对 IsaacLab 本体的项目专用扩展 (不在本仓库路径下,
                                实际位于 D:\IsaacLab\source\isaaclab_tasks\...\config\g1\,
                                这里存一份快照方便追溯)
  scripts/
    train_fusion_v2.py      融合训练入口 (PPO 训练 alpha 网络)
    play_fusion_v2.py       回放/诊断入口 (--fixed-alpha / --dagger-ckpt / --max-steps)
    collect_dagger_data.py  DAgger 数据采集 (支持纯 rough 驱动 / 用已训好的 alpha 网络驱动迭代)
    train_dagger_offline.py DAgger 离线监督训练 (纯 PyTorch, 不需要 Isaac)
    diag_fixed_alpha.py     09-12 的早期诊断脚本 (固定几个 alpha 定值对比表现)
  docs/
    PROBLEMS.md              本文档
    results_summary.csv      本阶段所有实验的关键指标汇总
```

大量 checkpoint (`model_*.pt`)、DAgger 数据集 (`dagger_dataset*.npz`, 每份约 268MB)、完整训练日志未随本次提交上传——单文件普遍超过 GitHub 100MB 限制，且日志里九成内容是 Isaac Sim 扩展加载的噪声。`docs/results_summary.csv` 保留了从中提炼出的关键数字。

## 这几天做了什么、遇到了什么问题

### 1. rough_policy 独立训练 (09-10 ~ 09-11)

把 SMP 的扩散去噪打分 (`smp_guidance_reward`) 直接接进了地形策略的重训过程，作为风格奖励项，而不是等融合阶段再补——地形策略最终摔倒率从 100% 降到 24%，风格奖励项增长约 10 倍。过程中两次被系统 OOM 杀掉，用 `--resume` 从 checkpoint 恢复；本机 15.7GB 内存下 `train.py` 这类脚本必须留意批量/env 数配置。

### 2. 三次融合训练迭代，均以"全程摔倒"告终 (09-11 ~ 09-13)

| 轮次 | 现象 | 排查到的原因 |
|---|---|---|
| 三路融合 v1 (09-11) | 训完回放全程摔倒 | `w_joint_acc` 权重错了 8 个数量级，压过所有其他奖励信号 |
| 三路融合 v2 (09-11) | 修完上面那个问题重训，依然全程摔倒 | α 网络塌缩到 93% 权重压舞蹈——舞蹈参考轨迹固定、容易拟合，比认真走地形更好骗分；融合奖励里当时**完全没有摔倒惩罚**；切换奖励阈值是拿旧策略标定的，跟新策略实测的 leg_divergence 量级对不上 |
| 两路融合 (09-12) | 加完摔倒惩罚、重新标定切换阈值后训到 iter 2730/3000 仍 100% 摔倒 | 用户判断这不是奖励微调能解决的，停止训练，退回两路(去掉舞蹈)先验证主干是否成立 |

### 3. 移植同行工作 Hiking in the Wild 的三个机制 (09-13)

来源: [project-instinct/InstinctLab](https://github.com/project-instinct/InstinctLab)（论文 [Hiking in the Wild](https://project-instinct.github.io/hiking-in-the-wild/)）。移植了：

- **flat-patch-sampling 速度指令** (`flatpatch_command.py`)：只朝地形上真实可达的平坦点走，而不是随机给一个可能压根不可达的速度指令。IsaacLab 自带的 `ROUGH_TERRAINS_CFG` 原本完全没有配置 flat patch。
- **is_alive 奖励**：跟其他项解耦的"活着就加分"，区别于只在摔倒瞬间扣一次分的 `termination_penalty`。
- **简化版 foot volume points 穿透惩罚** (`foot_penetration.py`)：原论文版本依赖地形边缘检测生成的虚拟障碍物（未移植，工作量大），这里简化成脚下 raycaster 网格直接跟地形表面比高度。

三项逐一冒烟测试后接入训练，跑完 3000 轮：**摔倒率全程仍钉在~100%，地形课程等级从未推进**。结论：这三个机制解决的是"reward hacking"类问题（不合理的指令/缺乏存活激励/脚部穿模），跟这次持续摔倒不是同一个病根。

### 4. 隔离诊断定位真正的病根 (09-14)

用 `G1FusionEnv` 自带的 `alpha_override` 把两个冻结策略单独拉出来测（各 64 个并行环境、2000 步窗口）：

| 策略 | 摔倒率 | 平均存活步数 |
|---|---|---|
| 纯 SMP（人类风格，未在此地形训练过） | 100.0% | 88.2 |
| 纯地形感知策略 | 58.3% | 683.3 |

> 踩过一个坑：第一次测纯地形策略只给了 600 步窗口，只观测到 8 个 episode 结束、误判成"100%摔倒"——地形策略存活时间远长于 SMP，评估窗口要跟策略自身的时间尺度匹配，拉到 2000 步后才看到真实数字。

**关键发现**：PPO 训出来的 α 网络把约 85% 权重压在了两者中明显更弱的 SMP 上，只给了称职的地形策略 15%。跟"压舞蹈"是完全同一种 reward hacking 形状——`r_sim`/`r_smp_style` 这类"贴近 SMP 动作"的软奖励比"活下来"这种硬约束更容易稳定拿分。

### 5. 移植用户早期实验 (fusion-label-exp 第一阶段) 的 DAgger 方案 (09-14)

本仓库第一阶段（[../docs/EXPERIMENT_REPORT.md](../docs/EXPERIMENT_REPORT.md)）已经用详尽的排除法证明过：

- **纯 RL 软奖励学不出随地形切换的 α**——排除了数据/观测/地形/评估等所有外因后，α2 在平地和崎岖地形上仍固定在 0.88 左右。
- **"两个专家动作的分歧"信号没有区分度**——这正是本阶段 `switch/leg_divergence` 机制在用的同一个信号，第一阶段标定后发现平地/崎岖两种地形下的分布几乎完全重叠。
- 换成 **DAgger 监督训练 + height_scan 地形粗糙度**（物理直连信号）后，α 网络成功学会了随地形切换（平地 α2=0.40、崎岖 α2=0.80，平地摔倒率 0.4%）。

这一阶段复刻了这套三段式流水线（`collect_dagger_data.py` → `train_dagger_offline.py` → `play_fusion_v2.py --dagger-ckpt`），做了三次尝试：

| 尝试 | 目标信号 | 离线评估分化 | 端到端部署 |
|---|---|---|---|
| 单轮 | SMP 打分（用户想法：低分→偏 rough） | 自然 0.27 / 不自然 0.76 | 摔倒率 100%，α2 标准差仅 0.042（塌缩回常数） |
| 单轮 | height_scan 粗糙度（复刻第一阶段配方） | 安全 0.30 / 危险 0.87 | 摔倒率 99.8%，α2 标准差仅 0.045（同样塌缩，且比 SMP 信号版本更差） |
| 两轮迭代 | height_scan 粗糙度 + 用第一轮网络重新采集数据聚合 | 安全 0.36 / 危险 0.80 | 摔倒率 98.5%，α2 标准差 0.035（有改善但仍远未解决） |

三次尝试的离线评估都能看到清晰的情境分化，**但一接回真实闭环部署，α 预测统一塌缩回一个接近常数的值**。目前最可能的解释：训练数据由单一专家策略驱动采集，跟实际部署时"α 混合动作驱动"的状态分布不一致，网络在没见过的状态上退回全局均值；真正的迭代式数据聚合（第二轮）方向对但幅度远不如第一阶段实验明显，压缩程度比第一阶段严重得多，原因尚不清楚。

## 当前结论

**目前尝试过的所有融合方案（PPO 训练 α、DAgger 单轮、DAgger 两轮迭代）都没有跑赢"直接单独使用地形感知策略"这个最朴素的基线**（58.3% 摔倒率、平均存活 683 步）。完整数字见 [results_summary.csv](results_summary.csv)。

## 未解决 / 待讨论的问题

1. **动作空间线性混合两个未做相位同步的闭环策略，是否本身就是不稳定的控制律？** 两策略在同一真实状态下对腿部关节的分歧（leg_divergence）长期维持在较大且稳定的量级，说明二者步态节奏从未对齐过；即便 α 决策完全正确，中间态的混合动作本身也可能是物理上不连贯的（`diag_fixed_alpha.py` 就是 09-12 就已经在问这个问题，当时还没有确凿证据）。
2. 要不要收紧 α 的取值范围（比如强制 α_smp 上限 0.3），放弃"学出真正动态切换"，退而求其次做一个风格偏置而非能力互补的方案？
3. DAgger 数据聚合迭代继续投入的性价比——两轮之间的改善幅度（存活步数 76→160）不算大。
4. 是否需要提前规划向论文原版深度摄像头感知（而非 height_scan 这种仿真特权信息）迁移的时间点，这关系到真机部署路线图，跟"能不能站稳"是两个独立问题。

## 如何复现

前置同第一阶段：Isaac Lab (isaacsim 6.1) / rsl_rl，已训练好的 SMP、rough_policy checkpoint。

```bat
:: 融合训练 (PPO 训 alpha)
python scripts/train_fusion_v2.py --no-dance --num-envs 96 --max-iterations 3000 --headless

:: 固定 alpha 诊断 (单独测每个冻结子策略)
python scripts/play_fusion_v2.py --no-dance --fixed-alpha rough --num-envs 64 --max-steps 2000 --headless

:: DAgger: 采集 -> 离线训练 -> 端到端验证
python scripts/collect_dagger_data.py --num-envs 256 --steps 3000 --headless
python scripts/train_dagger_offline.py --data dagger_data/dagger_dataset.npz --signal height_scan --epochs 150 --auto_thresh
python scripts/play_fusion_v2.py --no-dance --dagger-ckpt dagger_data/dagger_alpha_net.pt --num-envs 64 --max-steps 2000 --headless
```
