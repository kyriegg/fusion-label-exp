# Function Label 融合实验(G1 人形 · Isaac Lab)

冻结一个**模仿 policy** 与一个**感知 policy**,只训练一个轻量 α 网络输出融合系数,
生成 hybrid 动作驱动机器人,目标是让融合动作同时继承双方优点、并能随地形切换。

```
hybrid = α1 ⊙ label1(模仿) + α2 ⊙ label2(感知)
```

完整实验记录见 **[docs/EXPERIMENT_REPORT.md](docs/EXPERIMENT_REPORT.md)**。

## 核心结论(v1–v6)

1. **α 融合可控** —— 约束权重决定 α 走向,v1↔v2 的 α2 从 0.05 干净翻到 0.92。
2. **学习式融合优于朴素平均** —— v2 融合(966 步 / 摔倒 18%)优于 0.5/0.5 平均(850 步 / 47%)。
3. **纯 RL + 软奖励无法习得条件切换**(负结果) —— 穷尽排除数据/观测/地形/评估等外因后,
   α2 在平地与崎岖上仍稳定 0.88,未随地形切换。指向下一步:DAgger 监督式硬训练。

## 代码

| 文件 | 作用 |
|---|---|
| `isaac/fusion_env_g1.py` | 融合环境:双冻结 policy + α 融合 + 四项约束 reward + 切换奖励 |
| `isaac/train_isaac_g1.py` | PPO 训练入口(含地形重构:50%平面+50%崎岖) |
| `isaac/eval_policies.py` | 四模式对比评估(flat/rough/avg/fusion)+ 关节数据录制,支持平地/崎岖切换 |
| `isaac/collect_expert.py` | 单专家在各自主场采集关节运动数据 |
| `isaac/analyze_joints.py` | 关节数据分析(逐关节统计 + 轨迹图,不依赖 Isaac) |

## 复现

前置: Isaac Lab 2.3+,已训练 `Isaac-Velocity-Flat-G1-v0` 与 `Isaac-Velocity-Rough-G1-v0`
两个策略并导出 jit(`exported/policy.pt`)。

```bash
# 训练融合网络
isaaclab.bat -p isaac/train_isaac_g1.py \
    --flat_policy <flat/policy.pt> --rough_policy <rough/policy.pt> \
    --num_envs 1024 --headless --max_iterations 1500

# 评估(崎岖 / 平地)+ 录关节数据
isaaclab.bat -p isaac/eval_policies.py \
    --flat_policy <flat/policy.pt> --rough_policy <rough/policy.pt> \
    --num_envs 256 --steps 2000 --headless --terrain_obs --terrain rough --modes fusion --record
```

## 下一步

DAgger 监督式训练(路径2)兑现"α 随地形切换";可选上下半身解耦;接入真实模仿/感知网络。
