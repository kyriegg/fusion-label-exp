# 关节运动数据说明与采集清单

本文件记录 G1 人形在不同策略、不同地形下的**干净关节运动数据**：已采了哪些、
怎么读、字段含义、以及为得到完整对照还需补采哪些。

---

## 1. 数据从哪来（两种采集脚本）

| 脚本 | 用途 | 独有字段 | 输出文件名 |
|---|---|---|---|
| `isaac/collect_expert.py` | 单个专家策略在**自己主场**跑 | 无(纯运动数据) | `expert_joints_<tag>.npz` |
| `isaac/eval_policies.py --record` | 融合策略评估时录 | `label1/label2/hybrid/alpha` | `joints_<mode>_<terrain>.npz` |

两者都存在 `D:\IsaacLab\eval_results\` 下。区别：专家脚本采的是"某个策略走路的样子"，
融合脚本额外记录了"两路标签 + 融合系数"，能看融合内部发生了什么。

---

## 2. npz 里有哪些字段

数组形状统一为 **(T 帧, E 环境, J 关节=37)**，另有一维 `joint_names`(37个关节名)。

### 两种文件都有的运动字段

| 字段 | 形状 | 含义 |
|---|---|---|
| `joint_pos` | (T,E,37) | 关节角位置 (rad) |
| `joint_vel` | (T,E,37) | 关节角速度 (rad/s) |
| `joint_acc` | (T,E,37) | 关节角加速度 (rad/s²) |
| `joint_torque` | (T,E,37) | 关节力矩 (Nm) |
| `foot_force` | (T,E,2,3) | 左右脚接触力向量 |
| `root_lin_vel` | (T,E,3) | 机身线速度(本体系) |
| `root_ang_vel` | (T,E,3) | 机身角速度(本体系) |
| `projected_gravity` | (T,E,3) | 重力在本体系投影(姿态) |
| `done` | (T,E) | 该帧是否回合结束(摔倒/超时) |

### 仅融合文件(`joints_*.npz`)额外有

| 字段 | 形状 | 含义 |
|---|---|---|
| `label1` | (T,E,37) | 模仿 policy 输出的关节目标 |
| `label2` | (T,E,37) | 感知 policy 输出的关节目标 |
| `hybrid` | (T,E,37) | 融合后实际下发的关节目标 |
| `alpha` | (T,E,2) | 融合系数 [α1, α2] |
| `action` | (T,E,37) | (专家文件里) 策略原始输出 |

> 注：`collect_expert.py` 记 `action`；`eval_policies.py --record` 记 `label1/2/hybrid/alpha`。

---

## 3. 37 个关节的分组(重要)

采数据、做分析时按功能分组，避免手部噪声干扰：

- **下肢(12个,locomotion 核心)**: `{left,right}_{hip_pitch,hip_roll,hip_yaw,knee,ankle_pitch,ankle_roll}_joint`
- **躯干(1个)**: `torso_joint`
- **上肢手臂(8个)**: `{left,right}_{shoulder_pitch,shoulder_roll,shoulder_yaw,elbow_pitch,elbow_roll}_joint`
- **手部(16个,对 locomotion 无意义,分析时应排除)**:
  `{left,right}_{zero,one,two,three,four,five,six}_joint` 等

> 已知问题：手部关节在两专家间差异巨大但纯属噪声；`six` 关节还会死顶限位空耗力矩。
> 相似性/切换等分析**只用下肢12关节**。

---

## 4. 采集清单：已采 / 待补

目标是得到"干净的、纯地形、可对照"的关节数据。理想的完整矩阵是
**{模仿专家, 感知专家, 融合策略} × {纯平地, 纯崎岖}** 共 6 份。

| 数据 | 策略 | 地形 | 文件 | 状态 |
|---|---|---|---|---|
| 融合×崎岖 | fusion(v6) | 纯崎岖 | `joints_fusion_rough.npz` | ✅ 已采 |
| 融合×平地 | fusion(v6) | 纯平地 | `joints_fusion_flat.npz` | ✅ 已采 |
| 模仿专家×平地 | flat policy | 纯平地 | `expert_joints_flat.npz` | ✅ 已采 |
| 感知专家×崎岖 | rough policy | 纯崎岖 | `expert_joints_rough.npz` | ✅ 已采 |
| **DAgger×崎岖** | dagger α网络 | 纯崎岖 | `joints_dagger_rough.npz` | ✅ 已采 |
| **DAgger×平地** | dagger α网络 | 纯平地 | `joints_dagger_flat.npz` | ✅ 已采 |
| 模仿专家×崎岖 | flat policy | 纯崎岖 | (可选对照) | ⬜ 可选,未采 |
| 感知专家×平地 | rough policy | 纯平地 | (可选对照) | ⬜ 可选,未采 |

> **主对照矩阵已完整**：{模仿专家, 感知专家, 融合(v6), DAgger} × {平地, 崎岖} 的核心数据已采。
> DAgger 数据是最终成果 —— 同一 α 网络在两地形上 α2 分化(平地 0.40 / 崎岖 0.80),
> 即"随地形切换"的直接证据。剩余两份专家跨地形对照可选,暂未采。

> **另有 DAgger 训练数据集** `dagger_dataset.npz`(19.2 万样本, 含 raw_obs+height_scan+label1+label2)
> 和训好的 α 网络 `dagger_alpha_net.pt`,均已本地备份。

### 专家主场性能(采集时记录)

| 专家 | 主场 | 平均存活步数 | 速度跟踪误差(m/s) | 力矩均值(Nm) | 力矩峰值(Nm) |
|---|---|---|---|---|---|
| flat (模仿) | 平地 | 1500 (满,从不摔) | 0.124 | 6.07 | 111 |
| rough (感知) | 崎岖 | 1333 | 0.218 | 7.80 | 300 (顶限位) |

> 平地专家在平地上从不摔倒、力矩小而平、速度跟踪最准;崎岖专家应对地形则付出明显更高的
> 关节代价(力矩峰值顶到上限)。量化了"地形适应有成本"。

---

## 5. 补采命令

**模仿专家 × 平地**（真·纯平面任务）:
```bat
isaaclab.bat -p "D:\robot paper\fusion_label_exp\isaac\collect_expert.py" ^
  --policy "D:\IsaacLab\logs\rsl_rl\g1_flat\2026-07-04_20-46-16\exported\policy.pt" ^
  --task Isaac-Velocity-Flat-G1-v0 --tag flat --num_envs 64 --steps 1500 --headless
```

**感知专家 × 崎岖**:
```bat
isaaclab.bat -p "D:\robot paper\fusion_label_exp\isaac\collect_expert.py" ^
  --policy "D:\IsaacLab\logs\rsl_rl\g1_rough\2026-07-22_12-29-05\exported\policy.pt" ^
  --task Isaac-Velocity-Rough-G1-v0 --tag rough --num_envs 64 --steps 1500 --headless
```

每条产出 `expert_joints_<tag>.npz` + `expert_stats_<tag>.csv`(逐关节统计) + 终端性能摘要。

---

## 6. 怎么读数据

**快速查看**（普通 python，不需要 Isaac）:
```python
import numpy as np
d = np.load(r"D:\IsaacLab\eval_results\joints_fusion_rough.npz", allow_pickle=True)
print(list(d.keys()))                 # 有哪些字段
print(d["joint_pos"].shape)           # (T, E, 37)
print(d["joint_names"])               # 37 个关节名
# 第0环境、右膝的力矩时间序列
j = list(d["joint_names"]).index("right_knee_joint")
tau = d["joint_torque"][:, 0, j]
```

**一键出统计表+图**（用现成分析脚本）:
```bat
python "D:\robot paper\fusion_label_exp\isaac\analyze_joints.py" ^
  --npz eval_results\joints_fusion_rough.npz --dump_timeseries
```
产出 `joint_stats_*.csv`(逐关节：活动范围/速度/力矩/饱和率/标签偏差)、
`joint_timeseries_*.csv`(逐帧长表，可进 Excel)、两张曲线图。

---

## 7. 备份提醒

npz 是大文件，已被仓库 `.gitignore` 排除，**不会上传 GitHub**。
请自行把 `D:\IsaacLab\eval_results\` 下的 `*.npz` 备份到安全位置——
下一步 DAgger 监督训练(路径2)会用到这些关节数据和 v6 的 checkpoint。
