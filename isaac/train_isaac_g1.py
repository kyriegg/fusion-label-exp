r"""
G1 人形 融合训练入口

用法 (D:/IsaacLab 目录下):
  isaaclab.bat -p "D:/robot paper/fusion_label_exp/isaac/train_isaac_g1.py" ^
      --flat_policy  <g1_flat 的 exported/policy.pt> ^
      --rough_policy <g1_rough 的 exported/policy.pt> ^
      --num_envs 1024 --headless
"""
from __future__ import annotations
import argparse, os, sys
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--flat_policy", required=True)
parser.add_argument("--rough_policy", required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--max_iterations", type=int, default=1500)
parser.add_argument("--task", default="Isaac-Velocity-Rough-G1-v0")
parser.add_argument("--robot", default="g1")
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

# v2 调整 (v1 结果: alpha2 塌陷到 0.05, 摔倒率 36.7%):
#   1. w_sim 1.0 -> 0.3, sigma 0.5 -> 1.0  (削弱"抄模仿"的收益)
#   2. w_penetrate 2 -> 6, w_float 1 -> 3  (加重摔倒/悬空代价)
#   3. 相似性奖励排除手部关节 (见 fusion_env_g1.py, 手指噪声会稀释 alpha 信号)
REWARD_CFG = {
    "w_sim": 0.3, "sim_sigma": 1.0,
    "w_penetrate": 6.0, "force_thresh": 1.0,
    "w_float": 3.0,
    "w_jitter": 1.0, "w_joint_acc": 2.5e-7, "w_alpha_rate": 0.1,
    # 想保留全部关节做对照实验就设成 []
    "sim_exclude_patterns": ["_zero_", "_one_", "_two_", "_three_",
                             "_four_", "_five_", "_six_"],
    # v3: 给 alpha 网络喂地形扫描
    "use_terrain_obs": True,
    # v4 (DAgger 路径1): 切换奖励 —— 让 alpha2 跟踪"下肢两专家分歧"导出的目标
    #   下肢分歧小(平地)-> 目标 a2 低(偏模仿); 分歧大(崎岖)-> 目标 a2 高(偏感知)
    "w_switch": 8.0,          # 切换奖励权重 (要盖过其他项, 让切换占主导)
    "switch_a_lo": 0.15,      # 平地目标 alpha2
    "switch_a_hi": 0.90,      # 崎岖目标 alpha2
    "switch_d_lo": 0.05,      # 下肢分歧归一化下界 (映射到 a_lo)
    "switch_d_hi": 0.40,      # 下肢分歧归一化上界 (映射到 a_hi)
    "switch_ema": 0.0,        # 目标时间平滑 (0=不平滑)
}


def main():
    device = "cuda:0"
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)

    # ================= 地形: 强制 50%平面 + 50%崎岖 =================
    # 病根: G1 rough 任务默认 6 种子地形全崎岖, 且开了地形缓存(use_cache),
    # 光改 terrain_generator 会被缓存覆盖回默认。解法: 直接替换 generator +
    # 关掉缓存, 让 gym.make 用新配置重新生成。
    try:
        import isaaclab.terrains as terrain_gen
        from isaaclab.terrains import TerrainGeneratorCfg

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
            },
        )
        env_cfg.scene.terrain.terrain_type = "generator"
        env_cfg.scene.terrain.terrain_generator = mixed
        env_cfg.scene.terrain.max_init_terrain_level = 9
        print("[fusion] v6: 地形设为 50%平面+50%崎岖, use_cache=False", flush=True)
    except Exception as e:
        print(f"[warn] 地形设置失败: {e}", flush=True)

    base_env = gym.make(args.task, cfg=env_cfg)

    # 验证 gym.make 后实际生效的地形
    try:
        real = base_env.unwrapped.scene["terrain"].cfg.terrain_generator
        keys = list(real.sub_terrains.keys()) if real else None
        print(f"[DIAG-after] 实际地形 sub_terrains = {keys}, "
              f"use_cache = {getattr(real, 'use_cache', '?')}", flush=True)
        if not real or "flat" not in real.sub_terrains:
            print("!!! [WARN] 平面未生效, 训练仍是纯崎岖, 请把此行贴给助手 !!!",
                  flush=True)
    except Exception as e:
        print(f"[DIAG-after] {e}", flush=True)

    env = G1FusionEnv(base_env, args.flat_policy, args.rough_policy,
                      REWARD_CFG, robot=args.robot, device=device)
    print(f"[fusion] num_envs={env.num_envs} obs={env.num_obs} act={env.num_actions}")

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.experiment_name = "g1_fusion_stage1"
    agent_cfg.max_iterations = args.max_iterations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg,
                                             metadata.version("rsl-rl-lib"))

    log_dir = os.path.join("logs", "rsl_rl", "g1_fusion_stage1",
                           datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=device)
    runner.learn(num_learning_iterations=args.max_iterations)

    os.makedirs("outputs", exist_ok=True)
    alg = runner.alg
    net = getattr(alg, "actor", None) or getattr(alg, "actor_critic", None)
    torch.save(net.state_dict(), "outputs/g1_fusion_alpha_net.pt")
    print("FusionNet 已导出 -> outputs/g1_fusion_alpha_net.pt | 日志:", log_dir)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
