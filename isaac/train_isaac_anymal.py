r"""
Anymal-C 替身融合实验 训练入口 (自动适配已装 rsl_rl 版本)

关键改动: PPO 配置不再手写, 直接从任务注册表加载官方 agent 配置 —— 官方
train.py 能跑通就说明这份配置与你装的 rsl_rl 兼容, 我们只改实验名和迭代数。

用法 (在 D:/IsaacLab 目录下):
  isaaclab.bat -p "D:/robot paper\fusion_label_exp\isaac\train_isaac_anymal.py" ^
      --flat_policy <flat的exported\policy.pt> ^
      --rough_policy <rough的exported\policy.pt> --num_envs 1024 --headless
"""
from __future__ import annotations
import argparse, os, sys
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--flat_policy", required=True)
parser.add_argument("--rough_policy", required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--max_iterations", type=int, default=1000)
parser.add_argument("--task", default="Isaac-Velocity-Rough-Anymal-C-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- 以下 import 必须在 app 启动之后 ----
import gymnasium as gym                                   # noqa: E402
import torch                                              # noqa: E402
from rsl_rl.runners import OnPolicyRunner                 # noqa: E402
import isaaclab_tasks                                     # noqa: F401, E402
from isaaclab_tasks.utils.parse_cfg import (              # noqa: E402
    parse_env_cfg, load_cfg_from_registry)
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg  # noqa: E402
from importlib import metadata                               # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from isaac.fusion_env_anymal import AnymalFusionEnv       # noqa: E402

REWARD_CFG = {
    "w_sim": 1.0, "sim_sigma": 0.5,
    "w_penetrate": 2.0, "force_thresh": 1.0,
    "w_float": 1.0,
    "w_jitter": 1.0, "w_joint_acc": 2.5e-7, "w_alpha_rate": 0.1,
}


def main():
    device = "cuda:0"

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    base_env = gym.make(args.task, cfg=env_cfg)

    env = AnymalFusionEnv(base_env,
                          flat_policy_jit=args.flat_policy,
                          rough_policy_jit=args.rough_policy,
                          reward_cfg=REWARD_CFG, device=device)
    print(f"[fusion] num_envs={env.num_envs} obs={env.num_obs} act={env.num_actions}")

    # ---- 官方 agent 配置, 保证与已装 rsl_rl 的 schema 一致 ----
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    # 旧格式配置 -> 新版 rsl_rl 格式 (官方 train.py 同款转换, 之前漏了这步)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.experiment_name = "anymal_fusion_stage1"
    agent_cfg.max_iterations = args.max_iterations
    train_cfg = agent_cfg.to_dict()

    log_dir = os.path.join("logs", "rsl_rl", "anymal_fusion_stage1",
                           datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=device)
    runner.learn(num_learning_iterations=args.max_iterations)

    # ---- 导出 α 网络 (兼容新旧两种字段名) ----
    os.makedirs("outputs", exist_ok=True)
    alg = runner.alg
    net = getattr(alg, "actor", None) or getattr(alg, "actor_critic", None)
    torch.save(net.state_dict(), "outputs/fusion_alpha_net.pt")
    print("FusionNet 已导出 -> outputs/fusion_alpha_net.pt | 日志:", log_dir)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
