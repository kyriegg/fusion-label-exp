"""
Isaac 第一阶段训练入口 (rsl_rl PPO)。

用法 (Isaac Lab):
  ./isaaclab.sh -p fusion_label_exp/isaac/train_isaac.py --num_envs 4096 --headless

流程:
  1. AppLauncher 起 sim
  2. 加载并冻结 模仿 policy / 感知 policy
  3. 创建 FusionLabelEnv (hybrid label 驱动机器人, 四项约束做 reward)
  4. rsl_rl OnPolicyRunner + FusionActorCritic 训练 α 网络
  5. 训练完导出 FusionNet 权重
"""
from __future__ import annotations
import argparse

# ---- Isaac Lab 启动 (必须在其他 isaac import 之前) ----
# from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=3000)
parser.add_argument("--imitation_ckpt", default="checkpoints/imitation_policy.pt")
parser.add_argument("--perception_ckpt", default="checkpoints/perception_policy.pt")
parser.add_argument("--fusion_mode", choices=["scalar", "per_dim"], default="scalar")
# AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# app_launcher = AppLauncher(args)
# simulation_app = app_launcher.app

import torch  # noqa: E402
# from rsl_rl.runners import OnPolicyRunner                     # noqa: E402
# from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper             # noqa: E402

from models.frozen_policies import ImitationPolicy, PerceptionPolicy  # noqa: E402
from isaac.fusion_env import FusionLabelEnv                            # noqa: E402
from isaac.fusion_actor import FusionActorCritic                       # noqa: E402


PPO_CFG = {
    # rsl_rl runner 配置 (train_cfg)
    "seed": 42,
    "runner": {
        "num_steps_per_env": 24,
        "max_iterations": 3000,
        "save_interval": 100,
        "experiment_name": "stage1_function_label",
        "policy_class_name": "FusionActorCritic",
        "algorithm_class_name": "PPO",
    },
    "algorithm": {
        "clip_param": 0.2, "entropy_coef": 0.005,
        "num_learning_epochs": 5, "num_mini_batches": 4,
        "learning_rate": 3.0e-4, "schedule": "adaptive",
        "gamma": 0.99, "lam": 0.95,
        "desired_kl": 0.01, "max_grad_norm": 1.0,
        "value_loss_coef": 1.0, "use_clipped_value_loss": True,
    },
    "policy": {"hidden_dims": [256, 256], "init_noise_std": 0.3},
}

REWARD_CFG = {
    # 四项约束权重 (与离线版 configs/default.yaml 对齐, 可先照抄再调)
    "w_sim": 1.0, "sim_sigma": 0.5,
    "w_penetrate": 5.0, "force_thresh": 1.0,
    "w_float": 3.0, "float_max_height": 0.03,
    "w_jitter": 0.5, "w_joint_acc": 2.5e-7, "w_alpha_rate": 0.1,
    "contact_vel_thresh": 0.15,
}


def main():
    device = "cuda:0"

    # ---- 冻结的两套 policy ----
    # 维度按你们真实网络填
    imi = ImitationPolicy(motion_dim=..., label_dim=...,
                          ckpt=args.imitation_ckpt).to(device)
    per = PerceptionPolicy(vel_dim=..., terrain_dim=..., label_dim=...,
                           ckpt=args.perception_ckpt).to(device)

    # ---- 环境 ----
    # env_cfg = FusionLabelEnvCfg()          # TODO: 复用你们 locomotion 任务的场景配置
    # env_cfg.scene.num_envs = args.num_envs
    # env_cfg.rewards = REWARD_CFG
    # env_cfg.fusion_mode = args.fusion_mode
    # env = FusionLabelEnv(env_cfg, imi, per)
    # env = RslRlVecEnvWrapper(env)

    # ---- PPO ----
    # PPO_CFG["runner"]["max_iterations"] = args.max_iterations
    # runner = OnPolicyRunner(env, PPO_CFG, log_dir="logs/stage1", device=device)
    # 把自定义 ActorCritic 注册给 runner:
    #   rsl_rl 新版支持 policy_class_name 直接找到 import 的类;
    #   老版可 monkey-patch: rsl_rl.modules.FusionActorCritic = FusionActorCritic
    # runner.learn(num_learning_iterations=args.max_iterations)

    # ---- 导出 function label 网络 ----
    # runner.alg.actor_critic.export_fusion_net("outputs/fusion_net_stage1.pt")
    # simulation_app.close()
    raise SystemExit(
        "骨架脚本: 取消注释并接入你们的 Isaac Lab 场景配置后运行。\n"
        "需要填的位置: env_cfg(场景/机器人/传感器)、两套 policy 的维度与 ckpt。")


if __name__ == "__main__":
    main()
