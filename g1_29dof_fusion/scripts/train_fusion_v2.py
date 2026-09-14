#!/usr/bin/env python
"""
三策略融合训练脚本 (SMP + 地形 + 舞蹈参考轨迹)

用法 (isaaclab_venv 下):
    D:\\IsaacLab\\isaaclab_venv\\Scripts\\python.exe train_fusion_v2.py ^
        --smp-policy path/to/smp_model.pt ^
        --rough-policy path/to/rough_policy.pt ^
        --dance-ref path/to/macarena_q.csv ^
        --num-envs 256 ^
        --max-iterations 3000 ^
        --headless

2026-09-11 修复记录 (之前这个脚本从没跑通过):
  1. isaaclab.envs/isaaclab_tasks 依赖 Omniverse Kit app 跑起来才能 import,
     原代码在 create_isaac_env() 里 `from isaaclab.app import AppLauncher` 却
     从没真的实例化它 - 跟 D:\\robot paper\\fusion_label_exp\\isaac\\train_isaac_g1.py
     (已验证能跑) 的写法对比就能看出来: 必须先构造 AppLauncher/simulation_app,
     再 import 任何 isaaclab.* / isaaclab_tasks.* / rsl_rl.* 模块。所以这个文件
     现在把 argparse + AppLauncher 挪到最前面, 其余 import 全部延后。
  2. 基础环境原来用的是官方 G1RoughEnvCfg (G1_MINIMAL_CFG, 37维动作, 含手指),
     跟 smp_policy/舞蹈参考轨迹的 29 维对不上, 加法会直接崩。现在换成本项目自己
     加的 G1_29DOF_RoughEnvCfg (isaaclab_tasks 里新增的
     Isaac-Velocity-Rough-G1-29DOF-v0 任务), 动作空间原生就是 29 维。
  3. create_runner() 原来手写 `from rsl_rl.modules import ActorCritic` 构造策略网络,
     但装的 rsl_rl 5.5.1 已经没有 ActorCritic 这个类了 (整个换成 MLP/Model +
     配置字典驱动的新架构)。现在改用 isaaclab_rl.rsl_rl 提供的
     RslRlOnPolicyRunnerCfg/RslRlPpoActorCriticCfg/RslRlPpoAlgorithmCfg
     (跟 G1_29DOF_RoughPPORunnerCfg 训练地形策略时用的是同一套, 已验证能跑),
     经 handle_deprecated_rsl_rl_cfg() 转成新版 actor/critic/obs_groups 格式。
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="三策略融合训练 (SMP + 地形 + 舞蹈)")
parser.add_argument(
    "--smp-policy", type=str,
    default="D:/robot paper/smp/logs/rsl_rl/smp_forward_g1/2026-08-31_15-01-57_smp_forward_g1/model_4999.pt",
    help="SMP 自然行走策略路径 (原始 rsl_rl checkpoint)",
)
parser.add_argument(
    "--rough-policy", type=str,
    default="D:/robot paper/fusion_label_exp/logs/rsl_rl/g1_29dof_rough/2026-09-11_00-24-06/model_3599.pt",
    help="地形感知策略路径 (原始 rsl_rl checkpoint, 2026-09-11 用 smp_style 奖励重训的 29 维版本)",
)
parser.add_argument(
    "--dance-ref", type=str,
    default="D:/robot paper/fusion_label_exp/reference/macarena/q.csv",
    help="舞蹈参考轨迹 CSV 路径 (被 --no-dance 覆盖)",
)
parser.add_argument(
    "--no-dance", action="store_true",
    help="两路融合模式 (SMP风格 + 地形感知), 不加载舞蹈参考轨迹, alpha降为2维。"
         "2026-09-12: 三路融合(+舞蹈)反复训练后 alpha 网络塌缩到93%压舞蹈、摔倒率100%不降"
         "(舞蹈相似度奖励是拟合固定参考轨迹, 比认真走地形好赚分), 用户决定先退回两路融合验证"
         "感知+风格这条主干本身是否站得住, 舞蹈作为后续阶段再加。",
)
parser.add_argument("--num-envs", type=int, default=256, help="并行环境数量")
parser.add_argument("--max-iterations", type=int, default=3000, help="最大迭代次数")
parser.add_argument("--experiment-name", type=str, default="smp_rough_dance", help="实验名称")
parser.add_argument("--wandb-project", type=str, default="fusion-triple", help="WandB 项目名 (未装 wandb 时用 tensorboard, 见 --logger)")
parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb", "neptune"], help="日志后端")
parser.add_argument("--seed", type=int, default=42, help="随机种子")
parser.add_argument("--resume-checkpoint", type=str, default=None, help="续训: 上次的 alpha 网络 checkpoint 路径 (model_N.pt)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.no_dance:
    args.dance_ref = None

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- 以下全部要在 simulation_app 起来之后才能 import ----
import os  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, "D:/robot paper/fusion_label_exp/isaac")

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.g1_29dof_rough_env_cfg import (  # noqa: E402
    G1_29DOF_RoughEnvCfg,
)
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    handle_deprecated_rsl_rl_cfg,
)
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import smp_bridge  # noqa: E402
from fusion_env_g1_v2 import G1FusionEnv  # noqa: E402


def get_reward_cfg():
    """奖励函数配置"""
    return {
        # 相似性奖励
        "w_sim": 1.0,
        "sim_sigma": 0.1,

        # smp 扩散去噪打分 (人类风格约束) - 用户要求融合后的动作要持续保持 smp 的风格,
        # 不管 alpha 怎么在三个子策略间切换都要受这个约束, 所以权重给跟 w_sim 同量级,
        # 让它成为跟"贴近smp动作"同等重要的常驻信号, 而不是可有可无的小项。
        "w_smp_style": 1.0,

        # 舞蹈奖励
        "w_dance_style": 0.3,

        # 惩罚项
        "w_penetrate": 1.0,
        "force_thresh": 5.0,
        "w_float": 0.5,
        "w_jitter": 0.1,
        # 摔倒(提前terminate)的直接惩罚 - 之前完全没有这一项, alpha网络没有理由不去用
        # "全权重压舞蹈"这种容易拿高分但不看地形、一直摔的策略。量级参考已验证的
        # rough_policy训练里 termination_penalty=-200 (乘以dt=0.02约等于-4/次), 这里
        # 给到量级相近但稍大一点, 因为融合奖励里其他项本身数值都比rough训练小。
        "w_terminate": 5.0,
        # 每多活一步的持续正反馈 (is_alive, 跟 w_terminate 那种"只在摔倒那一步扣一次"不是
        # 同一回事) - 参考 InstinctLab parkour 奖励里的 is_alive=3.0, 但那是在 reward
        # manager 里按 dt 缩放过的; 这里跟 w_sim(=1.0) 同一套未缩放的手写奖励, 先给一个
        # 小量级、不会盖过其他持续信号的值, 等日志里 rew/alive 出来再看要不要调。
        "w_alive": 0.1,
        # 简化版 foot volume points 穿透惩罚 (见 foot_penetration.py / fusion_env_g1_v2.py),
        # 量级是"穿透深度(m) 求和", 跟已有的接触力惩罚 w_penetrate 不是同一物理量, 没法直接
        # 照抄权重 - 先给个保守值, 冒烟测试阶段这个值本身就很小(~0.0015), 等真实训练日志出来
        # 看 pen/foot_volume 的实际量级再重新标定。
        "w_foot_penetrate": 0.5,
        # 0.01 是草稿脚本里的遗留值, 跟已验证过的 rough-terrain 训练里同一个物理量
        # (关节加速度平方) 用的权重 (-1.25e-7, 见 g1_29dof_rough_env_cfg.py 的
        # dof_acc_l2) 差了 8 个数量级。第一次融合训练 smoke test 就看到
        # pen/jitter 原始值上万 (量级压过 rew/sim ~0.01), 这个惩罚项完全淹没了
        # 其他奖励信号, alpha 网络学不到东西 - 这是机器人训完还一直摔的主因。
        "w_joint_acc": 1.25e-7,
        "w_alpha_rate": 0.1,

        # 切换奖励: d_lo/d_hi 是 rough_policy 和 smp_policy 输出的下肢关节动作差异
        # (leg_divergence) 的归一化范围。旧值 0.05~0.40 是给另一套(未经过我们这次DOF/
        # 关节顺序修复的)策略调的, 实测这两个新策略的 leg_divergence 稳定在 1.5~2.3,
        # 远超 0.40, switch/target_a2 因此永远饱和在 switch_a_hi, 这个奖励项等于没在
        # 起真正的"按情境切换"作用。改成按实测范围重新标定。
        "w_switch": 0.5,
        "switch_a_lo": 0.15,
        "switch_a_hi": 0.90,
        "switch_d_lo": 0.5,
        "switch_d_hi": 2.5,
        "switch_ema": 0.3,

        # 地形观测
        "use_terrain_obs": True,
    }


def create_isaac_env(reward_cfg):
    """创建 Isaac Lab 基础环境和融合环境"""
    print("\n[1/3] 创建基础环境...")

    env_cfg = G1_29DOF_RoughEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    # hybrid/label_smp/dance_action 现在统一按 smp_bridge.JOINT_NAMES 顺序 (见
    # fusion_env_g1_v2.py 里 self._canon_ids 的注释), 所以这里融合训练自己的基础环境
    # 的动作空间也要按这个顺序, env.step(hybrid) 才会把值作用到正确的关节上。
    # preserve_order=True 让 find_joints 严格按 JOINT_NAMES 给的顺序返回。
    env_cfg.actions.joint_pos.joint_names = list(smp_bridge.JOINT_NAMES)
    env_cfg.actions.joint_pos.preserve_order = True
    # G1_29DOF_RoughEnvCfg 自带的 rewards.smp_style 是给单独训练rough_policy用的。
    # G1FusionEnv 自己的 _compute_reward 也会调 smp_bridge.smp_guidance_reward(计算
    # 实际融合后动作的风格分, 见 fusion_env_g1_v2.py), 两边都调用等于每步把
    # MotionFeatureBuffer 的滑动窗口更新了两次, 窗口跟真实时间步错位、特征失真。
    # base_env 自己算出来的奖励反正也被 G1FusionEnv.step() 里的 self.env.step(hybrid)
    # 直接丢弃(取的是 G1FusionEnv 自己的 _compute_reward 结果), 关掉这里的重复计算。
    env_cfg.rewards.smp_style = None

    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    print(f"基础环境创建成功: {args.num_envs} 个并行环境")

    print("\n[2/3] 创建融合环境...")
    fusion_env = G1FusionEnv(
        base_env=base_env,
        smp_policy_jit=args.smp_policy,
        rough_policy_jit=args.rough_policy,
        dance_ref_csv=args.dance_ref,
        reward_cfg=reward_cfg,
        robot="g1",
        device=args.device,
    )
    print("融合环境创建成功")
    return fusion_env


def create_runner(env):
    """创建 PPO 训练器 (alpha 融合网络, 不是 smp/rough 那两个已冻结的子策略)"""
    print("\n[3/3] 创建训练器...")

    num_obs = env.num_obs
    num_actions = env.num_actions
    print(f"观测维度: {num_obs}")
    print(f"动作维度 (alpha): {num_actions}")

    agent_cfg = RslRlOnPolicyRunnerCfg(
        seed=args.seed,
        device=args.device,
        num_steps_per_env=24,
        max_iterations=args.max_iterations,
        empirical_normalization=False,
        obs_groups={},  # 空字典触发默认行为: actor/critic 都用 env 提供的 "policy" 组
        save_interval=500,
        experiment_name=args.experiment_name,
        logger=args.logger,
        wandb_project=args.wandb_project,
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            entropy_coef=0.0,
            desired_kl=0.01,
            max_grad_norm=1.0,
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
        ),
    )
    import importlib.metadata as metadata
    handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    # 绝对路径, 不依赖启动时的 cwd - 之前用相对路径 "logs/rsl_rl/..." 好几次都莫名其妙存到了
    # C 盘 (cwd 不是每次都固定在 D 盘某处), 用户明确要求所有东西都放 D 盘, 一次性根治。
    log_dir = os.path.join("D:/robot paper/fusion_label_exp/logs/rsl_rl", args.experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    runner = OnPolicyRunner(env=env, train_cfg=agent_cfg.to_dict(), log_dir=log_dir, device=args.device)
    print(f"训练器创建成功, 日志目录: {log_dir}")

    if args.resume_checkpoint:
        print(f"续训: 加载 {args.resume_checkpoint}")
        runner.load(args.resume_checkpoint)
        print(f"已恢复到 iteration {runner.current_learning_iteration}")

    return runner


def main():
    print("=" * 70)
    print("两路融合训练 (SMP风格 + 地形感知)" if args.no_dance else "三路融合训练 (SMP风格 + 地形感知 + 舞蹈参考轨迹)")
    print("=" * 70)
    print(f"SMP 策略: {args.smp_policy}")
    print(f"地形策略: {args.rough_policy}")
    print(f"舞蹈参考: {'未启用 (--no-dance)' if args.no_dance else args.dance_ref}")
    print(f"并行环境: {args.num_envs}")
    print(f"最大迭代: {args.max_iterations}")
    print(f"设备: {args.device}")
    print("=" * 70)

    for path in [args.smp_policy, args.rough_policy]:
        if not os.path.exists(path):
            print(f"错误: 文件不存在: {path}")
            return

    reward_cfg = get_reward_cfg()

    env = create_isaac_env(reward_cfg)
    runner = create_runner(env)

    print("\n" + "=" * 70)
    print("开始训练...")
    print("=" * 70)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

    print("\n" + "=" * 70)
    print("训练完成!")
    print(f"模型保存在: logs/rsl_rl/{args.experiment_name}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
    simulation_app.close()
