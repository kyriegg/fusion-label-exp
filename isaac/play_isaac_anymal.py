r"""
融合控制回放 (带图形界面, 肉眼验收 + 录屏用)

用法 (在 D:\IsaacLab 目录下, 注意没有 --headless):
  isaaclab.bat -p "D:\robot paper\fusion_label_exp\isaac\play_isaac_anymal.py" ^
      --flat_policy <flat的policy.pt> --rough_policy <rough的policy.pt> --num_envs 16

不指定 --checkpoint 时自动加载 logs\rsl_rl\anymal_fusion_stage1 下最新的 model_*.pt。
运行中终端每隔几秒打印 alpha 均值, 窗口里观察机器人过崎岖地形的步态。
录屏: Win+Alt+R 开始/停止 (Xbox Game Bar)。
"""
from __future__ import annotations
import argparse, glob, os, re, sys
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--flat_policy", required=True)
parser.add_argument("--rough_policy", required=True)
parser.add_argument("--checkpoint", default=None, help="model_*.pt 路径, 默认自动找最新")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--task", default="Isaac-Velocity-Rough-Anymal-C-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym                                    # noqa: E402
import torch                                               # noqa: E402
from rsl_rl.runners import OnPolicyRunner                  # noqa: E402
import isaaclab_tasks                                      # noqa: F401, E402
from isaaclab_tasks.utils.parse_cfg import (               # noqa: E402
    parse_env_cfg, load_cfg_from_registry)
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg  # noqa: E402
from importlib import metadata                             # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from isaac.fusion_env_anymal import AnymalFusionEnv        # noqa: E402
# 不能 import 训练脚本(其顶层代码会再启动一个App), 配置直接内联
REWARD_CFG = {
    "w_sim": 1.0, "sim_sigma": 0.5,
    "w_penetrate": 2.0, "force_thresh": 1.0,
    "w_float": 1.0,
    "w_jitter": 1.0, "w_joint_acc": 2.5e-7, "w_alpha_rate": 0.1,
}


def find_latest_ckpt() -> str:
    cands = glob.glob(os.path.join("logs", "rsl_rl", "anymal_fusion_stage1",
                                   "*", "model_*.pt"))
    if not cands:
        raise FileNotFoundError("没找到 checkpoint, 用 --checkpoint 手动指定")
    def key(p):
        run = os.path.basename(os.path.dirname(p))
        it = int(re.search(r"model_(\d+)\.pt", p).group(1))
        return (run, it)
    return max(cands, key=key)


def main():
    device = "cuda:0"
    ckpt = args.checkpoint or find_latest_ckpt()
    print(f"[play] 加载 checkpoint: {ckpt}")

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    base_env = gym.make(args.task, cfg=env_cfg)
    env = AnymalFusionEnv(base_env, flat_policy_jit=args.flat_policy,
                          rough_policy_jit=args.rough_policy,
                          reward_cfg=REWARD_CFG, device=device)

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.experiment_name = "anymal_fusion_stage1"
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(ckpt, map_location=device)
    policy = runner.get_inference_policy(device=device)

    obs = env.get_observations().to(device)
    step, a2_sum = 0, 0.0
    print("[play] 开始回放, Ctrl+C 退出 | 观察: 机器人过崎岖地形 + α2 均值")
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _rew, _dones, extras = env.step(actions)
            obs = obs.to(device)
        a2 = extras.get("log", {}).get("alpha/a2_mean", float("nan"))
        a2_sum += a2
        step += 1
        if step % 100 == 0:
            print(f"  step {step:5d} | alpha2 当前={a2:.3f} 累计均值={a2_sum/step:.3f}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
