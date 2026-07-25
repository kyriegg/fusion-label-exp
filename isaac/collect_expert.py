r"""
单专家策略关节数据采集 (理解A: 各专家在各自主场)

直接用一个 jit policy 驱动官方任务, 录制逐帧关节运动数据。不走融合 env,
所以:
  - 平地用 Isaac-Velocity-Flat-G1-v0 (真·纯平面, 无 height_scan)
  - 崎岖用 Isaac-Velocity-Rough-G1-v0
  两个任务观测维度不同也没关系, 各自的专家跑各自的任务, 不涉及拼接。

用法 (在 D:\IsaacLab 目录下):
  # 平地专家 x 平地
  isaaclab.bat -p "D:\robot paper\fusion_label_exp\isaac\collect_expert.py" ^
      --policy "D:\IsaacLab\logs\rsl_rl\g1_flat\...\exported\policy.pt" ^
      --task Isaac-Velocity-Flat-G1-v0 --tag flat --num_envs 64 --steps 1500 --headless

  # 崎岖专家 x 崎岖
  isaaclab.bat -p "D:\robot paper\fusion_label_exp\isaac\collect_expert.py" ^
      --policy "D:\IsaacLab\logs\rsl_rl\g1_rough\...\exported\policy.pt" ^
      --task Isaac-Velocity-Rough-G1-v0 --tag rough --num_envs 64 --steps 1500 --headless

产出: eval_results/expert_joints_<tag>.npz  (逐帧关节数据)
      eval_results/expert_stats_<tag>.csv   (逐关节统计)
      终端打印性能摘要 (存活/摔倒率/速度跟踪)
"""
from __future__ import annotations
import argparse, os, sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--policy", required=True, help="jit policy.pt 路径")
parser.add_argument("--task", required=True,
                    help="Isaac-Velocity-Flat-G1-v0 或 Isaac-Velocity-Rough-G1-v0")
parser.add_argument("--tag", required=True, help="输出文件标签, 如 flat / rough")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--record_envs", type=int, default=8, help="录制前几个环境")
parser.add_argument("--out", default="eval_results")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym                                    # noqa: E402
import numpy as np                                         # noqa: E402
import torch                                               # noqa: E402
import isaaclab_tasks                                      # noqa: F401, E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg   # noqa: E402


@torch.inference_mode()
def main():
    device = "cuda:0"
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    env_cfg.seed = 12345
    env = gym.make(args.task, cfg=env_cfg)
    uenv = env.unwrapped

    policy = torch.jit.load(args.policy, map_location=device).eval()

    robot = uenv.scene["robot"]
    contact = uenv.scene["contact_forces"]
    foot_ids, foot_names = contact.find_bodies(".*ankle_roll.*")
    print(f"[collect] task={args.task} tag={args.tag}")
    print(f"[collect] 脚部 body: {foot_names}")

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    ne = min(args.record_envs, args.num_envs)
    rec = {k: [] for k in ["joint_pos", "joint_vel", "joint_acc", "joint_torque",
                           "foot_force", "root_lin_vel", "root_ang_vel",
                           "projected_gravity", "action", "done"]}
    n_done = 0
    vel_err_sum, vel_cnt = 0.0, 0

    def torque():
        d = robot.data
        t = getattr(d, "applied_torque", None)
        if t is None:
            t = getattr(d, "computed_torque", d.joint_effort_target)
        return t

    for step in range(args.steps):
        if step % 250 == 0:
            print(f"    ... {step}/{args.steps} 步", flush=True)
        act = policy(obs)
        obs_dict, _rew, terminated, truncated, _ex = env.step(act)
        obs = obs_dict["policy"]
        dones = (terminated | truncated)
        n_done += int(dones.sum().item())

        # 速度跟踪误差 (命令 vs 实际机身水平速度)
        try:
            cmd = uenv.command_manager.get_command("base_velocity")[:, :2]
            act_v = robot.data.root_lin_vel_b[:, :2]
            vel_err_sum += float((cmd - act_v).norm(dim=-1).mean())
            vel_cnt += 1
        except Exception:
            pass

        if step < args.steps:
            d = robot.data
            f = lambda t: t[:ne].detach().cpu().numpy().copy()
            rec["joint_pos"].append(f(d.joint_pos))
            rec["joint_vel"].append(f(d.joint_vel))
            rec["joint_acc"].append(f(d.joint_acc))
            rec["joint_torque"].append(f(torque()))
            rec["foot_force"].append(f(contact.data.net_forces_w[:, foot_ids]))
            rec["root_lin_vel"].append(f(d.root_lin_vel_b))
            rec["root_ang_vel"].append(f(d.root_ang_vel_b))
            rec["projected_gravity"].append(f(d.projected_gravity_b))
            rec["action"].append(f(act))
            rec["done"].append(dones[:ne].cpu().numpy().copy())

    # ---------------- 保存 npz ----------------
    os.makedirs(args.out, exist_ok=True)
    arrs = {k: np.stack(v) for k, v in rec.items() if len(v)}
    arrs["joint_names"] = np.array(robot.data.joint_names)
    npz_path = os.path.join(args.out, f"expert_joints_{args.tag}.npz")
    np.savez_compressed(npz_path, **arrs)
    T, E, J = arrs["joint_pos"].shape
    print(f"\n[collect] 关节数据 -> {npz_path}  (T={T}, E={E}, J={J})")

    # ---------------- 逐关节统计 ----------------
    import csv
    names = [str(x) for x in arrs["joint_names"]]
    flat = lambda a: a.reshape(-1, a.shape[-1])
    qpos, qvel, tau = flat(arrs["joint_pos"]), flat(arrs["joint_vel"]), flat(arrs["joint_torque"])
    stats_path = os.path.join(args.out, f"expert_stats_{args.tag}.csv")
    with open(stats_path, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(["joint", "位置均值", "活动范围", "位置min", "位置max",
                    "速度均值", "速度峰值", "力矩均值", "力矩峰值"])
        for j, nm in enumerate(names):
            p, v, t = qpos[:, j], qvel[:, j], tau[:, j]
            w.writerow([nm, round(float(p.mean()), 4), round(float(p.max()-p.min()), 4),
                        round(float(p.min()), 4), round(float(p.max()), 4),
                        round(float(np.abs(v).mean()), 4), round(float(np.abs(v).max()), 3),
                        round(float(np.abs(t).mean()), 3), round(float(np.abs(t).max()), 2)])
    print(f"[collect] 逐关节统计 -> {stats_path}")

    # ---------------- 性能摘要 ----------------
    avg_ep = round(args.steps * args.num_envs / max(n_done, 1), 1)
    vel_err = round(vel_err_sum / max(vel_cnt, 1), 4)
    print("\n" + "=" * 50)
    print(f"专家性能摘要 [{args.tag}]  (任务 {args.task})")
    print(f"  回合数        : {n_done}")
    print(f"  平均存活步数  : {avg_ep}  (上限 ~{uenv.max_episode_length})")
    print(f"  速度跟踪误差  : {vel_err} m/s")
    print(f"  关节力矩均值  : {round(float(np.abs(tau).mean()), 3)} Nm")
    print(f"  关节力矩峰值  : {round(float(np.abs(tau).max()), 2)} Nm")
    print("=" * 50)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
