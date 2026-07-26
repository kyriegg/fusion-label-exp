"""
rsl_rl 兼容的 ActorCritic: actor 就是 FusionNet (输出 α 的 raw logits)。

关键点:
  - actor 的输入 = env 观测 (concat[label1, label2, ...])
  - actor 的输出 = α raw logits, 归一化在 env 侧做 (softmax)
  - 这样 PPO 训练的"策略"天然就是 function label 网络, 与会议方案一致
  - critic 用独立 MLP (标准 asymmetric 也行, 先对称起步)

训练完成后, FusionNet 权重可单独导出, 直接用于第二阶段的
hybrid label 数据生成。
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch.distributions import Normal


class FusionActorCritic(nn.Module):
    """接口对齐 rsl_rl.modules.ActorCritic"""
    is_recurrent = False

    def __init__(self, num_obs: int, num_critic_obs: int, num_actions: int,
                 hidden_dims=(256, 256), init_noise_std: float = 0.3, **kwargs):
        super().__init__()

        def mlp(in_d, out_d):
            layers, d = [], in_d
            for h in hidden_dims:
                layers += [nn.Linear(d, h), nn.ELU()]
                d = h
            layers.append(nn.Linear(d, out_d))
            return nn.Sequential(*layers)

        self.actor = mlp(num_obs, num_actions)         # <- 这就是 FusionNet
        self.critic = mlp(num_critic_obs, 1)

        # actor 末层零初始化: 起始 α1≈α2≈0.5
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    # ---- rsl_rl 要求的接口 ----
    def reset(self, dones=None): pass

    def update_distribution(self, obs):
        mean = self.actor(obs)
        self.distribution = Normal(mean, self.std.clamp(min=1e-3))

    def act(self, obs, **kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        return self.actor(obs)

    def evaluate(self, critic_obs, **kwargs):
        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self): return self.distribution.mean
    @property
    def action_std(self): return self.distribution.stddev
    @property
    def entropy(self): return self.distribution.entropy().sum(dim=-1)

    # ---- 第一阶段结束后导出 FusionNet ----
    def export_fusion_net(self, path: str):
        torch.save({"fusion_actor": self.actor.state_dict()}, path)
        print(f"FusionNet (function label) 已导出 -> {path}")
