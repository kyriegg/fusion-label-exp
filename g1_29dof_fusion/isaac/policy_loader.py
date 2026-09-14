"""Load a frozen inference-only policy from a raw rsl_rl training checkpoint.

fusion_env_g1_v2.py used to load smp_policy/rough_policy via torch.jit.load(),
but D:\\robot paper\\fusion_label_exp\\exported\\rough_policy.pt and
D:\\robot paper\\smp\\logs\\...\\model_*.pt are both plain rsl_rl training
checkpoints (dict with actor_state_dict/critic_state_dict/optimizer_state_dict/
iter/infos) - never exported to TorchScript, so torch.jit.load() throws
immediately. This reconstructs the actor MLP (+ optional obs normalizer)
straight from actor_state_dict, matching rsl_rl 5.5.1's module layout
(rsl_rl.modules.MLP: Sequential named "0","2","4",... for the Linear layers;
rsl_rl.modules.EmpiricalNormalization: buffers "_mean"/"_var"/"_std"/"count").
Confirmed by inspecting both checkpoints directly: hidden dims are always
[512, 256, 128] / activation "elu" for this project's policies (matches
G1_29DOF_RoughPPORunnerCfg and the smp training cfg), but hidden dims are
still inferred from the checkpoint's own weight shapes rather than hardcoded,
so this keeps working if that ever changes.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules import MLP, EmpiricalNormalization


class FrozenPolicy(nn.Module):
    """Deterministic (mean-action) inference wrapper around a loaded actor MLP."""

    def __init__(self, mlp: MLP, normalizer: EmpiricalNormalization | None):
        super().__init__()
        self.mlp = mlp
        self.normalizer = normalizer

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.normalizer is not None:
            obs = self.normalizer(obs)
        return self.mlp(obs)


def load_rsl_rl_policy(ckpt_path: str, device: torch.device | str) -> FrozenPolicy:
    """Load ckpt_path's actor_state_dict into a frozen, eval-mode FrozenPolicy."""
    device = torch.device(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["actor_state_dict"]

    input_dim = sd["mlp.0.weight"].shape[1]
    output_dim = sd["mlp.6.weight"].shape[0]
    hidden_dims = [sd["mlp.0.weight"].shape[0], sd["mlp.2.weight"].shape[0], sd["mlp.4.weight"].shape[0]]

    mlp = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims, activation="elu").to(device)
    mlp_state = {k[len("mlp."):]: v for k, v in sd.items() if k.startswith("mlp.")}
    mlp.load_state_dict(mlp_state)

    normalizer = None
    if "obs_normalizer._mean" in sd:
        normalizer = EmpiricalNormalization(shape=(input_dim,)).to(device)
        norm_state = {k[len("obs_normalizer."):]: v for k, v in sd.items() if k.startswith("obs_normalizer.")}
        normalizer.load_state_dict(norm_state)
        normalizer.eval()

    policy = FrozenPolicy(mlp, normalizer).to(device)
    policy.eval()
    policy.requires_grad_(False)
    return policy
