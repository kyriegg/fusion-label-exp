"""Offline-trained (DAgger-style) alpha network for the smp+rough fusion, plus load/save helpers.

Ported from the user's own earlier fusion-label-exp repo (github.com/kyriegg/fusion-label-exp,
isaac/dagger/train_dagger_offline.py, 2026-07-26), which proved pure-PPO-trained alpha networks
cannot learn terrain-conditional switching (alpha converges to one fixed global-compromise value
regardless of terrain - see project memory reference_fusion_label_exp_v1.md), and that supervised
("DAgger") training against a target alpha derived from a real physical/quality signal does work.

That old recipe used height_scan roughness (terrain property) as the signal: flat -> low target a2
(favor imitation), rough -> high target a2 (favor perception). This adaptation (2026-09-14, user's
own idea) swaps the signal for smp's own diffusion-guidance score (smp_bridge.smp_guidance_reward,
"how close does the robot's actual recent motion look to real human motion") - a signal that,
unlike "expert action disagreement" (which the old repo calibrated and found has ~no discriminative
power between terrains, see the same memory doc), is not just measuring "these are two different
networks" - it's an independently-learned prior over real human motion. Direction (user-confirmed):
low smp score (motion looks unnatural, e.g. scrambling for balance) -> high target a2 (favor rough/
perception); high smp score (motion looks natural) -> low target a2 (favor smp/style).

Network input is deliberately [label_smp, label_rough, raw_obs] only - the same as the old recipe -
NOT the smp score itself, so the trained network doesn't need to re-run the smp diffusion critic at
inference time; the score is only ever used to build the training *target*.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AlphaNet(nn.Module):
    """Input: [label_smp, label_rough, raw_obs] concatenated. Output: alpha2 in (0,1) (alpha1 = 1-alpha2)."""

    def __init__(self, in_dim: int, hid: tuple[int, ...] = (256, 128)):
        super().__init__()
        layers = []
        dd = in_dim
        for h in hid:
            layers += [nn.Linear(dd, h), nn.ELU()]
            dd = h
        layers.append(nn.Linear(dd, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))  # (N, 1), a2 in (0, 1)


def save_alpha_net(net: AlphaNet, in_dim: int, hid: tuple[int, ...], out_path: str) -> None:
    torch.save({"alpha_net": net.state_dict(), "in_dim": in_dim, "arch": "sigmoid_a2", "hid": list(hid)}, out_path)


def load_alpha_net(path: str, device: str) -> AlphaNet:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = AlphaNet(ckpt["in_dim"], tuple(ckpt["hid"])).to(device)
    net.load_state_dict(ckpt["alpha_net"])
    net.eval()
    return net


def alpha_net_input(label_smp: torch.Tensor, label_rough: torch.Tensor, raw_obs: torch.Tensor) -> torch.Tensor:
    return torch.cat([label_smp, label_rough, raw_obs], dim=-1)


def alpha2_to_alpha(alpha2: torch.Tensor) -> torch.Tensor:
    """(N, 1) alpha2 -> (N, 2) [alpha1, alpha2], matching G1FusionEnv's alpha_override shape."""
    return torch.cat([1.0 - alpha2, alpha2], dim=-1)
