"""Temporal-convolution denoiser and the MSE behavior-cloning baseline."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=timesteps.dtype) / half)
    args = timesteps[:, None] * freqs[None, :]
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    if dim % 2:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class TrajectoryDenoiser(nn.Module):
    """1D conv denoiser over an action chunk, conditioned on a state vector and the noise level."""

    def __init__(self, action_dim: int, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.hidden = hidden
        self.time_net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Mish(),
            nn.Linear(hidden, hidden),
        )
        self.cond_proj = nn.Linear(obs_dim, hidden)
        self.film = nn.Linear(hidden, hidden * 2)
        # The first layer reads only the action chunk. Condition and noise level
        # scale those features. A horizon embedding lets the chunk rise and return.
        self.conv1 = nn.Conv1d(action_dim, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.out_conv = nn.Conv1d(hidden, action_dim, kernel_size=1)
        self.act = nn.Mish()

    def forward(self, noisy_trajectory: torch.Tensor, timestep: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # noisy_trajectory: (B, T, A), timestep: (B,) in [0, 1], cond: (B, obs)
        length = noisy_trajectory.shape[1]
        t_emb = self.time_net(sinusoidal_embedding(timestep, self.hidden))
        c_emb = self.cond_proj(cond) + t_emb
        scale, shift = self.film(c_emb).chunk(2, dim=-1)
        h = self.conv1(noisy_trajectory.transpose(1, 2))
        index = torch.arange(length, device=noisy_trajectory.device, dtype=noisy_trajectory.dtype)
        pe = sinusoidal_embedding(index, self.hidden).transpose(0, 1).unsqueeze(0)
        h = h + pe
        h = h * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        h = self.act(h)
        h = self.act(self.conv2(h)) + h
        h = self.act(self.conv3(h))
        return self.out_conv(h).transpose(1, 2)


class VisionStub(nn.Module):
    """Small conv encoder so the latency table can include an eye in front of the action head.

    Weights are random. The stub stands in for a vision embedding; it is not TinySurgicalNet
    and it is not the serving benchmark.
    """

    def __init__(self, obs_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.Mish(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.Mish(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.Mish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, obs_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class MLPBehaviorClone(nn.Module):
    """Deterministic chunk regression. MSE on two corridors returns their average."""

    def __init__(self, obs_dim: int, pred_horizon: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Mish(),
            nn.Linear(hidden, hidden),
            nn.Mish(),
            nn.Linear(hidden, pred_horizon * action_dim),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        flat = self.net(cond)
        return flat.view(-1, self.pred_horizon, self.action_dim)
