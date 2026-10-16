"""Cosine schedule, epsilon-prediction training, and DDIM sampling."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class DiffusionSchedule:
    def __init__(self, steps: int = 50, device: torch.device | None = None):
        self.steps = steps
        self.alpha_bar = cosine_alpha_bar(steps).to(device or torch.device("cpu"))

    def to(self, device: torch.device) -> "DiffusionSchedule":
        self.alpha_bar = self.alpha_bar.to(device)
        return self


def cosine_alpha_bar(steps: int, s: float = 0.008) -> torch.Tensor:
    t = torch.arange(steps + 1, dtype=torch.float64)
    f = torch.cos(((t / steps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
    alpha_bar = f / f[0]
    alpha_bar = alpha_bar.clamp(1e-4, 0.9999)
    return alpha_bar[1:].float()


def q_sample(x0: torch.Tensor, k: torch.Tensor, alpha_bar: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    ab = alpha_bar[k].view(-1, 1, 1).to(dtype=x0.dtype)
    return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise


def ddim_indices(train_steps: int, sample_steps: int) -> torch.Tensor:
    raw = torch.linspace(train_steps - 1, 0, sample_steps)
    idx = torch.unique(raw.round().long())
    return torch.sort(idx, descending=True).values


def ddim_step(x: torch.Tensor, eps: torch.Tensor, k: int, k_prev: int, alpha_bar: torch.Tensor) -> torch.Tensor:
    ab = alpha_bar[k].to(dtype=x.dtype)
    ab_prev = x.new_tensor(1.0) if k_prev < 0 else alpha_bar[k_prev].to(dtype=x.dtype)
    x0 = (x - (1.0 - ab).sqrt() * eps) / ab.sqrt()
    x0 = x0.clamp(-6.0, 6.0)
    return ab_prev.sqrt() * x0 + (1.0 - ab_prev).sqrt() * eps


def train_epsilon(
    model: nn.Module,
    obs: torch.Tensor,
    actions: torch.Tensor,
    schedule: DiffusionSchedule,
    epochs: int,
    batch_size: int,
    lr: float,
) -> list[float]:
    device = actions.device
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: list[float] = []
    n = actions.shape[0]
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        seen = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            x0 = actions[idx]
            cond = obs[idx]
            k = torch.randint(0, schedule.steps, (x0.shape[0],), device=device)
            noise = torch.randn_like(x0)
            xt = q_sample(x0, k, schedule.alpha_bar, noise)
            t_norm = k.float() / max(schedule.steps - 1, 1)
            pred = model(xt, t_norm, cond)
            loss = torch.mean((pred - noise) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * x0.shape[0]
            seen += x0.shape[0]
        losses.append(total / max(seen, 1))
    return losses


@torch.inference_mode()
def sample_ddim(
    model: nn.Module,
    cond: torch.Tensor,
    schedule: DiffusionSchedule,
    horizon: int,
    action_dim: int,
    sample_steps: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    model.eval()
    device = cond.device
    indices = ddim_indices(schedule.steps, sample_steps).to(device)
    x = torch.randn(
        cond.shape[0],
        horizon,
        action_dim,
        device=device,
        dtype=cond.dtype,
        generator=generator,
    )
    for i, k_t in enumerate(indices.tolist()):
        k_prev = int(indices[i + 1]) if i + 1 < indices.numel() else -1
        t_norm = torch.full((cond.shape[0],), float(k_t) / max(schedule.steps - 1, 1), device=device)
        eps = model(x, t_norm, cond)
        x = ddim_step(x, eps, int(k_t), k_prev, schedule.alpha_bar)
    return x
