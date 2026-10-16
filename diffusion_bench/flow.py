"""Optimal-transport conditional flow matching on the same action-chunk backbone.

The interpolant is the straight line x_t = (1 - t) * noise + t * action.
The regression target is the constant velocity action - noise. Sampling is Euler
integration from t=0 to t=1. No beta schedule is used.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def train_velocity(
    model: nn.Module,
    obs: torch.Tensor,
    actions: torch.Tensor,
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
            target_action = actions[idx]
            cond = obs[idx]
            noise = torch.randn_like(target_action)
            t = torch.rand(target_action.shape[0], 1, 1, device=device)
            x_t = (1.0 - t) * noise + t * target_action
            velocity = target_action - noise
            pred = model(x_t, t.view(-1), cond)
            loss = torch.mean((pred - velocity) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * target_action.shape[0]
            seen += target_action.shape[0]
        losses.append(total / max(seen, 1))
    return losses


@torch.inference_mode()
def sample_euler(
    model: nn.Module,
    cond: torch.Tensor,
    horizon: int,
    action_dim: int,
    steps: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Integrate dx/dt = v_theta from noise (t=0) to the action chunk (t=1)."""
    if steps < 1:
        raise ValueError("Euler needs at least one function evaluation")
    model.eval()
    device = cond.device
    x = torch.randn(
        cond.shape[0],
        horizon,
        action_dim,
        device=device,
        dtype=cond.dtype,
        generator=generator,
    )
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((cond.shape[0],), i * dt, device=device)
        velocity = model(x, t, cond)
        x = x + dt * velocity
    return x
