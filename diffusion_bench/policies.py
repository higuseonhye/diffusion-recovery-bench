"""Rollouts for open-loop CONTINUE and receding-horizon REPLAN."""

from __future__ import annotations

import numpy as np
import torch

from .diffusion import DiffusionSchedule, sample_ddim
from .flow import sample_euler
from .models import MLPBehaviorClone, TrajectoryDenoiser
from .task import (
    ACTION_DIM,
    DISTURB_AFTER_STEPS,
    GOAL,
    GOAL_TOL,
    N_ACTIONS,
    OBS_DIM,
    PRED_HORIZON,
    START,
    observation,
    path_hits_disk,
    segment_hits_disk,
)


class Normalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def to(self, device: torch.device) -> "Normalizer":
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.mean) / self.std

    def decode_action(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.std + self.mean


def fit_normalizers(obs: np.ndarray, act: np.ndarray) -> tuple[Normalizer, Normalizer]:
    obs_mean = obs.mean(axis=0)
    obs_std = np.maximum(obs.std(axis=0), 1e-3)
    act_mean = act.mean(axis=(0, 1))
    act_std = np.maximum(act.std(axis=(0, 1)), 1e-3)
    return Normalizer(obs_mean, obs_std), Normalizer(act_mean, act_std)


class ChunkPolicy:
    def chunk(self, xy: np.ndarray, generator: torch.Generator | None) -> np.ndarray:
        raise NotImplementedError


class DiffusionChunkPolicy(ChunkPolicy):
    def __init__(
        self,
        model: TrajectoryDenoiser,
        schedule: DiffusionSchedule,
        obs_norm: Normalizer,
        act_norm: Normalizer,
        sample_steps: int,
        device: torch.device,
    ):
        self.model = model
        self.schedule = schedule
        self.obs_norm = obs_norm
        self.act_norm = act_norm
        self.sample_steps = sample_steps
        self.device = device

    def chunk(self, xy: np.ndarray, generator: torch.Generator | None) -> np.ndarray:
        obs = torch.tensor(observation(xy), dtype=torch.float32, device=self.device).unsqueeze(0)
        cond = self.obs_norm.encode(obs)
        normed = sample_ddim(
            self.model,
            cond,
            self.schedule,
            PRED_HORIZON,
            ACTION_DIM,
            self.sample_steps,
            generator=generator,
        )
        out = self.act_norm.decode_action(normed)[0].detach().cpu().numpy()
        return np.clip(out, -0.40, 0.40)


class FlowChunkPolicy(ChunkPolicy):
    def __init__(
        self,
        model: TrajectoryDenoiser,
        obs_norm: Normalizer,
        act_norm: Normalizer,
        sample_steps: int,
        device: torch.device,
    ):
        self.model = model
        self.obs_norm = obs_norm
        self.act_norm = act_norm
        self.sample_steps = sample_steps
        self.device = device

    def chunk(self, xy: np.ndarray, generator: torch.Generator | None) -> np.ndarray:
        obs = torch.tensor(observation(xy), dtype=torch.float32, device=self.device).unsqueeze(0)
        cond = self.obs_norm.encode(obs)
        normed = sample_euler(
            self.model,
            cond,
            PRED_HORIZON,
            ACTION_DIM,
            self.sample_steps,
            generator=generator,
        )
        out = self.act_norm.decode_action(normed)[0].detach().cpu().numpy()
        return np.clip(out, -0.40, 0.40)


class MLPChunkPolicy(ChunkPolicy):
    def __init__(self, model: MLPBehaviorClone, obs_norm: Normalizer, act_norm: Normalizer, device: torch.device):
        self.model = model
        self.obs_norm = obs_norm
        self.act_norm = act_norm
        self.device = device

    def chunk(self, xy: np.ndarray, generator: torch.Generator | None) -> np.ndarray:
        del generator
        obs = torch.tensor(observation(xy), dtype=torch.float32, device=self.device).unsqueeze(0)
        cond = self.obs_norm.encode(obs)
        with torch.inference_mode():
            normed = self.model(cond)
        out = self.act_norm.decode_action(normed)[0].detach().cpu().numpy()
        return np.clip(out, -0.40, 0.40)


def _generator(device: torch.device, seed: int) -> torch.Generator:
    gen = torch.Generator(device="cpu" if device.type == "cpu" else device)
    gen.manual_seed(seed)
    return gen


def rollout(
    policy: ChunkPolicy,
    *,
    mode: str,
    t_exec: int,
    seed: int,
    device: torch.device,
    disturb_m: float = 0.0,
    disturb_after: int = DISTURB_AFTER_STEPS,
    max_steps: int | None = None,
) -> dict:
    """Integrate relative chunks. CONTINUE denoises once. REPLAN denoises every `t_exec` steps."""
    if mode not in {"continue", "replan"}:
        raise ValueError(mode)
    if max_steps is None:
        max_steps = N_ACTIONS if mode == "continue" else N_ACTIONS * 2
    pos = START.astype(np.float64).copy()
    path = [pos.copy()]
    generator = _generator(device, seed)
    chunk: np.ndarray | None = None
    cursor = 0
    prev_offset = np.zeros(2, dtype=np.float64)
    collided = False
    goal_step = None
    for step in range(max_steps):
        if chunk is None or cursor >= t_exec or cursor >= PRED_HORIZON:
            if mode == "continue" and chunk is not None:
                break
            chunk = policy.chunk(pos, generator)
            cursor = 0
            prev_offset = np.zeros(2, dtype=np.float64)
        delta = np.clip(chunk[cursor] - prev_offset, -0.05, 0.05)
        prev_offset = chunk[cursor]
        cursor += 1
        nxt = pos + delta
        if segment_hits_disk(pos, nxt):
            collided = True
        pos = nxt
        if disturb_m and (step + 1) == disturb_after:
            pos = pos + np.array([0.0, disturb_m], dtype=np.float64)
        path.append(pos.copy())
        if np.linalg.norm(pos - GOAL) <= GOAL_TOL:
            if goal_step is None:
                goal_step = step + 1
            if mode == "replan":
                break
    points = np.stack(path)
    if path_hits_disk(points):
        collided = True
    final_error = float(np.linalg.norm(points[-1] - GOAL))
    success = (not collided) and final_error <= GOAL_TOL
    steps_after = None
    if goal_step is not None and success:
        steps_after = int(goal_step - disturb_after) if disturb_m else int(goal_step)
        steps_after = max(steps_after, 0)
    mean_y = float(points[:, 1].mean())
    if mean_y > 0.01:
        side = "left"
    elif mean_y < -0.01:
        side = "right"
    else:
        side = "collapsed"
    return {
        "success": success,
        "collided": collided,
        "final_error_m": final_error,
        "steps_to_goal_after_event": steps_after,
        "side": side,
        "path": points.astype(np.float32),
    }


def summarize_rollouts(rows: list[dict], disturb_after: int = DISTURB_AFTER_STEPS) -> dict:
    n = len(rows)
    successes = [r for r in rows if r["success"]]
    errors_cm = [r["final_error_m"] * 100.0 for r in rows]
    steps = [r["steps_to_goal_after_event"] for r in successes if r["steps_to_goal_after_event"] is not None]
    sides = [r["side"] for r in rows]
    return {
        "trials": n,
        "success_rate": len(successes) / n,
        "collision_rate": sum(1 for r in rows if r["collided"]) / n,
        "median_final_error_cm": float(np.median(errors_cm)),
        "mean_final_error_cm": float(np.mean(errors_cm)),
        "median_steps_to_goal": float(np.median(steps)) if steps else None,
        "median_time_to_goal_at_20hz_s": (float(np.median(steps)) * 0.05) if steps else None,
        "left_fraction": sides.count("left") / n,
        "right_fraction": sides.count("right") / n,
        "collapsed_fraction": sides.count("collapsed") / n,
        "disturb_after_steps": disturb_after,
    }
