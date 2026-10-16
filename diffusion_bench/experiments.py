"""Latency, multimodal separation, and perturbation-recovery experiments."""

from __future__ import annotations

import time

import numpy as np
import torch

from .diffusion import DiffusionSchedule, sample_ddim, train_epsilon
from .models import MLPBehaviorClone, TrajectoryDenoiser, VisionStub
from .policies import (
    DiffusionChunkPolicy,
    MLPChunkPolicy,
    fit_normalizers,
    rollout,
    summarize_rollouts,
)
from .task import (
    ACTION_DIM,
    DISTURB_AFTER_STEPS,
    N_ACTIONS,
    OBS_DIM,
    PERTURBATION_M,
    PRED_HORIZON,
    START,
    build_dataset,
    geometry_checks,
    min_clearance,
    observation,
)

SURGICAL_OBS_DIM = 128
SURGICAL_ACTION_DIM = 7
HZ_50_BUDGET_MS = 20.0
HZ_20_BUDGET_MS = 50.0


def _percentiles(samples_ms: list[float]) -> dict:
    arr = np.asarray(samples_ms, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def run_latency(
    device: torch.device,
    steps_list: list[int],
    n_runs: int = 100,
    warmup: int = 20,
    train_steps: int = 50,
) -> dict:
    """Time one closed-loop denoise. The surgical-shaped head is randomly initialized."""
    head = TrajectoryDenoiser(SURGICAL_ACTION_DIM, SURGICAL_OBS_DIM, hidden=64).to(device).eval()
    eye = VisionStub(SURGICAL_OBS_DIM).to(device).eval()
    schedule = DiffusionSchedule(train_steps, device=device)
    cond = torch.randn(1, SURGICAL_OBS_DIM, device=device)
    image = torch.randn(1, 3, 128, 128, device=device)
    rows = []
    for include_eye in (False, True):
        for k in steps_list:
            for _ in range(warmup):
                if include_eye:
                    cond = eye(image)
                sample_ddim(head, cond, schedule, PRED_HORIZON, SURGICAL_ACTION_DIM, k)
            _sync(device)
            samples = []
            for _ in range(n_runs):
                _sync(device)
                t0 = time.perf_counter()
                if include_eye:
                    cond = eye(image)
                sample_ddim(head, cond, schedule, PRED_HORIZON, SURGICAL_ACTION_DIM, k)
                _sync(device)
                samples.append((time.perf_counter() - t0) * 1000.0)
            stats = _percentiles(samples)
            p99 = stats["p99_ms"]
            rows.append(
                {
                    "condition": "eye_and_action_head" if include_eye else "action_head",
                    "requested_steps": k,
                    "executed_steps": int(ddim_len(train_steps, k)),
                    "runs": n_runs,
                    **stats,
                    "meets_50hz_20ms": p99 < HZ_50_BUDGET_MS,
                    "meets_20hz_50ms": p99 < HZ_20_BUDGET_MS,
                }
            )
    return {
        "device": str(device),
        "train_diffusion_steps": train_steps,
        "pred_horizon": PRED_HORIZON,
        "action_dim": SURGICAL_ACTION_DIM,
        "obs_dim": SURGICAL_OBS_DIM,
        "image": [3, 128, 128],
        "warmup": warmup,
        "budget_50hz_ms": HZ_50_BUDGET_MS,
        "budget_20hz_ms": HZ_20_BUDGET_MS,
        "weights": "random_initialization",
        "rows": rows,
    }


def ddim_len(train_steps: int, sample_steps: int) -> int:
    from .diffusion import ddim_indices

    return int(ddim_indices(train_steps, sample_steps).numel())


def _to_device(obs: np.ndarray, act: np.ndarray, obs_norm, act_norm, device: torch.device):
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    act_t = torch.tensor(act, dtype=torch.float32, device=device)
    return obs_norm.encode(obs_t), act_norm.encode(act_t)


def train_policies(device: torch.device, seed: int, epochs: int, diffusion_steps: int) -> dict:
    rng = np.random.default_rng(seed)
    data = build_dataset(rng)
    obs_norm, act_norm = fit_normalizers(data["obs"], data["act"])
    obs_norm.to(device)
    act_norm.to(device)
    obs_t, act_t = _to_device(data["obs"], data["act"], obs_norm, act_norm, device)
    schedule = DiffusionSchedule(diffusion_steps, device=device)

    denoiser = TrajectoryDenoiser(ACTION_DIM, OBS_DIM, hidden=128).to(device)
    mlp = MLPBehaviorClone(OBS_DIM, PRED_HORIZON, ACTION_DIM, hidden=256).to(device)
    diff_losses = train_epsilon(denoiser, obs_t, act_t, schedule, epochs=epochs, batch_size=256, lr=1e-3)
    mlp_losses = _train_mlp(mlp, obs_t, act_t, epochs=epochs, batch_size=256, lr=1e-3)
    denoiser.eval()
    mlp.eval()
    return {
        "denoiser": denoiser,
        "mlp": mlp,
        "schedule": schedule,
        "obs_norm": obs_norm,
        "act_norm": act_norm,
        "dataset": data,
        "diffusion_loss_last": diff_losses[-1],
        "mlp_loss_last": mlp_losses[-1],
        "diffusion_loss_first": diff_losses[0],
        "mlp_loss_first": mlp_losses[0],
    }


def _train_mlp(model, obs, actions, epochs: int, batch_size: int, lr: float) -> list[float]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    n = actions.shape[0]
    device = actions.device
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        seen = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            pred = model(obs[idx])
            loss = torch.mean((pred - actions[idx]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item()) * idx.shape[0]
            seen += idx.shape[0]
        losses.append(total / max(seen, 1))
    model.eval()
    return losses


def run_multimodal(bundle: dict, device: torch.device, trials: int, seed: int, sample_steps: int) -> dict:
    diffusion = DiffusionChunkPolicy(
        bundle["denoiser"],
        bundle["schedule"],
        bundle["obs_norm"],
        bundle["act_norm"],
        sample_steps,
        device,
    )
    mlp = MLPChunkPolicy(bundle["mlp"], bundle["obs_norm"], bundle["act_norm"], device)
    mlp_rows = [rollout(mlp, mode="continue", t_exec=N_ACTIONS, seed=seed, device=device, max_steps=N_ACTIONS)]
    diff_rows = [
        rollout(diffusion, mode="continue", t_exec=N_ACTIONS, seed=seed + i, device=device, max_steps=N_ACTIONS)
        for i in range(trials)
    ]
    mlp_summary = summarize_rollouts(mlp_rows)
    mlp_summary["collision_rate"] = 1.0 if mlp_rows[0]["collided"] else 0.0
    mlp_summary["min_clearance_cm"] = min_clearance(mlp_rows[0]["path"]) * 100.0
    diff_summary = summarize_rollouts(diff_rows)
    diff_summary["min_clearance_cm"] = float(np.median([min_clearance(r["path"]) for r in diff_rows]) * 100.0)
    return {
        "sample_steps": sample_steps,
        "start_m": START.tolist(),
        "mlp": mlp_summary,
        "diffusion": diff_summary,
        "paths": {
            "mlp": _path(mlp_rows[0]["path"]),
            "diffusion": [_path(r["path"]) for r in diff_rows[:8]],
        },
    }


def run_recovery(bundle: dict, device: torch.device, trials: int, seed: int, sample_steps: int) -> dict:
    diffusion = DiffusionChunkPolicy(
        bundle["denoiser"],
        bundle["schedule"],
        bundle["obs_norm"],
        bundle["act_norm"],
        sample_steps,
        device,
    )
    grouped: dict[tuple[str, int], list] = {
        ("continue", N_ACTIONS): [],
        ("replan", 4): [],
        ("replan", 8): [],
    }
    for i in range(trials):
        trial_seed = seed + 1000 + i
        grouped[("continue", N_ACTIONS)].append(
            rollout(
                diffusion,
                mode="continue",
                t_exec=N_ACTIONS,
                seed=trial_seed,
                device=device,
                disturb_m=PERTURBATION_M,
                max_steps=N_ACTIONS,
            )
        )
        for t_exec in (4, 8):
            grouped[("replan", t_exec)].append(
                rollout(
                    diffusion,
                    mode="replan",
                    t_exec=t_exec,
                    seed=trial_seed,
                    device=device,
                    disturb_m=PERTURBATION_M,
                    max_steps=N_ACTIONS * 2,
                )
            )
    rows = []
    for (mode, t_exec), rollouts in grouped.items():
        summary = summarize_rollouts(rollouts)
        summary["mode"] = "CONTINUE" if mode == "continue" else "REPLAN"
        summary["t_exec"] = t_exec
        summary["perturbation_cm"] = PERTURBATION_M * 100.0
        rows.append(summary)

    examples = {}
    continue_rows = grouped[("continue", N_ACTIONS)]
    for t_exec in (4, 8):
        chosen = 0
        for i, (left, right) in enumerate(zip(continue_rows, grouped[("replan", t_exec)])):
            if (not left["success"]) and right["success"]:
                chosen = i
                break
        examples[str(t_exec)] = {
            "trial": chosen,
            "continue": _path(continue_rows[chosen]["path"]),
            "replan": _path(grouped[("replan", t_exec)][chosen]["path"]),
        }
    return {
        "sample_steps": sample_steps,
        "perturbation_cm": PERTURBATION_M * 100.0,
        "perturbation_axis": "+y",
        "disturb_after_steps": DISTURB_AFTER_STEPS,
        "goal_tolerance_cm": 2.0,
        "rows": rows,
        "examples": examples,
    }


def _path(points: np.ndarray) -> list[list[float]]:
    return np.round(np.asarray(points, dtype=np.float64), 5).tolist()


def probe_origin(bundle: dict, device: torch.device, sample_steps: int) -> dict:
    """One deterministic MLP path and one diffusion sample, used as a training sanity check."""
    diffusion = DiffusionChunkPolicy(
        bundle["denoiser"], bundle["schedule"], bundle["obs_norm"], bundle["act_norm"], sample_steps, device
    )
    mlp = MLPChunkPolicy(bundle["mlp"], bundle["obs_norm"], bundle["act_norm"], device)
    mlp_row = rollout(mlp, mode="continue", t_exec=N_ACTIONS, seed=0, device=device, max_steps=N_ACTIONS)
    diff_row = rollout(diffusion, mode="continue", t_exec=N_ACTIONS, seed=0, device=device, max_steps=N_ACTIONS)
    return {
        "mlp_collided": mlp_row["collided"],
        "diffusion_collided": diff_row["collided"],
        "mlp_side": mlp_row["side"],
        "diffusion_side": diff_row["side"],
        "obs_dim": int(observation(START).shape[0]),
    }
