"""Measure Euler flow matching against the saved DDIM policy on the same task."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from . import __version__
from .diffusion import DiffusionSchedule, sample_ddim
from .experiments import _percentiles, _sync
from .flow import sample_euler, train_velocity
from .models import TrajectoryDenoiser
from .policies import DiffusionChunkPolicy, FlowChunkPolicy, Normalizer, rollout, summarize_rollouts
from .task import (
    ACTION_DIM,
    N_ACTIONS,
    OBS_DIM,
    PERTURBATION_M,
    PRED_HORIZON,
    START,
    build_dataset,
    min_clearance,
    observation,
)

SURGICAL_OBS = 128
SURGICAL_ACTION = 7
BUDGET_100HZ_MS = 10.0
BUDGET_50HZ_MS = 20.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare DDIM and optimal-transport flow matching on the planar bench.")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--checkpoint", default="results/planar_policy.pt")
    parser.add_argument("--output", default="results/flow_matching_report.json")
    parser.add_argument("--report", default="FLOW_MATCHING_RESULTS.md")
    parser.add_argument("--figure", default="docs/figures/flow_matching_compare.png")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--latency-runs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _time_sampler(fn, steps_list: list[int], n_runs: int, warmup: int, device: torch.device) -> list[dict]:
    rows = []
    for steps in steps_list:
        for _ in range(warmup):
            fn(steps)
        _sync(device)
        samples = []
        for _ in range(n_runs):
            _sync(device)
            t0 = time.perf_counter()
            fn(steps)
            _sync(device)
            samples.append((time.perf_counter() - t0) * 1000.0)
        stats = _percentiles(samples)
        p99 = stats["p99_ms"]
        rows.append(
            {
                "steps": steps,
                "runs": n_runs,
                **stats,
                "meets_100hz_10ms": p99 < BUDGET_100HZ_MS,
                "meets_50hz_20ms": p99 < BUDGET_50HZ_MS,
            }
        )
    return rows


@torch.inference_mode()
def run_latency(device: torch.device, n_runs: int) -> dict:
    head = TrajectoryDenoiser(SURGICAL_ACTION, SURGICAL_OBS, hidden=64).to(device).eval()
    schedule = DiffusionSchedule(50, device=device)
    cond = torch.randn(1, SURGICAL_OBS, device=device)
    steps = [1, 2, 4, 10, 16]

    def ddim(k: int) -> None:
        sample_ddim(head, cond, schedule, PRED_HORIZON, SURGICAL_ACTION, k)

    def euler(k: int) -> None:
        sample_euler(head, cond, PRED_HORIZON, SURGICAL_ACTION, k)

    return {
        "device": str(device),
        "action_dim": SURGICAL_ACTION,
        "obs_dim": SURGICAL_OBS,
        "pred_horizon": PRED_HORIZON,
        "weights": "random_initialization",
        "budget_100hz_ms": BUDGET_100HZ_MS,
        "budget_50hz_ms": BUDGET_50HZ_MS,
        "ddim": _time_sampler(ddim, steps, n_runs, warmup=15, device=device),
        "flow": _time_sampler(euler, steps, n_runs, warmup=15, device=device),
    }


def _quality(policy, device: torch.device, trials: int, seed: int) -> dict:
    rows = [
        rollout(policy, mode="continue", t_exec=N_ACTIONS, seed=seed + i, device=device, max_steps=N_ACTIONS)
        for i in range(trials)
    ]
    summary = summarize_rollouts(rows)
    summary["min_clearance_cm"] = float(np.median([min_clearance(r["path"]) for r in rows]) * 100.0)
    summary["paths"] = [np.round(r["path"], 5).tolist() for r in rows[:6]]
    return summary


def _recovery(policy, device: torch.device, trials: int, seed: int) -> list[dict]:
    grouped = {("CONTINUE", N_ACTIONS): [], ("REPLAN", 4): [], ("REPLAN", 8): []}
    for i in range(trials):
        trial_seed = seed + 1000 + i
        grouped[("CONTINUE", N_ACTIONS)].append(
            rollout(
                policy,
                mode="continue",
                t_exec=N_ACTIONS,
                seed=trial_seed,
                device=device,
                disturb_m=PERTURBATION_M,
                max_steps=N_ACTIONS,
            )
        )
        for t_exec in (4, 8):
            grouped[("REPLAN", t_exec)].append(
                rollout(
                    policy,
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
        summary["mode"] = mode
        summary["t_exec"] = t_exec
        rows.append(summary)
    return rows


def _load_diffusion(path: Path, device: torch.device, sample_steps: int) -> DiffusionChunkPolicy:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = TrajectoryDenoiser(ACTION_DIM, OBS_DIM, hidden=128).to(device)
    model.load_state_dict(checkpoint["denoiser"])
    model.eval()
    schedule = DiffusionSchedule(int(checkpoint["diffusion_steps"]), device=device)
    obs_norm = Normalizer(checkpoint["obs_mean"].numpy(), checkpoint["obs_std"].numpy()).to(device)
    act_norm = Normalizer(checkpoint["act_mean"].numpy(), checkpoint["act_std"].numpy()).to(device)
    return DiffusionChunkPolicy(model, schedule, obs_norm, act_norm, sample_steps, device)


def _train_flow(device: torch.device, seed: int, epochs: int) -> tuple[FlowChunkPolicy, dict]:
    rng = np.random.default_rng(seed)
    data = build_dataset(rng)
    obs_norm, act_norm = _fit(data["obs"], data["act"], device)
    obs = obs_norm.encode(torch.tensor(data["obs"], dtype=torch.float32, device=device))
    act = act_norm.encode(torch.tensor(data["act"], dtype=torch.float32, device=device))
    model = TrajectoryDenoiser(ACTION_DIM, OBS_DIM, hidden=128).to(device)
    losses = train_velocity(model, obs, act, epochs=epochs, batch_size=256, lr=1e-3)
    model.eval()
    policy = FlowChunkPolicy(model, obs_norm, act_norm, sample_steps=2, device=device)
    info = {
        "epochs": epochs,
        "samples": int(data["samples"]),
        "loss_first": losses[0],
        "loss_last": losses[-1],
    }
    return policy, info, model, obs_norm, act_norm


def _fit(obs: np.ndarray, act: np.ndarray, device: torch.device):
    obs_mean = obs.mean(axis=0)
    obs_std = np.maximum(obs.std(axis=0), 1e-3)
    act_mean = act.mean(axis=(0, 1))
    act_std = np.maximum(act.std(axis=(0, 1)), 1e-3)
    return (
        Normalizer(obs_mean, obs_std).to(device),
        Normalizer(act_mean, act_std).to(device),
    )


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    return f"{float(value):.{digits}f}"


def _pct(rate: float) -> str:
    return f"{100.0 * rate:.1f}%"


def render_markdown(report: dict) -> str:
    lines = [
        "# Flow matching vs DDIM",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "Same 1D temporal denoiser. DDIM regresses noise along a cosine schedule. "
        "Flow matching regresses the straight-line velocity from noise to the action chunk "
        "and integrates it with Euler steps. Behavior numbers use the saved DDIM checkpoint "
        "and a flow model trained for the same epoch budget on the same synthetic corridors.",
        "",
        "## Latency, action head only",
        "",
        "Random weights, batch 1, action dim 7, observation dim 128, horizon 16. "
        "100 Hz requires P99 < 10 ms. 50 Hz requires P99 < 20 ms. "
        "One Euler step and one DDIM step are both a single forward pass.",
        "",
        "| sampler | steps | p50_ms | p99_ms | 100Hz | 50Hz |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name in ("ddim", "flow"):
        for row in report["latency"][name]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        str(row["steps"]),
                        _fmt(row["p50_ms"]),
                        _fmt(row["p99_ms"]),
                        "pass" if row["meets_100hz_10ms"] else "fail",
                        "pass" if row["meets_50hz_20ms"] else "fail",
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Undisturbed corridors",
            "",
            "32 stochastic samples from the origin. Success means the goal ball is reached and the disk is missed.",
            "",
            "| sampler | steps | collision | success | median_error_cm | left | right | collapsed |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report["quality"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["sampler"],
                    str(row["steps"]),
                    _pct(row["collision_rate"]),
                    _pct(row["success_rate"]),
                    _fmt(row["median_final_error_cm"]),
                    _pct(row["left_fraction"]),
                    _pct(row["right_fraction"]),
                    _pct(row["collapsed_fraction"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 6 cm jump",
            "",
            "Same seeds as the diffusion recovery track (`seed + 1000`).",
            "",
            "| sampler | steps | mode | T_exec | success | collision | median_error_cm |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for block in report["recovery"]:
        for row in block["rows"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        block["sampler"],
                        str(block["steps"]),
                        row["mode"],
                        str(row["t_exec"]),
                        _pct(row["success_rate"]),
                        _pct(row["collision_rate"]),
                        _fmt(row["median_final_error_cm"]),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Latency does not measure trajectory quality. A 2-step sampler is fast whether or not the chunk clears the disk.",
            "- The earlier expectation that flow matching at 2 steps would land near 2.5 ms restates the measured DDIM K=2 cost. This file is the direct Euler timing.",
            "- 100 Hz is a 10 ms control period. It describes this action head, not a camera encoder plus a robot.",
            "- The planar point mass, the 4 cm disk, and the instantaneous 6 cm jump are unchanged. Isaac Sim was not run.",
            "- This is a software benchmark, not a surgical device.",
            "",
        ]
    )
    return "\n".join(lines)


def write_figure(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))
    steps = [row["steps"] for row in report["latency"]["ddim"]]
    x = np.arange(len(steps))
    width = 0.36
    ax = axes[0]
    ddim_p50 = [row["p50_ms"] for row in report["latency"]["ddim"]]
    flow_p50 = [row["p50_ms"] for row in report["latency"]["flow"]]
    ddim_p99 = [row["p99_ms"] for row in report["latency"]["ddim"]]
    flow_p99 = [row["p99_ms"] for row in report["latency"]["flow"]]
    ax.bar(x - width / 2, ddim_p50, width, color="#2b5c8f", label="DDIM P50")
    ax.bar(x + width / 2, flow_p50, width, color="#1b8a5a", label="Flow P50")
    ax.scatter(x - width / 2, ddim_p99, color="#2b5c8f", s=18, zorder=3, label="DDIM P99")
    ax.scatter(x + width / 2, flow_p99, color="#1b8a5a", s=18, zorder=3, label="Flow P99")
    ax.axhline(10.0, color="#6b4c9a", linestyle="--", linewidth=1.2, label="100 Hz (10 ms)")
    ax.axhline(20.0, color="#b4232c", linestyle=":", linewidth=1.2, label="50 Hz (20 ms)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in steps])
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("Latency (ms)")
    ax.set_ylim(0, 32)
    ax.set_title("Action-head latency", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)

    ax = axes[1]
    labels = []
    success = []
    colors = []
    for row in report["quality"]:
        labels.append(f"{row['sampler']}\n{row['steps']}")
        success.append(100.0 * row["success_rate"])
        colors.append("#2b5c8f" if row["sampler"] == "ddim" else "#1b8a5a")
    bars = ax.bar(labels, success, color=colors)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Success (%)")
    ax.set_title("Corridor success from the origin", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{bar.get_height():.0f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.synthetic:
        print("Pass --synthetic.", flush=True)
        return 2
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"Missing {checkpoint}. Run the diffusion bench first.", flush=True)
        return 2

    print("Latency: DDIM vs Euler, same head.", flush=True)
    latency = run_latency(device, args.latency_runs)
    for name in ("ddim", "flow"):
        for row in latency[name]:
            print(
                f"  {name:4s} steps={row['steps']:2d}  P99 {row['p99_ms']:.2f} ms  "
                f"100Hz {'pass' if row['meets_100hz_10ms'] else 'FAIL'}",
                flush=True,
            )

    print("DDIM quality at K=2, 4, 10 from the saved checkpoint.", flush=True)
    quality = []
    recovery = []
    for steps in (2, 4, 10):
        policy = _load_diffusion(checkpoint, device, steps)
        summary = _quality(policy, device, args.trials, args.seed)
        print(
            f"  ddim K={steps:2d}  collision {summary['collision_rate']:.0%}  success {summary['success_rate']:.0%}",
            flush=True,
        )
        quality.append({"sampler": "ddim", "steps": steps, **{k: v for k, v in summary.items() if k != "paths"}, "paths": summary["paths"]})
        if steps in (2, 10):
            rows = _recovery(policy, device, args.trials, args.seed)
            recovery.append({"sampler": "ddim", "steps": steps, "rows": rows})
            replan = next(r for r in rows if r["mode"] == "REPLAN" and r["t_exec"] == 4)
            print(f"    REPLAN T_exec=4 success {replan['success_rate']:.0%}", flush=True)

    print(f"Training flow matching for {args.epochs} epochs.", flush=True)
    flow_policy, train_info, model, obs_norm, act_norm = _train_flow(device, args.seed, args.epochs)
    print(f"  velocity loss {train_info['loss_first']:.4f} -> {train_info['loss_last']:.4f}", flush=True)
    for steps in (1, 2, 4):
        flow_policy.sample_steps = steps
        summary = _quality(flow_policy, device, args.trials, args.seed)
        print(
            f"  flow NFE={steps}  collision {summary['collision_rate']:.0%}  success {summary['success_rate']:.0%}",
            flush=True,
        )
        quality.append({"sampler": "flow", "steps": steps, **{k: v for k, v in summary.items() if k != "paths"}, "paths": summary["paths"]})
        if steps in (1, 2):
            rows = _recovery(flow_policy, device, args.trials, args.seed)
            recovery.append({"sampler": "flow", "steps": steps, "rows": rows})
            replan = next(r for r in rows if r["mode"] == "REPLAN" and r["t_exec"] == 4)
            print(f"    REPLAN T_exec=4 success {replan['success_rate']:.0%}", flush=True)

    torch.save(
        {
            "denoiser": model.state_dict(),
            "obs_mean": obs_norm.mean.detach().cpu(),
            "obs_std": obs_norm.std.detach().cpu(),
            "act_mean": act_norm.mean.detach().cpu(),
            "act_std": act_norm.std.detach().cpu(),
            "epochs": args.epochs,
        },
        Path(args.output).parent / "flow_policy.pt",
    )

    report = {
        "suite": "diffusion_recovery_bench",
        "comparison": "flow_matching_vs_ddim",
        "version": __version__,
        "synthetic": True,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "platform": platform.platform(),
        },
        "origin_observation_dim": int(observation(START).shape[0]),
        "training": train_info,
        "latency": latency,
        "quality": quality,
        "recovery": recovery,
    }
    out = Path(args.output)
    md = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md.write_text(render_markdown(report), encoding="utf-8")
    write_figure(report, Path(args.figure))
    print(f"Wrote {out}", flush=True)
    print(f"Wrote {md}", flush=True)
    print(f"Wrote {args.figure}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
