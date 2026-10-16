"""Charts from a finished benchmark report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .task import GOAL, GOAL_TOL, OBSTACLE_CENTER, OBSTACLE_RADIUS, START


def write_figures(report: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        _latency(report, out_dir / "ddim_latency.png"),
        _multimodal(report, out_dir / "multimodal_separation.png"),
        _recovery(report, out_dir / "perturbation_recovery.png"),
    ]
    return paths


def _style(ax) -> None:
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)


def _latency(report: dict, path: Path) -> Path:
    rows = [row for row in report["latency"]["rows"] if row["condition"] == "action_head"]
    eye = [row for row in report["latency"]["rows"] if row["condition"] == "eye_and_action_head"]
    labels = [str(row["requested_steps"]) for row in rows]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    head_bars = ax.bar(x - width / 2, [row["p99_ms"] for row in rows], width, color="#2b5c8f", label="Action head P99")
    eye_bars = ax.bar(x + width / 2, [row["p99_ms"] for row in eye], width, color="#1b8a5a", label="Eye + action head P99")
    ax.axhline(20.0, color="#b4232c", linestyle="--", linewidth=1.4, label="50 Hz budget (20 ms)")
    ax.axhline(50.0, color="#c47b2b", linestyle=":", linewidth=1.4, label="20 Hz budget (50 ms)")
    peak = max(row["p99_ms"] for row in rows + eye)
    ax.set_ylim(0, max(24.0, peak * 1.25))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Requested DDIM steps K")
    ax.set_ylabel("P99 latency (ms)")
    ax.set_title("Denoising latency against the control period", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="upper left", frameon=False)
    _style(ax)
    for bars in (head_bars, eye_bars):
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.3, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _draw_world(ax) -> None:
    disk = plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS, color="#d9534f", alpha=0.35, label="Obstacle")
    goal = plt.Circle(GOAL, GOAL_TOL, color="#1b8a5a", alpha=0.35, label="Goal ball")
    ax.add_patch(disk)
    ax.add_patch(goal)
    ax.scatter([START[0]], [START[1]], c="#1c2830", s=28, zorder=4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.02, 0.30)
    ax.set_ylim(-0.18, 0.20)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    _style(ax)


def _multimodal(report: dict, path: Path) -> Path:
    paths = report["multimodal"]["paths"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    _draw_world(ax)
    for index, points in enumerate(paths["diffusion"]):
        arr = np.asarray(points)
        ax.plot(arr[:, 0], arr[:, 1], color="#2b5c8f", alpha=0.75, linewidth=1.4, label="Diffusion sample" if index == 0 else None)
    mlp = np.asarray(paths["mlp"])
    ax.plot(mlp[:, 0], mlp[:, 1], color="#d9534f", linewidth=2.2, label="MLP mean chunk")
    ax.set_title("Same observation, two policies", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _recovery(report: dict, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4))
    for ax, t_exec in zip(axes, ("4", "8")):
        example = report["recovery"]["examples"][t_exec]
        _draw_world(ax)
        cont = np.asarray(example["continue"])
        replan = np.asarray(example["replan"])
        ax.plot(cont[:, 0], cont[:, 1], color="#d9534f", linestyle="--", linewidth=2.0, label="CONTINUE")
        ax.plot(replan[:, 0], replan[:, 1], color="#1b8a5a", linewidth=2.0, label="REPLAN")
        if len(cont) > 4:
            ax.scatter([cont[4, 0]], [cont[4, 1]], c="#c47b2b", s=36, zorder=5, label="+6 cm")
        ax.set_title(f"T_exec = {t_exec}", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.suptitle("6 cm lateral jump", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
