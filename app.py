"""Local board for the diffusion-recovery measurements and a live 0–8 cm replay."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from diffusion_bench.diffusion import DiffusionSchedule
from diffusion_bench.models import MLPBehaviorClone, TrajectoryDenoiser
from diffusion_bench.policies import DiffusionChunkPolicy, MLPChunkPolicy, Normalizer, rollout
from diffusion_bench.plots import _draw_world
from diffusion_bench.task import ACTION_DIM, N_ACTIONS, OBS_DIM, PRED_HORIZON

ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "results" / "benchmark_report.json"
CHECKPOINT_PATH = ROOT / "results" / "planar_policy.pt"

st.set_page_config(page_title="Physical Control Bench", page_icon="◎", layout="wide")


def _load_report() -> dict | None:
    if not REPORT_PATH.exists():
        return None
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


@st.cache_resource
def _load_policy(path_str: str):
    checkpoint = torch.load(path_str, map_location="cpu", weights_only=False)
    device = torch.device("cpu")
    denoiser = TrajectoryDenoiser(ACTION_DIM, OBS_DIM, hidden=128)
    mlp = MLPBehaviorClone(OBS_DIM, PRED_HORIZON, ACTION_DIM, hidden=256)
    denoiser.load_state_dict(checkpoint["denoiser"])
    mlp.load_state_dict(checkpoint["mlp"])
    denoiser.eval()
    mlp.eval()
    schedule = DiffusionSchedule(int(checkpoint["diffusion_steps"]), device=device)
    obs_norm = Normalizer(checkpoint["obs_mean"].numpy(), checkpoint["obs_std"].numpy()).to(device)
    act_norm = Normalizer(checkpoint["act_mean"].numpy(), checkpoint["act_std"].numpy()).to(device)
    diffusion = DiffusionChunkPolicy(
        denoiser, schedule, obs_norm, act_norm, int(checkpoint["sample_steps"]), device
    )
    clone = MLPChunkPolicy(mlp, obs_norm, act_norm, device)
    return diffusion, clone


def _replay(diffusion, centimeters: float, t_exec: int, seed: int = 7):
    disturb = centimeters / 100.0
    continued = rollout(
        diffusion,
        mode="continue",
        t_exec=N_ACTIONS,
        seed=seed,
        device=torch.device("cpu"),
        disturb_m=disturb,
        max_steps=N_ACTIONS,
    )
    replanned = rollout(
        diffusion,
        mode="replan",
        t_exec=t_exec,
        seed=seed,
        device=torch.device("cpu"),
        disturb_m=disturb,
        max_steps=N_ACTIONS * 2,
    )
    return continued, replanned


def _figure(continued, replanned, centimeters: float):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    _draw_world(ax)
    cont = continued["path"]
    replan = replanned["path"]
    ax.plot(cont[:, 0], cont[:, 1], color="#d9534f", linestyle="--", linewidth=2.0, label="CONTINUE")
    ax.plot(replan[:, 0], replan[:, 1], color="#1b8a5a", linewidth=2.0, label="REPLAN")
    ax.set_title(f"Lateral jump {centimeters:.1f} cm", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    return fig


def main() -> None:
    st.title("Physical control bench")
    st.caption("Planar diffusion policy. The table is the last measured run. The plot is a live replay of the saved weights.")
    report = _load_report()
    if report is None or not CHECKPOINT_PATH.exists():
        st.warning("Run `python -m diffusion_bench.runner --synthetic` in this directory first.")
        return

    multi = report["multimodal"]
    recovery_rows = report["recovery"]["rows"]
    head = [row for row in report["latency"]["rows"] if row["condition"] == "action_head" and row["requested_steps"] == 10]
    p99 = head[0]["p99_ms"] if head else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Action-head P99 at K=10", f"{p99:.2f} ms" if p99 is not None else "—")
    c2.metric("MLP collision", f"{100 * multi['mlp']['collision_rate']:.0f}%")
    c3.metric("Diffusion collision", f"{100 * multi['diffusion']['collision_rate']:.0f}%")
    replan4 = next(row for row in recovery_rows if row["mode"] == "REPLAN" and row["t_exec"] == 4)
    c4.metric("REPLAN T_exec=4 success", f"{100 * replan4['success_rate']:.0f}%")

    st.subheader("6 cm recovery table")
    st.dataframe(
        [
            {
                "mode": row["mode"],
                "T_exec": row["t_exec"],
                "success_%": round(100 * row["success_rate"], 1),
                "collision_%": round(100 * row["collision_rate"], 1),
                "median_error_cm": round(row["median_final_error_cm"], 2),
            }
            for row in recovery_rows
        ],
        width="stretch",
    )

    st.subheader("Replay")
    centimeters = st.slider("Lateral jump (cm)", min_value=0.0, max_value=8.0, value=6.0, step=0.5)
    t_exec = st.radio("REPLAN execute horizon", options=[4, 8], horizontal=True)
    diffusion, _mlp = _load_policy(str(CHECKPOINT_PATH))
    continued, replanned = _replay(diffusion, centimeters, int(t_exec))
    left, right = st.columns(2)
    left.pyplot(_figure(continued, replanned, centimeters))
    right.markdown(
        f"CONTINUE final error **{continued['final_error_m'] * 100:.1f} cm**, "
        f"collision **{continued['collided']}**.\n\n"
        f"REPLAN final error **{replanned['final_error_m'] * 100:.1f} cm**, "
        f"collision **{replanned['collided']}**."
    )
    st.caption(
        "Axes are meters. A slider value of 6 cm is 0.06 m, which is 60 mm on the "
        "240 mm path (25% of the travel). That label converts meters to centimeters. "
        "It does not magnify a smaller motion, and it is not a sub-millimeter surgical amplitude. "
        "CONTINUE integrates the chunk sampled at the origin. "
        "REPLAN samples a new chunk from the state after the jump. "
        "A single slider replay is not the table above."
    )


main()
