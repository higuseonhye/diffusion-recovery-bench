"""JSON, CSV, and Markdown writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path


NOTES = [
    "The planar task is a point mass in meters. Isaac Sim, contact dynamics, and a 7-DoF arm were not executed.",
    "Exp 1 times a randomly initialized temporal-conv denoiser (action dim 7, observation dim 128, horizon 16). The eye row adds a random 128x128 conv stub. Neither row is the surgical-vision serving benchmark.",
    "Exp 2 and Exp 3 train that same denoiser family on synthetic left/right corridors. The MLP baseline is MSE regression on the same chunks.",
    "CONTINUE denoises once and integrates that chunk. REPLAN denoises again every T_exec steps from the measured state. The 6 cm event is an instantaneous +y state jump after 4 steps.",
    "Success means the final point is inside the 2 cm goal ball and the polyline never intersects the 4 cm obstacle.",
    "steps_to_goal is a control-cycle count. The 20 Hz seconds column multiplies that count by 50 ms; it is not a measured wall-clock duration.",
    "This is a software benchmark, not a clinical validation and not a surgical device.",
]


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return f"{float(value):.{digits}f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def _pct(rate: float) -> str:
    return f"{100.0 * rate:.1f}%"


def render_markdown(report: dict) -> str:
    env = report["environment"]
    latency = report["latency"]
    multi = report["multimodal"]
    recovery = report["recovery"]
    train = report["training"]
    head_rows = [row for row in latency["rows"] if row["condition"] == "action_head"]
    eye_rows = [row for row in latency["rows"] if row["condition"] == "eye_and_action_head"]
    head_ok = sum(1 for row in head_rows if row["meets_50hz_20ms"])
    mlp = multi["mlp"]
    diff = multi["diffusion"]

    latency_table = []
    for row in latency["rows"]:
        latency_table.append(
            [
                row["condition"],
                str(row["requested_steps"]),
                str(row["executed_steps"]),
                _fmt(row["p50_ms"], 2),
                _fmt(row["p95_ms"], 2),
                _fmt(row["p99_ms"], 2),
                "pass" if row["meets_50hz_20ms"] else "fail",
                "pass" if row["meets_20hz_50ms"] else "fail",
            ]
        )
    recovery_table = []
    for row in recovery["rows"]:
        recovery_table.append(
            [
                row["mode"],
                str(row["t_exec"]),
                str(row["trials"]),
                _pct(row["success_rate"]),
                _pct(row["collision_rate"]),
                _fmt(row["median_final_error_cm"], 2),
                _fmt(row["median_steps_to_goal"], 1),
                _fmt(row["median_time_to_goal_at_20hz_s"], 2),
            ]
        )

    lines = [
        "# Diffusion recovery benchmark results",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "Synthetic proxy for action-chunk diffusion latency, multimodal corridor selection, "
        "and receding-horizon recovery after a 6 cm lateral jump. "
        f"The action head met the 20 ms / 50 Hz P99 budget on {head_ok} of {len(head_rows)} DDIM step settings. "
        f"MLP collision rate was {_pct(mlp['collision_rate'])}; "
        f"diffusion collision rate was {_pct(diff['collision_rate'])}.",
        "",
        "## Environment",
        "",
        _table(
            ["python", "torch", "numpy", "device", "torch threads", "platform"],
            [[
                env["python"],
                env["torch"],
                env["numpy"],
                env["device"],
                str(env["torch_num_threads"]),
                env["platform"],
            ]],
        ),
        "",
        "## Task",
        "",
        _table(
            ["start_m", "goal_m", "obstacle_radius_m", "goal_tol_m", "perturbation_m", "disturb_after_steps"],
            [[
                "[0.00, 0.00]",
                "[0.24, 0.00]",
                "0.04",
                "0.02",
                "0.06",
                str(recovery["disturb_after_steps"]),
            ]],
        ),
        "",
        f"Dataset windows: {train['samples']} "
        f"(nominal corridors {train['nominal_kept']}, recovery bridges {train['recovery_kept']}). "
        f"Diffusion epsilon loss {train['diffusion_loss_first']:.4f} -> {train['diffusion_loss_last']:.4f}. "
        f"MLP MSE {train['mlp_loss_first']:.4f} -> {train['mlp_loss_last']:.4f}. "
        f"Trained for {train['epochs']} epochs, diffusion schedule length {train['diffusion_steps']}.",
        "",
        "## Exp 1 — DDIM latency",
        "",
        "Batch 1. P99 is over the timed runs after warmup. "
        f"50 Hz requires P99 < {latency['budget_50hz_ms']:.0f} ms. "
        f"20 Hz requires P99 < {latency['budget_20hz_ms']:.0f} ms. "
        "Weights are randomly initialized; latency does not depend on fitting the corridors.",
        "",
        _table(
            ["condition", "K", "executed", "p50_ms", "p95_ms", "p99_ms", "50Hz", "20Hz"],
            latency_table,
        ),
        "",
        "## Exp 2 — Multimodal separation",
        "",
        "Same start state, two demonstration sides. MLP regresses the chunk with MSE. "
        f"Diffusion draws {diff['trials']} DDIM samples (K={multi['sample_steps']}). "
        "A collapsed path is one whose mean y stays inside ±1 cm.",
        "",
        _table(
            ["method", "trials", "collision", "success", "median_error_cm", "left", "right", "collapsed", "median_clearance_cm"],
            [
                [
                    "mlp_bc",
                    str(mlp["trials"]),
                    _pct(mlp["collision_rate"]),
                    _pct(mlp["success_rate"]),
                    _fmt(mlp["median_final_error_cm"], 2),
                    _pct(mlp["left_fraction"]),
                    _pct(mlp["right_fraction"]),
                    _pct(mlp["collapsed_fraction"]),
                    _fmt(mlp["min_clearance_cm"], 2),
                ],
                [
                    "diffusion",
                    str(diff["trials"]),
                    _pct(diff["collision_rate"]),
                    _pct(diff["success_rate"]),
                    _fmt(diff["median_final_error_cm"], 2),
                    _pct(diff["left_fraction"]),
                    _pct(diff["right_fraction"]),
                    _pct(diff["collapsed_fraction"]),
                    _fmt(diff["min_clearance_cm"], 2),
                ],
            ],
        ),
        "",
        "## Exp 3 — 6 cm jump, CONTINUE vs REPLAN",
        "",
        "The same diffusion policy is executed two ways. "
        f"After {recovery['disturb_after_steps']} steps the state jumps +6 cm in y. "
        "CONTINUE keeps the original chunk. REPLAN reads the new state every T_exec steps. "
        "Median steps and the 20 Hz time column are computed on successes only.",
        "",
        _table(
            ["mode", "T_exec", "trials", "success", "collision", "median_error_cm", "median_steps", "median_s_at_20hz"],
            recovery_table,
        ),
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in NOTES)
    lines.append("")
    del eye_rows
    return "\n".join(lines)


def write_csv(report: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    latency_path = directory / "latency.csv"
    with latency_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "condition",
                "requested_steps",
                "executed_steps",
                "p50_ms",
                "p95_ms",
                "p99_ms",
                "meets_50hz_20ms",
                "meets_20hz_50ms",
            ],
        )
        writer.writeheader()
        for row in report["latency"]["rows"]:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    multi = report["multimodal"]
    with (directory / "multimodal.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "method",
            "trials",
            "collision_rate",
            "success_rate",
            "median_final_error_cm",
            "left_fraction",
            "right_fraction",
            "collapsed_fraction",
            "min_clearance_cm",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in ("mlp", "diffusion"):
            row = {"method": name, **{key: multi[name][key] for key in fields if key != "method"}}
            writer.writerow(row)

    with (directory / "recovery.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "mode",
            "t_exec",
            "trials",
            "success_rate",
            "collision_rate",
            "median_final_error_cm",
            "mean_final_error_cm",
            "median_steps_to_goal",
            "median_time_to_goal_at_20hz_s",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["recovery"]["rows"]:
            writer.writerow({key: row.get(key) for key in fields})


def write_report(report: dict, json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    write_csv(report, json_path.parent)
