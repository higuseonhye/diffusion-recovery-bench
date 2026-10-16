"""Single entry point for latency, multimodal separation, and recovery."""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from . import __version__
from .experiments import probe_origin, run_latency, run_multimodal, run_recovery, train_policies
from .plots import write_figures
from .report import write_report
from .task import geometry_checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark toy diffusion-policy latency, mode separation, and replanning.")
    parser.add_argument("--synthetic", action="store_true", help="Required. Use the synthetic planar task.")
    parser.add_argument("--output", default="results/benchmark_report.json")
    parser.add_argument("--report", default="BENCHMARK_RESULTS.md")
    parser.add_argument("--figures", default="docs/figures")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--latency-runs", type=int, default=80)
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def collect_environment(device: torch.device) -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.synthetic:
        print("Pass --synthetic. This suite does not start a simulator.", file=sys.stderr)
        return 2

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"diffusion_recovery_bench {__version__} on {device}.", flush=True)

    print("Checking corridor geometry...", flush=True)
    geometry = geometry_checks()
    print(
        f"Mean path clearance {geometry['mean_clearance_m'] * 100:.1f} cm; "
        f"left corridor clearance {geometry['left_clearance_m'] * 100:.1f} cm.",
        flush=True,
    )

    print("Exp 1/3: DDIM latency...", flush=True)
    latency = run_latency(device, steps_list=[2, 5, 10, 16, 50], n_runs=args.latency_runs)
    for row in latency["rows"]:
        if row["condition"] != "action_head":
            continue
        flag = "pass" if row["meets_50hz_20ms"] else "FAIL"
        print(
            f"  K={row['requested_steps']:2d}  P50 {row['p50_ms']:.2f} ms  "
            f"P99 {row['p99_ms']:.2f} ms  50Hz {flag}",
            flush=True,
        )

    print(f"Training planar policies for {args.epochs} epochs...", flush=True)
    bundle = train_policies(device, seed=args.seed, epochs=args.epochs, diffusion_steps=args.diffusion_steps)
    probe = probe_origin(bundle, device, sample_steps=args.sample_steps)
    print(
        f"  origin probe: MLP collided={probe['mlp_collided']} ({probe['mlp_side']}), "
        f"diffusion collided={probe['diffusion_collided']} ({probe['diffusion_side']}).",
        flush=True,
    )

    print("Exp 2/3: multimodal rollouts...", flush=True)
    multimodal = run_multimodal(bundle, device, trials=args.trials, seed=args.seed, sample_steps=args.sample_steps)
    print(
        f"  MLP collision {multimodal['mlp']['collision_rate']:.0%}, "
        f"diffusion collision {multimodal['diffusion']['collision_rate']:.0%}, "
        f"diffusion success {multimodal['diffusion']['success_rate']:.0%}.",
        flush=True,
    )

    print("Exp 3/3: 6 cm CONTINUE vs REPLAN...", flush=True)
    recovery = run_recovery(bundle, device, trials=args.trials, seed=args.seed, sample_steps=args.sample_steps)
    for row in recovery["rows"]:
        print(
            f"  {row['mode']:8s} T_exec={row['t_exec']:2d}  "
            f"success {row['success_rate']:.0%}  "
            f"median error {row['median_final_error_cm']:.1f} cm",
            flush=True,
        )

    dataset = bundle["dataset"]
    report = {
        "suite": "diffusion_recovery_bench",
        "version": __version__,
        "schema_version": 1,
        "synthetic": True,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "argv": sys.argv if argv is None else ["diffusion_bench.runner", *argv],
        "environment": collect_environment(device),
        "geometry": geometry,
        "training": {
            "epochs": args.epochs,
            "diffusion_steps": args.diffusion_steps,
            "samples": dataset["samples"],
            "nominal_kept": dataset["nominal_kept"],
            "recovery_kept": dataset["recovery_kept"],
            "diffusion_loss_first": bundle["diffusion_loss_first"],
            "diffusion_loss_last": bundle["diffusion_loss_last"],
            "mlp_loss_first": bundle["mlp_loss_first"],
            "mlp_loss_last": bundle["mlp_loss_last"],
            "origin_probe": probe,
        },
        "latency": latency,
        "multimodal": multimodal,
        "recovery": recovery,
    }
    json_path = Path(args.output)
    markdown_path = Path(args.report)
    write_report(report, json_path, markdown_path)
    figures = write_figures(report, Path(args.figures))
    _save_checkpoint(bundle, json_path.parent / "planar_policy.pt", args)
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {markdown_path}", flush=True)
    for figure in figures:
        print(f"Wrote {figure}", flush=True)
    return 0


def _save_checkpoint(bundle: dict, path: Path, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "denoiser": bundle["denoiser"].state_dict(),
            "mlp": bundle["mlp"].state_dict(),
            "obs_mean": bundle["obs_norm"].mean.detach().cpu(),
            "obs_std": bundle["obs_norm"].std.detach().cpu(),
            "act_mean": bundle["act_norm"].mean.detach().cpu(),
            "act_std": bundle["act_norm"].std.detach().cpu(),
            "diffusion_steps": args.diffusion_steps,
            "sample_steps": args.sample_steps,
        },
        path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
