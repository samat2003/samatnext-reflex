#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run seed sweeps for ReflexFormer temporal-hard experiments."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def load_eval_metrics(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["metrics"]


def load_batch1_p95(path: Path) -> float:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    for row in payload["rows"]:
        if int(row["batch"]) == 1:
            return float(row["latency_ms_p95"])
    raise ValueError(f"benchmark JSON has no batch=1 row: {path}")


def summarize(rows: list[dict]) -> dict:
    keys = [
        "accuracy",
        "macro_f1",
        "failsafe_recall",
        "safety_critical_miss_rate",
        "max_predicted_class_share",
        "batch1_p95_ms",
    ]
    summary = {}
    for key in keys:
        values = [float(r[key]) for r in rows if r.get(key) is not None]
        if not values:
            continue
        summary[key] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
        }
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run ReflexFormer temporal-hard seed sweep")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--preset", choices=["s", "m"], default="s")
    p.add_argument("--difficulty", default="temporal_hard")
    p.add_argument("--train-size", type=int, default=20_000)
    p.add_argument("--val-size", type=int, default=3_000)
    p.add_argument("--test-size", type=int, default=3_000)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--device", default="auto")
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.02)
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    p.add_argument("--out-csv", type=Path, default=RESULTS_DIR / "reflexformer_temporal_hard_seed_sweep.csv")
    p.add_argument("--out-json", type=Path, default=RESULTS_DIR / "reflexformer_temporal_hard_seed_sweep.json")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lr = args.lr if args.lr is not None else (2e-4 if args.preset == "s" else 1e-4)
    rows = []
    for seed in args.seeds:
        data_dir = ROOT / f"data/reflexformer_m_seed_{seed}_{args.difficulty}"
        out_dir = ROOT / f"checkpoints/reflexformer_{args.preset}_temporal_hard_seed_{seed}"
        eval_json = RESULTS_DIR / f"reflexformer_{args.preset}_temporal_hard_seed_{seed}_eval.json"
        bench_json = RESULTS_DIR / f"reflexformer_{args.preset}_temporal_hard_seed_{seed}_benchmark.json"
        run(
            [
                sys.executable,
                "experiments/reflexformer_m/make_dataset.py",
                "--seq-len",
                str(args.seq_len),
                "--train-size",
                str(args.train_size),
                "--val-size",
                str(args.val_size),
                "--test-size",
                str(args.test_size),
                "--seed",
                str(seed),
                "--difficulty",
                args.difficulty,
                "--out-dir",
                str(data_dir.relative_to(ROOT)),
            ]
        )
        run(
            [
                sys.executable,
                "experiments/reflexformer_m/train.py",
                "--data-dir",
                str(data_dir.relative_to(ROOT)),
                "--out-dir",
                str(out_dir.relative_to(ROOT)),
                "--preset",
                args.preset,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(lr),
                "--weight-decay",
                str(args.weight_decay),
                "--device",
                args.device,
                "--class-weights",
                "none",
                "--failsafe-weight",
                "1.0",
                "--label-smoothing",
                str(args.label_smoothing),
                "--patience",
                str(args.patience),
                "--grad-clip",
                str(args.grad_clip),
                "--normalize",
                "--add-deltas",
            ]
        )
        run(
            [
                sys.executable,
                "experiments/reflexformer_m/eval.py",
                "--data-dir",
                str(data_dir.relative_to(ROOT)),
                "--checkpoint",
                str((out_dir / "best.pt").relative_to(ROOT)),
                "--device",
                args.device,
                "--json-out",
                str(eval_json.relative_to(ROOT)),
            ]
        )
        metrics = load_eval_metrics(eval_json)
        batch1_p95 = None
        if args.benchmark:
            run(
                [
                    sys.executable,
                    "experiments/reflexformer_m/benchmark.py",
                    "--checkpoint",
                    str((out_dir / "best.pt").relative_to(ROOT)),
                    "--batch-sizes",
                    "1",
                    "--seq-len",
                    str(args.seq_len),
                    "--device",
                    args.device,
                    "--dtype",
                    args.dtype,
                    "--json",
                    str(bench_json.relative_to(ROOT)),
                ]
            )
            batch1_p95 = load_batch1_p95(bench_json)
        row = {
            "seed": seed,
            "preset": args.preset,
            "difficulty": args.difficulty,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "failsafe_recall": metrics["failsafe_recall"],
            "safety_critical_miss_rate": metrics["safety_critical_miss_rate"],
            "max_predicted_class_share": metrics["max_predicted_class_share"],
            "batch1_p95_ms": batch1_p95,
            "eval_json": str(eval_json),
        }
        rows.append(row)
        print(
            f"seed={seed} macro_f1={row['macro_f1']:.4f} "
            f"failsafe_recall={row['failsafe_recall']:.4f} "
            f"miss_rate={row['safety_critical_miss_rate']:.4f} "
            f"max_share={row['max_predicted_class_share']:.4f}"
        )

    summary = summarize(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "rows": rows, "summary": summary}, f, indent=2, default=str)
        f.write("\n")
    print(f"wrote CSV: {args.out_csv}")
    print(f"wrote JSON: {args.out_json}")
    print("mean/std:")
    for key, item in summary.items():
        print(f"  {key}: mean={item['mean']:.4f} std={item['std']:.4f} n={item['n']}")


if __name__ == "__main__":
    main()
