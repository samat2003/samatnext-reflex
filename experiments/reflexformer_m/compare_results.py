#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Print markdown summaries for ReflexFormer temporal-hard results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> object | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def metric_row(name: str, metrics: dict) -> dict:
    return {
        "name": name,
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "failsafe_recall": metrics.get("failsafe_recall")
        or metrics.get("per_class", [{}, {}, {}, {}])[3].get("recall"),
        "safety_critical_miss_rate": metrics.get("safety_critical_miss_rate"),
        "max_predicted_class_share": metrics.get("max_predicted_class_share"),
    }


def print_table(rows: list[dict]) -> None:
    print("| Model | Accuracy | Macro F1 | FAILSAFE Recall | Safety-Critical Miss Rate | Max Pred Share |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        def fmt(value: object) -> str:
            return "n/a" if value is None else f"{float(value):.4f}"

        print(
            f"| {row['name']} | {fmt(row.get('accuracy'))} | {fmt(row.get('macro_f1'))} | "
            f"{fmt(row.get('failsafe_recall'))} | {fmt(row.get('safety_critical_miss_rate'))} | "
            f"{fmt(row.get('max_predicted_class_share'))} |"
        )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare ReflexFormer temporal-hard result files")
    p.add_argument("--baseline-json", type=Path, default=Path("results/reflexformer_m_baselines.json"))
    p.add_argument("--s-eval-json", type=Path, default=Path("results/reflexformer_s_v1_eval.json"))
    p.add_argument("--m-eval-json", type=Path, default=Path("results/reflexformer_m_stable_eval.json"))
    p.add_argument("--seed-sweep-json", type=Path, default=Path("results/reflexformer_temporal_hard_seed_sweep.json"))
    return p


def main() -> None:
    args = build_argparser().parse_args()
    rows = []
    baselines = load_json(args.baseline_json)
    if isinstance(baselines, list):
        for item in baselines:
            rows.append(metric_row(item.get("name", "baseline"), item))
    s_eval = load_json(args.s_eval_json)
    if isinstance(s_eval, dict):
        rows.append(metric_row("ReflexFormer-S", s_eval["metrics"]))
    m_eval = load_json(args.m_eval_json)
    if isinstance(m_eval, dict):
        rows.append(metric_row("ReflexFormer-M", m_eval["metrics"]))
    if rows:
        print("## Model Comparison")
        print_table(rows)
    else:
        print("No baseline/eval result files found.")

    sweep = load_json(args.seed_sweep_json)
    if isinstance(sweep, dict) and sweep.get("summary"):
        print()
        print("## Seed Sweep Summary")
        print("| Metric | Mean | Std | N |")
        print("|---|---:|---:|---:|")
        for key, item in sweep["summary"].items():
            print(f"| {key} | {item['mean']:.4f} | {item['std']:.4f} | {item['n']} |")


if __name__ == "__main__":
    main()
