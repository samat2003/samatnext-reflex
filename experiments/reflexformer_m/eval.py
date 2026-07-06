#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Evaluate ReflexFormer-M on the synthetic test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.reflexformer_m.model import CLASS_NAMES, apply_preset
from experiments.reflexformer_m.train import build_model, classification_report, load_split, prepare_features, resolve_device


@torch.inference_mode()
def evaluate_checkpoint(args: argparse.Namespace) -> dict:
    device = resolve_device(args.device)
    x_raw, y = load_split(args.data_dir, "test")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = argparse.Namespace(**checkpoint.get("args", {}))
    for name, default in {
        "d_model": 384,
        "n_layers": 10,
        "n_heads": 6,
        "mlp_ratio": 3.0,
        "dropout": 0.1,
        "pooling": "cls",
    }.items():
        if not hasattr(ckpt_args, name):
            setattr(ckpt_args, name, default)
    add_deltas = args.add_deltas or bool(getattr(ckpt_args, "add_deltas", False))
    normalizer = checkpoint.get("normalizer") if args.normalize else None
    x = prepare_features(x_raw, normalizer, add_deltas)
    model = build_model(ckpt_args, feature_dim=x.shape[2], seq_len=x.shape[1]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    ys, preds = [], []
    for batch_x, batch_y in loader:
        logits = model(batch_x.to(device))
        pred = logits.argmax(dim=-1).cpu()
        ys.append(batch_y)
        preds.append(pred)
    y_true = torch.cat(ys)
    y_pred = torch.cat(preds)
    metrics = classification_report(y_true, y_pred, len(CLASS_NAMES))
    metrics["invalid_output_rate"] = 0.0
    low = CLASS_NAMES.index("LOW_RISK")
    caution = CLASS_NAMES.index("CAUTION")
    avoid = CLASS_NAMES.index("AVOIDANCE_NEEDED")
    failsafe = CLASS_NAMES.index("FAILSAFE_NEEDED")
    false_safe_on_failsafe = int(((y_true == failsafe) & ((y_pred == low) | (y_pred == caution))).sum().item())
    false_low_on_critical = int((((y_true == avoid) | (y_true == failsafe)) & (y_pred == low)).sum().item())
    critical_mask = (y_true == avoid) | (y_true == failsafe)
    critical_miss = critical_mask & ((y_pred == low) | (y_pred == caution))
    metrics["false_safe_on_failsafe_needed"] = false_safe_on_failsafe
    metrics["false_low_risk_on_critical"] = false_low_on_critical
    metrics["failsafe_recall"] = metrics["per_class"][failsafe]["recall"]
    metrics["safety_critical_miss_rate"] = (
        critical_miss.float().sum().item() / critical_mask.float().sum().item()
        if critical_mask.any()
        else 0.0
    )
    return metrics


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate ReflexFormer-M")
    p.add_argument("--data-dir", type=Path, default=Path("data/reflexformer_m"))
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/reflexformer_m/best.pt"))
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--normalize", action="store_true", default=True)
    p.add_argument("--no-normalize", dest="normalize", action="store_false")
    p.add_argument("--add-deltas", action="store_true")
    p.add_argument("--json-out", type=Path, default=None)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    metrics = evaluate_checkpoint(args)
    print(f"test accuracy: {metrics['accuracy']:.4f}")
    print(f"macro F1: {metrics['macro_f1']:.4f}")
    print(f"invalid output rate: {metrics['invalid_output_rate']:.4f}")
    print(f"max predicted class share: {metrics['max_predicted_class_share']:.4f}")
    print(f"FAILSAFE_NEEDED recall: {metrics['failsafe_recall']:.4f}")
    print(f"false SAFE on FAILSAFE_NEEDED: {metrics['false_safe_on_failsafe_needed']}")
    print(f"false LOW_RISK on AVOIDANCE_NEEDED/FAILSAFE_NEEDED: {metrics['false_low_risk_on_critical']}")
    print(f"safety-critical miss rate: {metrics['safety_critical_miss_rate']:.4f}")
    print("class distribution:", {name: metrics["true_counts"][i] for i, name in enumerate(CLASS_NAMES)})
    print("predicted distribution:", {name: metrics["pred_counts"][i] for i, name in enumerate(CLASS_NAMES)})
    print("per-class precision/recall/F1:")
    for name, m in zip(CLASS_NAMES, metrics["per_class"]):
        print(f"  {name}: precision={m['precision']:.4f} recall={m['recall']:.4f} f1={m['f1']:.4f} support={m['support']}")
    print("confusion matrix rows=true cols=pred:")
    print("  " + " ".join(CLASS_NAMES))
    for name, row in zip(CLASS_NAMES, metrics["confusion_matrix"]):
        print(f"  {name}: {row}")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(args.checkpoint),
            "data_dir": str(args.data_dir),
            "metrics": metrics,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        print(f"wrote JSON: {args.json_out}")


if __name__ == "__main__":
    main()
