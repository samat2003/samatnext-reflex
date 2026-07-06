#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Small ReflexFormer-M ablation runner."""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.reflexformer_m.model import CLASS_NAMES, ReflexFormerMClassifier, count_parameters
from experiments.reflexformer_m.train import classification_report, load_split, resolve_device


def crop_seq(x: torch.Tensor, seq_len: int) -> torch.Tensor:
    return x[:, -seq_len:, :].contiguous()


@torch.inference_mode()
def eval_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    ys, preds = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        preds.append(logits.argmax(dim=-1).cpu())
        ys.append(y)
    return classification_report(torch.cat(ys), torch.cat(preds), len(CLASS_NAMES))


def batch1_latency_ms(model: nn.Module, seq_len: int, feature_dim: int, device: torch.device, dtype: torch.dtype) -> float:
    x = torch.randn(1, seq_len, feature_dim, device=device, dtype=dtype)
    model.eval()
    with torch.inference_mode():
        for _ in range(20):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(80):
                _ = model(x)
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / 80
        t0 = time.perf_counter()
        for _ in range(40):
            _ = model(x)
        return (time.perf_counter() - t0) * 1000.0 / 40


def run_one(args: argparse.Namespace, cfg: dict, train_x: torch.Tensor, train_y: torch.Tensor, val_x: torch.Tensor, val_y: torch.Tensor) -> dict:
    device = resolve_device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tx = crop_seq(train_x[: args.max_train], cfg["seq_len"])
    ty = train_y[: args.max_train]
    vx = crop_seq(val_x[: args.max_val], cfg["seq_len"])
    vy = val_y[: args.max_val]

    train_loader = DataLoader(TensorDataset(tx, ty), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(vx, vy), batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = ReflexFormerMClassifier(
        seq_len=cfg["seq_len"],
        feature_dim=tx.shape[2],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        mlp_ratio=args.mlp_ratio,
        dropout=0.1,
        pooling=cfg["pooling"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.03)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    started = time.perf_counter()
    for _ in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    metrics = eval_model(model, val_loader, device)
    model = model.to(dtype=dtype)
    latency = batch1_latency_ms(model, cfg["seq_len"], tx.shape[2], device, dtype)
    return {
        **cfg,
        "parameter_count": count_parameters(model),
        "val_macro_f1": metrics["macro_f1"],
        "val_accuracy": metrics["accuracy"],
        "batch1_latency_ms": latency,
        "examples_per_sec": 1000.0 / latency,
        "seconds": time.perf_counter() - started,
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run small ReflexFormer-M ablations")
    p.add_argument("--data-dir", type=Path, default=Path("data/reflexformer_m"))
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-train", type=int, default=2048)
    p.add_argument("--max-val", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--mlp-ratio", type=float, default=2.0)
    p.add_argument("--quick", action="store_true", help="Run a small representative subset.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    train_x, train_y = load_split(args.data_dir, "train")
    val_x, val_y = load_split(args.data_dir, "val")
    grids = {
        "seq_len": [32, 64, 128],
        "n_layers": [4, 8, 10],
        "d_model": [192, 256, 384],
        "pooling": ["cls", "mean"],
    }
    configs = [
        {"seq_len": s, "n_layers": l, "d_model": d, "n_heads": 6 if d % 6 == 0 else 4, "pooling": p}
        for s, l, d, p in itertools.product(grids["seq_len"], grids["n_layers"], grids["d_model"], grids["pooling"])
    ]
    if args.quick:
        configs = [
            {"seq_len": 32, "n_layers": 4, "d_model": 192, "n_heads": 6, "pooling": "cls"},
            {"seq_len": 64, "n_layers": 8, "d_model": 256, "n_heads": 4, "pooling": "cls"},
            {"seq_len": 128, "n_layers": 10, "d_model": 384, "n_heads": 6, "pooling": "cls"},
            {"seq_len": 128, "n_layers": 10, "d_model": 384, "n_heads": 6, "pooling": "mean"},
        ]

    print("| seq_len | layers | d_model | pooling | params | val_macro_f1 | batch1_ms | examples_sec |")
    print("|---:|---:|---:|---|---:|---:|---:|---:|")
    for cfg in configs:
        result = run_one(args, cfg, train_x, train_y, val_x, val_y)
        print(
            f"| {result['seq_len']} | {result['n_layers']} | {result['d_model']} | {result['pooling']} | "
            f"{result['parameter_count']} | {result['val_macro_f1']:.4f} | "
            f"{result['batch1_latency_ms']:.4f} | {result['examples_per_sec']:.2f} |"
        )


if __name__ == "__main__":
    main()
