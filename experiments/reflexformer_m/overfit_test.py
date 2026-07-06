#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Quick overfit test for ReflexFormer presets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.reflexformer_m.model import apply_preset, count_parameters
from experiments.reflexformer_m.train import build_model, fit_normalizer, load_split, prepare_features, resolve_device


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Overfit ReflexFormer on 256 examples")
    p.add_argument("--data-dir", type=Path, default=Path("data/reflexformer_m"))
    p.add_argument("--preset", choices=["s", "m"], default="s")
    p.add_argument("--device", default="auto")
    p.add_argument("--examples", type=int, default=256)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--normalize", action="store_true", default=True)
    p.add_argument("--no-normalize", dest="normalize", action="store_false")
    p.add_argument("--norm-clip", type=float, default=8.0)
    p.add_argument("--add-deltas", action="store_true", default=True)
    p.add_argument("--no-add-deltas", dest="add_deltas", action="store_false")
    return p


def main() -> None:
    args = apply_preset(build_argparser().parse_args())
    args.dropout = 0.0
    args.pooling = "cls"
    device = resolve_device(args.device)
    x_raw, y = load_split(args.data_dir, "train")
    x_raw = x_raw[: args.examples]
    y = y[: args.examples]
    normalizer = fit_normalizer(x_raw, args.norm_clip) if args.normalize else None
    x = prepare_features(x_raw, normalizer, args.add_deltas)
    model = build_model(args, x.shape[2], x.shape[1]).to(device)
    print(f"preset={args.preset} params={count_parameters(model):,} examples={len(x)} feature_dim={x.shape[2]}")
    loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    step = 0
    while step < args.steps:
        model.train()
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = loss_fn(model(bx), by)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            if step >= args.steps:
                break
    model.eval()
    preds = []
    with torch.inference_mode():
        for bx, _ in DataLoader(TensorDataset(x, y), batch_size=args.batch_size):
            preds.append(model(bx.to(device)).argmax(dim=-1).cpu())
    acc = (torch.cat(preds) == y).float().mean().item()
    print(f"overfit train accuracy: {acc:.4f}")
    if acc < 0.95:
        raise SystemExit("FAIL: overfit accuracy below 0.95")
    print("PASS")


if __name__ == "__main__":
    main()
